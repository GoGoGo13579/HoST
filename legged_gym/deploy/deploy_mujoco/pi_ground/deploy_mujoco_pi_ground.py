import sys
import os
import argparse
import time
import numpy as np
import yaml
import mujoco
import mujoco.viewer
import threading

# ONNX推理
try:
    import onnxruntime as ort
except ImportError:
    print("请安装onnxruntime: pip install onnxruntime")
    sys.exit(1)

# 全局控制
policy_started = False


def wait_for_enter():
    """等待启动策略"""
    global policy_started
    input("按回车键开始执行策略...")
    policy_started = True
    print("策略已启动！")


def load_config(config_path):
    """加载配置"""
    with open(config_path, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def get_gravity_orientation(quaternion):
    """获取重力方向 (从四元数计算投影重力方向)"""
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def get_observations(data, num_joints, last_actions, action_rescale, obs_scales,
                     qpos_start, qvel_start, has_free_base):
    """获取观测值 (与训练时Pi的obs结构对齐: 3+3+12+12+12+1=43)"""
    # 关节状态
    q = data.qpos[qpos_start:qpos_start+num_joints]
    dq = data.qvel[qvel_start:qvel_start+num_joints]

    # 基座状态
    if has_free_base:
        quat = data.qpos[3:7]
        ang_vel = data.qvel[3:6]
    else:
        quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        ang_vel = np.zeros(3, dtype=np.float32)

    # 重力方向
    gravity_ori = get_gravity_orientation(quat)

    # 拼接观测
    obs = np.concatenate([
        ang_vel * obs_scales['ang_vel'],
        gravity_ori,
        q * obs_scales['dof_pos'],
        dq * obs_scales['dof_vel'],
        last_actions,
        [action_rescale]
    ])

    return obs.astype(np.float32)


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """PD控制"""
    return (target_q - q) * kp + (target_dq - dq) * kd


def main():
    parser = argparse.ArgumentParser(description='部署Pi地面运动模型 (MuJoCo)')
    parser.add_argument('--model_path', type=str, default=None,
                       help='ONNX模型文件路径')
    parser.add_argument('--config', type=str, default=None,
                       help='配置文件路径，默认使用脚本同目录下的 pi_ground_config.yaml')
    args = parser.parse_args()

    # 配置文件路径（相对脚本目录）
    config_path = args.config or os.path.join(os.path.dirname(__file__), 'pi_ground_config.yaml')
    # 加载配置
    config = load_config(config_path)
    config_dir = os.path.dirname(os.path.abspath(config_path))

    # 模型路径
    model_path = args.model_path or config['model']['default_model_path']
    if not os.path.isabs(model_path):
        model_path = os.path.normpath(os.path.join(config_dir, model_path))
    if not os.path.exists(model_path):
        print(f"模型文件未找到: {model_path}")
        print("请将训练得到的Pi地面策略导出为ONNX并放置到配置指定路径，或使用 --model_path 指定。")
        sys.exit(1)

    # 加载ONNX模型
    print(f"加载ONNX模型: {model_path}")
    ort_session = ort.InferenceSession(model_path)

    # 设置MuJoCo
    xml_path = config['simulation']['xml_path']
    if not os.path.isabs(xml_path):
        xml_path = os.path.normpath(os.path.join(config_dir, xml_path))
    if not os.path.exists(xml_path):
        print(f"机器人MJCF/URDF文件未找到: {xml_path}")
        print("请先将Pi的URDF转换为MuJoCo MJCF XML，或提供MuJoCo可解析的文件路径。")
        sys.exit(1)

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # 输出关节名称
    print("\n关节名称:")
    for i in range(model.njnt):
        joint_name = model.joint(i).name
        print(f"关节 {i}: {joint_name}")

    # 仿真参数
    model.opt.timestep = config['simulation']['simulation_dt']
    # 更稳定的设置
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
    model.opt.iterations = max(model.opt.iterations, 50)
    if hasattr(model.opt, 'ls_iterations'):
        model.opt.ls_iterations = max(model.opt.ls_iterations, 8)
    control_decimation = config['simulation']['control_decimation']
    num_joints = config['robot']['num_joints']

    # 控制与观测配置
    action_scale = config['control']['action_scale']
    tau_limit = float(config['control'].get('tau_limit', 40.0))
    obs_scales = config['observation_scales']

    # PD参数 (长度需与关节数匹配)
    kp = np.array(config['control']['kp'], dtype=np.float32)
    kd = np.array(config['control']['kd'], dtype=np.float32)
    if len(kp) != num_joints or len(kd) != num_joints:
        print("kp/kd长度与num_joints不一致，请检查配置文件。")
        sys.exit(1)

    # 计算基座自由度与关节起始索引
    has_free_base = (model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE)
    qpos_start = model.nq - num_joints
    qvel_start = model.nv - num_joints

    # 初始姿态
    initial_pose = config['initial_pose']
    if has_free_base:
        data.qpos[0:3] = initial_pose['position']
        data.qpos[3:7] = initial_pose['orientation']  # [w,x,y,z]
    # 初始化关节角
    init_joint = np.array(initial_pose['joint_angles'], dtype=np.float32)
    if len(init_joint) != num_joints:
        print("initial_pose.joint_angles长度与num_joints不一致，请检查配置文件。")
        sys.exit(1)
    data.qpos[qpos_start:qpos_start+num_joints] = init_joint
    data.qvel[:] = 0
    # 额外阻尼：增加基座与关节的速度阻尼，降低爆加速度风险
    try:
        if model.dof_damping is not None and model.dof_damping.shape[0] == model.nv:
            extra_base_damp = 0.5 if has_free_base else 0.0
            extra_joint_damp = 0.05
            # 基座前6个DOF阻尼
            if has_free_base and model.nv >= (6 + num_joints):
                model.dof_damping[0:6] = np.maximum(model.dof_damping[0:6], extra_base_damp)
            # 关节段阻尼
            model.dof_damping[qvel_start:qvel_start+num_joints] = np.maximum(
                model.dof_damping[qvel_start:qvel_start+num_joints], extra_joint_damp
            )
    except Exception:
        pass
    mujoco.mj_forward(model, data)

    # 控制变量
    sim_counter = 0
    target_joint_pos = data.qpos[qpos_start:qpos_start+num_joints].copy()

    # 历史观测缓冲区
    history_length = int(config['observations']['history_length'])
    obs_dim_single = int(config['observations']['num_single_obs'])
    obs_dim_total = history_length * obs_dim_single

    # 初始化缓冲区
    obs_history = np.zeros((history_length, obs_dim_single), dtype=np.float32)
    last_actions = np.zeros(num_joints, dtype=np.float32)

    # 等待用户输入
    input_thread = threading.Thread(target=wait_for_enter, daemon=True)
    input_thread.start()

    print("开始仿真...")
    print(f"观测维度: 单步{obs_dim_single}维, 总计{obs_dim_total}维")
    print("按Enter键启动策略，按ESC键退出")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            # PD控制
            if policy_started:
                tau = pd_control(
                    target_joint_pos,
                    data.qpos[qpos_start:qpos_start+num_joints],
                    kp,
                    np.zeros(num_joints),
                    data.qvel[qvel_start:qvel_start+num_joints],
                    kd
                )
            else:
                tau = np.zeros(num_joints)

            # 施加力矩：若存在 actuator 则写入 ctrl，否则直接作用到 DOF 力数组
            # 力矩限幅
            tau = np.clip(tau, -tau_limit, tau_limit)
            if model.nu >= num_joints:
                data.ctrl[:num_joints] = tau
            else:
                dof_start = model.nv - num_joints
                data.qfrc_applied[:] = 0
                data.qfrc_applied[dof_start:dof_start+num_joints] = tau

            # 仿真步进
            mujoco.mj_step(model, data)
            sim_counter += 1

            # 策略推理
            if sim_counter % control_decimation == 0:
                # 获取当前观测
                current_obs = get_observations(
                    data, num_joints, last_actions, action_scale, obs_scales,
                    qpos_start, qvel_start, has_free_base
                )

                # 更新历史
                obs_history[:-1] = obs_history[1:]
                obs_history[-1] = current_obs

                # 推理
                if policy_started:
                    obs_input = obs_history.flatten().reshape(1, -1)
                    # 运行推理
                    try:
                        policy_output = ort_session.run(None, {'actor_obs': obs_input})
                    except Exception:
                        # 兼容不同导出名
                        policy_output = ort_session.run(None, {ort_session.get_inputs()[0].name: obs_input})
                    actions = policy_output[0].flatten()
                else:
                    actions = np.zeros(num_joints, dtype=np.float32)

                # 更新目标位置
                target_joint_pos = data.qpos[qpos_start:qpos_start+num_joints] + actions * action_scale

                # 更新历史动作
                last_actions = actions.copy()

            viewer.sync()
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    print("仿真结束")


if __name__ == "__main__":
    main()



