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
    """获取重力方向"""
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def get_observations(data, num_joints, last_actions, action_rescale, obs_scales):
    """获取观测值"""
    # 关节状态
    q = data.qpos[7:7+num_joints]
    dq = data.qvel[6:6+num_joints]
    
    # 基座状态
    quat = data.qpos[3:7]
    ang_vel = data.qvel[3:6]
    
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
    parser = argparse.ArgumentParser(description='部署G1地面运动模型')
    parser.add_argument('--model_path', type=str, default=None, 
                       help='ONNX模型文件路径')
    parser.add_argument('--config', type=str, 
                       default='/home/shenlan/HoST/legged_gym/deploy/deploy_mujoco/g1_platform/g1_platfrom_config.yaml',
                       help='配置文件路径')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 模型路径
    model_path = args.model_path or config['model']['default_model_path']
    if not os.path.exists(model_path):
        print(f"模型文件未找到: {model_path}")
        sys.exit(1)
    
    # 加载ONNX模型
    print(f"加载ONNX模型: {model_path}")
    ort_session = ort.InferenceSession(model_path)
    
    # 设置MuJoCo
    xml_path = config['simulation']['xml_path']
    if not os.path.exists(xml_path):
        print(f"XML文件未找到: {xml_path}")
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
    control_decimation = config['simulation']['control_decimation']
    num_joints = config['robot']['num_joints']
    kp = np.array(config['control']['kp'], dtype=np.float32)
    kd = np.array(config['control']['kd'], dtype=np.float32)
    action_scale = config['control']['action_scale']
    obs_scales = config['observation_scales']

    # 初始姿态
    initial_pose = config['initial_pose']
    data.qpos[:3] = initial_pose['position']
    data.qpos[3:7] = initial_pose['orientation']
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    
    # 控制变量
    sim_counter = 0
    target_joint_pos = np.array(initial_pose['joint_angles'], dtype=np.float32)
    
    # 历史观测缓冲区
    history_length = 6
    obs_dim_single = 76
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
                    data.qpos[7:7+num_joints], 
                    kp,
                    np.zeros(num_joints), 
                    data.qvel[6:6+num_joints], 
                    kd
                )
            else:
                tau = np.zeros(num_joints)
            data.ctrl[:] = tau
            data.ctrl[17] = 0.0
            data.ctrl[22] = 0.0
            
            # 仿真步进
            mujoco.mj_step(model, data)
            sim_counter += 1
            
            # 策略推理
            if sim_counter % control_decimation == 0:
                # 获取当前观测
                current_obs = get_observations(data, num_joints, last_actions, action_scale, obs_scales)
                
                # 更新历史
                obs_history[:-1] = obs_history[1:]
                obs_history[-1] = current_obs
                
                # 推理
                if policy_started:
                    obs_input = obs_history.flatten().reshape(1, -1)
                    # obs_input = np.flip(obs_input, axis=1)
                    # 运行推理
                    policy_output = ort_session.run(None, {'actor_obs': obs_input})
                    # policy_output = ort_session.run(None, {'input': obs_input})
                    actions = policy_output[0].flatten()  # 移除批次维度
                else:
                    actions = np.zeros(num_joints, dtype=np.float32)
                
                # 更新目标位置
                target_joint_pos = data.qpos[7:7+num_joints] + actions * action_scale
                
                # 更新历史动作
                last_actions = actions.copy()
            
            viewer.sync()
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
    
    print("仿真结束")


if __name__ == "__main__":
    main()
