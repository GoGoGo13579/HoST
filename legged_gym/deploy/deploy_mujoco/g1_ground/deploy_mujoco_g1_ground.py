import sys
import os
import argparse
import time
import numpy as np
import yaml
import mujoco
import mujoco.viewer
import threading

# 使用ONNX运行时进行模型推理
try:
    import onnxruntime as ort
except ImportError:
    print("请安装onnxruntime: pip install onnxruntime")
    sys.exit(1)

# 全局控制变量
policy_started = False


def wait_for_enter():
    """等待用户按回车键启动策略"""
    global policy_started
    input("按回车键开始执行策略...")
    policy_started = True
    print("策略已启动！")


def load_config(config_path):
    """从yaml文件加载配置"""
    with open(config_path, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def get_gravity_orientation(quaternion):
    """从四元数获取机体坐标系下的重力方向"""
    # quaternion格式为 [w, x, y, z]
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
    """从mujoco数据获取观测值 - 匹配训练时的76维观测"""
    # 关节位置和速度
    q = data.qpos[7:7+num_joints]  # 关节位置 (23维)
    dq = data.qvel[6:6+num_joints]  # 关节速度 (23维)
    
    # 基座方向和角速度
    quat = data.qpos[3:7]  # 四元数 [w,x,y,z]
    ang_vel = data.qvel[3:6]  # 角速度 (3维)
    
    # 机体坐标系下的重力方向
    gravity_ori = get_gravity_orientation(quat)  # (3维)
    
    # 动作缩放因子（添加小噪声）
    action_scale_with_noise = action_rescale
    
    # 拼接所有观测值 - 匹配训练时的顺序和缩放
    obs = np.concatenate([
        ang_vel * obs_scales['ang_vel'],        # 基座角速度 * 0.25 (3维)
        gravity_ori,                            # 重力方向 (3维) - 无缩放
        q * obs_scales['dof_pos'],              # 关节位置 * 1.0 (23维)
        dq * obs_scales['dof_vel'],             # 关节速度 * 0.05 (23维)
        last_actions,                           # 上一步动作 (23维) - 无缩放
        [action_scale_with_noise]               # 动作缩放因子 (1维) - 无缩放
    ])
    
    return obs.astype(np.float32)  # 总计76维


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """PD控制器计算关节力矩"""
    return (target_q - q) * kp + (target_dq - dq) * kd


def main():
    parser = argparse.ArgumentParser(description='部署G1机器人地面运动的ONNX模型')
    parser.add_argument('--model_path', type=str, default=None, 
                       help='ONNX模型文件路径')
    parser.add_argument('--config', type=str, 
                       default='/home/shenlan/HoST/legged_gym/deploy/deploy_mujoco/g1_ground/g1_ground_config.yaml',
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
    
    # 设置仿真参数
    model.opt.timestep = config['simulation']['simulation_dt']
    control_decimation = config['simulation']['control_decimation']
    num_joints = config['robot']['num_joints']
    # 控制参数
    kp = config['control']['kp']
    kd = config['control']['kd']
    
    # 输出关节名称
    print("\n关节名称:")
    for i in range(model.njnt):
        joint_name = model.joint(i).name
        print(f"关节 {i}: {joint_name}")
    
    action_scale = config['control']['action_scale']
    # 观测缩放因子
    obs_scales = config['observation_scales']
    # 初始化机器人姿态
    initial_pose = config['initial_pose']
    data.qpos[:3] = initial_pose['position']
    data.qpos[3:7] = initial_pose['orientation']
    data.qvel[:] = 0
    
    mujoco.mj_forward(model, data)
    
    # 控制变量
    sim_counter = 0
    target_joint_pos = np.array(initial_pose['joint_angles'], dtype=np.float32)
    
    # 历史观测缓冲区设置
    history_length = 6  # 配置文件中的num_actor_history
    obs_dim_single = 76  # 单步观测维度
    obs_dim_total = history_length * obs_dim_single  # 总观测维度456
    
    # 初始化历史缓冲区和动作缓冲区
    obs_history = np.zeros((history_length, obs_dim_single), dtype=np.float32)
    last_actions = np.zeros(num_joints, dtype=np.float32)
    
    # 在单独线程中等待用户输入
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
            
            # 仿真步进
            mujoco.mj_step(model, data)
            sim_counter += 1
            
            # 以控制频率运行策略
            if sim_counter % control_decimation == 0:
                # 获取当前单步观测值
                current_obs = get_observations(data, num_joints, last_actions, action_scale, obs_scales)
                
                # 更新历史缓冲区（滑动窗口）
                obs_history[:-1] = obs_history[1:]  # 向前移动历史
                obs_history[-1] = current_obs       # 添加当前观测
                
                # 如果策略已启动，运行推理；否则输出零动作
                if policy_started:
                    # 展平历史观测作为网络输入
                    obs_input = obs_history.flatten().reshape(1, -1)  # 形状: (1, 456)
                    # obs_input = np.flip(obs_input, axis=1)
                    # 运行推理
                    policy_output = ort_session.run(None, {'actor_obs': obs_input})
                    actions = policy_output[0].flatten()  # 移除批次维度
                else:
                    # 策略未启动，输出零动作
                    actions = np.zeros(num_joints, dtype=np.float32)
                
                # 缩放动作并更新目标位置，target = cur + action * beta
                target_joint_pos = data.qpos[7:7+num_joints] + actions * action_scale
                
                # 更新上一步动作
                last_actions = actions.copy()
            
            viewer.sync()
            
            # 休眠以保持实时性
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
    
    print("仿真结束")


if __name__ == "__main__":
    main()
