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


def get_observations(data, map_23dof_to_27dof, last_actions_23dof, action_rescale, obs_scales):
    """获取观测值"""
    # 获取27dof机器人关节状态
    q_27dof = data.qpos[7:7+27]
    dq_27dof = data.qvel[6:6+27]
    
    # 映射到23dof策略格式
    q_23dof = convert_27dof_to_23dof_states(q_27dof, map_23dof_to_27dof)
    dq_23dof = convert_27dof_to_23dof_states(dq_27dof, map_23dof_to_27dof)
    
    # 基座状态
    quat = data.qpos[3:7]
    ang_vel = data.qvel[3:6]
    
    # 重力方向
    gravity_ori = get_gravity_orientation(quat)
        
    # 拼接观测（使用23dof格式）
    obs = np.concatenate([
        ang_vel * obs_scales['ang_vel'],
        gravity_ori,
        q_23dof * obs_scales['dof_pos'],
        dq_23dof * obs_scales['dof_vel'],
        last_actions_23dof,  # 23dof格式的动作
        [action_rescale]
    ])
    
    return obs.astype(np.float32)


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """PD控制"""
    return (target_q - q) * kp + (target_dq - dq) * kd


def create_joint_mapping():
    """创建23dof到27dof的关节映射
    
    23dof关节顺序 (策略输出，基于关节索引1-23):
    1-6: 左腿 (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)
    7-12: 右腿 (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)  
    13: 腰部 (waist_yaw)
    14-18: 左臂 (shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll)
    19-23: 右臂 (shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll)
    
    27dof关节顺序 (实际机器人，基于关节索引1-27，不包括floating_base):
    1-6: 左腿 (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)
    7-12: 右腿 (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)
    13: 腰部 (waist_yaw)  
    14-17: 左臂 (shoulder_pitch, shoulder_roll, shoulder_yaw, elbow)
    18-20: 左手腕 (wrist_roll, wrist_pitch, wrist_yaw)
    21-24: 右臂 (shoulder_pitch, shoulder_roll, shoulder_yaw, elbow) 
    25-27: 右手腕 (wrist_roll, wrist_pitch, wrist_yaw)
    
    Returns:
        map_23dof_to_27dof: 23dof策略索引到27dof机器人索引的映射
    """
    # 23dof策略输出到27dof机器人的映射 (数组索引，从0开始)
    map_23dof_to_27dof = {
        # 左腿 (0-5 -> 0-5)，对应关节1-6
        0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5,
        # 右腿 (6-11 -> 6-11)，对应关节7-12  
        6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11,
        # 腰部 (12 -> 12)，对应关节13
        12: 12,
        # 左臂 (13-17 -> 13-16,17)，对应关节14-17,18(wrist_roll)
        13: 13, 14: 14, 15: 15, 16: 16, 17: 17,
        # 右臂 (18-22 -> 20,21,22,23,24)，对应关节21-25，跳过18-19的左手腕pitch,yaw关节
        18: 20, 19: 21, 20: 22, 21: 23, 22: 24
    }

        # 找出27dof机器人中无对应策略输出的关节
    mapped_27dof_joints = set(map_23dof_to_27dof.values())
    unmapped_joints_27dof = [i for i in range(27) if i not in mapped_27dof_joints]
            
    return map_23dof_to_27dof, unmapped_joints_27dof


def convert_23dof_to_27dof_actions(actions_23dof, map_23dof_to_27dof, unmapped_joints_27dof, origin_qpos, data):
    actions_27dof = np.zeros(27, dtype=np.float32)
    
    # 有策略控制的关节直接映射
    for idx_23dof, idx_27dof in map_23dof_to_27dof.items():
        actions_27dof[idx_27dof] = actions_23dof[idx_23dof]

    # 无策略控制的关节：使用更平滑的方式回到默认位置
    for idx_27dof in unmapped_joints_27dof:
        pos_error = origin_qpos[idx_27dof] - data.qpos[7 + idx_27dof]
        # 限制最大移动幅度，避免突然的大幅度动作
        max_move = 0.1  # 限制每步最大
        actions_27dof[idx_27dof] = np.clip(pos_error, -max_move, max_move)
    
    return actions_27dof


def convert_27dof_to_23dof_states(states_27dof, map_23dof_to_27dof):
    states_23dof = np.zeros(23, dtype=np.float32)
    
    for idx_23dof, idx_27dof in map_23dof_to_27dof.items():
        states_23dof[idx_23dof] = states_27dof[idx_27dof]
            
    return states_23dof


def main():
    parser = argparse.ArgumentParser(description='部署G1地面运动模型')
    parser.add_argument('--model_path', type=str, default=None, 
                       help='ONNX模型文件路径')
    parser.add_argument('--config', type=str, 
                       default='/home/shenlan/HoST/legged_gym/deploy/deploy_mujoco/g1_platform_29dof/g1_platfrom_config.yaml',
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
    kp_27dof = np.array(config['control']['kp'], dtype=np.float32)
    kd_27dof = np.array(config['control']['kd'], dtype=np.float32)
    action_scale = config['control']['action_scale']
    obs_scales = config['observation_scales']
    
    # 创建关节映射
    map_23dof_to_27dof, unmapped_joints_27dof = create_joint_mapping()
    print(f"策略关节数: 23, 机器人关节数: 27")
    print("关节映射已创建")
    
    # 打印映射关系用于调试
    print("\n23dof策略 -> 27dof机器人映射:")
    for idx_23dof, idx_27dof in sorted(map_23dof_to_27dof.items()):
        print(f"  23dof关节{idx_23dof} -> 27dof关节{idx_27dof}")
        
    print("\n27dof机器人中无对应策略输出的关节:")
    for joint_idx in unmapped_joints_27dof:
        print(f"  27dof关节{joint_idx} (无策略控制，将使用默认位置)")
    

    # 初始姿态
    initial_pose = config['initial_pose']
    data.qpos[:3] = initial_pose['position']
    data.qpos[3:7] = initial_pose['orientation']
    data.qpos[7:7+27] = initial_pose['joint_angles']
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    
    # 控制变量
    sim_counter = 0
    target_joint_pos_27dof = np.array(initial_pose['joint_angles'], dtype=np.float32)  # 27dof机器人目标位置
    default_joint_pos_27dof = np.array(initial_pose['joint_angles'], dtype=np.float32)  # 默认关节位置，用于无策略控制的关节
    origin_qpos = np.zeros(27, dtype=np.float32)
    
    # 历史观测缓冲区
    history_length = 6
    obs_dim_single_23dof = 76  # 23dof格式的观测维度
    obs_dim_total = history_length * obs_dim_single_23dof
    
    # 初始化缓冲区
    obs_history_23dof = np.zeros((history_length, obs_dim_single_23dof), dtype=np.float32)
    last_actions_23dof = np.zeros(23, dtype=np.float32)  # 23dof格式的动作历史
    
    # 等待用户输入
    input_thread = threading.Thread(target=wait_for_enter, daemon=True)
    input_thread.start()
    
    print("开始仿真...")
    print(f"观测维度: 单步{obs_dim_single_23dof}维(23dof格式), 总计{obs_dim_total}维")
    print("按Enter键启动策略，按ESC键退出")
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            
            # PD控制
            if policy_started:
                tau_27dof = pd_control(
                    target_joint_pos_27dof, 
                    data.qpos[7:7+27], 
                    kp_27dof,
                    np.zeros(27), 
                    data.qvel[6:6+27], 
                    kd_27dof
                )
            else:
                tau_27dof = np.zeros(27)
            data.ctrl[:] = tau_27dof
            data.ctrl[17: 19] = 0
            data.ctrl[24: 26] = 0
            
            # 仿真步进
            mujoco.mj_step(model, data)
            sim_counter += 1
            
            # 策略推理
            if sim_counter % control_decimation == 0:
                # 获取当前观测（23dof格式）
                current_obs_23dof = get_observations(data, map_23dof_to_27dof, last_actions_23dof, action_scale, obs_scales)
                
                # 更新历史
                obs_history_23dof[:-1] = obs_history_23dof[1:]
                obs_history_23dof[-1] = current_obs_23dof
                
                # 推理
                if policy_started:
                    obs_input_23dof = obs_history_23dof.flatten().reshape(1, -1)
                    # obs_input_23dof = np.flip(obs_input_23dof, axis=1)
                    # 运行推理（获得23dof策略动作）
                    policy_output = ort_session.run(None, {'actor_obs': obs_input_23dof})
                    # policy_output = ort_session.run(None, {'input': obs_input_23dof})
                    actions_23dof = policy_output[0].flatten()  # 23dof策略动作
                    
                    # 映射到27dof机器人动作
                    actions_27dof = convert_23dof_to_27dof_actions(actions_23dof, map_23dof_to_27dof, unmapped_joints_27dof, origin_qpos, data)
                else:
                    actions_23dof = np.zeros(23, dtype=np.float32)
                    actions_27dof = np.zeros(27, dtype=np.float32)
                    origin_qpos = data.qpos[7: 7+27]
                
                # 更新目标位置（使用27dof机器人动作）
                target_joint_pos_27dof = data.qpos[7:7+27] + actions_27dof * action_scale
                
                # 更新历史动作（保存23dof策略动作）
                last_actions_23dof = actions_23dof.copy()
            
            viewer.sync()
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
    
    print("仿真结束")


if __name__ == "__main__":
    main()
