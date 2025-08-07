from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

import numpy as np
import time
import yaml
import argparse
import sys
import os
import threading

# ONNX推理
try:
    import onnxruntime as ort
except ImportError:
    print("请安装onnxruntime: pip install onnxruntime")
    sys.exit(1)

# 全局状态控制
current_state = "damping"  # damping, reset, policy
policy_started = False


def wait_for_keyboard_input():
    """等待键盘输入来切换状态"""
    global current_state, policy_started
    
    print("\n=== 控制说明 ===")
    print("1: 进入阻尼模式")
    print("2: 手臂复位")
    print("3: 执行policy")
    print("q: 退出程序")
    print("===============\n")
    
    while True:
        try:
            key = input("请输入命令 (1/2/3/q): ").strip()
            if key == '1':
                current_state = "damping"
                policy_started = False
                print("切换到阻尼模式")
            elif key == '2':
                current_state = "reset"
                policy_started = False
                print("手臂复位")
            elif key == '3':
                current_state = "policy"
                policy_started = True
                print("开始执行policy")
            elif key == 'q':
                print("退出程序")
                return
            else:
                print("无效输入，请输入1、2、3或q")
        except (EOFError, KeyboardInterrupt):
            print("退出输入线程")
            return


def load_config(config_path):
    """加载配置"""
    with open(config_path, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def get_gravity_orientation(quaternion):
    """获取重力方向，quaternion格式: [w, x, y, z]"""
    qw, qx, qy, qz = quaternion
    
    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    
    return gravity_orientation


def create_joint_mapping():
    """创建23dof到29dof的关节映射
    返回: map_23dof_to_29dof, unmapped_joints_29dof
    """
    # 23dof策略输出到29dof机器人的映射 (数组索引，从0开始)
    map_23dof_to_29dof = {
        # 左腿 (0-5 -> 0-5)
        0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5,
        # 右腿 (6-11 -> 6-11)  
        6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11,
        # 腰部 (12 -> 12)
        12: 12,
        # 左臂 (13-17 -> 13-16,17)
        13: 13, 14: 14, 15: 15, 16: 16, 17: 17,
        # 右臂 (18-22 -> 20,21,22,23,24)
        18: 20, 19: 21, 20: 22, 21: 23, 22: 24
    }

    # 29dof机器人中无对应策略输出的关节
    mapped_29dof_joints = set(map_23dof_to_29dof.values())
    unmapped_joints_29dof = [i for i in range(29) if i not in mapped_29dof_joints]
            
    return map_23dof_to_29dof, unmapped_joints_29dof


def convert_23dof_to_29dof_actions(actions_23dof, map_23dof_to_29dof, unmapped_joints_29dof, default_positions):
    """将23dof策略动作转换为29dof机器人动作"""
    actions_29dof = np.zeros(29, dtype=np.float32)
    
    # 有策略控制的关节直接映射
    for idx_23dof, idx_29dof in map_23dof_to_29dof.items():
        actions_29dof[idx_29dof] = actions_23dof[idx_23dof]

    # 无策略控制的关节：保持默认位置（手腕关节）
    for idx_29dof in unmapped_joints_29dof:
        actions_29dof[idx_29dof] = 0.0  # 相对动作为0，保持当前位置
    
    return actions_29dof


def convert_29dof_to_23dof_states(states_29dof, map_23dof_to_29dof):
    """将29dof机器人状态转换为23dof策略格式"""
    states_23dof = np.zeros(23, dtype=np.float32)
    
    for idx_23dof, idx_29dof in map_23dof_to_29dof.items():
        states_23dof[idx_23dof] = states_29dof[idx_29dof]
            
    return states_23dof


def get_observations(qj, dqj, quaternion, ang_vel, map_23dof_to_29dof, last_actions_23dof, action_rescale, obs_scales):
    """获取观测值（23dof策略格式）"""
    # 转换为23dof格式
    q_23dof = convert_29dof_to_23dof_states(qj, map_23dof_to_29dof)
    dq_23dof = convert_29dof_to_23dof_states(dqj, map_23dof_to_29dof)
    
    # 重力方向
    gravity_ori = get_gravity_orientation(quaternion)
        
    # 拼接观测（使用23dof格式）
    obs = np.concatenate([
        ang_vel * obs_scales['ang_vel'],
        gravity_ori,
        q_23dof * obs_scales['dof_pos'],
        dq_23dof * obs_scales['dof_vel'],
        last_actions_23dof,
        [action_rescale]
    ])
    
    return obs.astype(np.float32)


class G1RealController:
    def __init__(self, config):
        self.config = config
        self.num_joints = 29
        
        # SDK通信设置
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = unitree_hg_msg_dds__LowState_()
        
        # 发布器和订阅器
        self.lowcmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher.Init()
        
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.LowStateHandler, 10)
        
        # 等待连接
        self.wait_for_connection()
        
        # 初始化命令
        self.init_cmd()
        
        # 控制参数
        self.kp_27dof = np.array(config['control']['kp'], dtype=np.float32)
        self.kd_27dof = np.array(config['control']['kd'], dtype=np.float32)
        self.action_scale = config['control']['action_scale']
        self.obs_scales = config['observation_scales']
        
        # 关节映射
        self.map_23dof_to_29dof, self.unmapped_joints_29dof = create_joint_mapping()

        # 无效关节id
        self.invadlid_joints_id = config['robot']['invadlid_joints_id']
        
        # 默认关节位置（用于复位）
        self.default_joint_pos = np.array(config['initial_pose']['joint_angles'], dtype=np.float32)
        self.reset_joint_pos = self.default_joint_pos.copy()
        
        # 观测和动作缓冲
        self.history_length = 6
        self.obs_dim_single_23dof = 76
        self.obs_history_23dof = np.zeros((self.history_length, self.obs_dim_single_23dof), dtype=np.float32)
        self.last_actions_23dof = np.zeros(23, dtype=np.float32)
        
        # 机器人状态
        self.qj = np.zeros(self.num_joints, dtype=np.float32)
        self.dqj = np.zeros(self.num_joints, dtype=np.float32)
        self.quaternion = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # [w, x, y, z]
        self.ang_vel = np.zeros(3, dtype=np.float32)
        
        # 初始化LocoClient
        self.loco_client = LocoClient()
        self.loco_client.SetTimeout(10.0)
        self.loco_client.Init()
        
        print("G1实物控制器初始化完成")
        print(f"关节映射: 23dof策略 -> 29dof机器人")
        print(f"无策略控制的关节（手腕）: {self.unmapped_joints_29dof}")
    
    def LowStateHandler(self, msg: LowState_):
        """低级状态消息处理"""
        self.low_state = msg
        
        # 更新关节状态
        for i in range(self.num_joints):
            self.qj[i] = msg.motor_state[i].q
            self.dqj[i] = msg.motor_state[i].dq
        
        # 更新IMU状态
        self.quaternion = np.array([
            msg.imu_state.quaternion[0],  # w
            msg.imu_state.quaternion[1],  # x  
            msg.imu_state.quaternion[2],  # y
            msg.imu_state.quaternion[3]   # z
        ], dtype=np.float32)
        
        self.ang_vel = np.array([
            msg.imu_state.gyroscope[0],
            msg.imu_state.gyroscope[1], 
            msg.imu_state.gyroscope[2]
        ], dtype=np.float32)
    
    def wait_for_connection(self):
        """等待与机器人建立连接"""
        print("等待与机器人建立连接...")
        while self.low_state.tick == 0:
            time.sleep(0.01)
        print("成功连接到机器人")
    
    def init_cmd(self):
        """初始化控制命令"""
        # 设置所有电机为位置控制模式
        for i in range(self.num_joints):
            self.low_cmd.motor_cmd[i].mode = 0x01  # 位置控制模式
            self.low_cmd.motor_cmd[i].q = 0.0
            self.low_cmd.motor_cmd[i].dq = 0.0
            self.low_cmd.motor_cmd[i].tau = 0.0
            self.low_cmd.motor_cmd[i].kp = 0.0
            self.low_cmd.motor_cmd[i].kd = 0.0
    
    def send_cmd(self):
        """发送控制命令"""
        # 计算CRC校验
        self.low_cmd.crc = CRC().Crc(self.low_cmd)
        self.lowcmd_publisher.Write(self.low_cmd)
    
    def damping_mode(self):
        """阻尼模式：使用loco_client进入阻尼"""
        self.loco_client.Damp()
    
    def reset_mode(self):
        """复位模式：仅复位手臂关节"""
        for i in range(self.num_joints):
            if i >= 13:  # 只复位手臂关节 (索引13开始)
                self.low_cmd.motor_cmd[i].q = self.reset_joint_pos[i]
                self.low_cmd.motor_cmd[i].kp = 20.0  # 适中刚度
                self.low_cmd.motor_cmd[i].kd = 2.0
            else:  # 腿部保持阻尼
                self.low_cmd.motor_cmd[i].q = self.qj[i]
                self.low_cmd.motor_cmd[i].kp = 5.0
                self.low_cmd.motor_cmd[i].kd = 1.0
            
            self.low_cmd.motor_cmd[i].dq = 0.0
            self.low_cmd.motor_cmd[i].tau = 0.0
    
    def policy_mode(self, target_positions):
        """策略模式：执行策略输出的动作"""
        for i in range(self.num_joints):
            self.low_cmd.motor_cmd[i].q = target_positions[i]
            self.low_cmd.motor_cmd[i].dq = 0.0
            self.low_cmd.motor_cmd[i].kp = self.kp_27dof[i]
            self.low_cmd.motor_cmd[i].kd = self.kd_27dof[i]
            self.low_cmd.motor_cmd[i].tau = 0.0


def main():
    parser = argparse.ArgumentParser(description='G1 29dof实物部署')
    parser.add_argument('--model_path', type=str, default=None,
                       help='ONNX模型文件路径')
    parser.add_argument('--config', type=str,
                       default='/home/shenlan/HoST/legged_gym/deploy/deploy_real/g1_29dof/deploy_real_g1_29dof_config.yaml',
                       help='配置文件路径')
    args = parser.parse_args()
    
    # 创建配置文件（如果不存在）
    if not os.path.exists(args.config):
        print(f"配置文件不存在，创建默认配置: {args.config}")
        os.makedirs(os.path.dirname(args.config), exist_ok=True)
        # 使用平台部署的配置作为模板
        platform_config_path = '/home/shenlan/HoST/legged_gym/deploy/deploy_mujoco/g1_platform_29dof/g1_platfrom_config.yaml'
        if os.path.exists(platform_config_path):
            with open(platform_config_path, 'r') as f:
                config_content = f.read()
                # 修改模型路径为实物部署路径
                config_content = config_content.replace(
                    'default_model_path: "/home/shenlan/HoST/legged_gym/deploy/deploy_mujoco/g1_platform/models/policy_g1_platform_12000.onnx"',
                    'default_model_path: "/home/shenlan/HoST/legged_gym/deploy/deploy_real/g1_29dof/models/policy_g1_platform_12000.onnx"'
                )
            with open(args.config, 'w') as f:
                f.write(config_content)
    
    # 加载配置
    config = load_config(args.config)
    
    # 模型路径
    model_path = args.model_path or config['model']['default_model_path']
    if not os.path.exists(model_path):
        print(f"模型文件未找到: {model_path}")
        print("请确保ONNX模型文件存在")
        sys.exit(1)
    
    # 加载ONNX模型
    print(f"加载ONNX模型: {model_path}")
    ort_session = ort.InferenceSession(model_path)
    
    # 初始化DDS通信
    ChannelFactoryInitialize(0, config['communication']['network_interface'])
    
    # 创建控制器
    controller = G1RealController(config)
    
    # 启动键盘输入线程
    keyboard_thread = threading.Thread(target=wait_for_keyboard_input, daemon=True)
    keyboard_thread.start()
    
    # 控制参数
    control_dt = 0.02  # 50Hz控制频率
    control_counter = 0
    target_joint_pos_29dof = controller.qj.copy()
    
    print("开始控制循环...")
    print("当前状态: 阻尼模式")
    
    global current_state, policy_started
    
    try:
        while True:
            loop_start = time.time()
            
            # 状态机控制
            if current_state == "damping":
                controller.damping_mode()
                # loco_client处理命令发送，不需要额外发送低级命令
                control_counter += 1
                elapsed = time.time() - loop_start
                if elapsed < control_dt:
                    time.sleep(control_dt - elapsed)
                continue
                
            elif current_state == "reset":
                controller.reset_mode()
                
            elif current_state == "policy" and policy_started:
                # 策略推理（每50ms执行一次，即每2.5个控制周期）
                if control_counter % 2 == 0:
                    # 获取当前观测（23dof格式）
                    current_obs_23dof = get_observations(
                        controller.qj, controller.dqj, controller.quaternion, controller.ang_vel,
                        controller.map_23dof_to_29dof, controller.last_actions_23dof,
                        controller.action_scale, controller.obs_scales
                    )
                    
                    # 更新观测历史
                    controller.obs_history_23dof[:-1] = controller.obs_history_23dof[1:]
                    controller.obs_history_23dof[-1] = current_obs_23dof
                    
                    # 策略推理
                    obs_input_23dof = controller.obs_history_23dof.flatten().reshape(1, -1)
                    policy_output = ort_session.run(None, {'actor_obs': obs_input_23dof})
                    actions_23dof = policy_output[0].flatten()
                    
                    # 转换为29dof动作
                    actions_29dof = convert_23dof_to_29dof_actions(
                        actions_23dof, controller.map_23dof_to_29dof, 
                        controller.unmapped_joints_29dof, controller.default_joint_pos
                    )
                    
                    # 更新目标位置
                    target_joint_pos_29dof = controller.qj + actions_29dof * controller.action_scale
                    
                    # 更新动作历史
                    controller.last_actions_23dof = actions_23dof.copy()
                
                # 执行策略动作
                controller.policy_mode(target_joint_pos_29dof)
            
            # 发送控制命令
            controller.send_cmd()
            
            control_counter += 1
            
            # 控制频率
            elapsed = time.time() - loop_start
            if elapsed < control_dt:
                time.sleep(control_dt - elapsed)
            
    except KeyboardInterrupt:
        print("\n收到Ctrl+C，切换到阻尼模式并退出...")
        controller.damping_mode()
        time.sleep(0.1)
    
    print("程序退出")


if __name__ == "__main__":
    main()
