import time

import mujoco.viewer
import mujoco
import numpy as np
import torch
import yaml
import os
from pathlib import Path
from collections import deque
import matplotlib.pyplot as plt
import csv
import pickle

def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation

def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd

if __name__ == "__main__":
    # get config file name from command line
    import argparse

    current_path = os.path.dirname(__file__)

    config_path = os.path.join(current_path, "g1_ground_config.yaml")
    print("config_path: ", config_path)

    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # 从新的配置文件结构中提取参数
    xml_path = config["simulation"]["xml_path"]
    
    simulation_duration = 30.0  # 默认30秒
    simulation_dt = config["simulation"]["simulation_dt"]
    control_decimation = config["simulation"]["control_decimation"]

    kps = np.array(config["control"]["kp"], dtype=np.float32)
    kds = np.array(config["control"]["kd"], dtype=np.float32)

    default_angles = np.array(config["initial_pose"]["joint_angles"], dtype=np.float32)

    ang_vel_scale = config["observation_scales"]["ang_vel"]
    dof_pos_scale = config["observation_scales"]["dof_pos"]
    dof_vel_scale = config["observation_scales"]["dof_vel"]
    action_scale = config["control"]["action_scale"]

    num_actions = config["robot"]["num_joints"]
    num_obs = config["observations"]["total_obs_dim"]

    # 默认命令值
    cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    policy_mode = "host"

    print("xml_path   : ", xml_path)

    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    counter = 0
    delay_buffer = np.zeros((5, 1, 23))

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt
    d.qpos[0:3] = config["initial_pose"]["position"]
    d.qpos[3:7] = config["initial_pose"]["orientation"]
    mujoco.mj_forward(m, d)

    # load policy - 如果有模型文件的话
    policy_path = config["model"]["default_model_path"]
    if os.path.exists(policy_path):
        if policy_path.endswith('.onnx'):
            print("ONNX model detected, please convert to torchscript for this demo")
            print("Using default standing pose...")
            policy = None
        else:
            policy = torch.jit.load(policy_path)
    else:
        print(f"Policy file not found: {policy_path}")
        print("Using default standing pose...")
        policy = None

    if policy_mode == "unitree":
        obs_buf = np.zeros(num_obs, dtype=np.float32)
    elif policy_mode == "host":
        obs_buf = np.zeros(num_obs, dtype=np.float32)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:

            step_start = time.time()
            tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
            d.ctrl[:] = tau

            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)

            counter += 1
            if counter % control_decimation == 0:
                # Apply control signal here.

                # create observation
                qj = d.qpos[7:]      # 关节角度
                dqj = d.qvel[6:]     # 关节速度
                quat = d.qpos[3:7]   # 四元数

                # omega = d.qvel[3:6]  # 角速度w
                ang_vel = d.qvel[3:6]
                qj = qj * dof_pos_scale
                dqj = dqj * dof_vel_scale
                gravity_orientation = get_gravity_orientation(quat)
                ang_vel = ang_vel * ang_vel_scale

                if policy_mode == "host" and policy is not None:
                    current_obs = np.concatenate((
                                            ang_vel * 0.25,
                                            gravity_orientation,
                                            qj,
                                            dqj,
                                            action,
                                            np.array([0.3]) 
                                        ), axis=-1)

                    if counter / control_decimation <= 30.0:
                        current_obs = current_obs * 0

                    obs_buf = np.concatenate((current_obs, obs_buf[76:76*6]), axis=-1, dtype=np.float32)

                    obs_tensor = torch.from_numpy(obs_buf).unsqueeze(0)

                    action = policy(obs_tensor).detach().numpy().squeeze()

                    if counter / control_decimation <= 30.0:
                        action = np.zeros(num_actions, dtype=np.float32)

                    delay_idx = 2
                    action_scaled = action * action_scale
                    action_scaled_expanded = np.expand_dims(np.expand_dims(action_scaled, axis=0), axis=1)  # [1,1,23]
                    delay_buffer = np.concatenate((delay_buffer[1:], action_scaled_expanded), axis=0)

                    target_dof_pos = default_angles + delay_buffer[delay_idx, 0, :]
                else:
                    # 如果没有策略，保持默认姿势
                    target_dof_pos = default_angles.copy()

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)