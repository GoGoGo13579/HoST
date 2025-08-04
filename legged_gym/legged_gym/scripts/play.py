import sys
from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, export_policy_as_onnx, task_registry, Logger

import torch
import time

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Value

# Set to True to export policy
EXPORT_POLICY = True


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 100)
    env_cfg.terrain.num_rows = 4
    env_cfg.terrain.num_cols = 4
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.control.action_scale = 0.3
    env_cfg.curriculum.pull_force = False
    env_cfg.env.test = True

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, env_cfg=env_cfg, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    logger = Logger(env.dt)
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        # export policy as ONNX model (for deployment)
        obs_size = env.num_obs
        # Extract training step number from checkpoint path
        checkpoint_step = "unknown"
        if hasattr(args, 'checkpoint_path') and args.checkpoint_path:
            import re
            # Extract number from model_XXXX.pt pattern
            match = re.search(r'model_(\d+)\.pt', args.checkpoint_path)
            if match:
                checkpoint_step = match.group(1)
        
        exported_policy_name = f"policy_{args.task}_{checkpoint_step}.onnx"
        export_policy_as_onnx(ppo_runner.alg.actor_critic, path, obs_size, exported_policy_name)
        print('Exported policy as ONNX model to: ', path)
    
    for i in range(10*int(env.max_episode_length)):

        result = env.gym.fetch_results(env.sim, True)
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())


if __name__ == '__main__':
    args = get_args()
    play(args)
