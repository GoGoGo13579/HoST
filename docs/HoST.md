# HoST 使用与故障排查

## 环境与常见问题

### libpython 相关
- 检查 Conda 环境：
```bash
conda info -e
```
- 设置库路径：
```bash
export LD_LIBRARY_PATH=/home/shenlan/anaconda3/envs/hvgym/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=/shenlan/lib/x86_64-linux-gnu/libstdc++.so.6

conda env config vars set LD_LIBRARY_PATH=/home/shenlan/anaconda3/envs/hvgym/lib:$LD_LIBRARY_PATH
```

### 常见报错及处理
- AttributeError: module 'distutils' has no attribute 'version'
```bash
pip install setuptools==59.5.0
```
- AttributeError: module 'numpy' has no attribute 'float'
```bash
pip install "numpy<1.24"
```

### PyTorch/CUDA 检查
```bash
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

python -c "import torch; print(f'PyTorch版本: {torch.__version__}, CUDA可用: {torch.cuda.is_available()}, 计算能力: {torch.cuda.get_device_capability()}')"
```

---

## 训练与评估

### G1 平地起身
#### 训练
```bash
python legged_gym/legged_gym/scripts/train.py --task g1_ground --run_name test_g1 --headless --num_envs=8192 # [ground, platform, slope, wall]
```
#### 推理
```bash
python legged_gym/legged_gym/scripts/play.py --task g1_ground --num_envs 64 --checkpoint_path legged_gym/logs/g1_ground/Aug05_10-11-30_test_g1/model_12000.pt
```
#### 评估
```bash
python legged_gym/legged_gym/scripts/eval/eval_ground.py --task g1_ground --checkpoint_path legged_gym/logs/g1_ground/Jul27_13-40-19_test_g1/model_12000.pt
```
#### 动作收集
```bash
python legged_gym/legged_gym/scripts/visualization/motion_collection.py --task g1_ground --checkpoint_path legged_gym/logs/g1_ground/Jul27_13-40-19_test_g1/model_12000.pt
```
#### 头、手、脚轨迹可视化
```bash
python legged_gym/legged_gym/scripts/visualization/trajectory_hands_feet.py --terrain ground # [ground, platform, slope, wall]
python legged_gym/legged_gym/scripts/visualization/trajectory_head_pelvis.py --terrain ground # [ground, platform, slope, wall]
```

### G1 靠墙起身
#### 训练
```bash
python legged_gym/legged_gym/scripts/train.py --task g1_wall --run_name test_g1 --headless --num_envs=8192 --max_iterations=100000 # [ground, platform, slope, wall]
```
#### 推理
```bash
python legged_gym/legged_gym/scripts/play.py --task g1_wall --num_envs 64 --checkpoint_path legged_gym/logs/g1_wall/Jul29_13-19-23_test_g1/model_9000.pt
```
#### 评估
```bash
python legged_gym/legged_gym/scripts/eval/eval_wall.py --task g1_wall --checkpoint_path legged_gym/logs/g1_wall/Jul29_13-19-23_test_g1/model_9000.pt
```
#### 动作收集
```bash
python legged_gym/legged_gym/scripts/visualization/motion_collection.py --task g1_wall --checkpoint_path legged_gym/logs/g1_wall/Jul29_13-19-23_test_g1/model_9000.pt
```
#### 头、手、脚轨迹可视化
```bash
python legged_gym/legged_gym/scripts/visualization/trajectory_hands_feet.py --terrain wall # [ground, platform, slope, wall]
python legged_gym/legged_gym/scripts/visualization/trajectory_head_pelvis.py --terrain wall # [ground, platform, slope, wall]
```

### G1 斜面起身
#### 训练
```bash
python legged_gym/legged_gym/scripts/train.py --task g1_slope --run_name test_g1 --headless --num_envs=8192 --max_iterations=100000 # [ground, platform, slope, wall]
```
#### 推理
```bash
python legged_gym/legged_gym/scripts/play.py --task g1_slope --num_envs 64 --checkpoint_path legged_gym/logs/g1_slope/Jul30_15-40-29_test_g1/model_5000.pt
```
#### 评估
```bash
python legged_gym/legged_gym/scripts/eval/eval_slope.py --task g1_slope --checkpoint_path legged_gym/logs/g1_slope/Jul27_13-40-19_test_g1/model_12000.pt
```
#### 动作收集
```bash
python legged_gym/legged_gym/scripts/visualization/motion_collection.py --task g1_wall --checkpoint_path legged_gym/logs/g1_ground/Jul27_13-40-19_test_g1/model_12000.pt
```
#### 头、手、脚轨迹可视化
```bash
python legged_gym/legged_gym/scripts/visualization/trajectory_hands_feet.py --terrain ground # [ground, platform, slope, wall]
python legged_gym/legged_gym/scripts/visualization/trajectory_head_pelvis.py --terrain ground # [ground, platform, slope, wall]
```

### G1 平台起身
#### 训练
```bash
python legged_gym/legged_gym/scripts/train.py --task g1_platform --run_name test_g1 --headless --num_envs=4096 --max_iterations=100000 # [ground, platform, slope, wall]
```
#### 推理
```bash
python legged_gym/legged_gym/scripts/play.py --task g1_platform --num_envs 64 --checkpoint_path legged_gym/logs/g1_platform/Aug07_15-15-04_g1_0807/model_12500.pt
```
```bash
python legged_gym/legged_gym/scripts/play.py --task g1_platform --num_envs 64 --checkpoint_path legged_gym/logs/g1_wall/Jul29_13-19-23_test_g1/model_3500.pt
```
#### 评估
```bash
python legged_gym/legged_gym/scripts/eval/eval_ground.py --task g1_wall --checkpoint_path legged_gym/logs/g1_ground/Jul27_13-40-19_test_g1/model_12000.pt
```
#### 动作收集
```bash
python legged_gym/legged_gym/scripts/visualization/motion_collection.py --task g1_wall --checkpoint_path legged_gym/logs/g1_ground/Jul27_13-40-19_test_g1/model_12000.pt
```
#### 头、手、脚轨迹可视化
```bash
python legged_gym/legged_gym/scripts/visualization/trajectory_hands_feet.py --terrain ground # [ground, platform, slope, wall]
python legged_gym/legged_gym/scripts/visualization/trajectory_head_pelvis.py --terrain ground # [ground, platform, slope, wall]
```

### G1 俯卧起身（prone）
#### 训练
```bash
python legged_gym/legged_gym/scripts/train.py --task g1_ground_prone --run_name g1_ground_prone --headless --num_envs=8192 --max_iterations=12000 # [ground, platform, slope, wall, ground_prone]
python legged_gym/legged_gym/scripts/train.py --task g1_ground_prone --resume --load_run Aug12_11-10-11_g1_ground_prone --checkpoint 12000 --headless --run_name g1_ground_prone_-10
```
#### 推理
```bash
python legged_gym/legged_gym/scripts/play.py --task g1_ground_prone --num_envs 64 --checkpoint_path legged_gym/logs/g1_ground_prone/Aug18_19-00-52_g1_ground_prone_-10/model_24000.pt
```
#### 评估
```bash
python legged_gym/legged_gym/scripts/eval/eval_ground.py --task g1_ground_prone --checkpoint_path legged_gym/logs/g1_ground_prone/Aug12_11-10-11_g1_ground_prone/model_12000.pt
```
#### 动作收集
```bash
python legged_gym/legged_gym/scripts/visualization/motion_collection.py --task g1_ground_prone --checkpoint_path legged_gym/logs/g1_ground_prone/Aug12_11-10-11_g1_ground_prone/model_12000.pt
```
#### 头、手、脚轨迹可视化
```bash
python legged_gym/legged_gym/scripts/visualization/trajectory_hands_feet.py --terrain ground_prone # [ground, platform, slope, wall]
python legged_gym/legged_gym/scripts/visualization/trajectory_head_pelvis.py --terrain ground_prone # [ground, platform, slope, wall]
```

### Mini Pi 平地起身
#### 训练
```bash
python legged_gym/legged_gym/scripts/train.py --task pi_ground --run_name test_minipi_ground --headless --num_envs=4096 --max_iterations=100000 # [ground, platform, slope, wall]
```
#### 推理
```bash
python legged_gym/legged_gym/scripts/play.py --task pi_ground --num_envs 64 --checkpoint_path legged_gym/logs/Pi_ground/Jul30_15-56-21_test_minipi_ground/model_25000.pt
```
#### 评估
```bash
python legged_gym/legged_gym/scripts/play.py --task pi_ground --num_envs 64 --checkpoint_path legged_gym/logs/Pi_ground/Jul30_15-56-21_test_minipi_ground/model_49700.pt
```
#### 动作收集
```bash
python legged_gym/legged_gym/scripts/visualization/motion_collection.py --task pi_ground --checkpoint_path legged_gym/logs/g1_ground/Jul27_13-40-19_test_g1/model_12000.pt
```
#### 头、手、脚轨迹可视化
```bash
python legged_gym/legged_gym/scripts/visualization/trajectory_hands_feet.py --terrain ground # [ground, platform, slope, wall]
python legged_gym/legged_gym/scripts/visualization/trajectory_head_pelvis.py --terrain ground # [ground, platform, slope, wall]
```