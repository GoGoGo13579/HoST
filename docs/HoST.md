# For libpython error:

- Check conda path:
    ```bash
    conda info -e
    ```
- Set LD_LIBRARY_PATH:
    ```bash
    export LD_LIBRARY_PATH=/home/shenlan/anaconda3/envs/hvgym/lib:$LD_LIBRARY_PATH
    export LD_PRELOAD=/shenlan/lib/x86_64-linux-gnu/libstdc++.so.6

    conda env config vars set LD_LIBRARY_PATH=/home/shenlan/anaconda3/envs/hvgym/lib:$LD_LIBRARY_PATH

    ```
    


- AttributeError: module 'distutils' has no attribute 'version'
```bash
pip install setuptools==59.5.0
```
- AttributeError: module 'numpy' has no attribute 'float'.
```bash
pip install "numpy<1.24"
```
- 
可能会遇到的问题
```bash

pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

python -c "import torch; print(f'PyTorch版本: {torch.__version__}, CUDA可用: {torch.cuda.is_available()}, 计算能力: {torch.cuda.get_device_capability()}')"
```

------

# 训练
## G1 平地起身

### 训练
```bash
python legged_gym/legged_gym/scripts/train.py --task g1_ground --run_name test_g1 --headless --num_envs=1 # [ground, platform, slope, wall]
```

### 推理
```bash
python legged_gym/legged_gym/scripts/play.py --task g1_ground --num_envs 64 --checkpoint_path legged_gym/logs/g1_ground/Aug05_10-11-30_test_g1/model_12000.pt
```
### 评估
```bash
python legged_gym/legged_gym/scripts/eval/eval_ground.py --task g1_ground --checkpoint_path legged_gym/logs/g1_ground/Jul27_13-40-19_test_g1/model_12000.pt
```
### 动作收集
```bash
python legged_gym/legged_gym/scripts/visualization/motion_collection.py --task g1_ground --checkpoint_path legged_gym/logs/g1_ground/Jul27_13-40-19_test_g1/model_12000.pt 
```
### 头、手、脚轨迹可视化
```bash
python legged_gym/legged_gym/scripts/visualization/trajectory_hands_feet.py  --terrain ground # [ground, platform, slope, wall]

python legged_gym/legged_gym/scripts/visualization/trajectory_head_pelvis.py  --terrain ground # [ground, platform, slope, wall]
```

## G1 靠墙起身

### 训练
```bash
python legged_gym/legged_gym/scripts/train.py --task g1_wall --run_name test_g1 --headless --num_envs=8192 --max_iterations=100000 # [ground, platform, slope, wall]
```

### 推理
```bash
python legged_gym/legged_gym/scripts/play.py --task g1_wall --num_envs 64 --checkpoint_path legged_gym/logs/g1_wall/Jul29_13-19-23_test_g1/model_9000.pt
```
### 评估
```bash
python legged_gym/legged_gym/scripts/eval/eval_wall.py --task g1_wall --checkpoint_path legged_gym/logs/g1_wall/Jul29_13-19-23_test_g1/model_9000.pt
```
### 动作收集
```bash
python legged_gym/legged_gym/scripts/visualization/motion_collection.py --task g1_wall --checkpoint_path legged_gym/logs/g1_wall/Jul29_13-19-23_test_g1/model_9000.pt 
```
### 头、手、脚轨迹可视化
```bash
python legged_gym/legged_gym/scripts/visualization/trajectory_hands_feet.py  --terrain wall # [ground, platform, slope, wall]

python legged_gym/legged_gym/scripts/visualization/trajectory_head_pelvis.py  --terrain wall # [ground, platform, slope, wall]
```

## G1 斜面起身

### 训练
```bash
python legged_gym/legged_gym/scripts/train.py --task g1_slope --run_name test_g1 --headless --num_envs=8192 --max_iterations=100000 # [ground, platform, slope, wall]
```

### 推理
```bash
python legged_gym/legged_gym/scripts/play.py --task g1_slope --num_envs 64 --checkpoint_path legged_gym/logs/g1_slope/Jul30_15-40-29_test_g1/model_5000.pt
```
### 评估
```bash
python legged_gym/legged_gym/scripts/eval/eval_slope.py --task g1_slope --checkpoint_path legged_gym/logs/g1_slope/Jul27_13-40-19_test_g1/model_12000.pt
```
### 动作收集
```bash
python legged_gym/legged_gym/scripts/visualization/motion_collection.py --task g1_wall --checkpoint_path legged_gym/logs/g1_ground/Jul27_13-40-19_test_g1/model_12000.pt 
```
### 头、手、脚轨迹可视化
```bash
python legged_gym/legged_gym/scripts/visualization/trajectory_hands_feet.py  --terrain ground # [ground, platform, slope, wall]

python legged_gym/legged_gym/scripts/visualization/trajectory_head_pelvis.py  --terrain ground # [ground, platform, slope, wall]
```

## G1 平台起身

### 训练
```bash
python legged_gym/legged_gym/scripts/train.py --task g1_platform --run_name test_g1 --headless --num_envs=4096 --max_iterations=100000 # [ground, platform, slope, wall]
```

### 推理
```bash
python legged_gym/legged_gym/scripts/play.py --task g1_platform --num_envs 64 --checkpoint_path legged_gym/logs/g1_wall/Jul29_13-19-23_test_g1/model_3500.pt
```
### 评估
```bash
python legged_gym/legged_gym/scripts/eval/eval_ground.py --task g1_wall --checkpoint_path legged_gym/logs/g1_ground/Jul27_13-40-19_test_g1/model_12000.pt
```
### 动作收集
```bash
python legged_gym/legged_gym/scripts/visualization/motion_collection.py --task g1_wall --checkpoint_path legged_gym/logs/g1_ground/Jul27_13-40-19_test_g1/model_12000.pt 
```
### 头、手、脚轨迹可视化
```bash
python legged_gym/legged_gym/scripts/visualization/trajectory_hands_feet.py  --terrain ground # [ground, platform, slope, wall]

python legged_gym/legged_gym/scripts/visualization/trajectory_head_pelvis.py  --terrain ground # [ground, platform, slope, wall]
```


## Mini Pi 平地起身

python legged_gym/scripts/train.py --task pi_ground --run_name test_minipi_ground

### 训练
```bash
python legged_gym/legged_gym/scripts/train.py --task pi_ground --run_name test_minipi_ground --headless --num_envs=4096 --max_iterations=100000 # [ground, platform, slope, wall]
```

### 推理
```bash
python legged_gym/legged_gym/scripts/play.py --task pi_ground --num_envs 64 --checkpoint_path legged_gym/logs/Pi_ground/Jul30_15-56-21_test_minipi_ground/model_20300.pt
```
### 评估
```bash
python legged_gym/legged_gym/scripts/eval/eval_ground.py --task g1_wall --checkpoint_path legged_gym/logs/g1_ground/Jul27_13-40-19_test_g1/model_12000.pt
```
### 动作收集
```bash
python legged_gym/legged_gym/scripts/visualization/motion_collection.py --task g1_wall --checkpoint_path legged_gym/logs/g1_ground/Jul27_13-40-19_test_g1/model_12000.pt 
```
### 头、手、脚轨迹可视化
```bash
python legged_gym/legged_gym/scripts/visualization/trajectory_hands_feet.py  --terrain ground # [ground, platform, slope, wall]

python legged_gym/legged_gym/scripts/visualization/trajectory_head_pelvis.py  --terrain ground # [ground, platform, slope, wall]
```