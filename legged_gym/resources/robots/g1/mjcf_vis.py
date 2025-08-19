import mujoco
import mujoco.viewer
import time
import os

# 1. 指定MJCF模型文件路径
mjcf_file = "/home/shenlan/zzy_ws/HoST/legged_gym/resources/robots/pi_12dof/urdf/pi_12dof_release_v1.urdf"  # 替换为你的MJCF文件路径

# 2. 检查文件是否存在
if not os.path.isfile(mjcf_file):
    raise FileNotFoundError(f"MJCF文件未找到: {mjcf_file}")

# 3. 加载模型
model = mujoco.MjModel.from_xml_path(mjcf_file)
data = mujoco.MjData(model)

# 4. 创建可视化窗口
with mujoco.viewer.launch_passive(model, data) as viewer:
    print("按ESC退出窗口")
    print("鼠标操作指南:")
    print("  - 左键拖拽: 旋转视角")
    print("  - 右键拖拽: 平移视角")
    print("  - 滚轮: 缩放")
    
    # 5. 模拟循环
    while viewer.is_running():
        step_start = time.time()
        
        # 物理模拟步进
        mujoco.mj_step(model, data)
        
        # 同步更新可视化
        viewer.sync()
        
        # 控制运行速度 (步频60Hz)
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

print("可视化窗口已关闭")