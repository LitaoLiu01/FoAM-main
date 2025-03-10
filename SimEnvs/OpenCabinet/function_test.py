import mujoco_py
import os

# 载入模型
model = mujoco_py.load_model_from_path("/home/liulitao/Desktop/hingecabinet/penl.xml")
sim = mujoco_py.MjSim(model)

# 创建查看器
viewer = mujoco_py.MjViewer(sim)

# 运行仿真
while True:
    sim.step()
    viewer.render()
