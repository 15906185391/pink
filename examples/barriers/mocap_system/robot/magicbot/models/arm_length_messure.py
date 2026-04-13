import pinocchio as pin
import numpy as np
import os

# ⚠️ 替换为你自己的URDF路径
urdf_path = "/home/maowei/ws_mw/mocap/mocap_system/models/l_arm_v1.urdf"

# ✅ 只加载模型，不加载几何
model = pin.buildModelFromUrdf(urdf_path)
data = model.createData()

# 关节ID
jid_la1 = model.getJointId("joint_la1")
jid_la4 = model.getJointId("joint_la4")
jid_la6 = model.getJointId("joint_la6")

# TCP用Frame ID
fid_tcp = model.getFrameId("link_tcp_l")

# 初始姿态
q = pin.neutral(model)

# 前向运动学
pin.forwardKinematics(model, data, q)
pin.updateFramePlacements(model, data)

# 取出各点位置
p1 = data.oMi[jid_la1].translation
p3 = data.oMi[jid_la4].translation
p5 = data.oMi[jid_la6].translation
ptcp = data.oMf[fid_tcp].translation  # ✅ 使用Frame

def dist(a, b):
    return np.linalg.norm(a - b)

upper_arm_len = dist(p1, p3)
fore_arm_len = dist(p3, p5)
hand_len = dist(p5, ptcp)

print(f"上臂长度 (joint_la1 → joint_la3): {upper_arm_len:.4f} m")
print(f"小臂长度 (joint_la3 → joint_la5): {fore_arm_len:.4f} m")
print(f"手部长度 (joint_la5 → tcp): {hand_len:.4f} m")
print(f"臂展总长度: {upper_arm_len + fore_arm_len + hand_len:.4f} m")