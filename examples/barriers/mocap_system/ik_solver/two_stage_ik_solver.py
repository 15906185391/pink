import numpy as np
import pinocchio as pin
from numpy.linalg import norm, solve
from mocap_system.Utils import *
from mocap_system.config.config_info import Config

DT = 0.5
# DT = 0.1
DAMP = 1e-6


class IKSolver:
    def __init__(self, subModel, is_left=True, config=None):
        self.is_left = is_left
        self.config = config
        if config is None:
            self.config = Config()
        self.subModel = subModel
        self.model = subModel.model
        self.data = subModel.data
        self.qpos = np.zeros(self.model.nq)     # 初始关节角度
        self.num_link = self.model.nq   # 关节数量

        # 关节极限
        if (self.is_left):
            self.qpos_threshold = self.config.arm_l
        else:
            self.qpos_threshold = self.config.arm_r

        self.qpos_threshold = self.qpos_threshold[:self.num_link]

        # # 直接从 pinocchio model 获取位置限制
        # limit_range = self.model.upperPositionLimit - self.model.lowerPositionLimit
        # limit_ratio = 0.05  # 留出 5% 的安全边距
        # self.qpos_lower = self.model.lowerPositionLimit + limit_ratio * limit_range  # 留出 5% 的安全边距
        # self.qpos_upper = self.model.upperPositionLimit - limit_ratio * limit_range
        # self.qpos_threshold = list(zip(self.qpos_lower, self.qpos_upper))
        self.wrist_limit_l = self.config.wrist_l
        self.wrist_limit_r = self.config.wrist_r
        self.waist_limit = self.config.waist
        self.head_limit = self.config.head

        self.head_vel_limit = getattr(self.config, "head_vel", [1.0, 1.0])
        self.waist_vel_limit = getattr(self.config, "waist_vel", [1.0, 1.0])
        self.wrist_vel_limit = getattr(
            self.config, "wrist_vel", [3.5, 3.5])  # TODO
        self.pre_wrist = None
        self.pre_waist = None

        self.max_iter = 1
        self.eps = 1e-4

        self.weight = np.eye(self.num_link)

        self.deta_h = np.zeros(self.num_link)
        self.vel = np.zeros(self.num_link)
        self.end_idx = self.model.nq
        self.ee_translation = np.zeros(3)

        self.qpos_lower = np.array([t[0] for t in self.qpos_threshold])
        self.qpos_upper = np.array([t[1] for t in self.qpos_threshold])

    def inv_kin(self, target_pose, frame_id):

        # 参数设置
        damp = 1e-12
        eps = 1e-4
        oMdes = pin.SE3(target_pose[0:3, 0:3], target_pose[0:3, 3])
        # i = 0
        # while True:
        for i in range(self.max_iter):
            # 保证 qpos 是 numpy 数组（shape 一维）
            # self.qpos = np.asarray(self.qpos, dtype=float).reshape(-1)
            pin.forwardKinematics(self.model, self.data, self.qpos)     # 正运动学
            pin.updateFramePlacements(
                self.model, self.data)            # 更新坐标系位置
            oMf = self.data.oMf[frame_id]                               # 末端位姿
            if i == self.max_iter - 1:
                self.ee_translation = oMf.translation.copy()
            # 末端误差变换矩阵
            iMd = oMf.actInv(oMdes)
            err = pin.log(iMd).vector                           # 6D误差向量
            # print(f"iter {i}: error norm = {norm(err):.6f}, error = {[f'{x:.4f}' for x in err]}")

            # 角度部分限幅
            err[3:] = np.clip(err[3:], -0.4, 0.4)

            # 权重缩放
            err *= np.array([1, 1, 1, 0.8, 0.8, 0.8])

            # 收敛检测
            if np.linalg.norm(err) < eps:
                break

            J = pin.computeJointJacobian(
                self.model, self.data, self.qpos, self.end_idx)
            J = -pin.Jlog6(iMd.inverse()) @ J  # 映射到局部误差空间

            self.get_deta_h()
            # lam = damp * self.weight
            # v = -(np.linalg.pinv(J.T @ J + lam)) @ J.T @ err

            JtJ = J.T @ J
            diag_mask = np.eye(self.num_link)
            lam = damp * self.weight.diagonal()
            JtJ_reg = JtJ + np.diag(lam)

            v = solve(JtJ_reg, J.T @ err)
            v = -v

            v_output = self.limit_vel(v)
            q_next = pin.integrate(self.model, self.qpos, v_output * DT)
            self.qpos = self.limit_qpos(q_next)

    def get_deta_h(self):
        t = self.qpos_threshold
        q = self.qpos
        a = 1
        b = 1
        for i in range(self.num_link):
            deta = abs(((t[i][1] - t[i][0]) ** 2 * (2 * q[i] - t[i][0] - t[i][1])) / (
                4 * ((t[i][0] - q[i]) ** 2) * ((t[i][1] - q[i]) ** 2) + 1e-12))
            if (deta >= self.deta_h[i]):
                self.weight[i, i] = a + deta * b
            else:
                self.weight[i, i] = a

            self.weight[i, i] = min(1e20, self.weight[i, i])

    def limit_vel(self, v):
        """限幅滤波后的速度输出"""
        vel_ratio = 0.5
        max_vel = 1
        # max_vel = 0.04

        # 线性插值平滑速度，并限幅
        v_filtered = vel_ratio * \
            np.array(v) + (1 - vel_ratio) * np.array(self.vel)
        v_clamped = np.clip(v_filtered, -max_vel, max_vel)  # 支持正负速度

        # 更新历史速度
        self.vel = v_clamped.copy()

        return v_clamped

    # def limit_qpos(self, theta):
    #     for i in range(self.num_link):
    #         if (theta[i] > self.qpos_threshold[i][1]):
    #             theta[i] = self.qpos_threshold[i][1]
    #         elif (theta[i] < self.qpos_threshold[i][0]):
    #             theta[i] = self.qpos_threshold[i][0]
    #     return theta

    def limit_qpos(self, theta):
        theta = np.asarray(theta, dtype=float)
        theta = np.clip(theta, self.qpos_lower, self.qpos_upper)
        return theta

    def compute_wrist_angle(self, _tracker_pose, _controller_pose, is_left=True, _reset_flag=False, _dt=None):

        raw_angle = rotation_difference_angles(
            _tracker_pose[0:3, 0:3], _controller_pose[0:3, 0:3])
        if _reset_flag:
            self.init_raw_angle = raw_angle.copy()
            self.pre_wrist = None
            # print("reset wrist flag: ",self.init_raw_angle)
            # print("reset wrist flag")
        wrist_delta = np.array(raw_angle) - np.array(self.init_raw_angle)
        wrist_angle = [wrist_delta[0], wrist_delta[2]]
        if is_left:
            wrist_list, self.pre_wrist = limit_position_and_velocity(
                target=wrist_angle,
                prev=self.pre_wrist,
                pos_limits=self.wrist_limit_l,
                vel_limits=getattr(self, "waist_vel", self.wrist_vel_limit),
                dt=_dt,
                smooth_alpha=0.3
            )
        else:
            wrist_list, self.pre_wrist = limit_position_and_velocity(
                target=wrist_angle,
                prev=self.pre_wrist,
                pos_limits=self.wrist_limit_r,
                vel_limits=getattr(self, "waist_vel", self.wrist_vel_limit),
                dt=_dt,
                smooth_alpha=0.3
            )
        # self.pre_wrist = wrist_list.copy()
        return wrist_list

    def compute_waist_angle(self, _delta_T_pelvis=None, _reset_flag=False, _dt=None):

        raw_waist = rot_to_eul(_delta_T_pelvis)
        if _reset_flag:
            self.init_raw_waist = raw_waist.copy()
            self.pre_waist = None
        waist_delta = raw_waist - self.init_raw_waist
        waist_angle = [waist_delta[2], waist_delta[0]]

        waist_list, self.pre_waist = limit_position_and_velocity(
            target=waist_angle,
            prev=self.pre_waist,
            pos_limits=self.waist_limit,
            vel_limits=getattr(self, "waist_vel", self.waist_vel_limit),
            dt=_dt,
            smooth_alpha=0.3
        )
        # print("waist list: ", [round(x, 2) for x in waist_list])
        return waist_list

    def compute_head_angle(self, _delta_T_headset=None, _reset_flag=False, _dt=None):
        raw_head = rot_to_eul(_delta_T_headset)
        if _reset_flag:
            self.init_raw_head = raw_head.copy()
            self.pre_head = None
        head_delta = raw_head - self.init_raw_head
        head_angle = [head_delta[2], head_delta[1]]

        head_list, self.pre_head = limit_position_and_velocity(
            target=head_angle,
            prev=self.pre_head,
            pos_limits=self.head_limit,
            vel_limits=getattr(self, "head_vel", self.head_vel_limit),
            dt=_dt,
            smooth_alpha=0.3
        )
        return head_list 
