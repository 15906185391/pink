import numpy as np
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import TransformBroadcaster
from rclpy.node import Node
from rclpy import node
from scipy.spatial.transform import Rotation as R
import pinocchio as pin
from typing import List, Optional, Tuple

# 计算 B 相对 A 的齐次变换矩阵


def relative_transform_matrix(A, B):
    """
    计算 B 相对 A 的齐次变换矩阵 T_rel = A^{-1} B

    参数:
        A, B: 4x4 齐次变换矩阵
    返回:
        T_rel: 4x4 齐次变换矩阵，表示B在A坐标系下的相对位姿
    """
    R_A = A[0:3, 0:3]
    t_A = A[0:3, 3]

    R_B = B[0:3, 0:3]
    t_B = B[0:3, 3]

    R_rel = R_A.T @ R_B
    t_rel = R_A.T @ (t_B - t_A)

    T_rel = np.eye(4)
    T_rel[0:3, 0:3] = R_rel
    T_rel[0:3, 3] = t_rel

    return T_rel

# # 关节角度插值
# def interpolate_v2(pre_qpos, target_qpos, ratio):
#     step_qpos = [0 for _ in range(len(pre_qpos))]
#     for i in range(len(pre_qpos)):
#         step_qpos[i] = target_qpos[i] * ratio + pre_qpos[i] * (1 - ratio)
#     return step_qpos

# # 关节角度和速度插值
# def interpolate_v3(pre_qpos, target_qpos, pre_vel, ratio):
    vel = [0 for _ in range(len(target_qpos))]
    for i in range(len(target_qpos)):
        vel[i] = target_qpos[i] - pre_qpos[i]

    vel[i] = vel[i] * ratio + pre_vel[i] * (1 - ratio)

    step_qpos = [0 for _ in range(len(pre_qpos))]
    for i in range(len(target_qpos)):
        step_qpos[i] = pre_qpos[i] + vel[i]
    return step_qpos, vel

# 关节角度插值


def interpolate_v2(pre_qpos, target_qpos, ratio):
    step_qpos = [0 for _ in range(len(pre_qpos))]
    for i in range(len(pre_qpos)):
        step_qpos[i] = target_qpos[i] * ratio + pre_qpos[i] * (1 - ratio)
    return step_qpos

# 新增：速度限幅工具函数


def clamp_velocity(vel_list, max_vel):
    """
    对 vel_list 每个元素进行限幅。
    max_vel: 标量或与 vel_list 等长的序列。
    返回限幅后的速度列表。
    """
    if max_vel is None:
        return vel_list
    # 支持标量或列表
    if isinstance(max_vel, (int, float)):
        max_list = [abs(max_vel)] * len(vel_list)
    else:
        max_list = [abs(v) for v in max_vel]
    clamped = []
    for v, mv in zip(vel_list, max_list):
        mv = float(mv)
        if mv <= 0:
            clamped.append(0.0)
        else:
            clamped.append(np.sign(v) * min(abs(v), mv))
    return clamped

# 修改：关节角度和速度插值，增加速度限幅参数 max_vel


def interpolate_v3(pre_qpos, target_qpos, pre_vel, ratio, max_vel=None):
    """
    插值并限制速度：
      - 计算期望速度（target - pre）
      - 与上次速度按 ratio 融合： vel = desired*ratio + pre_vel*(1-ratio)
      - 对 vel 进行限幅（可传入标量或列表 max_vel）
      - 计算新的 qpos = pre_qpos + vel

    返回 (step_qpos, vel_limited)
    """
    n = len(target_qpos)
    # 计算期望速度（短期增量）
    desired = [target_qpos[i] - pre_qpos[i] for i in range(n)]
    # 融合上次速度与期望速度
    blended = [desired[i] * ratio +
               (pre_vel[i] if i < len(pre_vel) else 0.0) * (1 - ratio) for i in range(n)]
    # 对速度限幅
    vel_limited = clamp_velocity(blended, max_vel)
    # 更新位姿
    step_qpos = [pre_qpos[i] + vel_limited[i] for i in range(n)]
    return step_qpos, vel_limited


def interpolate_v4(pre_qpos, target_qpos, pre_vel, ratio):
    vel = [0 for _ in range(len(target_qpos))]
    for i in range(len(target_qpos)):
        vel[i] = target_qpos[i] - pre_qpos[i]

    vel[i] = vel[i] * ratio + pre_vel[i] * (1 - ratio)

    step_qpos = [0 for _ in range(len(pre_qpos))]
    for i in range(len(target_qpos)):
        step_qpos[i] = pre_qpos[i] + vel[i]
    return step_qpos, vel

# 角度滤波


def filter(target_qpos, pre_qpos):
    # 确保为 numpy 数组，避免列表与 numpy 标量直接相乘导致 TypeError
    t = np.asarray(target_qpos, dtype=float)
    if pre_qpos is None:
        return t
    p = np.asarray(pre_qpos, dtype=float)

    err = np.linalg.norm(t - p)
    max_err = 0.5
    a_min, a_max = 0.7, 1.0 
    # a_min, a_max = 0.1, 0.9 
    a = a_min + (a_max - a_min) * np.clip(err / max_err, 0.0, 1.0)

    qpos = a * t + (1.0 - a) * p
    return qpos

# 旋转矩阵转四元数


def rotation_matrix_to_quaternion(R):
    """
    将 3x3 旋转矩阵转为四元数 (x, y, z, w)
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return [x, y, z, w]

# 旋转矩阵转欧拉角


def euler_from_R_zyx(R):
    """
    从旋转矩阵求欧拉角（先Z，再X，再Y）
    返回 (z, x, y)，单位为度
    """
    R = np.asarray(R)
    if abs(R[1, 2]) < 1 - 1e-6:
        x = np.arcsin(-R[1, 2])
        z = np.arctan2(R[0, 2], R[2, 2])
        y = np.arctan2(R[1, 0], R[1, 1])
    else:
        # 万向节锁死情况
        x = np.pi/2 * np.sign(-R[1, 2])
        z = np.arctan2(-R[2, 0], R[0, 0])
        y = 0.0
    return [z, x, y]

# def rotation_difference_angles(R1, R2, degrees=False, seq='zyx'):
#     """
#     输入两个旋转矩阵，输出绕指定顺序的旋转角（默认zyx，单位为度）
#     推荐使用scipy的Rotation避免欧拉角跳变和锁死问题
#     """
#     from scipy.spatial.transform import Rotation as R
#     R_rel = R2 @ R1.T
#     rot = R.from_matrix(R_rel)
#     angles = rot.as_euler(seq, degrees=degrees)
#     return angles

# 计算相对旋转矩阵


def relative_rotation(R1, R2):
    """计算相对旋转矩阵 R_rel = R2 * R1.T"""
    return R2 @ R1.T

# 旋转矩阵和欧拉角之间的转换


def rotation_difference_angles(R1, R2):
    """输入两个旋转矩阵，输出绕 Z→X→Y 的旋转角（°）"""
    R_rel = relative_rotation(R1, R2)
    return euler_from_R_zyx(R_rel)

# 发布单个位姿


def publish_single_pose(node: Node, T, frame_id, publisher):
    msg = PoseStamped()
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.header.frame_id = "pelvis"  # 或 "reset_frame"
    msg.pose.position.x = T[0, 3]
    msg.pose.position.y = T[1, 3]
    msg.pose.position.z = T[2, 3]

    # 从旋转矩阵转四元数
    R = T[0:3, 0:3]
    quat = rotation_matrix_to_quaternion(R)

    msg.pose.orientation.x = quat[0]
    msg.pose.orientation.y = quat[1]
    msg.pose.orientation.z = quat[2]
    msg.pose.orientation.w = quat[3]

    publisher.publish(msg)

# 发布TF


def broadcast_tf(node: Node, T, parent_frame, child_frame):
    t = TransformStamped()
    t.header.stamp = node.get_clock().now().to_msg()
    t.header.frame_id = parent_frame
    t.child_frame_id = child_frame
    t.transform.translation.x = T[0, 3]
    t.transform.translation.y = T[1, 3]
    t.transform.translation.z = T[2, 3]
    R = T[0:3, 0:3]
    quat = rotation_matrix_to_quaternion(R)
    t.transform.rotation.x = quat[0]
    t.transform.rotation.y = quat[1]
    t.transform.rotation.z = quat[2]
    t.transform.rotation.w = quat[3]


def inv_transform(T: np.ndarray) -> np.ndarray:
    assert T.shape == (4, 4), "输入必须是4x4矩阵"
    R = T[:3, :3]
    t = T[:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t
    T_inv = np.eye(4)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv
    return T_inv


def eul_to_rot(euler_angles):
    rotation = R.from_euler('xyz', euler_angles)
    rotation_matrix = rotation.as_matrix()
    return rotation_matrix


def quat_to_eul(q, degrees=False):
    rotation = R.from_quat(q)
    euler_angles = rotation.as_euler('xyz', degrees=degrees)
    return euler_angles


def rot_to_eul(R_matrix, degrees=False):
    rotation = R.from_matrix(R_matrix)
    euler_angles = rotation.as_euler(
        'xyz', degrees=degrees)  # 默认使用zyx顺序，如果需要度为单位
    return euler_angles


def get_odd_data(data):
    pos = data[:3]
    if (len(data[3:]) == 4):
        eul = quat_to_eul(data[3:])
    else:
        eul = data[3:]
    rotation = eul_to_rot(eul)

    odd_matrix = np.zeros((4, 4))
    odd_matrix[3, 3] = 1
    odd_matrix[:3, :3] = rotation
    odd_matrix[:3, 3] = pos.transpose()
    return odd_matrix


def relative_pose_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    计算从坐标系A到坐标系B的相对变换矩阵 T_AB = B @ inv(A)

    参数:
        A (np.ndarray): 4x4 齐次变换矩阵 (A坐标系相对于参考系)
        B (np.ndarray): 4x4 齐次变换矩阵 (B坐标系相对于参考系)

    返回:
        T_AB (np.ndarray): 4x4 相对变换矩阵 (从A到B)
    """

    # # 相对变换 = B * A_inv（矩阵乘法）
    # T_AB = B @ A_inv

    # 假设 A, B 是 4x4 numpy 数组
    A_se3 = pin.SE3(A)
    B_se3 = pin.SE3(B)

    # Pinocchio推荐的相对变换写法
    # T_AB_se3 = A_se3.inverse() * B_se3
    # T_AB_se3 = B_se3 * A_se3.inverse()
    # 或者
    T_AB_se3 = B_se3.actInv(A_se3)

    # 得到4x4矩阵
    T_AB = T_AB_se3.homogeneous

    return T_AB


def compose_transform(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    组合两个4x4齐次变换矩阵，返回复合后的变换。
    """
    T1 = pin.SE3(A)
    T2 = pin.SE3(B)

    # 组合变换
    T_new = T1 * T2

    return T_new.homogeneous


def pose_matrix_to_eul_xyz(T):
    """
    将4x4齐次变换矩阵转为欧拉角(x, y, z)和平移(x, y, z)
    返回: (欧拉角, 平移) 都为长度3的np数组
    """
    R_mat = T[:3, :3]
    t = T[:3, 3]
    eul = R.from_matrix(R_mat).as_euler('xyz', degrees=False)
    return eul, t


def delta_pose_eul_xyz(A, B):
    """
    输入A、B为4x4齐次变换矩阵
    返回delta_eul, delta_xyz
    """
    eul_A, xyz_A = pose_matrix_to_eul_xyz(A)
    eul_B, xyz_B = pose_matrix_to_eul_xyz(B)
    delta_eul = eul_B - eul_A
    delta_xyz = xyz_B - xyz_A
    return delta_eul, delta_xyz


def compose_delta_pose_eul_xyz(T, delta_eul, delta_xyz):
    """
    输入T为4x4齐次变换矩阵，delta_eul和delta_xyz为长度3的数组
    返回新的4x4齐次变换矩阵
    """
    eul_T, xyz_T = pose_matrix_to_eul_xyz(T)
    delta_eul_mod = np.array([delta_eul[0], -delta_eul[1], -delta_eul[2]])
    # delta_eul_mod = np.array([delta_eul[2], delta_eul[1], delta_eul[0]])
    new_eul = eul_T + delta_eul
    # new_eul = eul_T + delta_eul_mod
    new_xyz = xyz_T + delta_xyz

    R_new = R.from_euler('xyz', new_eul, degrees=False).as_matrix()
    T_new = np.eye(4)
    T_new[:3, :3] = R_new
    T_new[:3, 3] = new_xyz
    return T_new


def get_relative_rotation_matrix(A, B):
    """
    输入A、B为4x4齐次变换矩阵
    返回两者之间的相对旋转矩阵 R_rel = R_B @ R_A.T
    """
    R_A = A[:3, :3]
    R_B = B[:3, :3]
    t_A = A[:3, 3]
    t_B = B[:3, 3]
    R_rel = R_B @ R_A.T
    # delta_t = t_B - t_A
    return R_rel


def get_relative_translation_rotation_matrix(A, B):
    """
    输入A、B为4x4齐次变换矩阵
    返回两者之间的相对旋转矩阵 R_rel = R_B @ R_A.T
    """
    R_A = A[:3, :3]
    R_B = B[:3, :3]
    t_A = A[:3, 3]
    t_B = B[:3, 3]
    R_rel = R_B @ R_A.T
    delta_t = t_B - t_A

    # # ==================== 保护措施 ====================
    # # 1. 检查位移是否过大
    # displacement_threshold = 0.5  # 米
    # if np.any(np.abs(delta_t) > displacement_threshold):
    #     print(f"⚠️  Protection triggered: Large displacement detected! "
    #           f"delta_t = {delta_t}, max component = {np.max(np.abs(delta_t)):.3f}m")
    #     return np.eye(3), np.zeros(3)

    # # 2. 检查旋转角度是否过大
    # angle_threshold = np.radians(45)  # 45 度转换为弧度
    # try:
    #     # 从旋转矩阵提取旋转角度（使用旋转轴 - 角表示）
    #     rotation = R.from_matrix(R_rel)
    #     angle = rotation.magnitude()  # 获取旋转角度的大小（弧度）

    #     if angle > angle_threshold:
    #         print(f"⚠️  Protection triggered: Large rotation detected! "
    #               f"angle = {np.degrees(angle):.2f}° (threshold: {np.degrees(angle_threshold):.2f}°)")
    #         return np.eye(3), np.zeros(3)
    # except Exception as e:
    #     print(f"⚠️  Warning: Error computing rotation angle: {e}")
    #     # 如果计算出错，保守处理，返回保护值
    #     return np.eye(3), np.zeros(3)

    # # ==================== 正常情况 ====================

    return R_rel, delta_t


def compose_relative_translation_rotation_matrix(A, R_rel, delta_t):
    """
    输入A为4x4齐次变换矩阵，R_rel为3x3相对旋转矩阵
    返回新的4x4齐次变换矩阵
    """
    R_A = A[:3, :3]
    t_A = A[:3, 3]
    R_new = R_rel @ R_A
    T_new = np.eye(4)
    T_new[:3, :3] = R_new
    T_new[:3, 3] = t_A + delta_t
    return T_new


class BezierTimeScaledPlanner:
    """
    Multi-segment quadratic Bezier planner
    Middle control points are PROVIDED by user
    """

    def __init__(self, vmax, amax, fps):
        """
        vmax : 标量
        amax : 标量
        fps  : 采样频率
        """
        self.vmax = vmax
        self.amax = amax
        self.fps = fps

    # =========================
    # 主接口
    # =========================
    def plan(self, p, p_mid):
        """
        p      : (N+1, D)  起点 + 终点
        p_mid  : (N,   D)  每段 Bezier 的中间控制点
        return : (M, D)    轨迹
        """
        p = np.asarray(p)
        p_mid = np.asarray(p_mid)

        assert p.shape[0] == p_mid.shape[0] + 1, \
            "p must be one longer than p_mid"

        traj = []

        for i in range(p_mid.shape[0]):
            P0 = p[i]
            P1 = p_mid[i]
            P2 = p[i + 1]

            # ===== Step 1: 路径长度估计 =====
            L = self._estimate_length(P0, P1, P2)

            # ===== Step 2: 时间计算 =====
            T = self._compute_time(L)

            # ===== Step 3: 采样点数 =====
            N = max(int(np.ceil(T * self.fps)), 2)

            # ===== Step 4: 时间缩放 =====
            s, _, _ = self._quintic_time_scaling(T, N)

            # ===== Step 5: Bezier 插值 =====
            seg_traj = self._bezier_quadratic(P0, P1, P2, s)
            traj.append(seg_traj)

        return np.vstack(traj)

    def _estimate_length(self, P0, P1, P2, n_sample=50):
        s = np.linspace(0.0, 1.0, n_sample)
        curve = self._bezier_quadratic(P0, P1, P2, s)
        return np.sum(np.linalg.norm(np.diff(curve, axis=0), axis=1))

    def _compute_time(self, L):
        T_vel = L / self.vmax
        T_acc = np.sqrt(L / self.amax)
        return max(T_vel, T_acc)

    def _quintic_time_scaling(self, T, N):
        t = np.linspace(0.0, T, N)
        tau = t / T

        s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
        sd = (30 / T) * tau**2 - (60 / T) * tau**3 + (30 / T) * tau**4
        sdd = (60 / T**2) * tau - (180 / T**2) * tau**2 + (120 / T**2) * tau**3

        return s, sd, sdd

    def _bezier_quadratic(self, P0, P1, P2, s):
        s = s[:, None]
        return (
            (1 - s)**2 * P0
            + 2 * (1 - s) * s * P1
            + s**2 * P2
        )


def limit_values(raw_values, limits):
    """
    通用限位函数
    :param raw_values: 原始数值列表
    :param limits: 每个元素的限幅区间列表，如 [(lo1, hi1), (lo2, hi2), ...]
    :return: 限幅后的数值列表
    """
    limited = raw_values.copy()
    try:
        for i in range(min(len(raw_values), len(limits))):
            lo, hi = limits[i]
            limited[i] = max(min(raw_values[i], hi), lo)
    except Exception:
        # 若配置缺失或格式异常，退回不做限幅
        limited = raw_values
    return limited


def limit_position_and_velocity(
    target: List[float],
    prev: Optional[List[float]],
    pos_limits: Optional[List[Tuple[float, float]]],
    vel_limits: Optional[List[float]],
    dt: float,
    smooth_alpha: Optional[float] = None
):
    """
    对向量做：1) 位置限幅（pos_limits），2) 基于 dt 的速度限幅（vel_limits），3) 可选的平滑插值。
    - target: 目标位置向量（list-like）
    - prev: 上一次输出（若 None 则直接返回限幅后目标）
    - pos_limits: 每轴 (lo, hi) 列表或 None
    - vel_limits: 每轴最大速度（绝对值），list 或 None
    - dt: 时间差（秒），必须 > 0
    - smooth_alpha: 若给出 (0..1) 则在速度限幅后应用指数平滑：out = prev + alpha*(limited - prev)
    返回 (limited_out_list, new_prev_list)
    """
    tgt = np.array(target, dtype=float)
    if pos_limits is not None:
        lim = []
        for i, val in enumerate(tgt):
            try:
                lo, hi = pos_limits[i]
                lim.append(np.clip(val, lo, hi))
            except Exception:
                lim.append(val)
        tgt = np.array(lim, dtype=float)

    if prev is None:
        out = tgt.copy()
    else:
        prev_a = np.array(prev, dtype=float)
        out = tgt.copy()
        if vel_limits is not None and dt > 0:
            max_delta = np.array(vel_limits, dtype=float) * float(dt)
            delta = tgt - prev_a
            clamped = np.clip(delta, -max_delta, max_delta)
            out = prev_a + clamped

    if smooth_alpha is not None and prev is not None:
        a = float(smooth_alpha)
        out = np.array(prev, dtype=float) + a * \
            (out - np.array(prev, dtype=float))

    return out.tolist(), out.tolist()

class SE3LowPassFilter:
    """First-order filter for SE3 poses to prevent sudden jumps in reference commands.

    Smoothly interpolates between the current filtered pose and the target pose
    using exponential filtering for position and SLERP for orientation.
    """

    def __init__(self, tau: float = 0.1):
        """Initialize the filter.

        Args:
            tau: Time constant in seconds. Larger values = slower, smoother tracking.
                 tau=0.1 means ~63% of the step is taken per 0.1 second.
        """
        self.tau = tau
        self.filtered_position: np.ndarray | None = None
        self.filtered_quaternion: pin.Quaternion | None = None

    def reset(self, pose: np.ndarray) -> None:
        """Reset the filter state to a specific pose.

        Args:
            pose: 4x4 homogeneous transformation matrix.
        """
        self.filtered_position = pose[:3, 3].copy()
        self.filtered_quaternion = pin.Quaternion(pose[:3, :3])

    def update(self, target_pose: np.ndarray, dt: float) -> np.ndarray:
        """Filter the target pose and return the smoothed result.

        Args:
            target_pose: 4x4 homogeneous transformation matrix (target).
            dt: Time step in seconds.

        Returns:
            4x4 homogeneous transformation matrix (filtered).
        """
        target_position = target_pose[:3, 3]
        target_quaternion = pin.Quaternion(target_pose[:3, :3])

        # Initialize on first call
        if self.filtered_position is None:
            self.filtered_position = target_position.copy()
            self.filtered_quaternion = target_quaternion
            return target_pose.copy()

        # Exponential filter coefficient: alpha = 1 - exp(-dt/tau)
        # alpha -> 0 as tau -> inf (slower), alpha -> 1 as tau -> 0 (faster)
        alpha = 1.0 - np.exp(-dt / self.tau)

        # Filter position (linear interpolation)
        self.filtered_position = self.filtered_position + alpha * (
            target_position - self.filtered_position
        )

        # Filter orientation (SLERP)
        self.filtered_quaternion = self.filtered_quaternion.slerp(
            alpha, target_quaternion
        )

        # Construct filtered pose
        filtered_pose = np.eye(4)
        filtered_pose[:3, :3] = self.filtered_quaternion.toRotationMatrix()
        filtered_pose[:3, 3] = self.filtered_position

        return filtered_pose