import numpy as np
import math
import os
import yaml
import copy
import pinocchio as pin
from mocap_system.Utils import *


class Interface:
    def __init__(self, name, module_name):
        self.name = name
        self.module_name = module_name


class SensorPose(Interface):
    """位姿接口，提供4x4矩阵、位置、姿态、增量"""

    def __init__(self, name, module_name):
        super().__init__(name, module_name)
        self.pose_matrix_init = np.eye(4)       # reset 时保存的初始世界位姿
        self.pose_matrix_init_inv = np.eye(4)       # reset 时保存的初始世界位姿

        self.pose_matrix = np.eye(4)            # 当前世界位姿
        self.pose_matrix_prev = np.eye(4)       # 上一时刻世界位姿

        self.translation_relative = np.zeros(3)  # 相对 init 的平移（在 reset frame 下）
        self.rotation_relative = np.eye(3)   # 相对 init 的位姿（在 reset frame 下）

        self.pose_matrix_relative = np.eye(4)  # 相对 reset_frame 的位姿（完整 4x4 矩阵）

        # 用于后续 update 中将 world -> reset_frame
        self.T_init_reset_inv = np.eye(4)

    def update(self, pose_matrix):
        # 更新历史状态
        self.pose_matrix_prev = self.pose_matrix.copy()
        self.pose_matrix = pose_matrix.copy()

    def get_delta_translation_rotation(self):
        """获取从上一帧到当前帧的增量平移和旋转"""
        delta_T = self.relative_transform(
            self.pose_matrix_prev, self.pose_matrix)
        translation = delta_T[0:3, 3].copy()
        rotation = delta_T[0:3, 0:3].copy()
        return translation, rotation

    @staticmethod
    def relative_transform(T_from, T_to):
        """
        计算 T_to 相对于 T_from 的 SE(3) 变换：
            T_relative = T_from^{-1} @ T_to

        表示：从 T_from 出发，如何到达 T_to？
        """
        R_A = T_from[0:3, 0:3]  # R_from
        t_A = T_from[0:3, 3]    # t_from
        R_B = T_to[0:3, 0:3]    # R_to
        t_B = T_to[0:3, 3]      # t_to

        R_rel = R_A.T @ R_B
        t_rel = R_A.T @ (t_B - t_A)

        T_rel = np.eye(4)
        T_rel[0:3, 0:3] = R_rel
        T_rel[0:3, 3] = t_rel
        return T_rel


class BooleanButton(Interface):
    """二值按钮接口"""

    def __init__(self, name, module_name, init_state=False):
        super().__init__(name, module_name)
        self.button_state = init_state
        self.button_state_prev = init_state

    def update(self, button_state):
        self.button_state_prev = self.button_state
        self.button_state = bool(button_state)

    def is_button_pressed(self):
        # return self.button_state and not self.button_state_prev
        return self.button_state

    def is_button_released(self):
        return not self.button_state and self.button_state_prev


class TravelButton(Interface):
    """模拟量按钮（扳机），有行程值"""

    def __init__(self, name, module_name, min_value=0.0, max_value=1.0, init_value=0.0):
        super().__init__(name, module_name)
        self.min_value = min_value
        self.max_value = max_value
        self.travel_value = init_value
        self.travel_value_prev = init_value

    def update(self, value):
        self.travel_value_prev = self.travel_value
        self.travel_value = max(self.min_value, min(self.max_value, value))

    def get_travel_value(self):
        return self.travel_value

    def get_delta(self):
        return self.travel_value - self.travel_value_prev

    def is_pressed_beyond(self, threshold):
        return self.travel_value >= threshold

    def is_released_below(self, threshold):
        return self.travel_value <= threshold


class Joystick(Interface):
    """二维摇杆接口"""

    def __init__(self, name, module_name, min_value=-1.0, max_value=1.0):
        super().__init__(name, module_name)
        self.min_value = min_value
        self.max_value = max_value
        self.x = 0.0
        self.y = 0.0
        self.x_prev = 0.0
        self.y_prev = 0.0

    def update(self, x, y):
        self.x_prev, self.y_prev = self.x, self.y
        self.x = max(self.min_value, min(self.max_value, x))
        self.y = max(self.min_value, min(self.max_value, y))

    def get_position(self):
        return self.x, self.y

    def get_delta(self):
        return self.x - self.x_prev, self.y - self.y_prev

    def get_magnitude(self):
        return min(1.0, (self.x ** 2 + self.y ** 2) ** 0.5)

    def get_direction(self):
        magnitude = self.get_magnitude()
        if magnitude == 0:
            return 0.0, 0.0
        return self.x / magnitude, self.y / magnitude

    def is_moved(self, threshold=0.1):
        return self.get_magnitude() >= threshold


class Module:
    def __init__(self, name, device_name):
        self.name = name
        self.device_name = device_name
        self.interfaces = []

    def add_interface(self, interface):
        self.interfaces.append(interface)

    def get_interfaces(self, name):
        return next((sm for sm in self.interfaces if sm.name == name), None)

    def update(self):
        for iface in self.interfaces:
            iface.update()


# 工厂映射表：根据 type 创建对应的接口类
INTERFACE_REGISTRY = {
    "SensorPose": SensorPose,
    "BooleanButton": BooleanButton,
    "TravelButton": TravelButton,
    "Joystick": Joystick,
}


class Device:
    def __init__(self, name):
        self.name = name
        self.modules = []
        self.T_global_reset = np.eye(4)
        self.T_global_reset_inv = np.eye(4)
        self.T_last_tracker_l = None
        self.T_last_tracker_r = None
        # self.T_corr = np.array([
        #     [0,  0,   1,  0],  # new X = old Z (forward)
        #     [1,  0,   0,  0],  # new Y = old -X (left)
        #     [0,  -1,  0,  0],   # new Z = old Y (up)
        #     [0,  0,   0,  1]
        # ])

        self.T_corr = np.array([
            [0,  0,   1,  0],  # new X = old Z (forward)
            [1,  0,   0,  0],  # new Y = old -X (left)
            [0,  -1,  0,  0],   # new Z = old Y (up)
            [0,  0,   0,  1]
        ])

        self.T_temp = np.eye(4, dtype=np.float64)

        self.T_tracker_l_corr = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])

        self.T_tracker_r_corr = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])

        self.init_tracker_l = None
        self.init_tracker_r = None
        self.T_init_pelvis_tracker = None
        self.T_init_headset = None

        self._iface_cache = {}

    # pico4ultra
    def reset(self):
        # # 获取当前世界位姿
        # T_headset = self.get_modules("Headset").get_interfaces("HeadsetPose").pose_matrix.copy()
        # T_left = self.get_modules("LeftController").get_interfaces("LeftControllerPose").pose_matrix.copy()
        # T_right = self.get_modules("RightController").get_interfaces("RightControllerPose").pose_matrix.copy()

        # T_Tracker_pelvis = self.get_modules("PelvisTracker").get_interfaces("PelvisTrackerPose").pose_matrix.copy()
        # T_Tracker_left = self.get_modules("LeftTracker").get_interfaces("LeftTrackerPose").pose_matrix.copy()
        # T_Tracker_right = self.get_modules("RightTracker").get_interfaces("RightTrackerPose").pose_matrix.copy()

        headset_module = self.get_modules("Headset")
        left_controller_module = self.get_modules("LeftController")
        right_controller_module = self.get_modules("RightController")
        pelvis_tracker_module = self.get_modules("PelvisTracker")
        left_tracker_module = self.get_modules("LeftTracker")
        right_tracker_module = self.get_modules("RightTracker")

        T_headset = headset_module.get_interfaces(
            "HeadsetPose").pose_matrix.copy()
        T_left = left_controller_module.get_interfaces(
            "LeftControllerPose").pose_matrix.copy()
        T_right = right_controller_module.get_interfaces(
            "RightControllerPose").pose_matrix.copy()

        T_Tracker_pelvis = pelvis_tracker_module.get_interfaces(
            "PelvisTrackerPose").pose_matrix.copy()
        T_Tracker_left = left_tracker_module.get_interfaces(
            "LeftTrackerPose").pose_matrix.copy()
        T_Tracker_right = right_tracker_module.get_interfaces(
            "RightTrackerPose").pose_matrix.copy()

        pos_h = T_headset[0:3, 3]  # 头显位置 → 作为 reset frame 的原点
        pos_l = T_left[0:3, 3]
        pos_r = T_right[0:3, 3]

        pos_tracker_pelvis = T_Tracker_pelvis[0:3, 3]
        pos_tracker_left = T_Tracker_left[0:3, 3]
        pos_tracker_right = T_Tracker_right[0:3, 3]

        def normalize(v):
            norm = np.linalg.norm(v)
            return v / norm if norm > 1e-8 else np.zeros_like(v)

        # -----------------------------
        # 构建统一的方向基底
        # -----------------------------
        y_axis = normalize(pos_l - pos_r)                    # Y: 左手 → 右手（右向）
        z_axis = normalize(pos_h - pos_tracker_pelvis)      # Z: 向上
        x_axis = normalize(np.cross(z_axis, y_axis))         # X: 前进方向（右手定则）

        if any(np.allclose(d, 0.) for d in [x_axis, y_axis, z_axis]):
            print("Failed to build valid coordinate frame during reset.")
            return

        # -----------------------------
        # 创建 T_reset：原点在 pos_h，方向由 x/y/z 定义
        # -----------------------------
        T_reset = np.eye(4)
        T_reset[0:3, 0] = x_axis
        T_reset[0:3, 1] = y_axis
        T_reset[0:3, 2] = z_axis
        T_reset[0:3, 3] = pos_tracker_pelvis  # 设置原点

        # 缓存 T_reset 和其逆：用于将世界坐标转到 reset frame
        self.T_global_reset = T_reset
        self.T_global_reset_inv = self.inv(T_reset)

        # 存储初始位姿及其逆

        # self.get_modules("Headset").get_interfaces("HeadsetPose").pose_matrix_init_inv = self.inv(T_headset)
        # self.get_modules("LeftController").get_interfaces("LeftControllerPose").pose_matrix_init_inv = self.inv(T_left)
        # self.get_modules("RightController").get_interfaces("RightControllerPose").pose_matrix_init_inv = self.inv(T_right)

        # self.get_modules("PelvisTracker").get_interfaces("PelvisTrackerPose").pose_matrix_init_inv = self.inv(T_Tracker_pelvis)
        # self.get_modules("LeftTracker").get_interfaces("LeftTrackerPose").pose_matrix_init_inv = self.inv(T_Tracker_left)
        # self.get_modules("RightTracker").get_interfaces("RightTrackerPose").pose_matrix_init_inv = self.inv(T_Tracker_right)

        headset_module.get_interfaces(
            "HeadsetPose").pose_matrix_init_inv = self.inv(T_headset)
        left_controller_module.get_interfaces(
            "LeftControllerPose").pose_matrix_init_inv = self.inv(T_left)
        right_controller_module.get_interfaces(
            "RightControllerPose").pose_matrix_init_inv = self.inv(T_right)

        pelvis_tracker_module.get_interfaces(
            "PelvisTrackerPose").pose_matrix_init_inv = self.inv(T_Tracker_pelvis)
        left_tracker_module.get_interfaces(
            "LeftTrackerPose").pose_matrix_init_inv = self.inv(T_Tracker_left)
        right_tracker_module.get_interfaces(
            "RightTrackerPose").pose_matrix_init_inv = self.inv(T_Tracker_right)

    def update_relative(self):
        if not hasattr(self, 'T_global_reset_inv'):
            print("Reset has not been called yet.")
            return None, None, None, None, None, None, None, None, None, None, None

        # # 获取当前世界位姿
        # T_head = self.get_modules("Headset").get_interfaces(
        #     "HeadsetPose").pose_matrix
        # T_left = self.get_modules("LeftController").get_interfaces(
        #     "LeftControllerPose").pose_matrix
        # T_right = self.get_modules("RightController").get_interfaces(
        #     "RightControllerPose").pose_matrix
        # T_tracker_pelvis = self.get_modules(
        #     "PelvisTracker").get_interfaces("PelvisTrackerPose").pose_matrix
        # T_tracker_left = self.get_modules(
        #     "LeftTracker").get_interfaces("LeftTrackerPose").pose_matrix
        # T_tracker_right = self.get_modules(
        #     "RightTracker").get_interfaces("RightTrackerPose").pose_matrix

        # # 将世界坐标转到 reset frame
        # T_head_rel = self.T_corr @ self.get_modules("PelvisTracker").get_interfaces(
        #     "PelvisTrackerPose").pose_matrix_init_inv @ T_head @ self.T_corr.T
        # T_left_rel = self.T_corr @ self.get_modules("PelvisTracker").get_interfaces(
        #     "PelvisTrackerPose").pose_matrix_init_inv @ T_left @ self.T_corr.T
        # T_right_rel = self.T_corr @ self.get_modules("PelvisTracker").get_interfaces(
        #     "PelvisTrackerPose").pose_matrix_init_inv @ T_right @ self.T_corr.T
        # T_tracker_pelvis_rel = self.T_corr @ self.get_modules("PelvisTracker").get_interfaces(
        #     "PelvisTrackerPose").pose_matrix_init_inv @ T_tracker_pelvis @ self.T_corr.T
        # T_tracker_left_rel = self.T_corr @ self.get_modules("PelvisTracker").get_interfaces(
        #     "PelvisTrackerPose").pose_matrix_init_inv @ T_tracker_left @ self.T_corr.T
        # T_tracker_right_rel = self.T_corr @ self.get_modules("PelvisTracker").get_interfaces(
        #     "PelvisTrackerPose").pose_matrix_init_inv @ T_tracker_right @ self.T_corr.T
        
        # self.get_modules("Headset").get_interfaces(
        #     "HeadsetPose").pose_matrix_relative = self.inv(T_tracker_pelvis_rel) @ T_head_rel
        # self.get_modules("LeftController").get_interfaces(
        #     "LeftControllerPose").pose_matrix_relative = self.inv(T_tracker_pelvis_rel) @ T_left_rel
        # self.get_modules("RightController").get_interfaces(
        #     "RightControllerPose").pose_matrix_relative = self.inv(T_tracker_pelvis_rel) @ T_right_rel
        # self.get_modules("PelvisTracker").get_interfaces(
        #     "PelvisTrackerPose").pose_matrix_relative = T_tracker_pelvis_rel
        # self.get_modules("LeftTracker").get_interfaces(
        #     "LeftTrackerPose").pose_matrix_relative = self.inv(T_tracker_pelvis_rel) @ T_tracker_left_rel
        # self.get_modules("RightTracker").get_interfaces(
        #     "RightTrackerPose").pose_matrix_relative = self.inv(T_tracker_pelvis_rel) @ T_tracker_right_rel

        # self.T_new_tracker_l = self.get_modules("LeftTracker").get_interfaces(
        #     "LeftTrackerPose").pose_matrix_relative.copy()  @ self.T_tracker_l_corr
        # self.T_new_tracker_r = self.get_modules("RightTracker").get_interfaces(
        #     "RightTrackerPose").pose_matrix_relative.copy() @ self.T_tracker_r_corr

        # self.T_pelvis_tracker = self.get_modules("PelvisTracker").get_interfaces(
        #     "PelvisTrackerPose").pose_matrix_relative.copy()
        # self.T_head_tracker = self.get_modules("Headset").get_interfaces(
        #     "HeadsetPose").pose_matrix_relative.copy()

        # if self.init_tracker_l is None or self.init_tracker_r is None:
        #     self.init_tracker_l = self.T_new_tracker_l.copy()
        #     self.init_tracker_r = self.T_new_tracker_r.copy()

        # if self.T_init_pelvis_tracker is None:
        #     self.T_init_pelvis_tracker = self.T_pelvis_tracker.copy()

        # delta_T_l, delta_t_l = get_relative_translation_rotation_matrix(
        #     self.init_tracker_l, self.T_new_tracker_l)
        # delta_T_r, delta_t_r = get_relative_translation_rotation_matrix(
        #     self.init_tracker_r, self.T_new_tracker_r)

        # delta_T_pelvis = get_relative_rotation_matrix(
        #     self.T_init_pelvis_tracker, self.T_pelvis_tracker)

        # return delta_T_l, delta_t_l, delta_T_r, delta_t_r, self.T_new_tracker_l, self.T_new_tracker_r, self.init_tracker_l, self.init_tracker_r, self.T_pelvis_tracker, self.T_head_tracker, delta_T_pelvis

        pelvis_tracker_module = self.get_modules("PelvisTracker")
        pelvis_pose = pelvis_tracker_module.get_interfaces("PelvisTrackerPose")
        
        T_corr = self.T_corr
        T_corr_T = T_corr.T
        pelvis_inv = pelvis_pose.pose_matrix_init_inv
        
        T_head = self.get_modules("Headset").get_interfaces("HeadsetPose").pose_matrix
        T_left = self.get_modules("LeftController").get_interfaces("LeftControllerPose").pose_matrix
        T_right = self.get_modules("RightController").get_interfaces("RightControllerPose").pose_matrix
        T_tracker_pelvis = pelvis_pose.pose_matrix
        T_tracker_left = self.get_modules("LeftTracker").get_interfaces("LeftTrackerPose").pose_matrix
        T_tracker_right = self.get_modules("RightTracker").get_interfaces("RightTrackerPose").pose_matrix

        common_transform = T_corr @ pelvis_inv
        
        T_head_rel = common_transform @ T_head @ T_corr_T
        T_left_rel = common_transform @ T_left @ T_corr_T
        T_right_rel = common_transform @ T_right @ T_corr_T
        T_tracker_pelvis_rel = common_transform @ T_tracker_pelvis @ T_corr_T
        T_tracker_left_rel = common_transform @ T_tracker_left @ T_corr_T
        T_tracker_right_rel = common_transform @ T_tracker_right @ T_corr_T

        tracker_pelvis_rel_inv = self.inv(T_tracker_pelvis_rel)
        
        self.get_modules("Headset").get_interfaces("HeadsetPose").pose_matrix_relative = tracker_pelvis_rel_inv @ T_head_rel
        self.get_modules("LeftController").get_interfaces("LeftControllerPose").pose_matrix_relative = tracker_pelvis_rel_inv @ T_left_rel
        self.get_modules("RightController").get_interfaces("RightControllerPose").pose_matrix_relative = tracker_pelvis_rel_inv @ T_right_rel
        pelvis_pose.pose_matrix_relative = T_tracker_pelvis_rel
        self.get_modules("LeftTracker").get_interfaces("LeftTrackerPose").pose_matrix_relative = tracker_pelvis_rel_inv @ T_tracker_left_rel
        self.get_modules("RightTracker").get_interfaces("RightTrackerPose").pose_matrix_relative = tracker_pelvis_rel_inv @ T_tracker_right_rel
        
        T_new_tracker_l = self.get_modules("LeftTracker").get_interfaces("LeftTrackerPose").pose_matrix_relative @ self.T_tracker_l_corr
        T_new_tracker_r = self.get_modules("RightTracker").get_interfaces("RightTrackerPose").pose_matrix_relative @ self.T_tracker_r_corr
        
        self.T_pelvis_tracker = pelvis_pose.pose_matrix_relative.copy()
        self.T_head_tracker = self.get_modules("Headset").get_interfaces("HeadsetPose").pose_matrix_relative.copy()
        
        if self.init_tracker_l is None or self.init_tracker_r is None:
            self.init_tracker_l = T_new_tracker_l.copy()
            self.init_tracker_r = T_new_tracker_r.copy()
            
        if self.T_init_pelvis_tracker is None:
            self.T_init_pelvis_tracker = self.T_pelvis_tracker.copy()
            
        if self.T_init_headset is None:
            self.T_init_headset = self.T_head_tracker.copy()
            
        
        
        delta_T_l, delta_t_l = get_relative_translation_rotation_matrix(self.init_tracker_l, T_new_tracker_l)
        delta_T_r, delta_t_r = get_relative_translation_rotation_matrix(self.init_tracker_r, T_new_tracker_r)
        
        delta_T_pelvis = get_relative_rotation_matrix(self.T_init_pelvis_tracker, self.T_pelvis_tracker)
        delta_T_head = get_relative_rotation_matrix(self.T_init_headset, self.T_head_tracker)
        
        return delta_T_l, delta_t_l, delta_T_r, delta_t_r, T_new_tracker_l, T_new_tracker_r, self.init_tracker_l, self.init_tracker_r, self.T_pelvis_tracker, self.T_head_tracker, delta_T_pelvis, delta_T_head
        

    def inv(self, T):
        T_inv = np.eye(4)
        R = T[0:3, 0:3]
        t = T[0:3, 3]
        T_inv[0:3, 0:3] = R.T
        T_inv[0:3, 3] = -R.T @ t
        return T_inv

    def add_module(self, module):
        self.modules.append(module)

    def get_modules(self, name):
        return next((sm for sm in self.modules if sm.name == name), None)

    @classmethod
    def from_yaml(cls, yaml_path: str):
        """从 YAML 文件创建一个 Device 对象"""
        with open(yaml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        device_cfg = config["device"]
        device = cls(
            name=device_cfg["name"]
        )

        for module_cfg in config["modules"]:
            module = Module(
                name=module_cfg["name"],
                device_name=device.name
            )

            for iface_cfg in module_cfg.get("interfaces", []):
                iface_class = INTERFACE_REGISTRY[iface_cfg["type"]]
                params = iface_cfg.get("params", {})

                interface = iface_class(
                    name=iface_cfg["name"],
                    module_name=module.name,
                    **params
                )
                module.add_interface(interface)

            device.add_module(module)

        return device

    def odd_trans(self, odd, is_left):
        odd_trans = np.zeros((4, 4))
        # if self.config.sensor_position == "Thumb_side":
        # 大拇指侧
        if (is_left):
            odd_trans[:3, :3] = eul_to_rot(
                np.array((-1.5708, 1.5708, 3.1415926)))  # 未知 x ；未知
        else:
            odd_trans[:3, :3] = eul_to_rot(
                np.array((-1.5708, 1.5708, 3.1415926)))
        odd_trans[3, 3] = 1
        odd_new = odd @ odd_trans
        return odd_new

    def prepare(self, pos_eul, base_pos_eul, is_left):
        self.init_pos_eul_base = pos_eul.copy()
        data_odd = get_odd_data(pos_eul)
        base_odd = get_odd_data(np.array(base_pos_eul))
        # data_odd_trans = np.linalg.inv(base_odd) @ data_odd
        data_odd_trans = inv_transform(base_odd) @ data_odd
        odd = self.odd_trans(data_odd_trans, is_left)
        return odd

    def update(self, pos_eul, base_pos_eul):
        self.pos_eul_base = pos_eul.copy()
        data_odd = get_odd_data(pos_eul)
        base_odd = get_odd_data(np.array(base_pos_eul))
        # data_odd_trans = np.linalg.pinv(self.init_odd) @  self.odd_trans(np.linalg.inv(base_odd) @ data_odd, self.is_left)
        data_odd_trans = self.inv_transform(
            self.init_odd) @  self.odd_trans(self.inv_transform(base_odd) @ data_odd, self.is_left)
        deta_eul = rot_to_eul(data_odd_trans[:3, :3])
        for i in range(3):
            self.deta_pos_eul[i] = data_odd_trans[i, 3] * self.ratio[i]
            self.deta_pos_eul[i+3] = deta_eul[i]


if __name__ == "__main__":
    path = os.path.join(os.path.dirname(__file__), "config/pico4_ultra.yaml")
    pico = Device.from_yaml(path)

    print("Device:", pico.name)
    print(pico.get_modules("RightController").get_interfaces(
        "ButtonA").is_button_pressed())
