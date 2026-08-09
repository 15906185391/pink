import sys
import os

sys.path = [p for p in sys.path if "code_decoupling_and_refactoring_20250922" not in p]

from pathlib import Path
CURRENT_DIR = str(Path(__file__).parent.absolute())
sys.path.insert(0, CURRENT_DIR)
sys.path.insert(0, os.path.join(CURRENT_DIR, "lcm_data_structure"))
print("✅ 已强制使用当前项目路径：", CURRENT_DIR)

from manipulation.arm.joint_command_t import joint_command_t as arm_joint_command_t
from manipulation.head.joint_command_t import joint_command_t as head_joint_command_t
from manipulation.waist.joint_command_t import joint_command_t as waist_joint_command_t
from manipulation.leg.joint_command_t import joint_command_t as leg_joint_command_t
from manipulation.gripper.gripper_command_t import gripper_command_t
from manipulation.gripper.single_gripper_joint_command_t import single_gripper_joint_command_t
from manipulation.joint.single_joint_command_t import single_joint_command_t

import lcm
import threading
import numpy as np
import time

class LCMHandler:
    def __init__(self):
        self.dim = 23
        self.joint_current_pos = np.zeros(23)
        self.joint_current_speed = np.zeros(23)

        self.motion_mode = 0
        self.com_mode = 1
        self.num_joints = 16

        self.left_arm_FT_original = [0.0 for _ in range(6)]
        self.right_arm_FT_original = [0.0 for _ in range(6)]

        self.left_arm_moving = False
        self.right_arm_moving = False
        self.head_moving = False
        self.waist_moving = False
        self.leg_moving = False
        self.left_gripper_moving = False
        self.right_gripper_moving = False

        self.interpolation_period = 2

        self.joint_current_pos_lock = threading.Lock()
        self.data_lock = threading.Lock()
        self.left_arm_state_updated = threading.Event()
        self.right_arm_state_updated = threading.Event()
        self.head_state_updated = threading.Event()
        self.waist_state_updated = threading.Event()
        self.leg_state_updated = threading.Event()

        # Manipulation LCM - 与控制器通信 (port 8880)
        self.manip_lcm = lcm.LCM('udpm://239.255.76.67:8880?ttl=1')

        # 订阅控制器转发的Manipulation状态话题
        self.manip_lcm.subscribe('MANIP_LEFT_ARM_STATE', self.manip_left_arm_state_listener)
        self.manip_lcm.subscribe('MANIP_RIGHT_ARM_STATE', self.manip_right_arm_state_listener)
        self.manip_lcm.subscribe('MANIP_HEAD_STATE', self.manip_head_state_listener)
        self.manip_lcm.subscribe('MANIP_WAIST_STATE', self.manip_waist_state_listener)
        self.manip_lcm.subscribe('MANIP_LEG_STATE', self.manip_leg_state_listener)
        self.manip_lcm.subscribe('MANIP_LEFT_GRIPPER_STATE', self.manip_left_gripper_state_listener)
        self.manip_lcm.subscribe('MANIP_RIGHT_GRIPPER_STATE', self.manip_right_gripper_state_listener)
        self.manip_lcm.subscribe('MANIP_ROBOT_INFO', self.manip_robot_info_listener)

        # 启动 Manipulation LCM 处理线程
        self.manip_lcm_thread_handle = threading.Thread(target=self.manip_lcm_handle, daemon=True)
        self.manip_lcm_thread_handle.start()

    def manip_lcm_handle(self):
        while True:
            self.manip_lcm.handle()

    # ==================== Manipulation 状态监听器 ====================

    def manip_left_arm_state_listener(self, channel, data):
        try:
            from manipulation.arm.joint_state_t import joint_state_t as arm_joint_state_t
            msg = arm_joint_state_t.decode(data)
            with self.joint_current_pos_lock:
                for i in range(7):
                    self.joint_current_pos[i] = msg.joints[i].posH
            self.left_arm_state_updated.set()
        except Exception as e:
            pass

    def manip_right_arm_state_listener(self, channel, data):
        try:
            from manipulation.arm.joint_state_t import joint_state_t as arm_joint_state_t
            msg = arm_joint_state_t.decode(data)
            with self.joint_current_pos_lock:
                for i in range(7):
                    self.joint_current_pos[7 + i] = msg.joints[i].posH
            self.right_arm_state_updated.set()
        except Exception as e:
            pass

    def manip_left_gripper_state_listener(self, channel, data):
        try:
            from manipulation.gripper.gripper_state_t import gripper_state_t
            msg = gripper_state_t.decode(data)
            with self.joint_current_pos_lock:
                self.joint_current_pos[14] = msg.pos[0]
        except Exception as e:
            pass

    def manip_right_gripper_state_listener(self, channel, data):
        try:
            from manipulation.gripper.gripper_state_t import gripper_state_t
            msg = gripper_state_t.decode(data)
            with self.joint_current_pos_lock:
                self.joint_current_pos[15] = msg.pos[0]
        except Exception as e:
            pass

    def manip_head_state_listener(self, channel, data):
        try:
            from manipulation.head.joint_state_t import joint_state_t as head_joint_state_t
            msg = head_joint_state_t.decode(data)
            with self.joint_current_pos_lock:
                for i in range(min(2, len(msg.joints))):
                    self.joint_current_pos[16 + i] = msg.joints[i].posH
            self.head_state_updated.set()
        except Exception as e:
            pass

    def manip_waist_state_listener(self, channel, data):
        try:
            from manipulation.waist.joint_state_t import joint_state_t as waist_joint_state_t
            msg = waist_joint_state_t.decode(data)
            with self.joint_current_pos_lock:
                for i in range(min(3, len(msg.joints))):
                    self.joint_current_pos[18 + i] = msg.joints[i].posH
            self.waist_state_updated.set()
        except Exception as e:
            pass

    def manip_leg_state_listener(self, channel, data):
        try:
            from manipulation.leg.joint_state_t import joint_state_t as leg_joint_state_t
            msg = leg_joint_state_t.decode(data)
            with self.joint_current_pos_lock:
                for i in range(min(2, len(msg.joints))):
                    self.joint_current_pos[21 + i] = msg.joints[i].posH
            self.leg_state_updated.set()
        except Exception as e:
            pass

    def manip_robot_info_listener(self, channel, data):
        pass

    # ==================== 发布接口 ====================

    def upper_body_data_publisher(self, package):
        self.publish_manipulation_commands(package)

    def publish_manipulation_commands(self, package):
        """
        将23维关节位置数据拆解并发布到控制器期望的Manipulation话题
        joint mapping:
        [0:7]  -> MANIP_LEFT_ARM_CMD (7 joints)
        [7:14] -> MANIP_RIGHT_ARM_CMD (7 joints)
        [14]   -> MANIP_LEFT_GRIPPER_CMD (1 position)
        [15]   -> MANIP_RIGHT_GRIPPER_CMD (1 position)
        [16:18] -> MANIP_HEAD_CMD (2 joints)
        [18:21] -> MANIP_WAIST_CMD (3 joints)
        [21:23] -> MANIP_LEG_CMD (2 joints)
        """
        ts = time.time_ns()

        # --- 左臂 ---
        if self.left_arm_moving:
            left_arm_msg = arm_joint_command_t()
            left_arm_msg.timestamp = ts
            left_arm_msg.joints = [single_joint_command_t() for _ in range(7)]
            for i in range(7):
                left_arm_msg.joints[i].pos = float(package[i])
                left_arm_msg.joints[i].vel = 0.0
            self.manip_lcm.publish('MANIP_LEFT_ARM_CMD', left_arm_msg.encode())

        # --- 右臂 ---
        if self.right_arm_moving:
            right_arm_msg = arm_joint_command_t()
            right_arm_msg.timestamp = ts
            right_arm_msg.joints = [single_joint_command_t() for _ in range(7)]
            for i in range(7):
                right_arm_msg.joints[i].pos = float(package[7 + i])
                right_arm_msg.joints[i].vel = 0.0
            self.manip_lcm.publish('MANIP_RIGHT_ARM_CMD', right_arm_msg.encode())

        # --- 左夹爪 ---
        if self.left_gripper_moving:
            left_gripper_msg = gripper_command_t()
            left_gripper_msg.timestamp = ts
            left_gripper_msg.cmd = single_gripper_joint_command_t()
            left_gripper_msg.cmd.pos = [float(package[14])]
            self.manip_lcm.publish('MANIP_LEFT_GRIPPER_CMD', left_gripper_msg.encode())

        # --- 右夹爪 ---
        if self.right_gripper_moving:
            right_gripper_msg = gripper_command_t()
            right_gripper_msg.timestamp = ts
            right_gripper_msg.cmd = single_gripper_joint_command_t()
            right_gripper_msg.cmd.pos = [float(package[15])]
            self.manip_lcm.publish('MANIP_RIGHT_GRIPPER_CMD', right_gripper_msg.encode())

        # --- 头部 (2 joints) ---
        if self.head_moving:
            head_msg = head_joint_command_t()
            head_msg.timestamp = ts
            head_msg.joints = [single_joint_command_t() for _ in range(2)]
            for i in range(2):
                head_msg.joints[i].pos = float(package[16 + i])
                head_msg.joints[i].vel = 0.0
            self.manip_lcm.publish('MANIP_HEAD_CMD', head_msg.encode())

        # --- 腰部 (3 joints) ---
        if self.waist_moving:
            waist_msg = waist_joint_command_t()
            waist_msg.timestamp = ts
            waist_msg.joints = [single_joint_command_t() for _ in range(3)]
            for i in range(3):
                waist_msg.joints[i].pos = float(package[18 + i])
                waist_msg.joints[i].vel = 0.0
            self.manip_lcm.publish('MANIP_WAIST_CMD', waist_msg.encode())

        # --- 腿部 (2 joints) ---
        if self.leg_moving:
            leg_msg = leg_joint_command_t()
            leg_msg.timestamp = ts
            leg_msg.joints = [single_joint_command_t() for _ in range(2)]
            for i in range(2):
                leg_msg.joints[i].pos = float(package[21 + i])
                leg_msg.joints[i].vel = 0.0
            self.manip_lcm.publish('MANIP_LEG_CMD', leg_msg.encode())
