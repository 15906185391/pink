import copy
import math
import time
import sys
import numpy as np

from mocap_system.config.config_info import Config

class MagicHand():
    def __init__(self, config=None):
        self.config = config
        if config is None:
            self.config = Config()
        self.hand_dof = 6
        self.init_finger_pos = np.array([3.0, 3.0, 3.0, 3.0, 0.92, 2.87], dtype=np.float64)
        self.limit_finger_pos = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.8], dtype=np.float64)
        self.finger_pos = self.init_finger_pos.copy()
        self.finger_vel = np.zeros(self.hand_dof)
        self.finger_acc = np.zeros(self.hand_dof)
        self.target_finger_pos = self.init_finger_pos.copy()
        self.finger_max_vel = np.ones(self.hand_dof) * 0.1
        self.finger_max_acc = np.ones(self.hand_dof) * 0.1
        self.four_finger_pos = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)
        self.last_two_finger_pos = np.array([0.5, 0.8], dtype=np.float64)
        
        
            
    def move_finger_to(self, finger_index, target_pos, speed=0.01, interval=0.01):
        """
        让指定手指以指定速度运动到目标位置
        :param finger_index: 手指索引（0 ~ hand_dof-1）
        :param target_pos: 目标位置（float）
        :param speed: 每次移动的最大步长
        :param interval: 每次移动的时间间隔（秒）
        """
        assert 0 <= finger_index < self.hand_dof, "手指索引超出范围"
        if abs(self.finger_pos[finger_index] - target_pos) > speed:
            delta = target_pos - self.finger_pos[finger_index]
            self.finger_pos[finger_index] += speed if delta > 0 else -speed
            
    def reset_to(self, step=0.01, interval=0.01):
        """
        让每个手指从当前位置以一定速度（step）返回指定目标位置
        :param target_pos: 目标位置列表，长度等于 hand_dof
        :param step: 每次移动的最大步长
        :param interval: 每次移动的时间间隔（秒）
        """
        target_pos = self.init_finger_pos.copy()
        assert len(target_pos) == self.hand_dof, "目标位置长度应等于手指自由度数"
        finished = [False] * self.hand_dof
        while not all(finished):
            for i in range(self.hand_dof):
                delta = target_pos[i] - self.finger_pos[i]
                if abs(delta) > step:
                    self.finger_pos[i] += step if delta > 0 else -step
                else:
                    self.finger_pos[i] = target_pos[i]
                    finished[i] = True
            # 可选：这里可以加上发布/更新动作
            time.sleep(interval)

    def move_four_fingers_to(self, speed=0.01, interval=0.01):
        """
        让前四个手指以指定速度运动到目标位置
        :param target_pos: 目标位置列表，长度应为4
        :param speed: 每次移动的最大步长
        :param interval: 每次移动的时间间隔（秒）
        """
        target_pos = self.four_finger_pos.copy()
        assert len(target_pos) == 4, "目标位置长度应为4"
        for i in range(4):
            self.move_finger_to(i, target_pos[i], speed, interval)
            
    def move_four_fingers_back(self, speed=0.01, interval=0.01):
        """
        让前四个手指以指定速度运动到目标位置
        :param target_pos: 目标位置列表，长度应为4
        :param speed: 每次移动的最大步长
        :param interval: 每次移动的时间间隔（秒）
        """
        target_pos = self.init_finger_pos[:4].copy()
        assert len(target_pos) == 4, "目标位置长度应为4"
        for i in range(4):
            self.move_finger_to(i, target_pos[i], speed, interval)
            
    def move_last_two_fingers_to(self, speed=0.01, interval=0.01):
        """
        让后两个手指以指定速度运动到目标位置
        :param target_pos: 目标位置列表，长度应为2
        :param speed: 每次移动的最大步长
        :param interval: 每次移动的时间间隔（秒）
        """
        target_pos = self.last_two_finger_pos.copy()
        assert len(target_pos) == 2, "目标位置长度应为2"
        for i in range(4, 6):
            self.move_finger_to(i, target_pos[i-4], speed, interval)
            
    def move_last_two_fingers_back(self, speed=0.01, interval=0.01):
        """
        让后两个手指以指定速度运动到目标位置
        :param target_pos: 目标位置列表，长度应为2
        :param speed: 每次移动的最大步长
        :param interval: 每次移动的时间间隔（秒）
        """
        target_pos = self.init_finger_pos[4:6].copy()
        assert len(target_pos) == 2, "目标位置长度应为2"
        for i in range(4, 6):
            self.move_finger_to(i, target_pos[i-4], speed, interval)
            
    def move_all_fingers_to(self, _grip_button, _trigger_button, speed=0.02, interval=0.01):
        """
        让所有手指以指定速度运动到目标位置
        :param target_pos: 目标位置列表，长度等于手指自由度数
        :param speed: 每次移动的最大步长
        :param interval: 每次移动的时间间隔（秒）
        """
        if _trigger_button is True:
            self.move_four_fingers_to(speed, interval)
        elif _trigger_button is False:
            self.move_four_fingers_back(speed, interval)
        if _grip_button is True:
            self.move_last_two_fingers_to(speed, interval)
        elif _grip_button is False:
            self.move_last_two_fingers_back(speed, interval)

class LeftHand(MagicHand):
    def __init__(self, config=None):
        super().__init__(config)
        # 可根据需要添加左手特有的属性或方法

class RightHand(MagicHand):
    def __init__(self, config=None):
        super().__init__(config)
        # 可根据需要添加右手特有的属性或方法