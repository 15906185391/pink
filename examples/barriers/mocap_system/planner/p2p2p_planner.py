# !/usr/bin/env python
# -*- coding: utf-8 -*-

import math
from mocap_system.planner.seven_planner_humanoid import RobotPlaner

class P2P2PPlanner:
    def __init__(self, config):
        self.config = config
           
    def get_pos_list_seven_segment(self, start, end, speed=45/180*math.pi, time_step=0.01):
        try:
            N=0
            pos_list = []
            pos_list.append(start)
            pos_list.append(end)
            planer = RobotPlaner()
            interpolate_trajectory = planer.joint_pos_ab_list_seven_segment(pos_list, speed, time_step)
            N = interpolate_trajectory.shape[0]
            return interpolate_trajectory, N
        except Exception as e:
            print("e")
            return None, 0
