import os
import re
import yaml
import copy
import time
import numpy as np
import pinocchio as pin

from mocap_system.robot import Robot, SubModel
from mocap_system.robot.magic_hand.magic_hand import RightHand, LeftHand
from mocap_system.config.config_info import Config
from mocap_system.ik_solver.two_stage_ik_solver import IKSolver
from mocap_system.ik_solver.qp_solver import QPIKSolver
from mocap_system.robot.lcm_data_structure.lcmHandle import LCMHandler
from mocap_system.planner.p2p2p_planner import P2P2PPlanner


class MagicBotGen1(Robot):
    def __init__(self):
        super().__init__("MagicBot Gen 1")

        current_path = os.path.dirname(__file__)
        self.yaml_path = os.path.join(current_path, "config/Gen1.yaml")
        with open(self.yaml_path, "r", encoding="utf-8") as f:
            self.yaml_data = yaml.safe_load(f)

        self.config = Config()

        urdf_path_l = os.path.join(
            current_path, self.yaml_data["urdf_path_l"]["value"])
        urdf_path_r = os.path.join(
            current_path, self.yaml_data["urdf_path_r"]["value"])
        self.dof = self.yaml_data["dof"]["value"]

        self.model_la = SubModel("LeftArm", urdf_path_l, self.yaml_path)
        self.model_ra = SubModel("RightArm", urdf_path_r, self.yaml_path)

        self.kin_la = IKSolver(self.model_la, is_left=True)
        self.kin_ra = IKSolver(self.model_ra, is_left=False)

        self.qpos_ik = np.zeros(self.dof)     # IK 计算得到的qpos
        self.pre_qpos = None        # 上一次的qpos
        self.p2p2p = P2P2PPlanner(self.yaml_path)
        self.init_qpos = np.asarray(self.yaml_data["init_qpos"]["value"])
        self.ready_pose = np.asarray(self.yaml_data["ready_pos"]["value"])
        self.ee_frame_name_l = self.yaml_data["ee_frame_name_l"]["value"]
        self.ee_frame_name_r = self.yaml_data["ee_frame_name_r"]["value"]
        self.lcm_handle = LCMHandler()

        self.move2first_steps = self.parse_steps("move2first_steps")
        self.move2stand_steps = self.parse_steps("move2stand_steps")

        self.left_hand = LeftHand(config=self.config)
        self.right_hand = RightHand(config=self.config)
        
        self._frame_id_tcp_l = self.model_la.model.getFrameId(self.ee_frame_name_l)
        self._frame_id_tcp_r = self.model_ra.model.getFrameId(self.ee_frame_name_r)
        
        self._kp_array = np.ones(self.dof) * 40
        self._kd_array = np.ones(self.dof) * 100
        
        self._qpos_mark_l = np.ones(5)
        self._qpos_mark_r = np.ones(5)

    def parse_steps(self, key: str):
        raw = self.yaml_data.get(key, [])
        steps = []

        def _parse_numeric_sequence(val):
            """支持多种形式：纯字符串、数字、列表（元素可以是数字或含分隔符的字符串、嵌套列表）"""
            if isinstance(val, (int, float)):
                return np.asarray([float(val)], dtype=float)
            if isinstance(val, str):
                parts = [p for p in re.split(
                    r'[;,\s,]+', val.strip()) if p != ""]
                return np.asarray([float(p) for p in parts], dtype=float)
            if isinstance(val, (list, tuple)):
                tokens = []
                for e in val:
                    if isinstance(e, (int, float)):
                        tokens.append(float(e))
                    elif isinstance(e, str):
                        parts = [p for p in re.split(
                            r'[;,\s,]+', e.strip()) if p != ""]
                        tokens.extend([float(p) for p in parts])
                    elif isinstance(e, (list, tuple)):
                        arr = _parse_numeric_sequence(e)
                        tokens.extend(arr.tolist())
                    else:
                        raise ValueError(
                            f"Unsupported element in sequence: {type(e)}")
                return np.asarray(tokens, dtype=float)
            raise ValueError(f"Unsupported full value type: {type(val)}")

        for s in raw:
            step = {}
            if "pre_from_prev" in s:
                step["pre_from_prev"] = bool(s["pre_from_prev"])
            if "full" in s:
                try:
                    step["full"] = _parse_numeric_sequence(s["full"])
                except Exception as e:
                    raise ValueError(
                        f"Failed to parse 'full' for key {key}: {e}")
            if "updates" in s:
                # YAML 的 map key 可能是字符串，转换为 int
                step["updates"] = {int(k): float(v)
                                   for k, v in s["updates"].items()}
            steps.append(step)
        return steps

    def _run_p2p_sequence(self, steps, wait_interval=1.0):
        """
        运行一系列 p2p 步骤。
        steps: list of dict, 每个 dict 支持:
            - 'updates': {idx: value, ...}   # 基于 tt_pre 进行局部更新
            - 'full': full_list_or_array     # 直接用完整目标数组替代
            - 'pre_from_prev': bool          # 是否用上一步结果作为 tt_pre (默认 False -> 从当前 lcm q 取)
        """
        prev = None
        for step in steps:
            # 等待获取当前 q
            while not self.lcm_handle.is_get_qpos:
                time.sleep(wait_interval)

            if step.get("pre_from_prev", False) and prev is not None:
                tt_pre = copy.copy(prev)
            else:
                tt_pre = copy.copy(self.lcm_handle.joint_current_pos)

            if "full" in step:
                tt = np.array(step["full"], dtype=float)
            else:
                tt = copy.copy(tt_pre)
                for idx, val in step.get("updates", {}).items():
                    tt[idx] = val

            formatted_numbers = [round(num, 2) for num in tt]
            self.p2p2p_move(tt_pre, tt)
            prev = tt

    def p2p2p_move(self, start, end):
        traj, n = self.p2p2p.get_pos_list_seven_segment(start, end)
        for i in range(n):
            self.lcm_handle.publish_joint_command(traj[i])
            time.sleep(0.004)

    # 刚开始预设复位动作
    def move2first(self):
        # 使用从 YAML 解析的 steps
        if not getattr(self, "move2first_steps", None):
            self.move2first_steps = self.parse_steps("move2first_steps")
        self._run_p2p_sequence(self.move2first_steps)

    # 暂停复位动作，直接移动到预设复位位姿
    def move2second(self):
        # 直接使用完整目标位姿 READYPOS
        steps = [
            {"full": self.ready_pose}
        ]
        self._run_p2p_sequence(steps)

    # 接受停止复位动作，直接移动到站立位姿 STANDPOS
    def move2stand(self):
        if not getattr(self, "move2stand_steps", None):
            self.move2stand_steps = self.parse_steps("move2stand_steps")
        self._run_p2p_sequence(self.move2stand_steps)

    # 移动至准备位姿
    def move2readypos(self):
        # 直接使用完整目标位姿 READYPOS
        steps = [
            {"full": self.init_qpos}
        ]
        self._run_p2p_sequence(steps)

    # 两阶段IK，输入左右手末端位姿、追踪器位姿、控制器位姿、骨盆增量位姿，输出全身关节角度列表
    def two_stage_ik_process(self, _ee_pose_l, _ee_pose_r, _tracker_pose_l, _controller_pose_l,
                             _tracker_pose_r, _controller_pose_r, _delta_T_pelvis, _delta_T_headset, _reset_flag, _dt=None):
        upper_body_list = np.zeros(self.dof)
        # 左臂逆运动学计算
        self.kin_la.qpos = self.qpos_ik[0:5]
        self.kin_la.inv_kin(target_pose=_ee_pose_l, frame_id=self._frame_id_tcp_l)
        
        self.kin_ra.qpos = self.qpos_ik[7:12]
        self.kin_ra.inv_kin(target_pose=_ee_pose_r, frame_id=self._frame_id_tcp_r)

        # 腕部和腰部角度计算
        wrist_l = self.kin_la.compute_wrist_angle(
            _tracker_pose_l, _controller_pose_l, is_left=True, _reset_flag=_reset_flag, _dt=_dt)
        wrist_r = self.kin_ra.compute_wrist_angle(
            _tracker_pose_r, _controller_pose_r, is_left=False, _reset_flag=_reset_flag, _dt=_dt)
        waist = self.kin_la.compute_waist_angle(
            _delta_T_pelvis=_delta_T_pelvis, _reset_flag=_reset_flag, _dt=_dt)
        
        head = self.kin_la.compute_head_angle(
            _delta_T_headset=_delta_T_headset, _reset_flag=_reset_flag, _dt=_dt)

        # 腕部关节角度
        upper_body_list[6] = wrist_l[0]
        upper_body_list[5] = wrist_l[1]
        upper_body_list[13] = wrist_r[0]
        upper_body_list[12] = wrist_r[1]

        # 腰部关节角度
        upper_body_list[26] = waist[0]
        upper_body_list[27] = waist[1]
        
        # 头部关节角度
        upper_body_list[28] = head[0]
        upper_body_list[29] = head[1]

        upper_body_list[0:5] = self.kin_la.qpos * self._qpos_mark_l
        upper_body_list[7:12] = self.kin_ra.qpos * self._qpos_mark_r
        
        return upper_body_list

    # 获取当前关节角度列表
    def get_current_qpos(self):
        res = self.lcm_handle.is_get_qpos
        if res:
            return self.lcm_handle.joint_current_pos
        else:
            print("get_current_qpos error")
            return False

    # 获取初始末端位姿
    def get_init_ee_pose(self, is_left=True, _end_index=5, arm_dof=7):
        if is_left:
            qpos = self.init_qpos[:_end_index]
            init_qpos_l = self.model_la.fk(qpos, "link_tcp_l")
            return init_qpos_l
        else:
            qpos = self.init_qpos[arm_dof:arm_dof+_end_index]
            init_qpos_r = self.model_ra.fk(qpos, "link_tcp_r")
            return init_qpos_r

    def get_ee_pose(self, q=None, is_left=True):
        if is_left:
            omf = self.model_la.fk(q, "link_tcp_l")
            return omf.translation.copy()
        else:
            omf = self.model_ra.fk(q, "link_tcp_r")
            return omf.translation.copy()

    # 关闭LCM通信

    def close(self):
        self.lcm_handle.stop_lcm()
