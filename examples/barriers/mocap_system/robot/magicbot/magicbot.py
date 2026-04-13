import os
import yaml
import numpy as np
import pinocchio as pin

from mocap_system.robot import Robot, SubModel
from mocap_system.Utils import *
from mocap_system.visualize import *
from mocap_system.ik_solver.two_stage_ik_solver import IKSolver
# from mocap_system.robot.lcm_data_structure.lcm_unit_gripper import *

class MagicBot(Robot):
    def __init__(self):
        super().__init__("MagicBot")
        current_path = os.path.dirname(__file__)
        self.yaml_path = os.path.join(current_path, "config/Z1.yaml") 
        with open(self.yaml_path, "r", encoding="utf-8") as f:
            self.yaml_data = yaml.safe_load(f) 
        urdf_path_l = os.path.join(current_path, self.yaml_data["urdf_path_l"]["value"])
        urdf_path_r = os.path.join(current_path, self.yaml_data["urdf_path_r"]["value"])
        self.dof = 30  

        self.model_la = SubModel("LeftArm", urdf_path_l, self.yaml_path)
        self.model_ra = SubModel("RightArm", urdf_path_r, self.yaml_path)

        self.kin_la = IKSolver(self.model_la, is_left=True)
        self.kin_ra = IKSolver(self.model_ra, is_left=False) 

        self.qpos_ik = np.zeros(self.dof)     # IK 计算得到的qpos
        self.pre_qpos = None        # 上一次的qpos
        self.init_qpos = np.asarray(self.yaml_data["init_qpos"]["value"])
        # self.init_qpos[2]=0.5*np.pi
        # self.init_qpos[3]=-0.5*np.pi
        # self.init_qpos[4]=-0.5*np.pi
        # self.init_qpos[9]=-0.5*np.pi
        # self.init_qpos[10]=0.5*np.pi
        # self.init_qpos[11]=0.5*np.pi

        self.lcm_unit = lcmUnit()
        self.p2p_m = P2PMotion(self.lcm_unit)
        
        
        # # 可视化
        # full_model_path = os.path.join(current_path, self.yaml_data["urdf_path_full"]["value"])
        # self.model, self.collision_model, self.visual_model = pin.buildModelsFromUrdf(
        #     full_model_path
        # )
        
        # try:
        #     self.viz = ViserVisualizer(self.model, collision_model = None, visual_model = self.visual_model)
        #     self.viz.initViewer(open=True)
        # except ImportError as err:
        #     print("Error while initializing the viewer. It seems you should install viser")
        #     print(err)
        #     sys.exit(0)
            
        # self.viz.loadViewerModel()
                
        

        self.RECOVERPOS =  [
            0.08, 0.2, 1.0, -0.3, -1.0, 0.0, 0.1, 
            0.08, -0.2, -1.0, 0.3, 1.0, 0.0, 0.1,
            3.08, 3.08, 3.08, 3.08, 0.93, 2.86, 
            3.08, 3.08, 3.08, 3.08, 0.93, 2.86, 
            0.0, 0.0, 0.0, 0.0]

        self.READYPOS =  self.yaml_data["ready_pos"]["value"]
        
        self.GO_HOME_READY = [
            0.03, 0.52, 1.55, -1.5, -1.55, 0.0, -0.0, 
            0.03, -0.53, -1.55, 1.5, 1.55, 0.0, -0.0, 
            3.07, 3.08, 3.08, 3.08, 0.92, 2.72, 
            3.08, 3.07, 3.08, 3.07, 0.93, 2.74, 
            0.01, 0.0, 
            -0.0, -0.0]
        
        # self.ready_pos =  [
        #     0.2, 0.15, 0.0, 1.0, 0.0, 0.0, 0.0, 
        #     0.2, -0.15, 0.0, 1.0, 0.0, 0.0, 0.0,
        #     3.08, 3.08, 3.08, 3.08, 0.93, 2.86,
        #     3.08, 3.08, 3.08, 3.08, 0.93, 2.86,
        #     0.0, 0.0, 0.0, 0.0]
        self.ready_pos =  [
            0.0, 0.5, 0, 0.07, -1.5707963, 0.52, 0.0, 
            0.0, -0.5, 0, -0.07, 1.5707963, -0.52, 0.0, 
            3.08, 3.08, 3.08, 3.08, 0.93, 2.86, 
            3.08, 3.07, 3.08, 3.07, 0.93, 2.87, 
            0.01, 0.0, -0.0, 0.0]

        self.waypoint = [
            1.0, 0.7, 0.0, 0.0, -1.0, 0.0, 0.0,
            1.0, -0.7, 0.0, 0.0, 1.0, 0.0, 0.0,
            3.08, 3.08, 3.08, 3.08, 0.93, 2.87,
            3.08, 3.08, 3.08, 3.08, 0.93, 2.87,
            0.0, 0.0, 0.0, 0.0]
        
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
            while not self.p2p_m.get_current_pos():
                time.sleep(wait_interval)

            if step.get("pre_from_prev", False) and prev is not None:
                tt_pre = copy.copy(prev)
            else:
                tt_pre = copy.copy(self.p2p_m.q)

            if "full" in step:
                tt = np.array(step["full"], dtype=float)
            else:
                tt = copy.copy(tt_pre)
                for idx, val in step.get("updates", {}).items():
                    tt[idx] = val

            formatted_numbers = [round(num, 2) for num in tt]
            print(f"p2p_motion.q: {formatted_numbers}")
            # 调用原有的 act_p2p 接口执行
            self.act_p2p(self.p2p_m, tt_pre, tt)
            prev = tt        
        
        
        

    def recover(self, q_target, q_waypoint, t_target):
        planner = BezierTimeScaledPlanner(5.0, 5.0, 500)

        q_start = np.asarray(self.lcm_unit.joint_current_pos)
        q_target = np.asarray(q_target)

        q_waypoint = np.asarray(q_waypoint)
        q_waypoint = np.atleast_2d(q_waypoint)

        p = np.vstack([q_start, q_target])

        traj_original = planner.plan(p, q_waypoint)

        t_original = traj_original.shape[0] / 500.0
        ratio = t_original / t_target

        traj = self.speed_adjust(traj_original, ratio)
        return traj


    def act_p2p(self, handler, start, end):
        data_list_t, d_len = handler.get_pos_list(start, end)
        handler.reset_pose()

    def to_ready(self):
        res = self.p2p_m.get_current_pos()
        # print(p2p_motion.q)
        while(True):
            res = self.p2p_m.get_current_pos()
            # print(f"to_ready get_current_pos res: {res}")
            if res:
                '''
                回位
                '''
                # act_p2p(p2p_motion, p2p_motion.q, RECOVERPOS)
                print(f"go ready")
                
                res = self.p2p_m.get_current_pos()
                tt = copy.copy(self.p2p_m.q)
                tt[1] = 0.3
                tt[8] = -0.3
                tt[5] = 0.52
                tt[12] = -0.52
                
                # tt[1] = 0
                # tt[8] = 0
                # tt[5] = 0
                # tt[12] = 0
                
                formatted_numbers = [round(num, 2) for num in tt]
                print(f"p2p_motion.q: {formatted_numbers}")
                self.act_p2p(self.p2p_m, self.p2p_m.q, tt)
                
                
                '''
                转 3 5关节, 保证4关节超前
                '''
                tt_pre = copy.copy(tt)
                res = self.p2p_m.get_current_pos()
                tt = copy.copy(tt_pre)
                tt[0] = 0.2
                tt[7] = 0.2
                # tt[2] = 1.5707963
                tt[2] = 0
                tt[4] = -1.5707963
                # tt[9] = -1.5707963
                tt[9] = 0
                tt[11] = 1.5707963
                tt[5] = 0
                tt[6] = 0
                tt[12] = 0
                tt[26] = 0.0
                tt[27] = 0.0
                tt[28] = 0.0
                tt[29] = 0.0
                
                formatted_numbers = [round(num, 2) for num in tt]
                print(f"p2p_motion.q: {formatted_numbers}")
                self.act_p2p(self.p2p_m, tt_pre, tt)
                
                '''
                后拉 抬手
                0关节往后
                4关节往前 实际显示向地面
                '''
                tt_pre = copy.copy(tt)
                res = self.p2p_m.get_current_pos()
                tt = copy.copy(tt_pre)
                tt[0] = 1
                tt[7] = 1
                tt[1] = 0.5
                tt[8] = -0.5
                # tt[3] = -1.7
                tt[3] = -1.7+1.575
                # tt[10] = 1.7
                tt[10] = 1.7-1.575
                tt[12] = 0
                
                formatted_numbers = [round(num, 2) for num in tt]
                print(f"p2p_motion.q: {formatted_numbers}")
                self.act_p2p(self.p2p_m, tt_pre, tt)
                
                '''
                手往前推 
                
                
                0关节往后
                4关节往前 实际显示向地面
                '''
                tt_pre = copy.copy(tt)
                res = self.p2p_m.get_current_pos()
                tt = copy.copy(tt_pre)
                tt[0] = 0
                tt[7] = 0
                tt[3] = -1.5+1.575
                tt[10] = 1.5-1.575
                tt[12] = 0
                # tt[29] = 0.34
                
                tt[5] = 0.52
                tt[12] = -0.52
                formatted_numbers = [round(num, 2) for num in tt]
                print(f"p2p_motion.q: {formatted_numbers}")
                self.act_p2p(self.p2p_m, tt_pre, tt)
                
                # '''
                # 抱箱子
                # '''
                # init_pos = [
                #     0.0, 0.5,  1.7,-1.2, -1.57, 0.0, -0.0, 
                #     0.0, -0.5, -1.7, 1.2, 1.57, 0.0, -0.0, 
                #     3.07, 3.08, 3.08, 3.08, 0.92, 2.81, 
                #     3.07, 3.08, 3.08, 3.08, 0.92, 2.83, 
                #     0.01, -0.01, 0.0, 0.0]
                # act_p2p(p2p_motion, tt, init_pos)
                
                break
            time.sleep(1)
        # p2p_motion.close()
        time.sleep(1)

    def to_ready_2(self):
        res = self.p2p_m.get_current_pos()
        print(self.p2p_m.q)
        while(True):
            res = self.p2p_m.get_current_pos()
            if res:
                '''
                回位
                '''
                self.act_p2p(self.p2p_m, self.p2p_m.q, self.READYPOS)
                print(f"go ready")
                
                break
            time.sleep(1)
        # p2p_motion.close()
        time.sleep(1)

    def to_recoverstand(self):
        while(True):
            res = self.p2p_m.get_current_pos()
            print(f"res: {res}")
            if res:
                '''
                回位
                '''
                # act_p2p(p2p_motion, p2p_motion.q, GO_HOME_READY)
                # print(f"go home")
                
                
                '''
                把手收回
                '''
                tt_pre = copy.copy(self.p2p_m.q)
                res = self.p2p_m.get_current_pos()
                tt = copy.copy(self.p2p_m.q)
                tt[0] = 1
                tt[7] = 1
                tt[1] = 1
                tt[8] = -1
                # tt[3] = 0
                # tt[10] = 0
                
                
                tt_pre = copy.copy(self.p2p_m.q)
                res = self.p2p_m.get_current_pos()
                tt = copy.copy(self.p2p_m.q)
                tt[0] = 1
                tt[7] = 1
                # tt[1] = 0.33
                # tt[8] = -0.33
                tt[1] = 1
                tt[8] = -1
                
                # tt[3] = -1.7+1.575
                # tt[10] = 1.7-1.575
                tt[3] = 1.575
                tt[10] = -1.575
                
                
                # tt[0] = 0
                # tt[7] = 0
                # tt[3] = 0
                # tt[10] = 0
                
                formatted_numbers = [round(num, 2) for num in tt]
                print(f"p2p_motion.q: {formatted_numbers}")
                self.act_p2p(self.p2p_m, tt_pre, tt)
                
                '''
                把肘子放下去
                '''
                tt_pre = copy.copy(tt)
                res = self.p2p_m.get_current_pos()
                tt = copy.copy(self.p2p_m.q)
                tt[0] = 0
                tt[7] = 0
                # tt[3] = 1.575
                # tt[10] = -1.575
                formatted_numbers = [round(num, 2) for num in tt]
                print(f"self.p2p_m.q: {formatted_numbers}")
                self.act_p2p(self.p2p_m, tt_pre, tt)
                '''
                把关节收回去
                '''
                tt_pre = copy.copy(tt)
                res = self.p2p_m.get_current_pos()
                tt = copy.copy(self.p2p_m.q)
                tt[0] = 0.03
                tt[7] = 0.03
                tt[2] = 0
                tt[4] = 0
                tt[9] = 0
                tt[11] = 0
                
                formatted_numbers = [round(num, 2) for num in tt]
                print(f"self.p2p_m.q: {formatted_numbers}")
                self.act_p2p(self.p2p_m, tt_pre, tt)
                '''
                到R
                '''
                tt_pre = copy.copy(tt)
                res = self.p2p_m.get_current_pos()
                
                formatted_numbers = [round(num, 2) for num in self.RECOVERPOS]
                print(f"self.p2p_m.q: {formatted_numbers}")
                self.act_p2p(self.p2p_m, tt_pre, self.RECOVERPOS)
                break
            time.sleep(1)
        # p2p_motion.close()
        time.sleep(1)

    def two_stage_ik_process(self, _tracker_l, _tracker_r):
        # 左臂逆运动学计算
        self.kin_la.qpos = self.qpos_ik[0:5]
        frame_id_tcp_l = self.model_la.model.getFrameId(self.yaml_data["ee_frame_name_l"]["value"])
        self.kin_la.inv_kin(target_pose=_tracker_l, frame_id=frame_id_tcp_l)

        self.kin_ra.qpos = self.qpos_ik[7:12]
        frame_id_tcp_r = self.model_ra.model.getFrameId(self.yaml_data["ee_frame_name_r"]["value"])
        self.kin_ra.inv_kin(target_pose=_tracker_r, frame_id=frame_id_tcp_r)

        arm_list = [0 for _ in range(14)]
        qpos_mark_l = [1, 1, 1, 1, 1]
        qpos_mark_r = [1, 1, 1, 1, 1]
        for i in range(5):
            arm_list[i] = self.kin_la.qpos[i] * qpos_mark_l[i]
            arm_list[i + 7] = self.kin_ra.qpos[i] * qpos_mark_r[i]

        return arm_list

    def get_p2p_ready_count(self):
        return self.p2p_m.ready_count

    def get_current_qpos(self):
        res = self.p2p_m.get_current_pos()       
        if res:
            return self.p2p_m.q
        else:
            print("get_current_qpos error") 
            return False

    def prepare_p2p(self):
        self.p2p_m.start_p2p_move()
        self.p2p_m.init_ready()

    def update_p2p(self, _init_qpos):
        return self.p2p_m.update_qpos(_init_qpos)
    
    def reset_ready_count(self):
        self.p2p_m.reset_ready_count()

    def skip_p2p(self):
        self.p2p_m.skip_p2p_motion()

    def close(self):
        self.p2p_m.close()

    def get_init_ee_pose(self, is_left=True):
        if is_left:
            qpos = np.zeros(self.model_la.model.nq)
            # print(f"qpos: {qpos}")
            # qpos[2] = 0.5* np.pi
            # qpos[3] = -0.5* np.pi
            # qpos[4] = -np.pi*0.5
            init_qpos_l = self.model_la.fk(qpos, "LINK_HAND_L")
            return init_qpos_l
        else:
            qpos = np.zeros(self.model_ra.model.nq)
            # qpos[2] = -0.5* np.pi
            # qpos[3] = 0.5* np.pi
            # qpos[4] = np.pi*0.5
            init_qpos_r = self.model_ra.fk(qpos, "LINK_HAND_R")
            return init_qpos_r


