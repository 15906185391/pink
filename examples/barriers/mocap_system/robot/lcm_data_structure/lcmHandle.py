import lcm
import threading
import numpy as np
import time
from mocap_system.robot.lcm_data_structure.upper_body_cmd_package import upper_body_cmd_package
from mocap_system.robot.lcm_data_structure.upper_body_data_package import upper_body_data_package
from mocap_system.robot.lcm_data_structure.left_tracker_pos import left_tracker_pos


class LCMHandler:
    def __init__(self, dim = 30):
        """LCM 通信处理类：接收机器人关节状态，发送关节指令"""
        self.dim = dim
        self.joint_current_pos = np.zeros(self.dim)
        self.joint_current_speed = np.zeros(self.dim)
        
        self.is_used = None
        self.error_code = None
        self.status = None
        self.update_once_arm = False
        self.running = True
        self.is_get_qpos = True

        self.plan_pre_qpos = None
        self.plan_pre_speed = None

        self.data_lock = threading.Lock()
        self.joint_current_pos_updated = threading.Event()

        # LCM 参数
        self.lcm_from_robot_period = 2  # ms
        self.lcm = lcm.LCM('udpm://239.255.76.67:7667?ttl=1')
        self.lcm.subscribe('upper_body_data', self.upper_body_data_listener)

        # 默认伺服模式
        self.default_control_mode = [4] * 30
        for idx in [5, 6, 12, 13]:  # 左右臂的6、7关节设为模式5
            self.default_control_mode[idx] = 5

        # 启动监听线程
        threading.Thread(target=self.lcm_handle, daemon=True).start()
        
    def lcm_handle(self):
        """持续处理 LCM 消息（阻塞式）"""
        while self.running:
            self.lcm.handle()

    def upper_body_data_listener(self, channel, data):
        """接收关节数据"""
        msg = upper_body_data_package.decode(data)
        if not msg.curJointPosVec:
            return
        with self.data_lock:
            self.joint_current_pos = np.array(msg.curJointPosVec)
            self.joint_current_speed = np.array(msg.curSpeedVec)
            self.is_used = np.array(msg.isUsed)
            self.error_code = np.array(msg.curErrCodeVec)
            self.status = np.array(msg.curStatusVec)
            self.joint_current_pos_updated.set()
            self.update_once_arm = True

    def publish_joint_command(self, qpos):
        """发布目标关节角度"""
        cmd = upper_body_cmd_package()
        cmd.isUsed = 0
        cmd.control_mode = self.default_control_mode
        cmd.jointPosVec = qpos.tolist()
        cmd.jointKp = (np.ones(self.dim) * 40).tolist()
        cmd.jointKd = (np.ones(self.dim) * 100).tolist()

        # 计算速度（简单差分）
        if self.plan_pre_qpos is None:
            speed = np.zeros_like(qpos)
        else:
            dt = self.lcm_from_robot_period / 1000
            speed = (qpos - self.plan_pre_qpos) / dt
        cmd.jointSpeedVec = speed.tolist()
                         
        cmd.jointCurrentVec = [0.0] * self.dim
        cmd.jointTorqueVec = [0.0] * self.dim

        # 缓存上一帧数据
        self.plan_pre_qpos = np.copy(qpos)
        self.plan_pre_speed = np.copy(speed)

        self.lcm.publish('upper_body_cmd', cmd.encode())
        
    def publish_tracker_pos(self, pos):
        msg = left_tracker_pos()
        msg.x = pos[0]
        msg.y = pos[1]
        msg.z = pos[2]
        self.lcm.publish('left_tracker_pos', msg.encode())
        # print(f"publish tracker pos: {pos}")

    def get_current_qpos(self):
        while not self.update_once_arm:
            time.sleep(0.1)
            print(f"get lcm position")
        print(f"current self.qpos: {self.joint_current_pos}")
        self.is_get_qpos = True
        return self.joint_current_pos
    
    def stop_lcm(self):
        self.running = False
