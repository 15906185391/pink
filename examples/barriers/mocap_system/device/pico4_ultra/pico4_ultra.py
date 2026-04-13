import os
import sys
import numpy as np
import time
import threading


from scipy.spatial.transform import Rotation as R
from mocap_system.stream.pico.streamer import Pico4UltraStreamer
from mocap_system.device.device_base import Device, SensorPose, BooleanButton, TravelButton, Joystick

class Pico4Ultra(Device):
    def __init__(self):
        super().__init__("Pico4 Ultra")
        # 在这里添加 Pico4 Ultra 特有的初始化代码
        self.device_type = "Pico4 Ultra"
        path = os.path.join(os.path.dirname(__file__), "config/pico4_ultra.yaml")   # 配置文件路径
        self.pico4 = Device.from_yaml(path)     # 设备配置
        
        print("Initialized mocap system.")

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

        self.T_controller_l_corr = np.array([
            [0, 0, -1, 0],
            [0, -1, 0, 0],
            [-1, 0, 0, 0],
            [0, 0, 0, 1]
        ])

        self.T_controller_r_corr = np.array([
            [0, 0, -1, 0],
            [0, -1, 0, 0],
            [-1, 0, 0, 0],
            [0, 0, 0, 1]
        ])

        print("Re-initialized mocap system.")

        self.stream_controller = None
        
        # 重连相关参数
        self.reconnect_enabled = True
        self.reconnect_interval = 3.0  # 重连间隔（秒）
        self.max_reconnect_attempts = 10  # 最大重连次数
        self.reconnect_attempts = 0
        self.last_reconnect_time = 0
        self.server_ip = None
        self.reconnect_thread = None
        self.stop_reconnect = False


    def connect(self, _server_ip):
        self.server_ip = _server_ip
        self.stream_controller = Pico4UltraStreamer(ip = _server_ip, device=self.pico4, port=12345, is_record=False)
        self.reconnect_attempts = 0
        # self.start_reconnect_monitor()
        
    def start_reconnect_monitor(self):
        """启动重连监控线程"""
        if self.reconnect_enabled and self.server_ip:
            self.stop_reconnect = False
            self.reconnect_thread = threading.Thread(target=self._reconnect_monitor_loop, daemon=True)
            self.reconnect_thread.start()
            print("Reconnect monitor started")
            
            
    def _reconnect_monitor_loop(self):
        """重连监控循环"""
        while not self.stop_reconnect:
            time.sleep(1.0)  # 每秒检查一次
            
            if self.stream_controller is None:
                continue
            
            # 检查连接状态
            if not self.stream_controller.is_connected() or self.stream_controller.check_data_timeout(timeout_threshold=5):
                print("connected:", self.stream_controller.is_connected(), "data timeout:", self.stream_controller.check_data_timeout(timeout_threshold=5))
                error_msg = self.stream_controller.get_connection_error()
                if error_msg:
                    print(f"⚠️ PICO connection lost: {error_msg}")
                else:
                    print(f"⚠️ PICO data timeout, attempting reconnect...")
                
                # 尝试重连
                if time.time() - self.last_reconnect_time > self.reconnect_interval:
                    self._attempt_reconnect()
            
            # 检查是否需要重置重连次数
            elif self.stream_controller.is_connected():
                self.reconnect_attempts = 0
                
    def _attempt_reconnect(self):
        """尝试重连"""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            print(f"❌ Max reconnect attempts ({self.max_reconnect_attempts}) reached. Giving up.")
            self.stop_reconnect = True
            return
        
        self.last_reconnect_time = time.time()
        self.reconnect_attempts += 1
        
        print(f"🔄 Attempting reconnect ({self.reconnect_attempts}/{self.max_reconnect_attempts})...")
        
        try:
            # 停止当前流
            if self.stream_controller:
                self.stream_controller.stop_streaming()
                self.stream_controller = None
            
            # 等待一小段时间
            time.sleep(0.5)
            
            # 重新连接
            self.stream_controller = Pico4UltraStreamer(
                ip=self.server_ip, 
                device=self.pico4, 
                port=12345, 
                is_record=False
            )
            
            # 等待连接建立
            timeout = 1
            start_time = time.time()
            while time.time() - start_time < timeout:
                if self.stream_controller.latest is not None:
                    print(f"✅ Reconnect successful!")
                    self.reconnect_attempts = 0
                    return
                time.sleep(0.1)
            
            print(f"⚠️ Reconnect attempt {self.reconnect_attempts} timed out")
            
        except Exception as e:
            print(f"❌ Reconnect failed: {e}")
    
    def stop_reconnect_monitor(self):
        """停止重连监控"""
        self.stop_reconnect = True
        if self.reconnect_thread:
            self.reconnect_thread.join(timeout=2.0)
        print("Reconnect monitor stopped")


    def get_button_state_by_name(self, button_name):
        if self.stream_controller is None:
            raise Exception("Device not connected.")
        if button_name == "X":
            return self.pico4.get_modules("LeftController").get_interfaces("ButtonX").is_button_pressed()
        elif button_name == "Y":
            return self.pico4.get_modules("LeftController").get_interfaces("ButtonY").is_button_pressed()
        elif button_name == "A":
            return self.pico4.get_modules("RightController").get_interfaces("ButtonA").is_button_pressed()
        elif button_name == "B":
            return self.pico4.get_modules("RightController").get_interfaces("ButtonB").is_button_pressed()
        elif button_name == "left_trigger_button":
            return self.pico4.get_modules("LeftController").get_interfaces("LeftTriggerIndex").is_button_pressed()
        elif button_name == "left_grip_button":
            return self.pico4.get_modules("LeftController").get_interfaces("LeftTriggerMiddle").is_button_pressed()
        elif button_name == "right_trigger_button":
            return self.pico4.get_modules("RightController").get_interfaces("RightTriggerIndex").is_button_pressed()
        elif button_name == "right_grip_button":
            return self.pico4.get_modules("RightController").get_interfaces("RightTriggerMiddle").is_button_pressed()

    def update_teleop_data(self):
        if not self.stream_controller.latest:
            print("no latest data")
            return False
        latest_data = self.stream_controller.latest
        try:
            # 更新头显位姿
            if 'head' in latest_data:
                head_pose = latest_data['head']
                headset_pose = self.pico4.get_modules("Headset").get_interfaces("HeadsetPose")
                if headset_pose and isinstance(headset_pose, SensorPose):
                    headset_pose.update(head_pose)

                    # print("Updated HeadsetPose:", head_pose)

            # 更新左手柄位姿
            if 'left_controller_matrix' in latest_data:
                left_pose = latest_data['left_controller_matrix']
                left_controller_pose = self.pico4.get_modules("LeftController").get_interfaces("LeftControllerPose")
                if left_controller_pose and isinstance(left_controller_pose, SensorPose):
                    left_controller_pose.update(left_pose)

                    # print("Updated LeftControllerPose:", left_pose)

            # 更新右手柄位姿
            if 'right_controller_matrix' in latest_data:
                right_pose = latest_data['right_controller_matrix']
                right_controller_pose = self.pico4.get_modules("RightController").get_interfaces("RightControllerPose")
                if right_controller_pose and isinstance(right_controller_pose, SensorPose):
                    right_controller_pose.update(right_pose)

                    # print("Updated RightControllerPose:", right_pose)

            # 更新左手柄按钮状态
            if 'left_btn_one' in latest_data:
                left_btn_one = latest_data['left_btn_one']
                button_x = self.pico4.get_modules("LeftController").get_interfaces("ButtonX")
                if button_x and isinstance(button_x, BooleanButton):
                    button_x.update(left_btn_one)
                    # print("Updated ButtonX:", left_btn_one)

            if 'left_btn_two' in latest_data:
                left_btn_two = latest_data['left_btn_two']
                button_y = self.pico4.get_modules("LeftController").get_interfaces("ButtonY")
                if button_y and isinstance(button_y, BooleanButton):
                    button_y.update(left_btn_two)
                    # print("Updated ButtonY:", left_btn_two)

            if 'left_trigger_index' in latest_data:
                left_trigger_index = latest_data['left_trigger_index']
                left_trigger_index_iface = self.pico4.get_modules("LeftController").get_interfaces("LeftTriggerIndex")
                if left_trigger_index_iface and isinstance(left_trigger_index_iface, BooleanButton):
                    left_trigger_index_iface.update(left_trigger_index)
                    # print("Updated LeftTriggerIndex:", left_trigger_index)

            if 'left_trigger_hand' in latest_data:
                left_trigger_hand = latest_data['left_trigger_hand']
                left_trigger_middle = self.pico4.get_modules("LeftController").get_interfaces("LeftTriggerMiddle")
                if left_trigger_middle and isinstance(left_trigger_middle, BooleanButton):
                    left_trigger_middle.update(left_trigger_hand)
                    # print("Updated LeftTriggerMiddle:", left_trigger_hand)

            # 更新右手柄按钮状态
            if 'right_btn_one' in latest_data:
                right_btn_one = latest_data['right_btn_one']
                button_a = self.pico4.get_modules("RightController").get_interfaces("ButtonA")
                if button_a and isinstance(button_a, BooleanButton):
                    button_a.update(right_btn_one)
                    # print("Updated ButtonA:", right_btn_one)

            if 'right_btn_two' in latest_data:
                right_btn_two = latest_data['right_btn_two']
                button_b = self.pico4.get_modules("RightController").get_interfaces("ButtonB")
                if button_b and isinstance(button_b, BooleanButton):
                    button_b.update(right_btn_two)
                    # print("Updated ButtonB:", right_btn_two)

            if 'right_trigger_index' in latest_data:
                right_trigger_index = latest_data['right_trigger_index']
                right_trigger_index_iface = self.pico4.get_modules("RightController").get_interfaces("RightTriggerIndex")
                if right_trigger_index_iface and isinstance(right_trigger_index_iface, BooleanButton):
                    right_trigger_index_iface.update(right_trigger_index)
                    # print("Updated RightTriggerIndex:", right_trigger_index)

            if 'right_trigger_hand' in latest_data:
                right_trigger_hand = latest_data['right_trigger_hand']
                right_trigger_middle = self.pico4.get_modules("RightController").get_interfaces("RightTriggerMiddle")
                if right_trigger_middle and isinstance(right_trigger_middle, BooleanButton):
                    right_trigger_middle.update(right_trigger_hand)
                    # print("Updated RightTriggerMiddle:", right_trigger_hand)

            # 更新左手柄摇杆状态
            if 'left_thumb_stick' in latest_data:
                left_thumb_stick = latest_data['left_thumb_stick']
                left_joystick = self.pico4.get_modules("LeftController").get_interfaces("LeftJoystick")
                if left_joystick and isinstance(left_joystick, Joystick):
                    if hasattr(left_thumb_stick, "x") and hasattr(left_thumb_stick, "y"):
                        left_joystick.update(left_thumb_stick.x, left_thumb_stick.y)
                        # print("Updated LeftJoystick:", left_thumb_stick)

            # 更新右手柄摇杆状态
            if 'right_thumb_stick' in latest_data:
                right_thumb_stick = latest_data['right_thumb_stick']
                right_joystick = self.pico4.get_modules("RightController").get_interfaces("RightJoystick")
                if right_joystick and isinstance(right_joystick, Joystick):
                    if hasattr(right_thumb_stick, "x") and hasattr(right_thumb_stick, "y"):
                        right_joystick.update(right_thumb_stick.x, right_thumb_stick.y)
                        # print("Updated RightJoystick:", right_thumb_stick)

            if 'tracker_list' in latest_data:
                tracker_pelvis = latest_data['tracker_list'][0]
                tracker_left = latest_data['tracker_list'][1]
                tracker_right = latest_data['tracker_list'][2]

                PelvisTrackerPose = self.pico4.get_modules("PelvisTracker").get_interfaces("PelvisTrackerPose")
                LeftTrackerPose = self.pico4.get_modules("LeftTracker").get_interfaces("LeftTrackerPose")
                RightTrackerPose = self.pico4.get_modules("RightTracker").get_interfaces("RightTrackerPose")


                if PelvisTrackerPose and isinstance(PelvisTrackerPose, SensorPose):
                    PelvisTrackerPose.update(tracker_pelvis)

                if LeftTrackerPose and isinstance(LeftTrackerPose, SensorPose):
                    LeftTrackerPose.update(tracker_left)

                if RightTrackerPose and isinstance(RightTrackerPose, SensorPose):
                    RightTrackerPose.update(tracker_right)

            return True
        except Exception as e:
            print(f"更新设备数据时出错: {e}")
            return False

    def update_relative(self):
        # delta_eul_l, delta_xyz_l, delta_eul_r, delta_xyz_r, T_new_tracker_l, T_new_tracker_r, init_tracker_l, init_tracker_r = self.pico4.update_relative()
        # return delta_eul_l, delta_xyz_l, delta_eul_r, delta_xyz_r, T_new_tracker_l, T_new_tracker_r, init_tracker_l, init_tracker_r
        delta_T_l, delta_t_l, delta_T_r, delta_t_r, T_new_tracker_l, T_new_tracker_r, init_tracker_l, init_tracker_r, T_pelvis_tracker, T_head_tracker, delta_T_pelvis, delta_T_headset = self.pico4.update_relative()
        return delta_T_l, delta_t_l, delta_T_r, delta_t_r, T_new_tracker_l, T_new_tracker_r, init_tracker_l, init_tracker_r, T_pelvis_tracker, T_head_tracker, delta_T_pelvis, delta_T_headset

    def send_hint_msg(self, state_, message_, type_):
        if self.stream_controller:
            self.stream_controller.update_send_hint_data(state=state_, message=message_, guideStepType=type_)
    
    def get_tracker_pose_rel(self, is_left):
        if is_left:
            return self.pico4.get_modules("LeftTracker").get_interfaces("LeftTrackerPose").pose_matrix_relative.copy() @ self.T_tracker_l_corr
        else:
            return self.pico4.get_modules("RightTracker").get_interfaces("RightTrackerPose").pose_matrix_relative.copy() @ self.T_tracker_r_corr

    def get_controller_pose_rel(self, is_left):
        if is_left:
            return self.pico4.get_modules("LeftController").get_interfaces("LeftControllerPose").pose_matrix_relative @ self.T_controller_l_corr
        else:
            return self.pico4.get_modules("RightController").get_interfaces("RightControllerPose").pose_matrix_relative @ self.T_controller_r_corr

    def reset(self):
        # 在这里添加重置设备的代码
        self.pico4.reset()
        self.reset_initial_pose()

    def reset_initial_pose(self):
        self.pico4.init_tracker_l = None
        self.pico4.init_tracker_r = None

    def close(self):
        # self.stream_controller.stop_streaming()
        # self.stream_controller = None
        # self.reset_initial_pose()
        
        self.stop_reconnect_monitor()
        if self.stream_controller:
            self.stream_controller.stop_streaming()
        self.stream_controller = None
        self.reset_initial_pose()