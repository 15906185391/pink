import os
import csv
import time
import subprocess
import signal
import threading

import numpy as np
from rclpy.node import Node
from collections import deque
from rclpy.action import ActionServer
from rclpy.action.server import ServerGoalHandle
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float32MultiArray

from mocap_system.log.log_manage import log_m
l_ = log_m

try:
    from humanoid_msgs.action import MocapCmd
    from mocap_system.robot.magicbot import MagicBotGen1
    from mocap_system.device.pico4_ultra import Pico4Ultra
    from mocap_system.config.config_info import Config

    # from mocap_system.visualize import *
    from mocap_system.Utils import *
except Exception as e:
    l_.import_error.update_log_info(
        "EXCEPTION", f": {e} Import Error in mocap.py")

"""
TODO:
存在的问题：
1.没有末端速度限制
2.接近奇异值时IK解抖动较大
3.PICO数据突变时保护
4.IK解算时，IK解算结果与实际运动轨迹不一致（表现为延迟）
5.碰撞
"""


class Mocap(Node):
    """
    Mocap node
    """

    # 初始化
    def __init__(self):
        super().__init__("mocap_teleop")

        self.robot = MagicBotGen1()
        # self.robot = MagicBot()
        self.teleop_device = Pico4Ultra()
        
        # 重连状态监控
        self.last_reconnect_check_time = 0
        self.reconnect_warning_shown = False

        # 周期性执行任务
        self.timer_update_device_input = self.create_timer(
            0.01, self.update_device_input)
        self.timer_update_state_machine = self.create_timer(
            0.01, self.update_state_machine)
        self.timer_process_data = self.create_timer(0.01, self.process_data)
        # self.timer_publish = self.create_timer(0.002, self.publish)
        self.timer_publish = self.create_timer(0.01, self.publish)

        # 插值相关属性
        self.interpolation_enabled = True
        self.last_processed_qpos = None  # 上次处理的关节位置
        self.next_target_qpos = None     # 下一个目标关节位置
        self.last_finger_pos_l = None  # 上次左手手指位置
        self.next_finger_pos_l = None  # 下次左手手指位置
        self.last_finger_pos_r = None  # 上次右手手指位置
        self.next_finger_pos_r = None  # 下次右手手指位置
        self.last_process_timestamp = 0  # 上次处理的时间戳
        self.process_period = 0.01       # 100Hz处理周期
        self.publish_period = 0.002      # 500Hz发布周期
        self.interpolation_alpha = 0.0   # 插值系数
        
        self.left_pub = self.create_publisher(
            PoseStamped, '/pico/left_hand_relative', 10)
        self.right_pub = self.create_publisher(
            PoseStamped, '/pico/right_hand_relative', 10)
        self.tracker_l_pub = self.create_publisher(
            PoseStamped, '/pico/left_final_goal', 10)
        self.tracker_r_pub = self.create_publisher(
            PoseStamped, '/pico/right_final_goal', 10)
        self.left_point_pub = self.create_publisher(
            PointStamped, '/pico/left_tracker_point', 10)
        self.right_point_pub = self.create_publisher(
            PointStamped, '/pico/right_tracker_point', 10)
        self.init_pos_l = self.create_publisher(
            PoseStamped, '/pico/left_hand_init', 10)
        self.init_pos_r = self.create_publisher(
            PoseStamped, '/pico/right_hand_init', 10)
        self.init_tracker_l_pub = self.create_publisher(
            PoseStamped, '/pico/left_tracker_init', 10)
        self.init_tracker_r_pub = self.create_publisher(
            PoseStamped, '/pico/right_tracker_init', 10)
        self.init_tracker_l_point_pub = self.create_publisher(
            PointStamped, '/pico/tracker_init', 10)
        self.init_tracker_r_point_pub = self.create_publisher(
            PointStamped, '/pico/tracker_init_r', 10)
        self.new_tracker_l_pub = self.create_publisher(
            PoseStamped, '/pico/left_tracker_new', 10)
        self.new_tracker_r_pub = self.create_publisher(
            PoseStamped, '/pico/right_tracker_new', 10)
        self.new_tracker_l_pub_point = self.create_publisher(
            PointStamped, '/pico/left_tracker_new_point', 10)
        self.new_tracker_r_pub_point = self.create_publisher(
            PointStamped, '/pico/right_tracker_new_point', 10)
        self.end_pos_l_pub = self.create_publisher(
            PoseStamped, '/pico/left_hand_end', 10)
        self.end_pos_r_pub = self.create_publisher(
            PoseStamped, '/pico/right_hand_end', 10)
        self.euler_pub = self.create_publisher(
            Float32MultiArray, '/pico/euler_angles', 10)
        self.pelvis_tracker_pub = self.create_publisher(
            PoseStamped, '/pico/pelvis_tracker_relative', 10)
        self.head_tracker_pub = self.create_publisher(
            PoseStamped, '/pico/head_tracker_relative', 10)
        self.wrist_pub = self.create_publisher(
            Float32MultiArray, '/pico/wrist_angles', 10)
        self.config_path = None   # 配置文件路径
        self.state = None            # 状态机状态

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

        # TELEOP CONTROL VARIABLES
        self.mocap_running = False          # 摇操启动开关
        self.teleop_switch = False          # PICO 连接开关
        self.teleop_connect_state = False   # PICO 连接状态
        self.mocap_running_err = False       # 摇操运行错误标志
        self.mocap_running_string = ""      # 摇操运行状态信息
        self.update_device_input_switch = False    # 更新设备输入开关
        self.calculate_ik_flag = False    # IK计算开关
        self.update_state_machine_flag = False    # 状态机更新开关

        # 动作服务
        self._mocap_action_server = ActionServer(
            self,
            MocapCmd,
            'mocap_action',
            self.execute_callback)

        self.config = Config()
        if self.config_path is not None:
            self.config.read_config_file(self.config_path)

        self.current_time = time.time()
        self.current_time_v2 = time.time()
        self.mocap_running = False        # 摇操启动开关
        self.language_flag = 0  # 0 中文 1 英文
        # self.teleop_device = None      # 控制变量
        self.is_send_data = False   # 是否发送数据
        self.is_start = False   # 是否开始
        # self.last_button_y_state = False    # 上一次左手柄Y按钮状态
        # self.send_cmd_vel = False    # 是否发送速度命令
        # self.is_full_body_mode = False   # 是否全身模式
        self.pub_lcm_flag = False    # 是否发布LCM数据
        # self.button_y_pressed_once = False  # Y 按钮是否第一次按下
        self.start_record = False       # 是否开始录制
        self.start_save = False         # 是否开始保存
        self.reset_pico_flag = True    # 是否重置PICO标志
        self.robot_reset_flag = False    # 机器人复位标志

        self.reset_flag = True    # 是否复位标志
        self.init_qpos_l: np.ndarray | None = None
        self.init_qpos_r: np.ndarray | None = None
        now_ns = self.get_clock().now().nanoseconds
        self.last_process_time = now_ns * 1e-9

        self.left_position = np.zeros(3)  # 左手末端位置
        
        self.pose_filter_l = SE3LowPassFilter(tau=0.1)  # SE3低通滤波器
        self.pose_filter_r = SE3LowPassFilter(tau=0.1)  # SE3低通滤波器

        # LCM Logger 控制
        self.lcm_logger_process = None  # LCM logger 进程
        self.last_button_a_state = False  # 上一次 A 按钮状态
        self.lcm_logger_log_dir = os.path.join(
            os.path.expanduser("~"), ".teleop", "log", "lcm_logs")
        os.makedirs(self.lcm_logger_log_dir, exist_ok=True)

    # ACTION SERVER CALLBACK

    def execute_callback(self, goal_handle: ServerGoalHandle) -> MocapCmd.Result:
        self.get_logger().info('Executing goal...')

        goal_key = goal_handle.request.cmd
        if goal_key == 1:
            self.start_mocap(goal_handle)
        elif goal_key == 2:
            self.stop_mocap(goal_handle)
        # else if goal_key == 3:
        #     self.reset_mocap(goal_handle)

        result = MocapCmd.Result()
        result.cmd = goal_key
        result.description = f"Processed goal {goal_key}"
        self.get_logger().info('Returning result...')
        return result

    # 启动遥操作
    def start_mocap(self, handle: ServerGoalHandle) -> None:
        goal_key = handle.request.cmd
        goal_value = handle.request.cmd_param

        # 判断是否可启动
        if self.mocap_running is True:
            print("Mocap is running")
            self.publish_feedback(handle, goal_key, "running")
            l_.mocap_info.update_log_info("MOCAP STATE INFO", "running")
            handle.abort()
            return
        self.mocap_running = True
        self.state = "A"
        l_.mocap_state.update_log_info(
            "INFO", "Mocap state: A (Start, wait for X to prepare)")
        self.publish_feedback(handle, goal_key, "idle")
        # 检查是否在46的状态下
        if self.config.is_real:
            if self.robot.get_current_qpos() is False:
                self.publish_feedback(
                    handle, goal_key, "can not get current pos")
                handle.abort()
                return

        # 连接PICO
        if self.teleop_device.stream_controller is None or self.teleop_device.stream_controller.running == False:
            self.teleop_device.stream_controller = None
            self.publish_feedback(handle, goal_key, "connect pico")
            l_.mocap_info.update_log_info("MOCAP STATE INFO", "connect pico")
            try:
                self.teleop_device.connect(goal_value)
                self.update_device_input_switch = True
            except Exception as e:
                self.get_logger().info('start_mocap err: {0}'.format(e))
        if self.teleop_device.stream_controller is None:
            self.publish_feedback(
                handle, goal_key, "Check PICO network or tracker")
            self.teleop_device.stream_controller = None
            handle.abort()
            # handle.destroy()
            return
        elif self.teleop_device.stream_controller.running == False:
            self.publish_feedback(
                handle, goal_key, "Check PICO running failed, check tracker.")
            self.teleop_device.stream_controller = None
            handle.abort()
            # handle.destroy()
            return
        else:
            self.language_flag = self.teleop_device.stream_controller.locale
            self.publish_feedback(handle, goal_key, "connected pico")
            if self.language_flag == 0:
                self.teleop_device.send_hint_msg(
                    1, "遥操作正在启动，机器人正在复位，请等待...", 1)
                l_.mocap_info.update_log_info(
                    "MOCAP STATE INFO", "遥操作正在启动，机器人正在复位，请等待...")
            elif self.language_flag == 1:
                self.teleop_device.send_hint_msg(
                    1, "Teleoperation is starting, robot is resetting, please wait...", 1)
            elif self.language_flag == 2:
                self.teleop_device.send_hint_msg(
                    1, "Die Teleoperation startet. Der Roboter wechselt jetzt in den Anfangszustand. Bitte warten Sie eine Minute.", 1)
            elif self.language_flag == 3:
                self.teleop_device.send_hint_msg(
                    1, "La teleoperazione sta iniziando. Il robot sta entrando nello stato iniziale. Attendere un minuto.", 1)

        # 运行到准备抓
        self.publish_feedback(handle, goal_key, "to_ready start")
        if self.config.is_real == True:
            self.robot.move2first()
        print("move to first done")
        l_.mocap_info.update_log_info("INFO", "Move to first start")
        l_.qpos_info.update_log_info(
            "INFO", "Move to first done, current qpos: %s" % self.robot.get_current_qpos())
        time.sleep(1)
        self.publish_feedback(handle, goal_key, "to_ready ok")
        # 启动摇操周期性任务
        self.teleop_switch = True          # PICO 连接开关
        self.teleop_connect_state = True   # PICO 连接状态

        # 周期性任务
        self.update_device_input_switch = True
        self.update_state_machine_flag = True

        self.publish_feedback(handle, goal_key, "teleop ok")
        self.publish_feedback(handle, goal_key, "running")
        if self.language_flag == 0:
            self.teleop_device.send_hint_msg(
                2, "机器人复位完成，请进入并保持准备姿态，按下左手柄X按钮准备开始。", 2)
        elif self.language_flag == 1:
            self.teleop_device.send_hint_msg(
                2, "Robot reset complete. Please enter and maintain the ready posture. Press the X button on the left handle to prepare for start.", 2)
        elif self.language_flag == 2:
            self.teleop_device.send_hint_msg(
                2, "Roboter-Reset abgeschlossen. Halten Sie die Bereitschaftsposition ein und drücken Sie die X-Taste am linken Griff, um sich auf den Start vorzubereiten.", 2)
        elif self.language_flag == 3:
            self.teleop_device.send_hint_msg(
                2, "Reset robot completato. Mantenere la posizione di standby e premere il pulsante X sulla maniglia sinistra per prepararsi all'avvio.", 2)

        l_.mocap_info.update_log_info(
            "MOCAP STATE INFO", "机器人复位完成，请进入并保持准备姿态，按下左手柄X按钮准备开始。")
        handle.succeed()
        self.state = "A"

    # 停止遥操作
    def stop_mocap(self, handle: ServerGoalHandle) -> None:
        print(self.mocap_running)
        for i in range(2):
            if self.language_flag == 0:
                self.teleop_device.send_hint_msg(
                    2, "遥操作正在关闭，机器人正在复位，请等待...", 1)
            elif self.language_flag == 1:
                self.teleop_device.send_hint_msg(
                    2, "Teleoperation is shutting down. The robot is resetting. Please wait...", 1)
            elif self.language_flag == 2:
                self.teleop_device.send_hint_msg(
                    2, "Die Teleoperation wurde beendet. Der Roboter kehrt in den Anfangszustand zurück.", 1)
            elif self.language_flag == 3:
                self.teleop_device.send_hint_msg(
                    2, "La teleoperazione è terminata. Il robot sta tornando allo stato iniziale.", 1)

            l_.mocap_info.update_log_info(
                "MOCAP STATE INFO", "遥操作正在关闭，机器人正在复位，请等待...")
            # self.send_cmd_vel = False
            self.is_full_body_mode = False
            time.sleep(0.1)

        goal_key = handle.request.cmd
        self.mocap_running = True
        self.update_device_input_switch = False
        self.update_state_machine_flag = False
        self.calculate_ik_flag = False
        self.reset_flag = True

        self.publish_feedback(handle, goal_key, "idle")
        self.is_start = False
        self.is_send_data = False
        if self.config.is_real:
            if self.robot.get_current_qpos() is False:
                self.publish_feedback(
                    handle, goal_key, "get current pos failed")
                handle.abort()
                return

        goal_key = handle.request.cmd
        self.publish_feedback(handle, goal_key, "0")
        if self.config.is_real == True:
            self.robot.move2stand()
        print("move to stand done")
        l_.qpos_info.update_log_info(
            "INFO", "Move to stand done, current qpos: %s" % self.robot.get_current_qpos())
        time.sleep(1)
        self.publish_feedback(handle, goal_key, "100")

        if self.language_flag == 0:
            self.teleop_device.send_hint_msg(2, "遥操作已关闭，暂无数据同步...", 0)
        elif self.language_flag == 1:
            self.teleop_device.send_hint_msg(
                2, "Teleoperation has been disabled. No data synchronization available at this time...", 0)
        elif self.language_flag == 2:
            self.teleop_device.send_hint_msg(
                2, "Die Teleoperation ist beendet.  Derzeit findet keine Datensynchronisierung statt.", 0)
        elif self.language_flag == 3:
            self.teleop_device.send_hint_msg(
                2, "La teleoperazione è terminata. Nessuna sincronizzazione dei dati ancora eseguita.", 0)

        l_.mocap_info.update_log_info("MOCAP STATE INFO", "遥操作已关闭，暂无数据同步...")
        time.sleep(0.1)
        # 断开PICO
        if self.teleop_device:
            self.teleop_device.close()
        self.teleop_switch = False          # PICO 连接开关
        self.teleop_connect_state = False   # PICO 连接状态
        self.publish_feedback(handle, goal_key, "stop_mocap")

        self.mocap_running = False
        handle.succeed()
        # self.robot.close()
        time.sleep(1)

    # 更新设备输入
    def update_device_input(self):
        if self.update_device_input_switch:
            
            # 检查 PICO 连接状态
            if self.teleop_device and self.teleop_device.stream_controller:
                if not self.teleop_device.stream_controller.is_connected():
                    if time.time() - self.last_reconnect_check_time > 2.0:
                        self.last_reconnect_check_time = time.time()
                        if self.language_flag == 0:
                            self.teleop_device.send_hint_msg(
                                3, "PICO 连接中断，正在尝试重连...", 0)
                        elif self.language_flag == 1:
                            self.teleop_device.send_hint_msg(
                                3, "PICO connection lost, attempting to reconnect...", 0)
                        l_.mocap_info.update_log_info(
                            "CONNECTION WARNING", "PICO connection lost, reconnecting...")
            
            
            
            
            self.teleop_device.update_teleop_data()
            self.button_x_state = self.teleop_device.get_button_state_by_name(
                "X")
            self.button_y_state = self.teleop_device.get_button_state_by_name(
                "Y")
            self.button_a_state = self.teleop_device.get_button_state_by_name(
                "A")
            self.button_b_state = self.teleop_device.get_button_state_by_name(
                "B")
            self.left_trigger_button_state = self.teleop_device.get_button_state_by_name(
                "left_trigger_button")
            self.left_grip_button_state = self.teleop_device.get_button_state_by_name(
                "left_grip_button")
            self.right_trigger_button_state = self.teleop_device.get_button_state_by_name(
                "right_trigger_button")
            self.right_grip_button_state = self.teleop_device.get_button_state_by_name(
                "right_grip_button")

            # # 检测 A 按钮的上升沿（按下瞬间）
            # if self.button_a_state and not self.last_button_a_state:
            #     if self.state == "B-2":  # 仅在 B-2 状态下响应
            #         if self.lcm_logger_process is None:
            #             self.start_record = True
            #             print(time.time())
            #             print("start record")
            #             # 启动 lcm-logger
            #             self.start_lcm_logger()
            # if self.button_b_state and not self.last_button_b_state:
            #     if self.state == "B-2":  # 仅在 B-2 状态下响应
            #         if self.lcm_logger_process is not None:
            #             self.start_record = False
            #             print(time.time())
            #             print("stop record")
            #             # 停止 lcm-logger
            #             self.stop_lcm_logger()

            # 使用独立的标志位，避免在高频循环中重复检查
            if not hasattr(self, '_logger_start_requested'):
                self._logger_start_requested = False
            if not hasattr(self, '_logger_stop_requested'):
                self._logger_stop_requested = False

            # 检测 A 按钮的上升沿（按下瞬间）
            if self.button_a_state and not self.last_button_a_state:
                if self.state == "B-2":  # 仅在 B-2 状态下响应
                    if self.lcm_logger_process is None:
                        self._logger_start_requested = True

            if self.button_b_state and not self.last_button_b_state:
                if self.state == "B-2":  # 仅在 B-2 状态下响应
                    if self.lcm_logger_process is not None:
                        self._logger_stop_requested = True

            # 异步处理 logger 启停（避免阻塞主循环）
            if self._logger_start_requested:
                self._logger_start_requested = False
                # self.start_record = True
                print(time.time())
                print("start record")
                # 在非阻塞模式下启动 lcm-logger
                threading.Thread(target=self.start_lcm_logger,
                                 daemon=True).start()

            if self._logger_stop_requested:
                self._logger_stop_requested = False
                # self.start_record = False
                print(time.time())
                print("stop record")
                # 在非阻塞模式下停止 lcm-logger
                threading.Thread(target=self.stop_lcm_logger,
                                 daemon=True).start()

            self.last_button_a_state = self.button_a_state
            self.last_button_b_state = self.button_b_state

        # 启动 LCM Logger

    # 记录 LCM 数据
    def start_lcm_logger(self):
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            log_filename = f"lcm_log_{timestamp}.lcm"
            log_filepath = os.path.join(self.lcm_logger_log_dir, log_filename)

            # 先检查 lcm-logger 是否存在
            import shutil
            if shutil.which('lcm-logger') is None:
                raise FileNotFoundError("lcm-logger command not found in PATH")

            # 启动 lcm-logger 命令（不使用 -c 参数，直接录制所有通道）
            self.lcm_logger_process = subprocess.Popen(
                ['lcm-logger', log_filepath],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                preexec_fn=os.setsid
            )

            # 等待一小段时间确认进程启动成功
            # time.sleep(0.1)
            if self.lcm_logger_process.poll() is not None:
                # 进程已异常退出
                returncode = self.lcm_logger_process.returncode
                raise RuntimeError(
                    f'lcm-logger exited immediately with code {returncode}')

            self.get_logger().info(f'LCM Logger started: {log_filepath}')
            l_.mocap_info.update_log_info(
                "LCM LOGGER", f"Logger started: {log_filename}")
            print(time.time())

            print(f"\n✅ LCM Logger started successfully!")
            print(f"   Log file: {log_filepath}")
            print(f"   Process PID: {self.lcm_logger_process.pid}")

        except FileNotFoundError:
            self.get_logger().error('lcm-logger command not found. Please install LCM.')
            l_.mocap_info.update_log_info(
                "LCM LOGGER ERROR", "lcm-logger command not found")
            print("\n❌ Error: lcm-logger command not found. Please install LCM.")
            self.lcm_logger_process = None
        except Exception as e:
            self.get_logger().error(f'Failed to start LCM Logger: {str(e)}')
            l_.mocap_info.update_log_info(
                "LCM LOGGER ERROR", f"Failed to start: {str(e)}")
            print(f"\n❌ Error starting LCM Logger: {str(e)}")
            self.lcm_logger_process = None

    # 停止 LCM Logger
    def stop_lcm_logger(self):
        if self.lcm_logger_process is not None:
            try:
                # 终止进程组
                os.killpg(os.getpgid(self.lcm_logger_process.pid),
                          signal.SIGTERM)
                self.lcm_logger_process.wait(timeout=5)

                self.get_logger().info('LCM Logger stopped')
                l_.mocap_info.update_log_info("LCM LOGGER", "Logger stopped")
                print("\n⏹️ LCM Logger stopped")
                print(time.time())

            except subprocess.TimeoutExpired:
                # 如果 SIGTERM 没有终止进程，使用 SIGKILL
                os.killpg(os.getpgid(self.lcm_logger_process.pid),
                          signal.SIGKILL)
                self.get_logger().warning('LCM Logger force killed')
                l_.mocap_info.update_log_info(
                    "LCM LOGGER", "Logger force killed")
                print("\n⚠️ LCM Logger force killed")
            except Exception as e:
                self.get_logger().error(f'Error stopping LCM Logger: {str(e)}')
                l_.mocap_info.update_log_info(
                    "LCM LOGGER ERROR", f"Error stopping: {str(e)}")
                print(f"\n❌ Error stopping LCM Logger: {str(e)}")
            finally:
                self.lcm_logger_process = None
        else:
            self.get_logger().warning('LCM Logger is not running')
            l_.mocap_info.update_log_info(
                "LCM LOGGER WARNING", "Logger is not running")
            print("\n⚠️ LCM Logger is not running")


    # 更新状态机
    def update_state_machine(self):
        if self.update_state_machine_flag == True:
            self.current_time = time.time()
            if self.mocap_running == False:
                return

            if self.state == "A":
                if self.button_x_state == True:
                    print("button x pressed in state A")
                    self.state = "B"
                    l_.mocap_state.update_log_info("INFO", "Mocap state: B")
                    if self.teleop_device:
                        if self.language_flag == 0:
                            self.teleop_device.send_hint_msg(
                                1, "请保持准备姿态，按下左手柄Y按钮以开始遥操作", 3)
                        elif self.language_flag == 1:
                            self.teleop_device.send_hint_msg(
                                1, "Please maintain the ready position and press the Y button on the left handle to begin teleoperation.", 3)
                        elif self.language_flag == 2:
                            self.teleop_device.send_hint_msg(
                                1, "Drücken Sie die Y-Taste am linken Griff, um zu starten.", 3)
                        elif self.language_flag == 3:
                            self.teleop_device.send_hint_msg(
                                1, "Premere il pulsante Y sulla maniglia sinistra per avviare.", 3)
                        l_.mocap_info.update_log_info(
                            "MOCAP STATE INFO", "请保持准备姿态，按下左手柄Y按钮以开始遥操作")

            elif self.state == "B":
                if self.reset_flag == True:
                    self.teleop_device.reset()                      # pico4 重置
                if self.button_y_state == True:
                    print("button y pressed in state B")
                    self.is_start = True
                    self.is_send_data = True
                    self.state = "B-1"
                    l_.mocap_state.update_log_info("INFO", "Mocap state: B-1")

            elif self.state == "B-1":
                if self.button_x_state == True:
                    print("button x pressed in state B-1")
                    self.state = "C"
                else:
                    self.pub_lcm_flag = False                    # 发布LCM
                    self.update_device_input_switch = True    # 更新设备输入开关
                    if self.is_send_data:
                        if self.teleop_device:
                            self.update_device_input()
                            if self.language_flag == 0:
                                self.teleop_device.send_hint_msg(
                                    2, "请保持准备姿态，等待关节就绪", 4)
                            elif self.language_flag == 1:
                                self.teleop_device.send_hint_msg(
                                    2, "Please maintain the ready position, waiting for joints to be ready (progress: ", 4)
                            elif self.language_flag == 2:
                                self.teleop_device.send_hint_msg(
                                    2, "Der Roboter gleicht die aktuelle Haltung ab. Warten Sie, bis die Aktion synchronisiert ist, bevor Sie die Teleoperation starten.", 4)
                            elif self.language_flag == 3:
                                self.teleop_device.send_hint_msg(
                                    2, "Il robot sta adattando la postura corrente. Attendere che l’azione sia sincronizzata prima di iniziare la teleoperazione.", 4)
                            l_.mocap_info.update_log_info(
                                "MOCAP STATE INFO", "请保持准备姿态，等待关节就绪")
                            if self.reset_flag == True:
                                self.robot.move2readypos()
                            if self.language_flag == 0:
                                self.teleop_device.send_hint_msg(
                                    1, "关节就绪！正在进行遥操作，按下左手柄X按钮可停止遥操作", 5)
                            elif self.language_flag == 1:
                                self.teleop_device.send_hint_msg(
                                    1, "Joints ready! Teleoperation in progress. Press the X button on the left handle to stop teleoperation.", 5)
                            elif self.language_flag == 2:
                                self.teleop_device.send_hint_msg(
                                    1, "Das Gelenk ist eingerastet und Sie können die Teleoperation starten. Wenn Sie die Teleoperation beenden möchten, drücken Sie X.", 5)
                            elif self.language_flag == 3:
                                self.teleop_device.send_hint_msg(
                                    1, "L’articolazione è in posizione e puoi avviare la teleoperazione. Per interromperla, premere X.", 5)
                            l_.mocap_info.update_log_info(
                                "MOCAP STATE INFO", "关节就绪！正在进行遥操作，按下左手柄X按钮可停止遥操作")
                            self.is_ready = True
                        self.state = "B-2"
                        l_.mocap_state.update_log_info(
                            "INFO", "Mocap state: B-2")
                        self.robot.qpos_ik = self.robot.init_qpos.copy()
                        self.robot.left_hand.finger_pos = self.robot.init_qpos[14:20].copy(
                        )
                        self.robot.right_hand.finger_pos = self.robot.init_qpos[20:26].copy(
                        )
                        self.robot.pre_qpos = self.robot.qpos_ik.copy()
                        print("pre qpos set to init qpos")
                        print("self init qpos ik", self.robot.qpos_ik)

            elif self.state == "B-2":
                if self.button_x_state == True:
                    print("button x pressed in state B-2")
                    self.state = "C"
                    l_.mocap_state.update_log_info("INFO", "Mocap state: C")
                else:
                    self.pub_lcm_flag = True                    # 发布LCM
                    self.calculate_ik_flag = True               # 计算IK
                    self.update_device_input_switch = True    # 更新设备输入开关
                    if not self.is_start:
                        return
                    if self.config.record_csv == True:
                        # if self.start_record == True:
                        self.robot.lcm_handle.publish_tracker_pos(self.left_position)

            elif self.state == "C":
                print("enter C")
                self.is_start = False
                self.calculate_ik_flag = False
                self.pub_lcm_flag = False
                self.is_send_data = False
                # self.send_cmd_vel = False
                # self.last_button_y_state = False
                self.is_full_body_mode = False
                self.state = "D"
                l_.mocap_state.update_log_info("INFO", "Mocap state: D")
                self.last_stop_time = self.current_time
                if self.teleop_device:
                    if self.language_flag == 0:
                        self.teleop_device.send_hint_msg(
                            3, "遥操作结束，等待3s后可重启", 6)
                    elif self.language_flag == 1:
                        self.teleop_device.send_hint_msg(
                            3, "Teleoperation complete. Wait 3 seconds before restarting.", 6)
                    elif self.language_flag == 2:
                        self.teleop_device.send_hint_msg(
                            3, "Der Roboter pausiert, bitte warten Sie 3 Sekunden.", 6)
                    elif self.language_flag == 3:
                        self.teleop_device.send_hint_msg(
                            3, "Il robot è in pausa, attendere 3 secondi.", 6)
                l_.mocap_info.update_log_info(
                    "MOCAP STATE INFO", "遥操作结束，等待3s后可重启")
                print(">>> STATE C → D (STOP)")
                self.close_mocap()

                if self.button_x_state:
                    print(">>> Button 1 ignored (already sending data)")

            elif self.state == "D":
                if self.current_time - getattr(self, "last_stop_time", 0) >= 3:
                    self.state = "E"
                    l_.mocap_state.update_log_info("INFO", "Mocap state: E")
                    print(">>> STATE D → E (auto after 3s)")
                    if self.language_flag == 0:
                        self.teleop_device.send_hint_msg(
                            2, "按下左手柄X按钮复位，或按下左手柄Y按钮继续遥操作", 7)
                    elif self.language_flag == 1:
                        self.teleop_device.send_hint_msg(
                            2, "Press the X button on the left handle to reset, or press the Y button on the left handle to continue teleoperation.", 7)
                    elif self.language_flag == 2:
                        self.teleop_device.send_hint_msg(
                            2, "Die Teleoperation wurde pausiert. Drücken Sie Y, um fortzufahren, oder X, um in den Anfangszustand zurückzukehren.", 7)
                    elif self.language_flag == 3:
                        self.teleop_device.send_hint_msg(
                            2, "La teleoperazione è stata messa in pausa. Premere Y per continuare o X per tornare allo stato iniziale.", 7)

                    l_.mocap_info.update_log_info(
                        "MOCAP STATE INFO", "按下左手柄X按钮复位，或按下左手柄Y按钮继续遥操作")
            elif self.state == "E":
                if (self.button_x_state == True):  # 按 0
                    if self.language_flag == 0:
                        self.teleop_device.send_hint_msg(
                            1, "机器人正在复位，请稍后...", 1)
                    elif self.language_flag == 1:
                        self.teleop_device.send_hint_msg(
                            1, "Robot resetting, please wait...", 1)
                    elif self.language_flag == 2:
                        self.teleop_device.send_hint_msg(
                            1, "Der Roboter wird zurückgesetzt, bitte warten...", 1)
                    elif self.language_flag == 3:
                        self.teleop_device.send_hint_msg(
                            1, "Il robot si sta resettando, attendere prego...", 1)
                    l_.mocap_info.update_log_info(
                        "MOCAP STATE INFO", "机器人正在复位，请稍后...")
                    self.robot.move2second()
                    self.teleop_device.reset_initial_pose()
                    print("move to second done")

                    self.is_send_data = True
                    self.reset_flag = True
                    self.state = "B"
                    l_.mocap_state.update_log_info("INFO", "Mocap state: B")
                    self.robot.pre_qpos = None
                    if self.teleop_device:
                        if self.language_flag == 0:
                            self.teleop_device.send_hint_msg(
                                2, "机器人复位完成，请进入并保持准备姿态，按下左手柄Y按钮以开始遥操作。", 8)
                        elif self.language_flag == 1:
                            self.teleop_device.send_hint_msg(
                                2, "Robot reset complete. Please enter and maintain the ready posture. Press the Y button on the left handle to start teleoperation.", 8)
                        elif self.language_flag == 2:
                            self.teleop_device.send_hint_msg(
                                2, "Der Reset des Roboters ist abgeschlossen. Bitte nehmen Sie die Bereitschaftsposition ein und halten Sie diese. Drücken Sie die Y-Taste am linken Griff, um den Fernbetrieb zu starten.", 8)
                        elif self.language_flag == 3:
                            self.teleop_device.send_hint_msg(
                                2, "Il ripristino del robot è stato completato. Assumere e mantenere la posizione di pronto. Premere il pulsante Y sulla maniglia sinistra per avviare il funzionamento remoto.", 8)
                    l_.mocap_info.update_log_info(
                        "MOCAP STATE INFO", "机器人复位完成，请进入并保持准备姿态，按下左手柄Y按钮以开始遥操作。")
                    print(">>> STATE E → B (auto after 5s)")

                if (self.button_y_state == True):
                    print("button y pressed in state E")
                    self.is_start = True
                    self.is_send_data = True
                    self.robot.pre_qpos = None
                    self.reset_flag = False
                    self.state = "B-2"
                    l_.mocap_state.update_log_info("INFO", "Mocap state: B-2")
                    if self.teleop_device:
                        if self.language_flag == 0:
                            self.teleop_device.send_hint_msg(
                                1, "正在进行遥操作，按下左手柄X按钮可停止遥操作", 5)
                        elif self.language_flag == 1:
                            self.teleop_device.send_hint_msg(
                                1, "Teleoperation is in progress. Press the X button on the left handle to stop teleoperation.", 5)
                        elif self.language_flag == 2:
                            self.teleop_device.send_hint_msg(
                                1, "Das Gelenk ist eingerastet und Sie können die Teleoperation starten. Wenn Sie die Teleoperation beenden möchten, drücken Sie X.", 5)
                        elif self.language_flag == 3:
                            self.teleop_device.send_hint_msg(
                                1, "L’articolazione è in posizione e puoi avviare la teleoperazione. Per interromperla, premere X.", 5)
                    l_.mocap_info.update_log_info(
                        "MOCAP STATE INFO", "正在进行遥操作，按下左手柄X按钮可停止遥操作")
                    print(">>> STATE E → B-2 (auto after 5s)")

            if self.button_x_state or self.button_y_state:
                self.current_time_v2 = self.current_time

    # 关闭遥操作
    def close_mocap(self):
        self.is_start = False
        self.is_send_data = False

    # # 处理数据
    # def process_data(self):
    #     if self.state is not None and self.calculate_ik_flag == True:
    #         delta_T_l, delta_t_l, delta_T_r, delta_t_r, T_new_tracker_l, T_new_tracker_r, init_tracker_l, init_tracker_r, T_pelvis_tracker, T_head_tracker, delta_T_pelvis = self.teleop_device.update_relative()

    #         # # (计算延迟和IK时间)
    #         # t_acq = time.perf_counter()

    #         if self.reset_flag:
    #             self.init_qpos_l = self.robot.get_init_ee_pose(
    #                 is_left=True).copy()
    #             self.init_qpos_r = self.robot.get_init_ee_pose(
    #                 is_left=False).copy()

    #         # 获取增量位姿和初始位姿，计算末端目标位姿
    #         self.final_T_l = compose_relative_translation_rotation_matrix(
    #             self.init_qpos_l.homogeneous, delta_T_l, delta_t_l)
    #         self.left_position = self.final_T_l[:3, 3]  # [x, y, z]
    #         self.final_T_r = compose_relative_translation_rotation_matrix(
    #             self.init_qpos_r.homogeneous, delta_T_r, delta_t_r)
    #         T_controller_l_rel = self.teleop_device.get_controller_pose_rel(
    #             is_left=True)
    #         T_controller_r_rel = self.teleop_device.get_controller_pose_rel(
    #             is_left=False)
    #         T_tracker_l_rel = self.teleop_device.get_tracker_pose_rel(
    #             is_left=True)
    #         T_tracker_r_rel = self.teleop_device.get_tracker_pose_rel(
    #             is_left=False)

    #         # 获取当前时间
    #         now_ns = self.get_clock().now().nanoseconds
    #         now = now_ns * 1e-9
    #         dt = max(now - getattr(self, "last_process_time", now), 1e-6)
    #         self.last_process_time = now

    #         # 更新插值相关数据
    #         old_target_qpos = self.next_target_qpos
            
    #         new_target_qpos = self.robot.two_stage_ik_process(
    #             self.final_T_l, self.final_T_r,
    #             T_tracker_l_rel, T_controller_l_rel,
    #             T_tracker_r_rel, T_controller_r_rel,
    #             delta_T_pelvis, _reset_flag=self.reset_flag, _dt=dt
    #         ).copy()
            
    #         # # 保存插值数据
    #         # self.last_processed_qpos = old_target_qpos if old_target_qpos is not None else new_target_qpos
    #         # self.next_target_qpos = new_target_qpos
    #         # self.last_process_timestamp = now

    #         # self.robot.qpos_ik = self.robot.two_stage_ik_process(
    #         #     self.final_T_l, self.final_T_r,
    #         #     T_tracker_l_rel, T_controller_l_rel,
    #         #     T_tracker_r_rel, T_controller_r_rel,
    #         #     delta_T_pelvis, _reset_flag=self.reset_flag, _dt=dt
    #         # ).copy()

    #         # 应用手指控制
    #         self.robot.left_hand.move_all_fingers_to(
    #             _grip_button=self.left_grip_button_state, _trigger_button=self.left_trigger_button_state)
    #         self.robot.right_hand.move_all_fingers_to(
    #             _grip_button=self.right_grip_button_state, _trigger_button=self.right_trigger_button_state)

    #         new_finger_pos_l = self.robot.left_hand.finger_pos.copy()
    #         new_finger_pos_r = self.robot.right_hand.finger_pos.copy()
            
    #         new_target_qpos[14:20] = new_finger_pos_l
    #         new_target_qpos[20:26] = new_finger_pos_r

    #         if old_target_qpos is not None:
    #             self.last_processed_qpos = old_target_qpos
    #             self.last_finger_pos_l = self.last_finger_pos_l if self.last_finger_pos_l is not None else new_finger_pos_l
    #             self.last_finger_pos_r = self.last_finger_pos_r if self.last_finger_pos_r is not None else new_finger_pos_r
    #         else:
    #             self.last_processed_qpos = new_target_qpos
    #             self.last_finger_pos_l = new_finger_pos_l
    #             self.last_finger_pos_r = new_finger_pos_r

    #         self.next_target_qpos = new_target_qpos
    #         self.next_finger_pos_l = new_finger_pos_l
    #         self.next_finger_pos_r = new_finger_pos_r
    #         self.last_process_timestamp = now
            
    #         self.robot.qpos_ik = new_target_qpos
    #         self.reset_flag = False
            
    #         # self.robot.qpos_ik[14:20] = self.robot.left_hand.finger_pos.copy()
    #         # self.robot.qpos_ik[20:26] = self.robot.right_hand.finger_pos.copy()

    #         # self.reset_flag = False

    #         # self.publish_single_pose(
    #         #     self.final_T_l, "left_tracker_frame", self.tracker_l_pub)
    #         # self.publish_single_pose(self.final_T_r, "right_tracker_frame", self.tracker_r_pub)
    #         # self.publish_point(
    #         #     self.final_T_l, "left_tracker_frame", self.left_point_pub)
    #         # self.publish_point(self.final_T_r, "right_tracker_frame", self.right_point_pub)
    #         # self.publish_single_pose(self.init_qpos_l.homogeneous, "left_hand_init_frame", self.init_pos_l)
    #         # # self.publish_single_pose(self.init_qpos_r.homogeneous, "right_hand_init_frame", self.init_pos_r)
    #         # self.publish_single_pose(init_tracker_l, "left_tracker_init_frame", self.init_tracker_l_pub)
    #         # # self.publish_single_pose(init_tracker_r, "right_tracker_init_frame", self.init_tracker_r_pub)
    #         # self.publish_point(init_tracker_l, "left_tracker_init_frame", self.init_tracker_l_point_pub)
    #         # # self.publish_point(init_tracker_r, "right_tracker_init_frame", self.init_tracker_r_point_pub)
    #         # self.publish_single_pose(T_new_tracker_l, "left_tracker_new_frame", self.new_tracker_l_pub)
    #         # # self.publish_single_pose(T_new_tracker_r, "right_tracker_new_frame", self.new_tracker_r_pub)
    #         # self.publish_point(T_new_tracker_l, "left_tracker_new_frame", self.new_tracker_l_pub_point)
    #         # # self.publish_point(T_new_tracker_r, "right_tracker_new_frame", self.new_tracker_r_pub_point)
    #         # self.publish_euler_from_matrix(self.final_T_l, self.euler_pub)
    #         # self.publish_single_pose(
    #         #     T_pelvis_tracker, "pelvis_tracker_frame", self.pelvis_tracker_pub)
    #         # self.publish_single_pose(
    #         #     T_head_tracker, "head_tracker_frame", self.head_tracker_pub)
    #     else:
    #         pass
        
        # 处理数据
    def process_data(self):
        if self.state is not None and self.calculate_ik_flag == True:
            delta_T_l, delta_t_l, delta_T_r, delta_t_r, T_new_tracker_l, T_new_tracker_r, init_tracker_l, init_tracker_r, T_pelvis_tracker, T_head_tracker, delta_T_pelvis, delta_T_headset = self.teleop_device.update_relative()
            
            if self.reset_flag:
                self.init_qpos_l = self.robot.get_init_ee_pose(is_left=True).copy()
                self.init_qpos_r = self.robot.get_init_ee_pose(is_left=False).copy()
                
            # 获取增量位姿和初始位姿，计算末端目标位姿
            self.final_T_l = compose_relative_translation_rotation_matrix(self.init_qpos_l.homogeneous, delta_T_l, delta_t_l)
            self.left_position = self.final_T_l[:3, 3]  # [x, y, z]
            self.final_T_r = compose_relative_translation_rotation_matrix(self.init_qpos_r.homogeneous, delta_T_r, delta_t_r)
            T_controller_l_rel = self.teleop_device.get_controller_pose_rel(is_left=True)
            T_controller_r_rel = self.teleop_device.get_controller_pose_rel(is_left=False)
            T_tracker_l_rel = self.teleop_device.get_tracker_pose_rel(is_left=True)
            T_tracker_r_rel = self.teleop_device.get_tracker_pose_rel(is_left=False)
            
            # 获取当前时间
            now_ns = self.get_clock().now().nanoseconds
            now = now_ns * 1e-9
            dt = max(now - getattr(self, "last_process_time", now), 1e-6)
            self.last_process_time = now
            
            self.smoothed_final_T_l = self.pose_filter_l.update(self.final_T_l, dt)
            self.smoothed_final_T_r = self.pose_filter_r.update(self.final_T_r, dt)
            
            # self.robot.qpos_ik = self.robot.two_stage_ik_process(
            #     self.final_T_l, self.final_T_r,
            #     T_tracker_l_rel, T_controller_l_rel,
            #     T_tracker_r_rel, T_controller_r_rel,
            #     delta_T_pelvis, _reset_flag=self.reset_flag, _delta_T_headset=delta_T_headset, _dt=dt
            # ).copy()
            
            self.robot.qpos_ik = self.robot.two_stage_ik_process(
                self.smoothed_final_T_l, self.smoothed_final_T_r,
                T_tracker_l_rel, T_controller_l_rel,
                T_tracker_r_rel, T_controller_r_rel,
                delta_T_pelvis, _reset_flag=self.reset_flag, _delta_T_headset=delta_T_headset, _dt=dt
            ).copy()
            
            # 应用手指控制
            self.robot.left_hand.move_all_fingers_to(_grip_button = self.left_grip_button_state, _trigger_button = self.left_trigger_button_state)
            self.robot.right_hand.move_all_fingers_to(_grip_button = self.right_grip_button_state, _trigger_button = self.right_trigger_button_state)
            
            self.robot.qpos_ik[14:20] = self.robot.left_hand.finger_pos.copy()
            self.robot.qpos_ik[20:26] = self.robot.right_hand.finger_pos.copy()
            
            self.reset_flag = False
            
            self.publish_single_pose(self.final_T_l, "left_tracker_frame", self.tracker_l_pub)
            self.publish_single_pose(self.final_T_r, "right_tracker_frame", self.tracker_r_pub)
            # self.publish_point(self.final_T_l, "left_tracker_frame", self.left_point_pub)
            # self.publish_point(self.final_T_r, "right_tracker_frame", self.right_point_pub)
            # self.publish_single_pose(self.init_qpos_l.homogeneous, "left_hand_init_frame", self.init_pos_l)
            # # self.publish_single_pose(self.init_qpos_r.homogeneous, "right_hand_init_frame", self.init_pos_r)
            # self.publish_single_pose(init_tracker_l, "left_tracker_init_frame", self.init_tracker_l_pub)
            # # self.publish_single_pose(init_tracker_r, "right_tracker_init_frame", self.init_tracker_r_pub)
            # self.publish_point(init_tracker_l, "left_tracker_init_frame", self.init_tracker_l_point_pub)
            # # self.publish_point(init_tracker_r, "right_tracker_init_frame", self.init_tracker_r_point_pub)
            self.publish_single_pose(T_new_tracker_l, "left_tracker_new_frame", self.new_tracker_l_pub)
            self.publish_single_pose(T_new_tracker_r, "right_tracker_new_frame", self.new_tracker_r_pub)
            # self.publish_point(T_new_tracker_l, "left_tracker_new_frame", self.new_tracker_l_pub_point)
            # self.publish_point(T_new_tracker_r, "right_tracker_new_frame", self.new_tracker_r_pub_point)
            # self.publish_euler_from_matrix(self.final_T_l, self.euler_pub)
            # self.publish_single_pose(T_pelvis_tracker, "pelvis_tracker_frame", self.pelvis_tracker_pub)
            # self.publish_single_pose(T_head_tracker, "head_tracker_frame", self.head_tracker_pub)
        else:
            pass

    # 发布数据
    def publish(self):

        if self.pub_lcm_flag == True:
            target_qpos = self.robot.qpos_ik.copy()
            if self.robot.pre_qpos is None:
                self.robot.pre_qpos = self.robot.get_current_qpos().copy()
            self.current_qpos = self.robot.get_current_qpos().copy()
            # qpos_pub = filter(target_qpos, self.robot.pre_qpos)
            qpos_pub = filter(target_qpos, self.current_qpos)  # 使用实际当前位置进行滤波
            self.robot.lcm_handle.publish_joint_command(qpos_pub)
            # self.robot.pre_qpos = qpos_pub
            self.robot.pre_qpos = self.current_qpos  # 保存实际位置，而非控制命令
    
    #     # 发布数据
    # def publish(self):
    #     if self.pub_lcm_flag == True:
    #         now_ns = self.get_clock().now().nanoseconds
    #         now = now_ns * 1e-9
            
    #         if self.interpolation_enabled and self.last_processed_qpos is not None and self.next_target_qpos is not None:
    #             elapsed_since_last_process = now - self.last_process_timestamp
                
    #             if elapsed_since_last_process < self.process_period:
    #                 alpha = elapsed_since_last_process / self.process_period
    #                 alpha = min(alpha, 1.0)
                    
    #                 filtered_last = filter(self.last_processed_qpos, self.robot.pre_qpos if self.robot.pre_qpos is not None else self.last_processed_qpos)
    #                 filtered_next = filter(self.next_target_qpos, filtered_last)
                    
    #                 interpolated_qpos = filtered_last * (1 - alpha) + filtered_next * alpha
                    
    #                 if self.last_finger_pos_l is not None and self.next_finger_pos_l is not None:
    #                     interpolated_finger_l = np.array(self.last_finger_pos_l) * (1 - alpha) + np.array(self.next_finger_pos_l) * alpha
    #                     interpolated_qpos[14:20] = interpolated_finger_l
                    
    #                 if self.last_finger_pos_r is not None and self.next_finger_pos_r is not None:
    #                     interpolated_finger_r = np.array(self.last_finger_pos_r) * (1 - alpha) + np.array(self.next_finger_pos_r) * alpha
    #                     interpolated_qpos[20:26] = interpolated_finger_r
    #             else:
    #                 interpolated_qpos = filter(self.next_target_qpos, self.robot.pre_qpos if self.robot.pre_qpos is not None else self.next_target_qpos)
    #                 if self.next_finger_pos_l is not None:
    #                     interpolated_qpos[14:20] = self.next_finger_pos_l
    #                 if self.next_finger_pos_r is not None:
    #                     interpolated_qpos[20:26] = self.next_finger_pos_r
    #         else:
    #             target_qpos = self.robot.qpos_ik.copy()
    #             if self.robot.pre_qpos is None:
    #                 self.robot.pre_qpos = self.robot.get_current_qpos().copy()
    #             interpolated_qpos = filter(target_qpos, self.robot.pre_qpos)
            
    #         self.robot.lcm_handle.publish_joint_command(interpolated_qpos)
    #         self.robot.pre_qpos = interpolated_qpos

    # 发布反馈
    def publish_feedback(self, handle, goal_key, str_value):
        try:
            feedback_msg = MocapCmd.Feedback()
            feedback_msg.cmd = goal_key
            feedback_msg.feedback = str_value
            self.get_logger().info(
                'Feedback: {0}'.format(feedback_msg.feedback))
            handle.publish_feedback(feedback_msg)
        except Exception as e:
            self.get_logger().info('publish_feedback err: {0}'.format(e))

    def publish_single_pose(self, T, frame_id, publisher):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
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

    def publish_point(self, T, frame_id, publisher):
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "pelvis"
        msg.point.x = T[0, 3]
        msg.point.y = T[1, 3]
        msg.point.z = T[2, 3]
        # print(f"publish point: {msg.point.x:.2f}, {msg.point.y:.2f}, {msg.point.z:.2f}")
        publisher.publish(msg)

    def publish_euler_from_matrix(self, T, publisher):
        """
        从齐次变换矩阵T提取旋转部分，转为欧拉角并发布
        """
        msg = Float32MultiArray()
        R_mat = T[:3, :3]
        eul = rot_to_eul(R_mat, degrees=False)  # 使用xyz顺序，单位为度
        msg.data = eul.tolist()
        publisher.publish(msg)

    def publish_visualization(self):
        self.robot.viz.display(self.robot.qpos_ik[:14])
        print("visualization updated")
