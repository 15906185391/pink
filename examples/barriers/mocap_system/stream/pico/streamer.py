import sys
import time 
import grpc
import numpy as np 
from threading import Thread, Event

from collections import deque
import csv
import os
from datetime import datetime

from .grpc_msg import xrTracking_pb2, xrTracking_pb2_grpc
# from grpc_msg import xrTracking_pb2, xrTracking_pb2_grpc
import google.protobuf.empty_pb2 as empty_pb2
from mocap_system.log.log_manage import log_m
l_ = log_m

# 处理 gRPC 矩阵消息，转换为 numpy 矩阵
def process_matrix(message):
    m = np.array([[message.m00, message.m01, message.m02, message.m03],
                    [message.m10, message.m11, message.m12, message.m13],
                    [message.m20, message.m21, message.m22, message.m23],
                    [0, 0, 0, 1]])
    return m

class Streamer:
    """
    gRPC 订阅者类，用于订阅 VR 控制器 的数据流
    """
    # 初始化
    def __init__(self, ip, port, device, is_record = False):
        self.ip = ip
        self.port = port
        self.device = device
        self.is_record = is_record
        self.record = []
        self.latest = None   
        self.running = True
        
        self.connected = False
        self.connection_error = None
        
        self._last_data_time = None
        # self.stream = None
        
        # 延迟和频率统计
        self.timestamps = deque(maxlen=500)  # 保存最近 500 个时间戳
        self.frame_intervals = deque(maxlen=500)  # 保存帧间隔
        self.latency_stats = {
            'count': 0,
            'sum': 0.0,
            'min': float('inf'),
            'max': 0.0,
            'last_frame_time': None,
            'start_time': None
        }
        
        
        
    def is_connected(self):
        """检查连接状态"""
        return self.connected and self.latest is not None
    
    
    def get_connection_error(self):
        """获取连接错误信息"""
        return self.connection_error
    
    # # 启动流
    # def start_streaming(self):
    #     stream_thread = Thread(target=self.stream)
    #     stream_thread.start()
    #     while self.latest is None and self.running:
    #         time.sleep(0.01)
    #     if self.latest is not None:
    #         print("== DATA IS FLOWING IN! ==")
    #         print("Ready to start streaming.")
    #     else:
    #         print("!! Failed to start streaming: no data received !!")

    def stop_streaming(self):
        self.running = False
        

    # 发送提示信息
    def update_send_hint_data(self, state, message):
        self.send_hint_data = xrTracking_pb2.HintMsg(state=state, text=message, guideStepType=0)
    
    
    
    def get_frequency_stats(self):
        """获取频率统计信息"""
        if len(self.frame_intervals) < 2:
            return None
        
        intervals = list(self.frame_intervals)
        avg_interval = sum(intervals) / len(intervals)
        frequency = 1.0 / avg_interval if avg_interval > 0 else 0
        
        return {
            'frequency_hz': frequency,
            'avg_interval_ms': avg_interval * 1000,
            'min_interval_ms': min(intervals) * 1000,
            'max_interval_ms': max(intervals) * 1000,
            'std_interval_ms': np.std(intervals) * 1000,
            'sample_count': len(intervals)
        }
    
    def get_latency_stats(self):
        """获取延迟统计信息"""
        stats = self.latency_stats
        if stats['count'] == 0:
            return None
        
        return {
            'avg_latency_ms': (stats['sum'] / stats['count']) * 1000,
            'min_latency_ms': stats['min'] * 1000,
            'max_latency_ms': stats['max'] * 1000,
            'total_frames': stats['count'],
            'duration_seconds': (stats['last_frame_time'] - stats['start_time']) if stats['start_time'] else 0
        }
    
    def save_to_csv(self, filename=None):
        """将延迟数据保存到 CSV 文件"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"latency_data_{timestamp}.csv"
        
        try:
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Frame Index', 'Timestamp', 'Frame Interval (ms)', 'Latency (ms)'])
                
                timestamps_list = list(self.timestamps)
                intervals_list = list(self.frame_intervals)
                
                for i in range(len(timestamps_list)):
                    interval_ms = intervals_list[i] * 1000 if i < len(intervals_list) else 0
                    writer.writerow([i, timestamps_list[i], interval_ms])
            
            print(f"Latency data saved to: {filename}")
            return filename
        except Exception as e:
            print(f"Error saving CSV: {e}")
            return None
    
class Pico4UltraStreamer(Streamer):
    """
    gRPC 订阅者类，用于订阅 VR 控制器 的数据流
    """

    # 初始化
    def __init__(self, ip, port,device, is_record = False): 
        super().__init__(ip, port, device, is_record)
        # Meta Quest IP 
        self.ip = ip
        self.port = port
        self.device = device
        self.is_record = is_record 
        self.recording = [] 
        self.latest = None 
        self.send_hint_data = xrTracking_pb2.HintMsg(state=0, text="idle", guideStepType=0)
        self.button_info_response = xrTracking_pb2.ButtonInfoResponse(locale=0)
        self.running = True
        self.locale = 0
        self.start_streaming()
        
        self._last_data_time = None

    def stream(self): 
        try:
            with grpc.insecure_channel(f"{self.ip}:{self.port}") as channel:
                stub = xrTracking_pb2_grpc.TrackingServiceStub(channel)
                responses = stub.StreamControllerUpdates(empty_pb2.Empty())
                responses_v2 = stub.StreamButtonInfo(self.update_send_data(), wait_for_ready=True) 
                
                
                # try:
                #     response = next(responses_v2)
                #     print("Received from server:", response.locale)
                #     self.locale = response.locale
                # except StopIteration:
                #     print("No response received.")
                # except grpc.RpcError as e:
                #     print("gRPC error:", e)
                    
                    
                response = next(responses_v2)
                self.locale = response.locale
                self.ready_to_show = response.readyToShow
                print(f"Received from server: locale={self.locale}, readyToShow={self.ready_to_show}")
                time.sleep(0.01)

                while not self.ready_to_show:
                    response = next(responses_v2)
                    self.locale = response.locale
                    self.ready_to_show = response.readyToShow
                    print(f"Received from server: locale={self.locale}, readyToShow={self.ready_to_show}")
                    time.sleep(0.04)  # 每 0.1 秒检查一次
                    
                prev_timestamp = None
                    
                for response in responses:
                    
                    # current_time = time.perf_counter()
                    current_time = time.time()
                    
                    # 更新数据时间戳
                    self._last_data_time = current_time
                    self.connected = True
                    self.connection_error = None
                    
                    if self.latency_stats['start_time'] is None:
                        self.latency_stats['start_time'] = current_time
                    
                    if prev_timestamp is not None:
                        frame_interval = current_time - prev_timestamp
                        self.frame_intervals.append(frame_interval)
                        
                        latency = current_time - self.latency_stats.get('last_server_time', current_time)
                        self.latency_stats['count'] += 1
                        self.latency_stats['sum'] += latency
                        self.latency_stats['min'] = min(self.latency_stats['min'], latency)
                        self.latency_stats['max'] = max(self.latency_stats['max'], latency)
                        self.latency_stats['last_frame_time'] = current_time
                    
                    self.timestamps.append(current_time)
                    prev_timestamp = current_time
                    self.latency_stats['last_server_time'] = current_time
                    
                    
                    transformations = {
                        "head": process_matrix(response.Head),
                        "left_controller_matrix": process_matrix(response.left_controller.matrix),  # 左控制器矩阵
                        "right_controller_matrix": process_matrix(response.right_controller.matrix), # 右控制器矩阵
                        "left_connect": response.left_controller.isConnected,   # 左控制器是否连接
                        "right_connect": response.right_controller.isConnected, # 右控制器是否连接
                        "left_btn_one": response.left_controller.btn_one.isPressed,       # 左控制器按钮1状态
                        "left_btn_two": response.left_controller.btn_two.isPressed,       # 左控制器按钮2状态
                        "left_trigger_index": response.left_controller.trigger_index.isPressed, # 左控制器食指扳机
                        "left_trigger_hand": response.left_controller.trigger_hand.isPressed,   # 左控制器手掌扳机
                        "right_btn_one": response.right_controller.btn_one.isPressed,      # 右控制器按钮1状态
                        "right_btn_two": response.right_controller.btn_two.isPressed,      # 右控制器按钮2状态
                        "right_trigger_index": response.right_controller.trigger_index.isPressed, # 右控制器食指扳机
                        "right_trigger_hand": response.right_controller.trigger_hand.isPressed,   # 右控制器手掌扳机
                        "left_thumb_stick": response.left_controller.thumb_stick.vector2,   # 左控制器摇杆
                        "right_thumb_stick": response.right_controller.thumb_stick.vector2, # 右控制器摇杆
                    }

                    transformations.update({"tracker_list": [None, None, None]})
                    tracker_count = 0
                    for i, tracker in enumerate(response.motionTracker):
                        transformations["tracker_list"][i] = process_matrix(tracker.matrix)
                        tracker_count += 1

                    if(tracker_count !=3):
                        log_m.tracker_state.update_log_info("ERROR", f"tracker count err : {tracker_count}\n")                        
                        print(f"tracker count err : {tracker_count}\n")
                        sys.exit()
                        self.running = False
                        # break

                    if self.record: 
                        self.recording.append(transformations)
                    self.latest = transformations 
                    if not (self.running is True):
                        break
        except grpc.RpcError as e:
            print(f"gRPC connection error: {e}")
            l_.mocap_info.update_log_info("CONNECTION ERROR", f"gRPC error: {e}")
            self.connected = False
            self.connection_error = str(e)
            self.running = False        
        
        
        except Exception as e:
            print(f"An error occurred: {e}")
            l_.mocap_info.update_log_info("EXCEPTION",f": {e} streamer error in buildStreamer")
            self.connected = False
            self.connection_error = str(e)
            self.running = False
            pass 
        
    # 启动流
    def start_streaming(self):
        stream_thread = Thread(target=self.stream)
        stream_thread.start()
        while self.latest is None and self.running:
            time.sleep(0.01)
        if self.latest is not None:
            print("== DATA IS FLOWING IN! ==")
            print("Ready to start streaming.")
        else:
            print("!! Failed to start streaming: no data received !!")
            
    def check_data_timeout(self, timeout_threshold=5):
        """检查数据是否超时未更新"""
        if not hasattr(self, '_last_data_time'):
            self._last_data_time = time.time()
            print("First data received, timestamp:", self._last_data_time)
            return False
        
        current_time = time.time()
        time_since_last_data = current_time - self._last_data_time
        
        if time_since_last_data > timeout_threshold:
            print("Data is older than", timeout_threshold, "seconds")
            return True
        
        return False    
        
    def get_latest(self): 
        return self.latest
        
    def get_recording(self): 
        return self.recording
    
    def print_frequency_stats(self):
        """打印频率统计信息"""
        freq_stats = self.get_frequency_stats()
        if freq_stats:
            print("\n=== Frequency Statistics ===")
            print(f"Average Frequency: {freq_stats['frequency_hz']:.2f} Hz")
            print(f"Average Interval: {freq_stats['avg_interval_ms']:.2f} ms")
            print(f"Min Interval: {freq_stats['min_interval_ms']:.2f} ms")
            print(f"Max Interval: {freq_stats['max_interval_ms']:.2f} ms")
            print(f"Std Deviation: {freq_stats['std_interval_ms']:.2f} ms")
            print(f"Sample Count: {freq_stats['sample_count']}")
            print("============================\n")
        return freq_stats
    
    def print_latency_stats(self):
        """打印延迟统计信息"""
        latency_stats = self.get_latency_stats()
        if latency_stats:
            print("\n=== Latency Statistics ===")
            print(f"Average Latency: {latency_stats['avg_latency_ms']:.2f} ms")
            print(f"Min Latency: {latency_stats['min_latency_ms']:.2f} ms")
            print(f"Max Latency: {latency_stats['max_latency_ms']:.2f} ms")
            print(f"Total Frames: {latency_stats['total_frames']}")
            print(f"Duration: {latency_stats['duration_seconds']:.2f} s")
            print("==========================\n")
        return latency_stats

    def update_send_hint_data(self,state, message, guideStepType):
        self.send_hint_data=xrTracking_pb2.HintMsg(state=state, text=message, guideStepType=guideStepType)

    def update_send_data(self):
        while True:
            message = xrTracking_pb2.ButtonInfo(
            hint=self.send_hint_data,
            )
            time.sleep(0.1)
            yield message

def main():
    ip = "10.140.235.183"   # 设备的IP
    port = 12345       # 服务器端口

    # 创建流对象
    streamer = Pico4UltraStreamer(ip, port, device="pico4_ultra", is_record=True)

    # 启动流
    streamer.start_streaming()

    # try:
    #     # 循环打印数据
    #     while True:
    #         if streamer.latest:
    #             print("Latest frame:")
    #             for k, v in streamer.latest.items():
    #                 print(f"  {k}: {v}")
    #         time.sleep(1.0)

    # except KeyboardInterrupt:
    #     print("Stopping streamer...")
    #     streamer.running = False
    
    try:
        stat_print_interval = 5.0
        last_stat_print_time = time.time()
        
        while True:
            current_time = time.time()
            
            if current_time - last_stat_print_time >= stat_print_interval:
                streamer.print_frequency_stats()
                streamer.print_latency_stats()
                last_stat_print_time = current_time
            
            if streamer.latest:
                pass
            # time.sleep(1.0)                               

    except KeyboardInterrupt:
        print("Stopping streamer...")
        streamer.running = False
        
        print("\nFinal Statistics:")
        streamer.print_frequency_stats()
        streamer.print_latency_stats()
        streamer.save_to_csv()

if __name__ == "__main__":
    main()

