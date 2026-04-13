import os
import rclpy
from mocap_system.mocap import Mocap

def main():
    # 可通过环境变量指定，例如： export CPU_AFFINITY="0,1"
    aff = "0,1,2,3,4,5,6,7"
    if aff:
        try:
            cpus = {int(x) for x in aff.split(",") if x.strip() != ""}
            # 优先使用标准库（Linux）
            os.sched_setaffinity(0, cpus)
        except AttributeError:
            # 如果系统不支持 sched_setaffinity，可尝试 psutil（需安装）
            try:
                import psutil
                p = psutil.Process()
                p.cpu_affinity(list(cpus))
            except Exception:
                pass    
    rclpy.init()

    robot_ctrl_ros = Mocap()

    # 运行节点
    rclpy.spin(robot_ctrl_ros)

    # 销毁节点，退出ROS2
    robot_ctrl_ros.destroy_node()
    rclpy.shutdown()

if __name__ =="__main__":

    main()