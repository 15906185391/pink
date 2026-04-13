
# -*- coding: utf-8 -*-
"""
QP-based Inverse Kinematics Solver with Self-Collision Avoidance
使用二次规划求解器实现带自碰撞避免的逆运动学
"""

import os
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import numpy as np
import pinocchio as pin
# import qpsolvers
from .loop_rate_limiters import RateLimiter
from .pink import Configuration, solve_ik
from .pink.barriers import SelfCollisionBarrier
from .pink.tasks import FrameTask, PostureTask
from .pink.utils import process_collision_pairs


class QPIKSolver:
    """
    基于二次规划(QP)的逆运动学求解器，支持自碰撞避免
    
    特性:
    - 使用 Pink 库进行 IK 求解
    - 支持多末端执行器任务
    - 内置自碰撞检测与避免
    - 姿态保持任务
    - 可视化支持 (Viser)
    """
    
    def __init__(
        self,
        urdf_path: str,
        srdf_path: Optional[str] = None,
        package_dirs: Optional[List[str]] = None,
        mesh_dir: Optional[str] = None,
        solver_name: str = "osqp",
        frequency: float = 50.0,
        enable_collision_avoidance: bool = True
    ):
        """
        初始化 QP-IK 求解器
        
        Args:
            urdf_path: URDF 文件路径
            srdf_path: SRDF 文件路径（用于碰撞对排除）
            package_dirs: ROS package 搜索路径列表
            mesh_dir: Mesh 文件目录
            solver_name: QP 求解器名称 ('osqp', 'quadprog', 'scs' 等)
            frequency: 控制频率 (Hz)
            enable_collision_avoidance: 是否启用自碰撞避免
        """
        # 保存配置参数
        self.urdf_path = Path(urdf_path).resolve()
        self.srdf_path = Path(srdf_path).resolve() if srdf_path else None
        self.package_dirs = package_dirs or [str(self.urdf_path.parent.parent)]
        self.mesh_dir = Path(mesh_dir).resolve() if mesh_dir else self.urdf_path.parent / "meshes"
        self.solver_name = solver_name
        self.frequency = frequency
        self.enable_collision_avoidance = enable_collision_avoidance
        
        # 时间管理
        self.dt = 1.0 / frequency
        self.t = 0.0
        
        # 加载机器人模型
        print(f"[QPIKSolver] 加载 URDF: {self.urdf_path}")
        self.robot = pin.RobotWrapper.BuildFromURDF(
            filename=str(self.urdf_path),
            package_dirs=self.package_dirs,
            root_joint=None,
        )
        
        # 处理碰撞对
        if self.srdf_path and self.srdf_path.exists():
            print(f"[QPIKSolver] 加载 SRDF: {self.srdf_path}")
            self.robot.collision_data = process_collision_pairs(
                self.robot.model, 
                self.robot.collision_model, 
                str(self.srdf_path)
            )
        else:
            print("[QPIKSolver] 未提供 SRDF 文件，使用默认碰撞对")
            self.robot.collision_data = self.robot.collision_model.createData()
        
        # 初始关节配置
        self.q_ref = np.zeros(self.robot.model.nq)
        self._set_default_configuration()
        
        # 创建 Pink 配置
        self.configuration = Configuration(
            self.robot.model,
            self.robot.data,
            self.q_ref.copy(),
            collision_model=self.robot.collision_model if enable_collision_avoidance else None,
            collision_data=self.robot.collision_data if enable_collision_avoidance else None,
        )
        
        # 初始化任务字典
        self.tasks: Dict[str, FrameTask] = {}
        self.posture_task: Optional[PostureTask] = None
        self.collision_barrier: Optional[SelfCollisionBarrier] = None
        
        # 速率限制器
        self.rate_limiter = RateLimiter(frequency=frequency)
        
        # 可视化服务器（可选）
        self.viser_server = None
        self.transform_controls: Dict[str, any] = {}
        
        print(f"[QPIKSolver] 初始化完成 - 关节数: {self.robot.model.nq}, 频率: {frequency}Hz")
    
    def _set_default_configuration(self):
        """设置默认的关节配置"""
        # MagicBot Gen1 默认配置（双臂展开）
        self.q_ref = np.array([
            0, 0, 1.5707963, -1.5707963, -1.5707963, 0, 0,   # 左臂
            0, 0, -1.5707963, 1.5707963, 1.5707963, 0, 0      # 右臂
        ])
        
        # 确保维度匹配
        if len(self.q_ref) != self.robot.model.nq:
            print(f"[警告] 默认配置维度 ({len(self.q_ref)}) 与模型关节数 ({self.robot.model.nq}) 不匹配")
            self.q_ref = np.zeros(self.robot.model.nq)
    
    def add_end_effector_task(
        self,
        frame_name: str,
        position_cost: float = 1.5,
        orientation_cost: float = 0.5,
        task_name: Optional[str] = None
    ) -> FrameTask:
        """
        添加末端执行器任务
        
        Args:
            frame_name: 框架名称（如 'link_la7'）
            position_cost: 位置代价权重
            orientation_cost: 姿态代价权重
            task_name: 任务名称（默认为 frame_name）
            
        Returns:
            创建的 FrameTask 对象
        """
        task_name = task_name or frame_name
        
        task = FrameTask(
            frame_name,
            position_cost=position_cost,
            orientation_cost=orientation_cost,
        )
        
        self.tasks[task_name] = task
        print(f"[QPIKSolver] 添加末端任务: {task_name} (frame={frame_name})")
        
        return task
    
    def setup_collision_barrier(
        self,
        gain: float = 10.0,
        safe_displacement_gain: float = 5.0,
        d_min: float = 0.02
    ):
        """
        设置自碰撞避免屏障
        
        Args:
            gain: 屏障增益
            safe_displacement_gain: 安全位移增益
            d_min: 最小安全距离 (m)
        """
        if not self.enable_collision_avoidance:
            print("[警告] 碰撞避免已禁用")
            return
        
        n_pairs = len(self.robot.collision_model.collisionPairs)
        self.collision_barrier = SelfCollisionBarrier(
            n_collision_pairs=n_pairs,
            gain=gain,
            safe_displacement_gain=safe_displacement_gain,
            d_min=d_min,
        )
        
        print(f"[QPIKSolver] 碰撞屏障已设置 - 碰撞对数量: {n_pairs}, d_min: {d_min}m")
    
    def setup_posture_task(self, cost: float = 1e-4):
        """
        设置姿态保持任务
        
        Args:
            cost: 姿态代价权重
        """
        self.posture_task = PostureTask(cost=cost)
        print(f"[QPIKSolver] 姿态任务已设置 - cost: {cost}")
    
    def set_task_target(
        self,
        task_name: str,
        position: np.ndarray,
        quaternion: Optional[np.ndarray] = None,
        rotation_matrix: Optional[np.ndarray] = None
    ):
        """
        设置任务目标位姿
        
        Args:
            task_name: 任务名称
            position: 目标位置 [x, y, z]
            quaternion: 目标四元数 [w, x, y, z]（可选）
            rotation_matrix: 目标旋转矩阵 3x3（可选，优先于 quaternion）
        """
        if task_name not in self.tasks:
            raise ValueError(f"任务 '{task_name}' 不存在，请先使用 add_end_effector_task 添加")
        
        task = self.tasks[task_name]
        
        # 设置位置
        task.transform_target_to_world.translation = position.copy()
        
        # 设置姿态
        if rotation_matrix is not None:
            task.transform_target_to_world.rotation = rotation_matrix.copy()
        elif quaternion is not None:
            # 将四元数转换为旋转矩阵
            quat = pin.Quaternion(quaternion[0], quaternion[1], quaternion[2], quaternion[3])
            task.transform_target_to_world.rotation = quat.toRotationMatrix()
    
    def set_task_target_from_transform(self, task_name: str, transform: np.ndarray):
        """
        从 4x4 变换矩阵设置任务目标
        
        Args:
            task_name: 任务名称
            transform: 4x4 齐次变换矩阵
        """
        if task_name not in self.tasks:
            raise ValueError(f"任务 '{task_name}' 不存在")
        
        position = transform[:3, 3]
        rotation = transform[:3, :3]
        
        self.set_task_target(task_name, position, rotation_matrix=rotation)
    
    def initialize_tasks(self):
        """初始化所有任务的目标为当前配置"""
        for task_name, task in self.tasks.items():
            task.set_target_from_configuration(self.configuration)
            print(f"[QPIKSolver] 任务 '{task_name}' 已初始化")
    
    def solve(self, dt: Optional[float] = None) -> np.ndarray:
        """
        求解逆运动学
        
        Args:
            dt: 时间步长（默认使用初始化时的值）
            
        Returns:
            关节速度向量
        """
        if dt is None:
            dt = self.dt
        
        # 构建任务列表
        active_tasks = list(self.tasks.values())
        if self.posture_task is not None:
            active_tasks.append(self.posture_task)
        
        # 构建屏障列表
        barriers = []
        if self.collision_barrier is not None:
            barriers.append(self.collision_barrier)
        
        # 求解 IK
        velocity = solve_ik(
            self.configuration,
            active_tasks,
            dt,
            solver=self.solver_name,
            barriers=barriers,
            safety_break=False,
        )
        
        # 积分更新配置
        self.configuration.integrate_inplace(velocity, dt)
        
        # 更新时间
        self.t += dt
        
        # return velocity
        
        self.configuration.integrate_inplace(velocity, dt)
        
        return self.configuration.q
        
        
    
    def get_current_qpos(self) -> np.ndarray:
        """获取当前关节位置"""
        return self.configuration.q.copy()
    
    def get_current_ee_pose(self, frame_name: str) -> np.ndarray:
        """
        获取指定框架的当前位姿
        
        Args:
            frame_name: 框架名称
            
        Returns:
            4x4 齐次变换矩阵
        """
        transform = self.configuration.get_transform_frame_to_world(frame_name)
        return transform.np.copy()
    
    def check_collision_status(self) -> Dict[str, float]:
        """
        检查碰撞状态
        
        Returns:
            包含碰撞信息的字典
        """
        if self.collision_barrier is None:
            return {"status": "disabled"}
        
        h = self.collision_barrier.compute_barrier(self.configuration)
        
        min_val = float(np.min(h))
        max_val = float(np.max(h))
        mean_val = float(np.mean(h))
        
        status = "safe"
        if min_val <= -0.02:
            status = "critical"
        elif min_val < 0.01:
            status = "warning"
        
        return {
            "status": status,
            "min_barrier": min_val,
            "max_barrier": max_val,
            "mean_barrier": mean_val,
        }
    
    def setup_viser_visualization(self, host: str = "0.0.0.0", port: int = 8080):
        """
        设置 Viser 可视化
        
        Args:
            host: 服务器主机地址
            port: 服务器端口
        """
        try:
            import viser
            from viser.extras import ViserUrdf
            import webbrowser
            import time
            
            # 创建服务器
            self.viser_server = viser.ViserServer(host=host, port=port)
            
            # 添加地面网格
            self.viser_server.scene.add_grid("/ground", width=2, height=2)
            
            # 加载 URDF 可视化
            self.urdf_vis = ViserUrdf(
                self.viser_server, 
                str(self.urdf_path), 
                root_node_name="/pelvis"
            )
            
            # 自动打开浏览器
            time.sleep(0.5)
            webbrowser.open(f"http://localhost:{port}")
            
            print(f"[QPIKSolver] Viser 可视化已启动 - http://localhost:{port}")
            
        except ImportError:
            print("[警告] viser 未安装，跳过可视化设置")
            self.viser_server = None
    
    def add_transform_control(
        self,
        control_name: str,
        position: Tuple[float, float, float] = (0.3, 0.0, 0.5),
        scale: float = 0.2
    ):
        """
        添加交互式变换控制器
        
        Args:
            control_name: 控制器名称
            position: 初始位置 (x, y, z)
            scale: 控制器缩放比例
        """
        if self.viser_server is None:
            print("[警告] 可视化服务器未初始化")
            return
        
        control = self.viser_server.scene.add_transform_controls(
            f"/{control_name}",
            scale=scale,
            position=position
        )
        
        self.transform_controls[control_name] = control
        print(f"[QPIKSolver] 添加变换控制器: {control_name}")
    
    # def update_task_from_control(self, task_name: str, control_name: str):
    #     """
    #     从变换控制器更新任务目标
        
    #     Args:
    #         task_name: 任务名称
    #         control_name: 控制器名称
    #     """
    #     if control_name not in self.transform_controls:
    #         raise ValueError(f"控制器 '{control_name}' 不存在")
        
    #     control = self.transform_controls[control_name]
        
    #     position = np.array(control.position)
    #     wxyz = np.array(control.wxyz)
        
    #     # 转换四元数顺序: viser (w,x,y,z) -> pinocchio (w,x,y,z)
    #     quaternion = pin.Quaternion(wxyz[0], wxyz[1], wxyz[2], wxyz[3])
        
    #     self.set_task_target(task_name, position, quaternion=quaternion.vector())
    
    def update_task_from_control(
        self, 
        task_name: str, 
        control_name: Optional[str] = None,
        target_pose: Optional[np.ndarray] = None
    ):
        """
        从变换控制器或直接传入位姿更新任务目标
        
        Args:
            task_name: 任务名称
            control_name: 控制器名称（可选，如果提供 target_pose 则忽略）
            target_pose: 4x4 齐次变换矩阵（可选，优先级高于 control_name）
        """
        # 优先使用直接传入的位姿
        if target_pose is not None:
            if target_pose.shape != (4, 4):
                raise ValueError(f"target_pose 必须是 4x4 矩阵，当前形状: {target_pose.shape}")
            
            # 从 4x4 变换矩阵提取位置和旋转
            position = target_pose[:3, 3]
            rotation_matrix = target_pose[:3, :3]
            
            self.set_task_target(task_name, position, rotation_matrix=rotation_matrix)
            return
        
        # 如果没有提供 target_pose，则从 viser 控制器读取
        if control_name is None:
            raise ValueError("必须提供 control_name 或 target_pose 其中之一")
        
        if control_name not in self.transform_controls:
            raise ValueError(f"控制器 '{control_name}' 不存在")
        
        control = self.transform_controls[control_name]
        
        position = np.array(control.position)
        wxyz = np.array(control.wxyz)
        
        # 转换四元数顺序: viser (w,x,y,z) -> pinocchio (w,x,y,z)
        quaternion = pin.Quaternion(wxyz[0], wxyz[1], wxyz[2], wxyz[3])
        
        self.set_task_target(task_name, position, quaternion=quaternion.vector())
    
    def update_visualization(self):
        """更新可视化显示"""
        if self.viser_server is None:
            return
        
        # 更新 URDF 配置
        if hasattr(self, 'urdf_vis'):
            self.urdf_vis.update_cfg(self.configuration.q)
        
        # 更新碰撞状态指示器
        collision_info = self.check_collision_status()
        if collision_info["status"] == "critical":
            color = (255, 0, 0)  # 红色
        elif collision_info["status"] == "warning":
            color = (255, 255, 0)  # 黄色
        else:
            color = (0, 255, 0)  # 绿色
        
        self.viser_server.scene.add_icosphere(
            "/collision_indicator",
            radius=0.05,
            color=color
        )
    
    def sleep(self):
        """等待下一个控制周期"""
        self.rate_limiter.sleep()
    
    def reset(self, qpos: Optional[np.ndarray] = None):
        """
        重置求解器状态
        
        Args:
            qpos: 重置到的关节位置（默认使用初始配置）
        """
        if qpos is None:
            qpos = self.q_ref.copy()
        
        self.configuration = Configuration(
            self.robot.model,
            self.robot.data,
            qpos,
            collision_model=self.robot.collision_model if self.enable_collision_avoidance else None,
            collision_data=self.robot.collision_data if self.enable_collision_avoidance else None,
        )
        
        self.t = 0.0
        self.initialize_tasks()
        
        print("[QPIKSolver] 求解器已重置")
    
    def close(self):
        """关闭求解器并清理资源"""
        if self.viser_server is not None:
            self.viser_server.stop()
        print("[QPIKSolver] 求解器已关闭")
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()


# ==================== 使用示例 ====================

if __name__ == "__main__":
    # URDF 路径
    urdf_path = Path(__file__).parent / "robots" / "magicbot-gen1_description" / "urdf" / "MAGICBOT_with_hand.urdf"
    srdf_path = Path(__file__).parent / "magic_gen1_spheres_collision.srdf"
    
    # 创建求解器
    with QPIKSolver(
        urdf_path=str(urdf_path),
        srdf_path=str(srdf_path) if srdf_path.exists() else None,
        solver_name="osqp",
        frequency=50.0,
        enable_collision_avoidance=True
    ) as ik_solver:
        
        # 添加末端执行器任务
        ik_solver.add_end_effector_task(
            frame_name="link_la7",
            position_cost=1.5,
            orientation_cost=0.5,
            task_name="left_hand"
        )
        
        ik_solver.add_end_effector_task(
            frame_name="link_ra7",
            position_cost=1.5,
            orientation_cost=0.5,
            task_name="right_hand"
        )
        
        # 设置碰撞避免
        ik_solver.setup_collision_barrier(
            gain=10.0,
            safe_displacement_gain=5.0,
            d_min=0.02
        )
        
        # 设置姿态任务
        ik_solver.setup_posture_task(cost=1e-4)
        
        # 初始化任务
        ik_solver.initialize_tasks()
        
        # 设置可视化
        ik_solver.setup_viser_visualization(port=8080)
        
        # 添加交互控制器
        ik_solver.add_transform_control(
            "ik_target_l",
            position=(0.3, 0.1, 0.56)
        )
        ik_solver.add_transform_control(
            "ik_target_r",
            position=(0.3, -0.1, 0.56)
        )
        
        # 主循环
        print("\n开始 IK 求解循环...")
        t = 0.0
        
        try:
            while True:
                # 从控制器更新任务目标
                ik_solver.update_task_from_control("left_hand", "ik_target_l")
                ik_solver.update_task_from_control("right_hand", "ik_target_r")
                
                # 求解 IK
                velocity = ik_solver.solve()
                
                # 检查碰撞状态
                collision_info = ik_solver.check_collision_status()
                if collision_info["status"] == "critical":
                    print(f"\n⚠️  严重警告: 检测到自碰撞!")
                    print(f"   最小屏障值: {collision_info['min_barrier']:.4f}")
                    break
                
                # 更新可视化
                ik_solver.update_visualization()
                
                # 等待下一周期
                ik_solver.sleep()
                t += ik_solver.dt
                
        except KeyboardInterrupt:
            print("\n用户中断，退出程序")