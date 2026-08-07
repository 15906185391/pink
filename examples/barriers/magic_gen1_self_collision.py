#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 Ivan Domrachev, Simeon Nedelchev
#
# /// script
# dependencies = ["daqp", "loop-rate-limiters", "meshcat", "pin-pink",
# "qpsolvers", "robot_descriptions"]
# ///

"""Two iiwa14-s with full-body self-collision avoidance using hpp-fcl."""

import os
from pathlib import Path
import yourdfpy

import numpy as np
import pinocchio as pin
import qpsolvers
from loop_rate_limiters import RateLimiter
# from robot_descriptions.iiwa14_description import PACKAGE_PATH, REPOSITORY_PATH

import meshcat_shapes
import pink
import viser
from viser.extras import ViserUrdf

from pink import solve_ik
from pink.barriers import SelfCollisionBarrier
from pink.tasks import FrameTask, PostureTask
from pink.utils import process_collision_pairs
from pink.visualization import start_meshcat_visualizer, start_viser_visualizer
from mocap_system.robot.magicbot import MagicBotGen1


if __name__ == "__main__":
    
    urdf_path = os.path.join(
        os.path.dirname(__file__),
        "..",  # 返回到 examples 目录
        "robots",
        "magicbot-gen1_description",
        "urdf",
        "MAGICBOT_with_hand.urdf",
    )
    
    urdf_path = Path(__file__).parent.parent / "robots" / "magicbot-gen1_description" / "urdf" / "MAGICBOT_with_hand.urdf"
    
    # Load URDF with search path for meshes
    urdf = yourdfpy.URDF.load(
        urdf_path,
        build_collision_scene_graph=True,
        load_meshes=True,
        mesh_dir=urdf_path.parent.parent / "meshes",
    )

    robot = pin.RobotWrapper.BuildFromURDF(
        filename=urdf_path,
        package_dirs=[os.path.dirname(os.path.dirname(__file__))],
        root_joint=None,
    )
    
    q_ref = np.array(
        [  0, 0, 1.5707963, -1.5707963, -1.5707963, 0, 0, 
            0, 0, -1.5707963, 1.5707963, 1.5707963, 0, 0]
    )
    
    pin.forwardKinematics(robot.model, robot.data, q_ref)
    pin.updateFramePlacements(robot.model, robot.data)
    
    # 获取左右手末端连杆的 frame ID
    left_ee_frame_name = "link_la7"
    right_ee_frame_name = "link_ra7"
    
    left_ee_frame_id = robot.model.getFrameId(left_ee_frame_name)
    right_ee_frame_id = robot.model.getFrameId(right_ee_frame_name)
    
    # 检查 frame 是否存在
    if left_ee_frame_id == robot.model.nframes:
        raise ValueError(f"找不到左末端连杆: {left_ee_frame_name}")
    if right_ee_frame_id == robot.model.nframes:
        raise ValueError(f"找不到右末端连杆: {right_ee_frame_name}")
    
    # 获取末端位姿 (SE3 变换矩阵)
    ee_l = robot.data.oMf[left_ee_frame_id]  # SE3 对象
    ee_r = robot.data.oMf[right_ee_frame_id]  # SE3 对象
    
    magicbot = MagicBotGen1()
    magicbot.get_current_qpos()
    magicbot.move2first()
    magicbot.move2readypos()
    

    srdf_path = (
        os.path.dirname(os.path.realpath(__file__))
        + "/magic_gen1_spheres_collision.srdf"
    )
    print(srdf_path)
    # viz = start_meshcat_visualizer(robot)
    # viz = start_viser_visualizer(robot)
    

    # Collisions: processing collisions from urdf (include all) and srdf
    # (exclude specified) and updating collision model and creating
    # corresponding collision data
    robot.collision_data = process_collision_pairs(
        robot.model, robot.collision_model, srdf_path
    )

    configuration = pink.Configuration(
        robot.model,
        robot.data,
        q_ref,
        collision_model=robot.collision_model,  # for self-collision barrier
        collision_data=robot.collision_data,
    )

    # Pink tasks
    left_end_effector_task = FrameTask(
        "link_la7",
        position_cost=5.0,  # [cost] / [m]
        orientation_cost=1.0,  # [cost] / [rad]
        gain=0.5,
    )
    right_end_effector_task = FrameTask(
        "link_ra7",
        position_cost=5.0,  # [cost] / [m]
        orientation_cost=1.0,  # [cost] / [rad]
        gain=0.5,
    )

    # Pink barriers
    collision_barrier = SelfCollisionBarrier(
        n_collision_pairs=len(robot.collision_model.collisionPairs),
        gain=10.0,
        safe_displacement_gain=5.0,
        d_min=0.02,
    )

    posture_task = PostureTask(
        cost=1e-4,  # [cost] / [rad]
    )
    

    barriers = [collision_barrier]
    tasks = [left_end_effector_task, right_end_effector_task, posture_task]

    
    # configuration = pink.Configuration(robot.model, robot.data, q_ref)
    
    for task in tasks:
        task.set_target_from_configuration(configuration)
    # viz.display(configuration.q)

    # viewer = viz.viewer
    # meshcat_shapes.frame(viewer["left_end_effector"], opacity=1.0)
    # meshcat_shapes.frame(viewer["right_end_effector"], opacity=1.0)
    # meshcat_shapes.frame(viewer["left_end_effector_target"], opacity=1.0)
    # meshcat_shapes.frame(viewer["right_end_effector_target"], opacity=1.0)

    # Select QP solver
    solver = qpsolvers.available_solvers[0]
    if "osqp" in qpsolvers.available_solvers:
        solver = "osqp"

    rate = RateLimiter(frequency=50.0, warn=False)
    dt = rate.period
    t = 0.0  # [s]
    # l_y_des = np.array([0.2, -0.1, 0.2])
    # r_y_des = np.array([0.2, 0.1, 0.2])

    # A = l_y_des.copy()
    # B = r_y_des.copy()

    # l_dy_des = np.zeros(3)
    # r_dy_des = np.zeros(3)
    
    # Set up visualizer.
    server = viser.ViserServer(host="0.0.0.0", port=8080)
    
    # 自动打开浏览器
    import webbrowser
    import time
    time.sleep(0.5)  # 等待服务器启动
    webbrowser.open(f"http://localhost:8080")
    server.scene.add_grid("/ground", width=2, height=2)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/pelvis")

    # # Target gizmo.
    # ik_target_l = server.scene.add_transform_controls(
    #     "/ik_target_l", scale=0.2, position=(0.3, 0.1, 0.56), wxyz=(0, 0, 1, 0)
    # )
    
    
    # ik_target_r = server.scene.add_transform_controls(
    #     "/ik_target_r", scale=0.2, position=(0.3, -0.1, 0.56), wxyz=(0, 0, 1, 0)
    # )
    
    # 将位姿转换为 viser 格式
    # 假设 ee_l 和 ee_r 是 4x4 变换矩阵或类似对象
    if hasattr(ee_l, 'translation'):
        # 如果是 Pinocchio SE3 对象
        pos_l = ee_l.translation
        rot_l = pin.Quaternion(ee_l.rotation)
        wxyz_l = np.array([rot_l.w, rot_l.x, rot_l.y, rot_l.z])
        
        pos_r = ee_r.translation
        rot_r = pin.Quaternion(ee_r.rotation)
        wxyz_r = np.array([rot_r.w, rot_r.x, rot_r.y, rot_r.z])
    elif isinstance(ee_l, np.ndarray) and ee_l.shape == (4, 4):
        # 如果是 4x4 变换矩阵
        pos_l = ee_l[:3, 3]
        rot_l = pin.Quaternion(ee_l[:3, :3])
        wxyz_l = np.array([rot_l.w, rot_l.x, rot_l.y, rot_l.z])
        
        pos_r = ee_r[:3, 3]
        rot_r = pin.Quaternion(ee_r[:3, :3])
        wxyz_r = np.array([rot_r.w, rot_r.x, rot_r.y, rot_r.z])
    else:
        # 默认值（如果无法解析）
        print("⚠️  警告: 无法解析 ee_l/ee_r 格式，使用默认位置")
        pos_l = np.array([0.3, 0.1, 0.56])
        wxyz_l = np.array([0, 0, 1, 0])
        pos_r = np.array([0.3, -0.1, 0.56])
        wxyz_r = np.array([0, 0, 1, 0])
    
    print(f"✓ 左手初始位姿: position={pos_l}, wxyz={wxyz_l}")
    print(f"✓ 右手初始位姿: position={pos_r}, wxyz={wxyz_r}")

    # Target gizmo - 使用真实位姿初始化
    ik_target_l = server.scene.add_transform_controls(
        "/ik_target_l", 
        scale=0.2, 
        position=pos_l,  
        wxyz=wxyz_l      
    )
    
    ik_target_r = server.scene.add_transform_controls(
        "/ik_target_r", 
        scale=0.2, 
        position=pos_r, 
        wxyz=wxyz_r    
    )
    
    print("✓ 变换控制器初始化完成")
    
    joint_cmd = np.zeros(30)
    
    joint_cmd[14:30] = magicbot.get_current_qpos()[14:30]
    
            # ==================== 轨迹平滑滤波器 ====================
    class TrajectorySmoother:
        """
        轨迹平滑器 - 使用指数移动平均和低通滤波
        
        原理：
        1. 指数移动平均（EMA）：对新旧值进行加权平均
        2. 速度限制：限制关节速度的最大变化率
        3. 加速度限制：限制加速度的变化
        """
        def __init__(self, n_joints=14, alpha=0.3, max_velocity=1.0, max_acceleration=5.0):
            """
            Args:
                n_joints: 关节数量
                alpha: EMA 系数 (0-1)，越小越平滑但响应越慢
                max_velocity: 最大关节速度 (rad/s)
                max_acceleration: 最大关节加速度 (rad/s²)
            """
            self.n_joints = n_joints
            self.alpha = alpha
            self.max_velocity = max_velocity
            self.max_acceleration = max_acceleration
            
            # 初始化状态
            self.prev_q = None          # 上一时刻位置
            self.prev_vel = np.zeros(n_joints)  # 上一时刻速度
            self.prev_acc = np.zeros(n_joints)  # 上一时刻加速度
            self.filtered_q = None      # 滤波后的位置
            
            print(f"[TrajectorySmoother] 初始化完成")
            print(f"  - 关节数: {n_joints}")
            print(f"  - EMA 系数: {alpha}")
            print(f"  - 最大速度: {max_velocity} rad/s")
            print(f"  - 最大加速度: {max_acceleration} rad/s²")
        
        def smooth(self, target_q, dt):
            """
            平滑目标关节角度
            
            Args:
                target_q: 目标关节角度数组
                dt: 时间步长
                
            Returns:
                smoothed_q: 平滑后的关节角度
            """
            if self.prev_q is None:
                # 第一次调用，直接返回目标值
                self.prev_q = target_q.copy()
                self.filtered_q = target_q.copy()
                return target_q.copy()
            
            # 步骤 1: 计算期望速度
            desired_vel = (target_q - self.prev_q) / dt
            
            # 步骤 2: 速度限制
            vel_magnitude = np.linalg.norm(desired_vel)
            if vel_magnitude > self.max_velocity:
                desired_vel = desired_vel * (self.max_velocity / vel_magnitude)
            
            # 步骤 3: 加速度限制
            desired_acc = (desired_vel - self.prev_vel) / dt
            acc_magnitude = np.linalg.norm(desired_acc)
            if acc_magnitude > self.max_acceleration:
                desired_acc = desired_acc * (self.max_acceleration / acc_magnitude)
                desired_vel = self.prev_vel + desired_acc * dt
            
            # 步骤 4: 指数移动平均滤波
            smoothed_q = self.alpha * target_q + (1 - self.alpha) * self.filtered_q
            
            # 步骤 5: 基于平滑位置的最终速度限制
            final_vel = (smoothed_q - self.prev_q) / dt
            final_vel_mag = np.linalg.norm(final_vel)
            if final_vel_mag > self.max_velocity:
                smoothed_q = self.prev_q + final_vel * (self.max_velocity / final_vel_mag) * dt
            
            # 更新状态
            self.prev_acc = desired_acc
            self.prev_vel = desired_vel
            self.prev_q = smoothed_q.copy()
            self.filtered_q = smoothed_q.copy()
            
            return smoothed_q
        
        def reset(self):
            """重置滤波器状态"""
            self.prev_q = None
            self.prev_vel = np.zeros(self.n_joints)
            self.prev_acc = np.zeros(self.n_joints)
            self.filtered_q = None
            print("[TrajectorySmoother] 已重置")
    
    # 创建轨迹平滑器实例
    smoother = TrajectorySmoother(
        n_joints=14,
        alpha=0.4,              # EMA 系数：0.3-0.5 之间比较合适
        max_velocity=2.0,       # 最大关节速度 2 rad/s
        max_acceleration=10.0   # 最大关节加速度 10 rad/s²
    )

    while True:
        target_position_l = np.array(ik_target_l.position)
        target_wxyz_l = np.array(ik_target_l.wxyz)
        left_end_effector_task.transform_target_to_world.translation = target_position_l
        # 使用 Pinocchio 的四元数转旋转矩阵
        quaternion = pin.Quaternion(target_wxyz_l[0], target_wxyz_l[1], target_wxyz_l[2], target_wxyz_l[3])
        left_end_effector_task.transform_target_to_world.rotation = quaternion.matrix()
        
        target_position_r=np.array(ik_target_r.position)
        target_wxyz_r=np.array(ik_target_r.wxyz)

        # left_end_effector_task.transform_target_to_world.translation = l_y_des
        right_end_effector_task.transform_target_to_world.translation = target_position_r
        # 使用 Pinocchio 的四元数转旋转矩阵
        quaternion = pin.Quaternion(target_wxyz_r[0], target_wxyz_r[1], target_wxyz_r[2], target_wxyz_r[3])
        right_end_effector_task.transform_target_to_world.rotation = quaternion.matrix()
        
        # h = collision_barrier.compute_barrier(configuration)
        # self.assertTrue(np.all(h[collision_barrier.non_colliding_objects_pair_id] > 0))

        # 检查碰撞屏障状态 (可选,用于调试)
        h = collision_barrier.compute_barrier(configuration)

        # 获取所有碰撞对的最小距离值
        min_val = np.min(h)
        max_val = np.max(h)
        mean_val = np.mean(h)

        # 在 viser 场景中添加状态指示器
        if min_val <= -0.02:
            server.scene.add_icosphere("/collision_warning", radius=0.05, color=(255, 0, 0))  # 红色 - 危险
            print(f"\n{'='*60}")
            print(f"⚠️  严重警告: 检测到自碰撞!")
            print(f"   最小屏障值: {min_val:.4f}")
            print(f"   最大屏障值: {max_val:.4f}")
            print(f"   平均屏障值: {mean_val:.4f}")
            print(f"{'='*60}\n")
            print("程序已停止以防止碰撞损坏。")
            break  # 退出循环，停止程序
        elif min_val < 0.01:
            server.scene.add_icosphere("/collision_warning", radius=0.05, color=(255, 255, 0))  # 黄色 - 警告
            print(f"⚡ 注意: 接近碰撞边界! 最小屏障值: {min_val:.4f}")
        else:
            server.scene.add_icosphere("/collision_warning", radius=0.05, color=(0, 255, 0))  # 绿色 - 安全
            
        velocity = solve_ik(
            configuration,
            tasks,
            dt,
            solver=solver,
            barriers=barriers,
            safety_break=False,
        )
        configuration.integrate_inplace(velocity, dt)
        
        # 获取 IK 求解后的关节角度
        raw_q = configuration.q.copy()
        
        # 应用平滑滤波
        smoothed_q = smoother.smooth(raw_q, dt)
        
        # 将平滑后的结果写回 configuration
        configuration.q[:] = smoothed_q

        urdf_vis.update_cfg(configuration.q)
        joint_cmd[:14] = configuration.q
        magicbot.lcm_handle.publish_joint_command(joint_cmd)
        rate.sleep()
        t += dt
