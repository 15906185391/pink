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
from robot_descriptions.iiwa14_description import PACKAGE_PATH, REPOSITORY_PATH

import meshcat_shapes
import pink
import viser
from viser.extras import ViserUrdf

from pink import solve_ik
from pink.barriers import SelfCollisionBarrier
from pink.tasks import FrameTask, PostureTask
from pink.utils import process_collision_pairs
from pink.visualization import start_meshcat_visualizer, start_viser_visualizer


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

    srdf_path = (
        os.path.dirname(os.path.realpath(__file__))
        + "/magic_gen1_spheres_collision.srdf"
    )
    print(srdf_path)
    # viz = start_meshcat_visualizer(robot)
    # viz = start_viser_visualizer(robot)
    q_ref = np.array(
        [  0, 0, 1.5707963, -1.5707963, -1.5707963, 0, 0, 
            0, 0, -1.5707963, 1.5707963, 1.5707963, 0, 0]
    )

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
        position_cost=1.5,  # [cost] / [m]
        orientation_cost=0.5,  # [cost] / [rad]
    )
    right_end_effector_task = FrameTask(
        "link_ra7",
        position_cost=1.5,  # [cost] / [m]
        orientation_cost=0.5,  # [cost] / [rad]
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

    rate = RateLimiter(frequency=50.0)
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

    # Target gizmo.
    ik_target_l = server.scene.add_transform_controls(
        "/ik_target_l", scale=0.2, position=(0.3, 0.1, 0.56), wxyz=(0, 0, 1, 0)
    )
    
    
    ik_target_r = server.scene.add_transform_controls(
        "/ik_target_r", scale=0.2, position=(0.3, -0.1, 0.56), wxyz=(0, 0, 1, 0)
    )
    
    
    
    target_link_name_l = "link_la7"
    
    l_y_des = np.array([0.392, 0.392, 0.6])
    r_y_des = np.array([0.392, 0.392, 0.6])
    
    l_dy_des = np.zeros(3)
    r_dy_des = np.zeros(3)

    while True:
        # # Make a sinusoidal trajectory between points A and B
        # mu = (1 + np.cos(t)) / 2
        # l_y_des[:] = (
        #     A + (B - A + 0.2 * np.array([0, 0, np.sin(mu * np.pi) ** 2])) * mu
        # )
        # r_y_des[:] = (
        #     B + (A - B + 0.2 * np.array([0, 0, -np.sin(mu * np.pi) ** 2])) * mu
        # )

        # left_end_effector_task.transform_target_to_world.translation = l_y_des
        # right_end_effector_task.transform_target_to_world.translation = r_y_des

        # Calculate desired trajectory
        A = 0.1
        B = 0.1
        # z -- 0.4 - 0.8
        l_y_des[:] = (
            0.2,
            0.1 + B * np.sin(t),
            0.2 + A * np.sin(t),
        )
        r_y_des[:] = (
            0.2,
            -0.1 - B * np.sin(t),
            0.2 + A * np.sin(t),
        )
        l_dy_des[:] = 0, B * np.cos(t), A * np.cos(t)
        r_dy_des[:] = 0, -B * np.cos(t), A * np.cos(t)
        
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
        # # Update visualization frames
        # viewer["left_end_effector"].set_transform(
        #     configuration.get_transform_frame_to_world(
        #         left_end_effector_task.frame
        #     ).np
        # )
        # viewer["right_end_effector"].set_transform(
        #     configuration.get_transform_frame_to_world(
        #         right_end_effector_task.frame
        #     ).np
        # )
        # viewer["left_end_effector_target"].set_transform(
        #     left_end_effector_task.transform_target_to_world.np
        # )
        # viewer["right_end_effector_target"].set_transform(
        #     right_end_effector_task.transform_target_to_world.np
        # )

        velocity = solve_ik(
            configuration,
            tasks,
            dt,
            solver=solver,
            barriers=barriers,
            safety_break=False,
        )
        configuration.integrate_inplace(velocity, dt)

        # Visualize result at fixed FPS
        # viz.display(configuration.q)
        urdf_vis.update_cfg(configuration.q)
        rate.sleep()
        t += dt
