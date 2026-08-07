#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
#
# /// script
# dependencies = ["daqp", "loop-rate-limiters", "pin-pink", "qpsolvers",
# "viser", "yourdfpy"]
# ///

"""PICO teleoperation for the wheel-arm robot using Pink IK."""

from pathlib import Path
import argparse
import sys
import time
import webbrowser

import numpy as np
import pinocchio as pin
import qpsolvers
import viser
import yourdfpy
from loop_rate_limiters import RateLimiter
from viser.extras import ViserUrdf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pink  # noqa: E402
from pink import solve_ik  # noqa: E402
from pink.barriers import SelfCollisionBarrier  # noqa: E402
from pink.exceptions import NoSolutionFound  # noqa: E402
from pink.tasks import FrameTask, PostureTask  # noqa: E402

from wheel_arm_self_collision import (  # noqa: E402
    JOINT_COLORS,
    JOINT_PLOT_ASPECT,
    JOINT_PLOT_UPDATE_HZ,
    JOINT_PLOT_WINDOW_S,
    JOINT_PRIMARY_PLOT_HEIGHT,
    JOINT_WRIST_PLOT_HEIGHT,
    LEFT_TCP,
    PACKAGE_NAME,
    RIGHT_TCP,
    SOLVE_FREQUENCY_HZ,
    LockedJointsTask,
    arm_joint_indices,
    arm_joint_names,
    build_primitive_collision_model,
    configure_self_collision,
    format_joint_table,
    initial_configuration,
    locked_joint_indices,
    pinocchio_to_yourdfpy_cfg,
    require_frame,
    resolve_package_uri,
    se3_to_viser_pose,
    select_solver,
)
from xrobotoolkit_teleop.common.xr_client import XrClient  # noqa: E402
from xrobotoolkit_teleop.utils.geometry import R_HEADSET_TO_WORLD  # noqa: E402


class MockXrClient:
    """Tiny XR source for testing the program without a PICO device."""

    def __init__(self) -> None:
        self.start_time = time.monotonic()

    def get_pose_by_name(self, name: str) -> np.ndarray:
        t = time.monotonic() - self.start_time
        y = 0.25 if name == "left_controller" else -0.25
        return np.array([0.05 * np.sin(t), y, 0.02 * np.cos(t), 0, 0, 0, 1])

    def get_key_value_by_name(self, name: str) -> float:
        return 1.0 if name in ("left_grip", "right_grip") else 0.0

    def get_button_state_by_name(self, name: str) -> bool:
        return False

    def close(self) -> None:
        pass


class RelativeTeleopTarget:
    """Map relative XR controller motion to a robot end-effector target."""

    def __init__(self) -> None:
        self.ref_controller: pin.SE3 | None = None
        self.ref_end_effector: pin.SE3 | None = None

    def reset(self) -> None:
        self.ref_controller = None
        self.ref_end_effector = None

    @property
    def active(self) -> bool:
        return self.ref_controller is not None and self.ref_end_effector is not None

    def update(
        self,
        controller_pose: pin.SE3,
        current_end_effector: pin.SE3,
        scale: float,
        orientation_enabled: bool,
    ) -> pin.SE3:
        if self.ref_controller is None or self.ref_end_effector is None:
            self.ref_controller = controller_pose.copy()
            self.ref_end_effector = current_end_effector.copy()
            return current_end_effector.copy()

        target = self.ref_end_effector.copy()
        target.translation = (
            self.ref_end_effector.translation
            + scale * (controller_pose.translation - self.ref_controller.translation)
        )
        if orientation_enabled:
            delta_rotation = controller_pose.rotation @ self.ref_controller.rotation.T
            target.rotation = delta_rotation @ self.ref_end_effector.rotation
        return target


def xr_pose_to_world_se3(xr_pose: np.ndarray) -> pin.SE3:
    """Convert [x, y, z, qx, qy, qz, qw] XR pose to robot-world SE3."""
    pose = np.asarray(xr_pose, dtype=float).reshape(-1)
    if pose.shape[0] != 7 or not np.all(np.isfinite(pose)):
        raise ValueError(f"Invalid XR pose: {xr_pose}")

    qx, qy, qz, qw = pose[3], pose[4], pose[5], pose[6]
    quat = pin.Quaternion(qw, qx, qy, qz)
    if quat.norm() < 1e-8:
        raise ValueError(f"Invalid XR quaternion: {xr_pose[3:7]}")
    quat.normalize()

    rotation = R_HEADSET_TO_WORLD @ quat.matrix() @ R_HEADSET_TO_WORLD.T
    translation = R_HEADSET_TO_WORLD @ pose[:3]
    return pin.SE3(rotation, translation)


def set_transform_handle_from_se3(handle, transform: pin.SE3) -> None:
    position, wxyz = se3_to_viser_pose(transform)
    handle.position = position
    handle.wxyz = wxyz


def format_teleop_status(
    enabled: bool,
    xr_ok: bool,
    left_active: bool,
    right_active: bool,
    scale: float,
    status: str,
    min_barrier: float,
) -> str:
    return "\n".join(
        [
            "### PICO Teleop",
            "",
            f"- Enabled: `{enabled}`",
            f"- XR data: `{'ok' if xr_ok else 'waiting/error'}`",
            f"- Left grip active: `{left_active}`",
            f"- Right grip active: `{right_active}`",
            f"- Scale: `{scale:.2f}`",
            f"- Collision: `{status}` (`{min_barrier:.4f}`)",
            "",
            "Hold left/right grip to drive each arm. Press `Y` or Reset Baseline to re-align.",
        ]
    )


def create_joint_plots(server: viser.ViserServer, joint_count: int):
    max_plot_samples = int(JOINT_PLOT_WINDOW_S * SOLVE_FREQUENCY_HZ)
    joint_time_history = np.full(max_plot_samples, np.nan)
    joint_position_history = np.full((joint_count, max_plot_samples), np.nan)
    joint_time_axis = np.linspace(-JOINT_PLOT_WINDOW_S, 0.0, max_plot_samples)
    joint_plot_axes = (
        {"label": "Time (s)"},
        {"label": "Joint angle (deg)"},
    )
    joint_plot_scales = {"x": {"time": False}, "y": {"range": (-180.0, 180.0)}}

    left_primary_plot = server.gui.add_uplot(
        (joint_time_axis, *joint_position_history[0:4]),
        (
            {"label": "time"},
            *(
                {"label": f"L{i}", "stroke": JOINT_COLORS[i - 1], "width": 2.0}
                for i in range(1, 5)
            ),
        ),
        title="Left Arm J1-J4",
        axes=joint_plot_axes,
        scales=joint_plot_scales,
        aspect=JOINT_PLOT_ASPECT,
        height=JOINT_PRIMARY_PLOT_HEIGHT,
    )
    left_wrist_plot = server.gui.add_uplot(
        (joint_time_axis, *joint_position_history[4:7]),
        (
            {"label": "time"},
            *(
                {"label": f"L{i}", "stroke": JOINT_COLORS[i - 1], "width": 2.0}
                for i in range(5, 8)
            ),
        ),
        title="Left Arm J5-J7",
        axes=joint_plot_axes,
        scales=joint_plot_scales,
        aspect=JOINT_PLOT_ASPECT,
        height=JOINT_WRIST_PLOT_HEIGHT,
    )
    right_primary_plot = server.gui.add_uplot(
        (joint_time_axis, *joint_position_history[7:11]),
        (
            {"label": "time"},
            *(
                {"label": f"R{i}", "stroke": JOINT_COLORS[i - 1], "width": 2.0}
                for i in range(1, 5)
            ),
        ),
        title="Right Arm J1-J4",
        axes=joint_plot_axes,
        scales=joint_plot_scales,
        aspect=JOINT_PLOT_ASPECT,
        height=JOINT_PRIMARY_PLOT_HEIGHT,
    )
    right_wrist_plot = server.gui.add_uplot(
        (joint_time_axis, *joint_position_history[11:14]),
        (
            {"label": "time"},
            *(
                {"label": f"R{i}", "stroke": JOINT_COLORS[i - 1], "width": 2.0}
                for i in range(5, 8)
            ),
        ),
        title="Right Arm J5-J7",
        axes=joint_plot_axes,
        scales=joint_plot_scales,
        aspect=JOINT_PLOT_ASPECT,
        height=JOINT_WRIST_PLOT_HEIGHT,
    )
    plots = (left_primary_plot, left_wrist_plot, right_primary_plot, right_wrist_plot)
    return joint_time_history, joint_position_history, plots


def update_joint_plots(
    plots,
    joint_time_history: np.ndarray,
    joint_position_history: np.ndarray,
    t: float,
) -> None:
    valid_times = joint_time_history[np.isfinite(joint_time_history)]
    if valid_times.size == 0:
        return
    plot_time = joint_time_history - valid_times[-1]
    left_primary_plot, left_wrist_plot, right_primary_plot, right_wrist_plot = plots
    left_primary_plot.data = (plot_time, *joint_position_history[0:4])
    left_wrist_plot.data = (plot_time, *joint_position_history[4:7])
    right_primary_plot.data = (plot_time, *joint_position_history[7:11])
    right_wrist_plot.data = (plot_time, *joint_position_history[11:14])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=8082, type=int)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--scale", default=1.0, type=float)
    parser.add_argument("--activation-threshold", default=0.9, type=float)
    parser.add_argument("--position-only", action="store_true")
    parser.add_argument("--mock-xr", action="store_true")
    parser.add_argument("--d-min", default=0.03, type=float)
    parser.add_argument("--initial-ignore-distance", default=None, type=float)
    parser.add_argument(
        "--collision-geometry",
        choices=("primitive", "stl"),
        default="primitive",
        help="Geometry used by the self-collision barrier.",
    )
    parser.add_argument(
        "--unlock-non-arm",
        action="store_true",
        help="Allow waist, leg and head joints to move during IK.",
    )
    args = parser.parse_args()
    if args.initial_ignore_distance is None:
        args.initial_ignore_distance = args.d_min

    urdf_path = (
        Path(__file__).parent.parent
        / "robots"
        / PACKAGE_NAME
        / "urdf"
        / "real_robot.urdf"
    ).resolve()
    package_dirs = [str(urdf_path.parents[2])]

    urdf = yourdfpy.URDF.load(
        str(urdf_path),
        build_collision_scene_graph=True,
        load_meshes=True,
        filename_handler=resolve_package_uri(urdf_path),
    )
    robot = pin.RobotWrapper.BuildFromURDF(
        filename=str(urdf_path),
        package_dirs=package_dirs,
        root_joint=None,
    )
    if args.collision_geometry == "primitive":
        robot.collision_model = build_primitive_collision_model(robot.model)

    q_ref = initial_configuration(robot.model)
    locked_q_indices, locked_v_indices = locked_joint_indices(robot.model)
    left_frame_id = require_frame(robot.model, LEFT_TCP)
    right_frame_id = require_frame(robot.model, RIGHT_TCP)
    robot.collision_data = configure_self_collision(
        robot, q_ref, args.initial_ignore_distance
    )
    arm_q_indices, _ = arm_joint_indices(robot.model)
    joint_names = arm_joint_names()

    print(f"URDF: {urdf_path}")
    print(f"nq={robot.model.nq}, collision pairs={len(robot.collision_model.collisionPairs)}")
    if args.unlock_non_arm:
        constraints = []
        print("Controlled joints: full model")
    else:
        constraints = [LockedJointsTask(locked_q_indices, locked_v_indices, q_ref)]
        print(
            "Controlled joints: left/right arms only "
            f"({robot.model.nv - len(locked_v_indices)} moving, "
            f"{len(locked_v_indices)} locked)"
        )

    configuration = pink.Configuration(
        robot.model,
        robot.data,
        q_ref,
        collision_model=robot.collision_model,
        collision_data=robot.collision_data,
    )

    left_task = FrameTask(LEFT_TCP, position_cost=5.0, orientation_cost=1.0, gain=0.5)
    right_task = FrameTask(RIGHT_TCP, position_cost=5.0, orientation_cost=1.0, gain=0.5)
    posture_task = PostureTask(cost=1e-4)
    tasks = [left_task, right_task, posture_task]
    for task in tasks:
        task.set_target_from_configuration(configuration)

    collision_barrier = SelfCollisionBarrier(
        n_collision_pairs=len(robot.collision_model.collisionPairs),
        gain=10.0,
        safe_displacement_gain=5.0,
        d_min=args.d_min,
    )
    barriers = [collision_barrier]

    xr_client = MockXrClient() if args.mock_xr else XrClient()
    solver = select_solver()
    rate = RateLimiter(frequency=SOLVE_FREQUENCY_HZ, warn=False)
    dt = rate.period

    server = viser.ViserServer(host=args.host, port=args.port)
    server.gui.configure_theme(control_layout="fixed", control_width="large")
    if not args.no_browser:
        time.sleep(0.5)
        webbrowser.open(f"http://localhost:{args.port}")

    server.scene.add_grid("/ground", width=2, height=2)
    enable_teleop_gui = server.gui.add_checkbox("Enable teleop", True)
    scale_gui = server.gui.add_slider(
        "Position scale",
        min=0.1,
        max=2.0,
        step=0.05,
        initial_value=args.scale,
    )
    server.gui.add_number("Solve frequency (Hz)", SOLVE_FREQUENCY_HZ, disabled=True)
    ik_time_gui = server.gui.add_number("IK time (ms)", 0.0, disabled=True)
    status_gui = server.gui.add_markdown(
        format_teleop_status(True, False, False, False, args.scale, "unknown", 0.0)
    )
    reset_button = server.gui.add_button("Reset Baseline")
    reset_state = {"requested": False}

    @reset_button.on_click
    def _(_) -> None:
        reset_state["requested"] = True

    joint_angles_deg = np.rad2deg(configuration.q[arm_q_indices])
    joint_values_gui = server.gui.add_markdown(format_joint_table(joint_angles_deg))
    joint_time_history, joint_position_history, joint_plots = create_joint_plots(
        server, len(joint_names)
    )

    urdf_vis = ViserUrdf(server, urdf, root_node_name="/real_robot")
    urdf_vis.update_cfg(pinocchio_to_yourdfpy_cfg(robot.model, configuration.q))

    left_pos, left_wxyz = se3_to_viser_pose(configuration.data.oMf[left_frame_id])
    right_pos, right_wxyz = se3_to_viser_pose(configuration.data.oMf[right_frame_id])
    left_target_handle = server.scene.add_transform_controls(
        "/pico_target_l", scale=0.16, fixed=True, position=left_pos, wxyz=left_wxyz
    )
    right_target_handle = server.scene.add_transform_controls(
        "/pico_target_r", scale=0.16, fixed=True, position=right_pos, wxyz=right_wxyz
    )
    server.scene.add_icosphere("/collision_status", radius=0.04, color=(0, 255, 0))

    print(f"Open http://localhost:{args.port}")
    print("PICO control: hold left/right grip to move each arm; press Y to reset baseline.")
    print(f"Left TCP:  position={left_pos}, wxyz={left_wxyz}")
    print(f"Right TCP: position={right_pos}, wxyz={right_wxyz}")

    left_mapper = RelativeTeleopTarget()
    right_mapper = RelativeTeleopTarget()
    last_y_button = False
    last_status = None
    last_print_time = 0.0
    last_plot_update_time = 0.0
    t = 0.0

    try:
        while True:
            current_left = configuration.get_transform_frame_to_world(LEFT_TCP).copy()
            current_right = configuration.get_transform_frame_to_world(RIGHT_TCP).copy()
            teleop_enabled = bool(enable_teleop_gui.value)
            scale = float(scale_gui.value)
            xr_ok = False
            left_active = False
            right_active = False

            try:
                y_button = bool(xr_client.get_button_state_by_name("Y"))
                reset_from_pico = y_button and not last_y_button
                last_y_button = y_button
                if reset_state["requested"] or reset_from_pico:
                    left_mapper.reset()
                    right_mapper.reset()
                    reset_state["requested"] = False
                    print("Teleop baseline reset.")

                left_grip = float(xr_client.get_key_value_by_name("left_grip"))
                right_grip = float(xr_client.get_key_value_by_name("right_grip"))
                left_active = teleop_enabled and left_grip >= args.activation_threshold
                right_active = teleop_enabled and right_grip >= args.activation_threshold

                left_controller_pose = xr_pose_to_world_se3(
                    xr_client.get_pose_by_name("left_controller")
                )
                right_controller_pose = xr_pose_to_world_se3(
                    xr_client.get_pose_by_name("right_controller")
                )
                xr_ok = True
            except Exception as exc:
                left_active = False
                right_active = False
                left_controller_pose = None
                right_controller_pose = None
                if time.monotonic() - last_print_time > 1.0:
                    print(f"Waiting for XR data: {exc}")
                    last_print_time = time.monotonic()

            if left_active and left_controller_pose is not None:
                left_target_pose = left_mapper.update(
                    left_controller_pose,
                    current_left,
                    scale,
                    orientation_enabled=not args.position_only,
                )
            else:
                left_mapper.reset()
                left_target_pose = current_left

            if right_active and right_controller_pose is not None:
                right_target_pose = right_mapper.update(
                    right_controller_pose,
                    current_right,
                    scale,
                    orientation_enabled=not args.position_only,
                )
            else:
                right_mapper.reset()
                right_target_pose = current_right

            left_task.transform_target_to_world = left_target_pose
            right_task.transform_target_to_world = right_target_pose
            set_transform_handle_from_se3(left_target_handle, left_target_pose)
            set_transform_handle_from_se3(right_target_handle, right_target_pose)

            h = collision_barrier.compute_barrier(configuration)
            min_barrier = float(np.min(h))
            if min_barrier <= 0.0:
                status = "collision"
                color = (255, 0, 0)
            elif min_barrier < 0.01:
                status = "warning"
                color = (255, 255, 0)
            else:
                status = "safe"
                color = (0, 255, 0)

            now = time.monotonic()
            if status != last_status:
                server.scene.add_icosphere("/collision_status", radius=0.04, color=color)
            if status != last_status or now - last_print_time > 1.0:
                print(f"collision status={status}, min barrier={min_barrier:.4f}")
                last_status = status
                last_print_time = now

            ik_start = time.perf_counter()
            try:
                velocity = solve_ik(
                    configuration,
                    tasks,
                    dt,
                    solver=solver,
                    barriers=barriers,
                    constraints=constraints,
                    safety_break=False,
                )
            except NoSolutionFound as exc:
                print(f"IK solver failed: {exc}")
                velocity = np.zeros(robot.model.nv)
            ik_time_gui.value = round((time.perf_counter() - ik_start) * 1000.0, 3)

            configuration.integrate_inplace(velocity, dt)
            if not args.unlock_non_arm:
                q_locked = configuration.q.copy()
                q_locked[locked_q_indices] = q_ref[locked_q_indices]
                configuration.update(q_locked)
            urdf_vis.update_cfg(pinocchio_to_yourdfpy_cfg(robot.model, configuration.q))

            joint_angles_deg = np.rad2deg(configuration.q[arm_q_indices])
            joint_time_history[:-1] = joint_time_history[1:]
            joint_time_history[-1] = t
            joint_position_history[:, :-1] = joint_position_history[:, 1:]
            joint_position_history[:, -1] = joint_angles_deg
            if t - last_plot_update_time >= 1.0 / JOINT_PLOT_UPDATE_HZ:
                joint_values_gui.content = format_joint_table(joint_angles_deg)
                update_joint_plots(
                    joint_plots,
                    joint_time_history,
                    joint_position_history,
                    t,
                )
                status_gui.content = format_teleop_status(
                    teleop_enabled,
                    xr_ok,
                    left_active,
                    right_active,
                    scale,
                    status,
                    min_barrier,
                )
                last_plot_update_time = t

            rate.sleep()
            t += dt
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        xr_client.close()


if __name__ == "__main__":
    main()
