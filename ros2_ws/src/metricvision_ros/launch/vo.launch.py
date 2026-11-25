"""Stereo VO on the wheelchair: live cameras or a bag, with rviz2 and optional recording.

Live (cameras + /odom already published by the wheelchair):
    ros2 launch metricvision_ros vo.launch.py
Replay a bag instead:
    ros2 launch metricvision_ros vo.launch.py bag:=/path/to/ros_bags/run_01
Every run is saved to results/live/<date_time>/ (change with output_dir:=..., "" = off). On Ctrl+C it holds
vo.tum (this VO), odom.tum (/odom, the reference), odom_at_vo.tum, metrics.md/json and trajectories.png;
with record:=true also bag/ (cameras, /odom, /tf and the VO output).
Options:
    approaches:=classical  (only the classical VO; default classical,learned)    plot:=false  (no live graph window)
    Save while running: the graph window's buttons, or `ros2 service call /vo/save std_srvs/srv/Trigger`.
    pipeline:=sparse|dense  window:=5  rviz:=false  record:=true  output_dir:=/some/folder
    params_file:=/path/to/my_params.yaml  left_topic:=...  right_topic:=...  reliable_qos:=false
"""

from __future__ import annotations

import os
from datetime import datetime

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.conditions import IfCondition, LaunchConfigurationEquals, LaunchConfigurationNotEquals
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from metricvision_ros.launcher import repo_root

CAMERA_TOPICS = ["/cam1/cam1/color/image_raw", "/cam2/cam2/color/image_raw"]
CAMERA_INFO_TOPICS = ["/cam1/cam1/color/camera_info", "/cam2/cam2/color/camera_info"]  # Needed to recalibrate from the bag.


def generate_launch_description() -> LaunchDescription:
    """Node + rviz2 + optional bag play / record."""
    share = get_package_share_directory("metricvision_ros")
    arg = LaunchConfiguration
    root = repo_root()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_output = os.path.join(str(root) if root else os.getcwd(), "results", "live", stamp)
    args = [
        DeclareLaunchArgument("params_file", default_value=os.path.join(share, "config", "vo_params.yaml")),
        DeclareLaunchArgument(
            "calibration",
            default_value="configs/wheelchair/rig_640x480_live.yaml",
            description="Rig calibration (repo-relative); the stereo rotation is re-estimated on start if it drifted.",
        ),
        DeclareLaunchArgument(
            "approaches",
            default_value="classical,learned",
            description="VO approaches on the same frames: classical (FAST/ORB+KLT), learned (SuperPoint+LightGlue, GPU). First drives /vo/*.",
        ),
        DeclareLaunchArgument("pipeline", default_value="sparse", choices=["sparse", "dense"]),
        DeclareLaunchArgument("window", default_value="-1", description="Sparse BA window (-1 = config, 0 = off)."),
        DeclareLaunchArgument("output_dir", default_value=default_output, description="Run folder (VO, /odom, comparison, bag)."),
        DeclareLaunchArgument("left_topic", default_value=CAMERA_TOPICS[0]),
        DeclareLaunchArgument("right_topic", default_value=CAMERA_TOPICS[1]),
        DeclareLaunchArgument("reliable_qos", default_value="true"),
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("plot", default_value="true", description="Live comparison graph window with Save buttons."),
        DeclareLaunchArgument("bag", default_value="", description="Play this bag instead of live cameras."),
        DeclareLaunchArgument("rate", default_value="1.0", description="Bag playback rate."),
        DeclareLaunchArgument("record", default_value="false", description="Record cameras + odom + VO output."),
    ]

    vo = Node(
        package="metricvision_ros",
        executable="vo_node",
        name="metricvision_vo",
        output="screen",
        emulate_tty=True,
        parameters=[
            arg("params_file"),
            {
                "pipeline": arg("pipeline"),
                "calibration": arg("calibration"),
                "approaches": arg("approaches"),
                "window": ParameterValue(arg("window"), value_type=int),
                "output_dir": arg("output_dir"),
                "left_topic": arg("left_topic"),
                "right_topic": arg("right_topic"),
                "reliable_qos": ParameterValue(arg("reliable_qos"), value_type=bool),
            },
        ],
    )

    plot = Node(
        package="metricvision_ros",
        executable="vo_plot",
        name="metricvision_vo_plot",
        parameters=[{"output_dir": arg("output_dir")}],
        condition=IfCondition(arg("plot")),
        output="log",
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", os.path.join(share, "rviz", "vo.rviz")],
        condition=IfCondition(arg("rviz")),
        output="log",
    )

    play = TimerAction(  # Give the node a few seconds to build the pipeline before the first images.
        period=5.0,
        actions=[
            ExecuteProcess(
                cmd=["ros2", "bag", "play", arg("bag"), "--rate", arg("rate"), "--topics", *CAMERA_TOPICS, *CAMERA_INFO_TOPICS, "/odom", "/Odometry"],
                output="screen",
            )
        ],
        condition=LaunchConfigurationNotEquals("bag", ""),
    )

    record = ExecuteProcess(
        cmd=[
            "ros2",
            "bag",
            "record",
            "-o",
            PathJoinSubstitution([arg("output_dir"), "bag"]),
            *CAMERA_TOPICS,
            *CAMERA_INFO_TOPICS,
            "/odom",
            "/Odometry",  # DLIO lidar-inertial odometry: the better reference.
            "/tf",
            "/tf_static",
            "/vo/odom",
            "/vo/path",
        ],
        output="screen",
        condition=IfCondition(arg("record")),
    )

    live_note = ExecuteProcess(
        cmd=["echo", "[metricvision] live mode: waiting for", arg("left_topic"), "+", arg("right_topic"), "| saving to", arg("output_dir")],
        output="screen",
        condition=LaunchConfigurationEquals("bag", ""),
    )

    return LaunchDescription([*args, vo, rviz, plot, play, record, live_note])
