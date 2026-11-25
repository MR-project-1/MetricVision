"""ROS 2 (Humble) node: live stereo VO on the wheelchair rig.

Subscribes to the two raw colour image topics, pairs them by header stamp
(``message_filters.ApproximateTimeSynchronizer``, slop = the offline ``max_pair_dt_s``), rectifies them with
the same :class:`~metricvision.datasets.rosbag_stereo.StereoRectifier` as the offline loader, and runs the
VO pipeline built from the same YAML config. Publishes, in the odometry body frame (``base_footprint``):

* ``/vo/odom`` (nav_msgs/Odometry): T_vo_odom_base, header frame ``vo_odom``,
* ``/vo/path`` (nav_msgs/Path): the trajectory so far,
* optionally the TF ``vo_odom -> vo_base_footprint`` (``publish_tf``; a separate child frame, so the wheel
  odometry's ``odom -> base_footprint`` is not fought over),
* ``/vo/debug_image`` (sensor_msgs/Image, bgr8): keypoints, stereo matches, PnP inliers/outliers, top-down
  VO vs wheel odometry and per-frame numbers (:class:`~metricvision.viz.VOVisualizer`); only rendered while
  someone subscribes (``rqt_image_view /vo/debug_image`` or the rviz2 config),
* ``/vo/landmarks`` (sensor_msgs/PointCloud2, ``vo_odom``): the PnP-inlier 3D points, accumulated on a
  voxel grid (``landmark_voxel_m``), i.e. the sparse map the poses were solved against,
* ``/vo/reference_path`` (nav_msgs/Path, ``vo_odom``) and ``/vo/reference_odom`` (nav_msgs/Odometry): the wheel
  odometry (``reference_topic``), anchored at the VO start, so both paths and poses overlay in rviz2.

Processing runs in a worker thread that always takes the newest pair, so the node drops frames instead of
building latency when the CPU falls behind (``dropped`` is logged). On shutdown, if ``output_dir`` is set, the
run is saved there: ``vo.tum`` (this VO), ``odom.tum`` (every /odom message, the reference), and the comparison
written by :mod:`metricvision.evaluation.live_report` (``odom_at_vo.tum``, ``metrics.json/md``, ``trajectories.png``).

Run it through the ROS 2 package (ros2_ws/src/metricvision_ros; launch file, params, rviz2 config):
    source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
    ros2 launch metricvision_ros vo.launch.py                       # live
    ros2 launch metricvision_ros vo.launch.py bag:=/abs/path/run_01  # bag replay
or directly from the repository root:
    uv run python -m metricvision.ros.vo_node --ros-args -p config:=configs/wheelchair.yaml -p pipeline:=sparse
"""

from __future__ import annotations

import array
import signal
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import PoseStamped, TransformStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path as PathMsg
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from std_srvs.srv import Trigger
from tf2_ros import TransformBroadcaster

from metricvision.config import ExperimentConfig
from metricvision.core.geometry import invert_pose
from metricvision.core.types import StereoFrame, Trajectory
from metricvision.datasets.rosbag_stereo import (
    RigCalibration,
    StereoRectifier,
    apply_resolution_adapter,
    image_message_to_gray,
    odometry_pose,
    resolution_adapter,
    stamp_seconds,
)
from metricvision.evaluation.live_report import write_live_report
from metricvision.evaluation.writers import write_tum
from metricvision.pipeline import build_odometry
from metricvision.viz import VOVisualizer


def _git_commit() -> str:
    """Commit of the checkout the node runs from (with "+dirty" for local changes), or "unknown"."""
    import subprocess

    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=2, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, timeout=2).stdout.strip()
        return commit + ("+dirty" if dirty else "")
    except Exception:  # noqa: BLE001 - git missing or not a repository (e.g. "dubious ownership" in a container).
        return "unknown"


def append_path_pose(path: PathMsg, pose: PoseStamped, min_step_m: float, min_turn_deg: float, max_poses: int) -> bool:
    """Add a pose to a growing nav_msgs/Path only when it moved, and keep the path bounded.

    rviz2 rebuilds the whole line for every Path message, so a path with one pose per camera frame, re-sent at the
    frame rate, grows until rviz2 can no longer keep up (and dies after some minutes of driving). Poses closer than
    ``min_step_m`` and ``min_turn_deg`` to the last kept one are skipped (a standing chair adds nothing), and once
    ``max_poses`` is exceeded the older half is thinned to every second pose, so the full run stays visible.

    Returns:
        True if the path changed.
    """
    poses = path.poses
    if poses:
        last = poses[-1].pose
        step = np.hypot(pose.pose.position.x - last.position.x, pose.pose.position.y - last.position.y)
        dot = abs(
            last.orientation.x * pose.pose.orientation.x
            + last.orientation.y * pose.pose.orientation.y
            + last.orientation.z * pose.pose.orientation.z
            + last.orientation.w * pose.pose.orientation.w
        )
        turn = np.degrees(2.0 * np.arccos(min(1.0, dot)))
        if step < min_step_m and turn < min_turn_deg:
            return False
    poses.append(pose)
    if len(poses) > max_poses:
        half = len(poses) // 2
        path.poses = poses[:half:2] + poses[half:]
    return True


cv2.setNumThreads(2)  # The wheelchair stack shares the CPU: one thread per core (71 in total) only adds contention.

APPROACH_FRONTENDS = {"classical": "fast_orb_klt", "learned": "superpoint_lightglue"}  # Approach -> sparse stereo front end.


class _ParallelApproach:
    """Another VO approach on the same rectified frames, in its own thread (newest frame wins, so it never lags).

    Its pipeline is built in the background (the learned front end loads SuperPoint/LightGlue onto the GPU, and
    downloads the weights on the very first run), publishes ``/vo_<name>/odom`` and ``/vo_<name>/path`` and keeps
    its trajectory and per-frame stats for the run folder.
    """

    def __init__(self, node: StereoVONode, name: str, config: ExperimentConfig) -> None:
        self.node, self.name, self.config = node, name, config
        self.trajectory = Trajectory()
        self.frames: list[tuple[float, int, int, float]] = []  # (stamp, tracking_ok, inliers, ms).
        self.processed = self.dropped = 0
        self.status = "loading"
        self.base_offset = np.eye(4)
        self.odometry: Any = None
        self.path = PathMsg()
        self.path.header.frame_id = node._odom_frame
        self.path_last = 0.0
        self.odom_pub = node.create_publisher(Odometry, f"/vo_{name}/odom", 10)
        self.path_pub = node.create_publisher(PathMsg, f"/vo_{name}/path", 10)
        self.cv = threading.Condition()
        self.job: tuple[float, Any, np.ndarray, np.ndarray] | None = None
        self.running = True
        threading.Thread(target=self.rebuild, args=(node._rectifier.camera,), daemon=True).start()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def rebuild(self, camera: Any) -> None:
        """(Re)create the pipeline for ``camera``; the path continues from its last pose."""
        try:
            odometry = build_odometry(self.config, camera)
            odometry.reset()
        except Exception as exc:  # noqa: BLE001 - e.g. no GPU / weights: keep the main VO running.
            self.status = f"failed: {exc}"
            self.node.get_logger().error(f"{self.name} approach disabled: {exc}")
            return
        with self.cv:
            if self.trajectory.poses:
                self.base_offset = self.trajectory.poses[-1]
            self.odometry = odometry
            self.status = "running"
        self.node.get_logger().info(f"{self.name} approach ready ({self.config.stereo_frontend.name}) -> /vo_{self.name}/odom, /vo_{self.name}/path")

    def submit(self, stamp: float, ros_stamp: Any, left: np.ndarray, right: np.ndarray) -> None:
        """Hand over the newest rectified pair (an unprocessed older one is dropped)."""
        with self.cv:
            if self.odometry is None:
                return
            if self.job is not None:
                self.dropped += 1
            self.job = (stamp, ros_stamp, left, right)
            self.cv.notify()

    def snapshot(self) -> Trajectory:
        with self.cv:
            return Trajectory(list(self.trajectory.timestamps), list(self.trajectory.poses))

    def stop(self) -> None:
        self.running = False
        with self.cv:
            self.cv.notify_all()
        self.thread.join(timeout=5.0)

    def _loop(self) -> None:
        node = self.node
        while self.running:
            with self.cv:
                while self.job is None and self.running:
                    self.cv.wait(0.2)
                if not self.running or self.job is None:
                    return
                stamp, ros_stamp, left, right = self.job
                self.job = None
                odometry, offset = self.odometry, self.base_offset
            try:
                result = odometry.process(StereoFrame(index=self.processed, timestamp=stamp, left=left, right=right))
            except Exception as exc:  # noqa: BLE001 - never take the main VO down.
                self.status = f"failed: {exc}"
                node.get_logger().error(f"{self.name} approach stopped: {exc}")
                return
            pose = offset @ node._body_from_camera @ result.world_from_camera @ node._camera_from_body
            st = result.stats
            with self.cv:
                self.trajectory.append(stamp, pose)
                self.frames.append((stamp, int(bool(st.tracking_ok)), int(st.num_pnp_inliers), float(st.total_ms)))
            self.processed += 1
            msg = Odometry()
            msg.header.stamp, msg.header.frame_id, msg.child_frame_id = ros_stamp, node._odom_frame, f"{self.name}_{node._base_frame}"
            msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z = (float(v) for v in pose[:3, 3])
            o = msg.pose.pose.orientation
            o.x, o.y, o.z, o.w = (float(v) for v in Rotation.from_matrix(pose[:3, :3]).as_quat())
            ps = PoseStamped()
            ps.header, ps.pose = msg.header, msg.pose.pose
            append_path_pose(self.path, ps, node._path_step, node._path_turn, node._path_max)
            try:
                self.odom_pub.publish(msg)
                now = time.monotonic()
                if now - self.path_last >= node._path_period:
                    self.path_last = now
                    self.path.header.stamp = ros_stamp
                    self.path_pub.publish(self.path)
            except Exception:
                if node.context.ok():
                    raise
                return


class StereoVONode(Node):
    """Live stereo visual odometry."""

    def __init__(self) -> None:
        """Declare parameters, build the pipeline, wire up subscriptions and publishers."""
        super().__init__("metricvision_vo")
        config_path = self.declare_parameter("config", "configs/wheelchair.yaml").value
        pipeline = self.declare_parameter("pipeline", "sparse").value  # sparse | dense.
        window = int(self.declare_parameter("window", -1).value)  # BA window override (-1 = config).
        self._odom_frame = self.declare_parameter("odom_frame", "vo_odom").value
        self._base_frame = self.declare_parameter("base_frame", "base_footprint").value
        self._publish_tf = bool(self.declare_parameter("publish_tf", False).value)
        self._output_dir = str(self.declare_parameter("output_dir", "").value)  # Run folder ("" = save nothing).
        self._save_video = bool(self.declare_parameter("save_video", True).value) and bool(self._output_dir)
        self._video_period = 1.0 / max(float(self.declare_parameter("video_rate_hz", 5.0).value), 0.5)
        self._video: cv2.VideoWriter | None = None
        self._video_last = 0.0
        self._checkpoint_s = float(self.declare_parameter("checkpoint_s", 10.0).value)
        self._checkpoint_last = time.monotonic()
        self._image_sizes: dict[str, list[int]] = {}
        self._selfcal_log: list[dict[str, Any]] = []
        # Reliable by default: best-effort DDS drops most fragmented ~1 MB images (only ~7% arrived in a bag replay test).
        reliable = bool(self.declare_parameter("reliable_qos", True).value)
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE) if reliable else qos_profile_sensor_data

        # Approaches run on the same frames: the first drives /vo/* (rviz, debug view, video), the others run in parallel
        # threads and publish /vo_<name>/*. classical = FAST/ORB + KLT, learned = SuperPoint + LightGlue (GPU); both
        # use the same rigid-rig LM (+ window BA), so the run compares the front ends under one back end.
        approaches = [a.strip() for a in str(self.declare_parameter("approaches", "classical,learned").value).split(",") if a.strip()]
        unknown = [a for a in approaches if a not in APPROACH_FRONTENDS]
        if not approaches or unknown:
            raise ValueError(f"approaches must be a comma list of {sorted(APPROACH_FRONTENDS)}, got {approaches}")
        base = ExperimentConfig.from_yaml(Path(config_path)).with_overrides(pipeline=pipeline, window=window if window >= 0 else None)
        config = base.with_overrides(stereo_frontend=APPROACH_FRONTENDS[approaches[0]])
        self._approach = approaches[0]
        params: dict[str, Any] = dict(config.dataset.params)
        left_topic = self.declare_parameter("left_topic", params.get("left_topic", "/cam1/cam1/color/image_raw")).value
        right_topic = self.declare_parameter("right_topic", params.get("right_topic", "/cam2/cam2/color/image_raw")).value
        calibration = self.declare_parameter("calibration", params["calibration"]).value
        slop = float(self.declare_parameter("max_pair_dt_s", params.get("max_pair_dt_s", 0.035)).value)

        rig = RigCalibration.load(calibration)
        self._expected_size = (rig.width, rig.height)  # The calibration is only valid at this resolution.
        self._rig_intrinsics = ((rig.k_left, rig.d_left), (rig.k_right, rig.d_right))
        self._config, self._pipeline_name = config, pipeline
        self._rect_alpha, self._rect_clahe = float(params.get("alpha", 0.0)), bool(params.get("clahe", False))
        self._base_offset = np.eye(4)  # VO pose before the last (re)start, so a rebuilt pipeline continues the path.
        self._apply_calibration(rig)
        self._run_info: dict[str, Any] = {
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "git_commit": _git_commit(),
            "pipeline": pipeline,
            "ba_window": window,
            "config": str(config_path),
            "calibration": str(calibration),
            "topics": {"left": left_topic, "right": right_topic},
            "stereo_pair_slop_s": slop,
            "approaches": {"main": approaches[0], "parallel": approaches[1:]},
        }
        if self._output_dir:
            calib_dir = Path(self._output_dir).resolve() / "calibration"
            calib_dir.mkdir(parents=True, exist_ok=True)
            (calib_dir / "rig_at_start.yaml").write_text(
                f"# Copy of {calibration} as loaded at the start of the run.\n"
                + yaml.safe_dump(rig.to_dict(), sort_keys=False, default_flow_style=None)
            )
        # Stereo self-calibration (see metricvision.calibration.stereo_selfcal): the wheelchair camera mount moves
        # between sessions, and a stale extrinsic leaves the rectified rows tens of pixels apart (no tracking at all).
        # Checked at startup and whenever tracking keeps failing; re-estimated if the rows are off.
        self._selfcal_enabled = bool(self.declare_parameter("auto_stereo_calib", True).value)
        self._selfcal_frames = int(self.declare_parameter("selfcal_frames", 6).value)
        self._selfcal_threshold = float(self.declare_parameter("selfcal_dy_threshold_px", 1.5).value)
        self._selfcal_pending = self._selfcal_enabled
        self._selfcal_images: list[tuple[np.ndarray, np.ndarray]] = []
        self._selfcal_last_sample = 0.0
        self._selfcal_last_run = -1e9

        self._odom_pub = self.create_publisher(Odometry, "/vo/odom", 10)
        self._path_pub = self.create_publisher(PathMsg, "/vo/path", 10)
        self._tf = TransformBroadcaster(self) if self._publish_tf else None
        self._debug_pub = self.create_publisher(Image, "/vo/debug_image", 2)
        self._landmark_pub = self.create_publisher(PointCloud2, "/vo/landmarks", 2)
        self._reference_pub = self.create_publisher(PathMsg, "/vo/reference_path", 10)
        self._reference_odom_pub = self.create_publisher(Odometry, "/vo/reference_odom", 10)  # /odom pose, vo_odom frame.
        # Each reference on its own, anchored at the VO start like the VO paths, so rviz2 shows all of them side by side.
        self._source_paths: dict[str, tuple[Any, PathMsg]] = {}
        for source in ("lidar", "wheel"):
            path = PathMsg()
            path.header.frame_id = self._odom_frame
            self._source_paths[source] = (self.create_publisher(PathMsg, f"/vo/{source}_path", 10), path)
        self._source_anchor: dict[str, np.ndarray] = {}
        self._source_last: dict[str, float] = {}
        self._source_prev: dict[str, tuple[float, np.ndarray]] = {}
        self._source_diverged: set[str] = set()
        self._save_lock = threading.Lock()
        self.create_service(Trigger, "/vo/save", self._on_save)
        self._voxel = float(self.declare_parameter("landmark_voxel_m", 0.05).value)
        self._landmarks: dict[tuple[int, int, int], np.ndarray] = {}  # Voxel -> point (vo_odom frame), oldest first.
        # rviz2 load limits (it re-renders every message in full; unbounded growth made it die after some minutes).
        self._landmark_max = int(self.declare_parameter("landmark_max_points", 60000).value)
        self._landmark_period = 1.0 / max(float(self.declare_parameter("landmark_rate_hz", 1.0).value), 0.05)
        self._landmark_last = 0.0
        self._path_step = float(self.declare_parameter("path_min_step_m", 0.02).value)
        self._path_turn = float(self.declare_parameter("path_min_turn_deg", 2.0).value)
        self._path_max = int(self.declare_parameter("path_max_poses", 4000).value)
        self._path_period = 1.0 / max(float(self.declare_parameter("path_rate_hz", 2.0).value), 0.05)
        self._path_last = [0.0, 0.0]  # Last publish time of /vo/path and /vo/reference_path.
        self._viz = VOVisualizer(self._body_from_camera, f"live | {pipeline}")
        self._viz_lock = threading.Lock()  # record() in the VO worker vs draw() in the debug worker.
        self._debug_scale = float(self.declare_parameter("debug_scale", 0.75).value)
        self._debug_period = 1.0 / max(float(self.declare_parameter("debug_rate_hz", 10.0).value), 0.1)
        self._debug_job: tuple[Any, int, np.ndarray, np.ndarray, Any] | None = None  # Newest frame to render.
        self._debug_cv = threading.Condition()
        self._reference_path = PathMsg()
        self._reference_path.header.frame_id = self._odom_frame
        self._odom_buffer: deque[tuple[float, np.ndarray]] = deque(maxlen=400)  # (stamp, T_odom_base).
        self._odom_log = Trajectory()  # Every /odom pose of the run, saved as the reference (odom.tum).
        self._reference0_inv: np.ndarray | None = None
        # Lidar-inertial odometry (DLIO on the wheelchair): a far better reference than the wheels, which record
        # nothing when the chair is pushed by hand. Saved as lidar.tum; preferred for the rviz2 overlay.
        self._lidar_buffer: deque[tuple[float, np.ndarray]] = deque(maxlen=800)
        self._lidar_log = Trajectory()
        self._overlay_source: str | None = None  # "lidar" or "wheel", fixed once the overlay is anchored.
        reference_topic = str(self.declare_parameter("reference_topic", "/odom").value)
        if reference_topic:
            self.create_subscription(Odometry, reference_topic, self._on_reference, 50)
        lidar_topic = str(self.declare_parameter("lidar_topic", "/Odometry").value)
        if lidar_topic:
            self.create_subscription(Odometry, lidar_topic, self._on_lidar, 200)
        self._path = PathMsg()
        self._path.header.frame_id = self._odom_frame
        self._trajectory = Trajectory()
        # Per-frame health (saved as frames.csv, summarised in the 5 s log line): why tracking works or not.
        self._frames: list[tuple[float, int, int, int, int, int, float, float, float, float, float]] = []
        self._frames_reported = 0

        self._lock = threading.Condition()
        self._latest: tuple[Image, Image] | None = None
        self._received = self._processed = self._dropped = 0
        self._processed_rate = 0.0  # Smoothed VO frames per second.
        self._last_done = time.monotonic()
        self._running = True
        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._worker.start()
        self._parallel = [_ParallelApproach(self, name, base.with_overrides(stereo_frontend=APPROACH_FRONTENDS[name])) for name in approaches[1:]]
        self._debug_worker = threading.Thread(target=self._debug_loop, daemon=True)  # Rendering never blocks the VO.
        self._debug_worker.start()

        self._topics = (left_topic, right_topic)
        self._reliable = reliable
        self._slop = slop
        self._image_counts = [0, 0]  # Raw images per camera, to tell "nothing arrives" from "nothing pairs".
        self._image_stamps = [0.0, 0.0]  # Header stamp of the latest image per camera (s).
        subscribers = [Subscriber(self, Image, topic, qos_profile=qos) for topic in self._topics]
        for side, sub in enumerate(subscribers):
            sub.registerCallback(lambda msg, side=side: self._on_image(side, msg))
        # Another streaming mode of the same camera (e.g. 1280x720) is mapped onto the calibrated one through its
        # camera_info; see resolution_adapter. Keyed by side -> (width, height) -> 2x3 map or None (not mappable).
        self._camera_info: list[CameraInfo | None] = [None, None]
        self._adapters: list[dict[tuple[int, int], np.ndarray | None]] = [{}, {}]
        for side, topic in enumerate(self._topics):
            info_topic = topic.rsplit("/", 1)[0] + "/camera_info"
            self.create_subscription(CameraInfo, info_topic, lambda msg, side=side: self._on_camera_info(side, msg), qos_profile_sensor_data)
        self._sync = ApproximateTimeSynchronizer(subscribers, queue_size=10, slop=slop)
        self._sync.registerCallback(self._on_pair)
        self.create_timer(5.0, self._report)
        self.get_logger().info(
            f"{pipeline} VO on {left_topic} + {right_topic} (slop {slop * 1e3:.0f} ms), rectified fx={self._rectifier.camera.fx:.1f} "
            f"B={self._rectifier.camera.baseline:.3f} m -> /vo/odom ({self._odom_frame} -> {self._base_frame})"
        )

    def _apply_calibration(self, rig: RigCalibration) -> None:
        """(Re)build rectification and the VO pipeline for ``rig``; the path continues from the last pose."""
        self._rig = rig
        self._rectifier = StereoRectifier(rig, alpha=self._rect_alpha, clahe=self._rect_clahe)
        self._body_from_camera = self._rectifier.body_from_camera
        self._camera_from_body = invert_pose(self._body_from_camera)
        self._odometry = build_odometry(self._config, self._rectifier.camera)
        self._odometry.reset()
        for approach in getattr(self, "_parallel", []):
            approach.rebuild(self._rectifier.camera)
        if hasattr(self, "_viz"):
            with self._viz_lock:
                self._viz = VOVisualizer(self._body_from_camera, f"live | {self._pipeline_name}")

    def _selfcal_collect(self, left: np.ndarray, right: np.ndarray) -> bool:
        """Gather raw pairs for the stereo check; returns True while still collecting (VO paused meanwhile)."""
        now = time.monotonic()
        if now - self._selfcal_last_sample >= 0.25:
            self._selfcal_images.append((left.copy(), right.copy()))
            self._selfcal_last_sample = now
        if len(self._selfcal_images) < self._selfcal_frames:
            return True
        self._selfcal_pending = False
        self._selfcal_last_run = now
        images, self._selfcal_images = self._selfcal_images, []
        self._run_selfcal(images)
        return False

    def _run_selfcal(self, images: list[tuple[np.ndarray, np.ndarray]]) -> None:
        """Measure the rectified row offset on ``images`` and re-estimate the stereo extrinsic if it is off."""
        from metricvision.calibration.stereo_selfcal import estimate_extrinsic, stereo_matches

        log = self.get_logger()
        pairs = [stereo_matches(left, right, self._rig) for left, right in images]
        xl = np.concatenate([a for a, _ in pairs])
        xr = np.concatenate([b for _, b in pairs])
        result = estimate_extrinsic(self._rig, xl, xr)
        if result is None:
            log.warn(f"stereo check: only {len(xl)} left/right matches (textureless view?); keeping the calibration")
            return
        if result.dy_before_px < self._selfcal_threshold:
            log.info(f"stereo check: calibration OK (rows {result.dy_before_px:.2f} px apart, {result.inliers} matches)")
            self._selfcal_log.append({"time": time.strftime("%H:%M:%S"), "rows_apart_px": round(result.dy_before_px, 2), "action": "kept"})
            return
        if not result.reliable:
            log.warn(
                f"stereo check: rows are {result.dy_before_px:.1f} px apart (calibration off) but the re-estimate is unreliable "
                f"({result.inliers} inliers, {result.sampson_rms_px:.2f} px RMS, {result.dy_after_px:.2f} px after): point the "
                "cameras at a textured, static scene"
            )
            return
        if self._trajectory.poses:
            self._base_offset = self._trajectory.poses[-1]
        self._apply_calibration(result.calibration)
        log.warn(
            f"stereo check: rows were {result.dy_before_px:.1f} px apart (a camera moved {result.rotation_change_deg:.2f} deg); "
            f"re-estimated from {result.inliers} matches -> {result.dy_after_px:.2f} px. VO restarted with the new calibration."
        )
        self._selfcal_log.append(
            {
                "time": time.strftime("%H:%M:%S"),
                "rows_apart_px": round(result.dy_before_px, 2),
                "after_px": round(result.dy_after_px, 2),
                "camera_moved_deg": round(result.rotation_change_deg, 2),
                "matches": result.inliers,
            }
        )
        if self._output_dir:
            folder = Path(self._output_dir).resolve() / "calibration"
            folder.mkdir(parents=True, exist_ok=True)
            header = (
                f"# Stereo self-calibration by the live VO node at {time.strftime('%Y-%m-%d %H:%M:%S')}: rotation re-estimated from "
                f"{result.inliers} live matches (Sampson RMS {result.sampson_rms_px:.2f} px, rows {result.dy_before_px:.1f} -> "
                f"{result.dy_after_px:.2f} px). Baseline length and body_from_left unchanged.\n"
            )
            (folder / "rig_selfcal.yaml").write_text(header + yaml.safe_dump(result.calibration.to_dict(), sort_keys=False, default_flow_style=None))

    def _on_image(self, side: int, msg: Image) -> None:
        """Count raw images per camera (diagnostics only)."""
        self._image_counts[side] += 1
        self._image_sizes.setdefault("left" if side == 0 else "right", [int(msg.width), int(msg.height)])
        self._image_stamps[side] = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def _on_camera_info(self, side: int, msg: CameraInfo) -> None:
        """Keep the latest camera_info per camera (needed only when it streams at another resolution)."""
        self._camera_info[side] = msg

    def _to_calibrated(self, side: int, msg: Image) -> np.ndarray | None:
        """Grayscale image at the calibrated resolution, or None (logged) when it cannot be mapped there."""
        gray = image_message_to_gray(msg)
        size = (int(msg.width), int(msg.height))
        if size == self._expected_size:
            return gray
        topic, (w, h) = self._topics[side], self._expected_size
        if size not in self._adapters[side]:
            info = self._camera_info[side]
            if info is None or (int(info.width), int(info.height)) != size:
                self.get_logger().error(
                    f"{topic} is {size[0]}x{size[1]} but the rig calibration is {w}x{h}, and no matching camera_info "
                    f"arrived yet: set that camera to {w}x{h} (realsense2_camera: rgb_camera.color_profile:={w}x{h}x15)",
                    throttle_duration_sec=5.0,
                )
                return None
            k_rig, d_rig = self._rig_intrinsics[side]
            affine = resolution_adapter(np.asarray(info.k, dtype=np.float64), np.asarray(info.d, dtype=np.float64), k_rig, d_rig)
            self._adapters[side][size] = affine
            log = self.get_logger()
            if affine is None:
                log.error(
                    f"{topic} streams {size[0]}x{size[1]}, which is not a scaled view of the calibrated {w}x{h} mode: "
                    f"set that camera to {w}x{h} (realsense2_camera: rgb_camera.color_profile:={w}x{h}x15)"
                )
            else:
                covered_top = max(0.0, affine[1, 2])
                covered_bottom = min(float(h), affine[1, 1] * size[1] + affine[1, 2])
                log.warn(
                    f"{topic} streams {size[0]}x{size[1]}; mapping it onto the {w}x{h} calibration via camera_info "
                    f"(scale {affine[0, 0]:.4f}, shift {affine[0, 2]:+.1f},{affine[1, 2]:+.1f} px; real image rows "
                    f"{covered_top:.0f}-{covered_bottom:.0f}). Running the camera at {w}x{h} is still better."
                )
        affine = self._adapters[side][size]
        return None if affine is None else apply_resolution_adapter(gray, affine, self._expected_size)

    def _on_pair(self, left: Image, right: Image) -> None:
        """Keep only the newest synchronised pair (the worker drops older ones)."""
        with self._lock:
            if self._latest is not None:
                self._dropped += 1
            self._latest = (left, right)
            self._received += 1
            self._lock.notify()

    def _on_reference(self, msg: Odometry) -> None:
        """Buffer wheel odometry for the overlay."""
        with self._lock:
            stamp, pose = stamp_seconds(msg), odometry_pose(msg)
            self._odom_buffer.append((stamp, pose))
            self._odom_log.append(stamp, pose)

    def _on_lidar(self, msg: Odometry) -> None:
        """Buffer and log the lidar-inertial odometry."""
        with self._lock:
            stamp, pose = stamp_seconds(msg), odometry_pose(msg)
            self._lidar_buffer.append((stamp, pose))
            self._lidar_log.append(stamp, pose)

    def _reference_at(self, stamp: float) -> np.ndarray | None:
        """Reference pose nearest to ``stamp`` (within 0.2 s): lidar if it publishes, else wheels; anchored at the first one used."""
        with self._lock:
            if self._overlay_source is None:
                self._overlay_source = "lidar" if self._lidar_buffer else ("wheel" if self._odom_buffer else None)
            buffer = self._lidar_buffer if self._overlay_source == "lidar" else self._odom_buffer
            if not buffer:
                return None
            t, pose = min(buffer, key=lambda item: abs(item[0] - stamp))
        if abs(t - stamp) > 0.2:
            return None
        if self._reference0_inv is None:
            self._reference0_inv = invert_pose(pose)
        return pose

    def _loop(self) -> None:
        """Worker: rectify, track, publish."""
        index = 0
        while self._running:
            with self._lock:
                while self._latest is None and self._running:
                    self._lock.wait(0.2)
                if not self._running:
                    return
                left_msg, right_msg = self._latest
                self._latest = None
            # remap would silently output garbage at another resolution, so map other modes first (or skip).
            left_raw, right_raw = self._to_calibrated(0, left_msg), self._to_calibrated(1, right_msg)
            if left_raw is None or right_raw is None:
                continue
            if self._selfcal_pending and self._selfcal_collect(left_raw, right_raw):
                continue
            left, right = self._rectifier.rectify(left_raw, right_raw)
            for approach in self._parallel:
                approach.submit(stamp_seconds(left_msg), left_msg.header.stamp, left, right)
            stamp = stamp_seconds(left_msg)
            result = self._odometry.process(StereoFrame(index=index, timestamp=stamp, left=left, right=right))
            pose = self._base_offset @ self._body_from_camera @ result.world_from_camera @ self._camera_from_body  # T_vo_odom_base.
            self._trajectory.append(stamp, pose)
            st, dbg = result.stats, result.debug
            self._frames.append(
                (
                    stamp,
                    int(bool(st.tracking_ok)),
                    int(st.num_matches),
                    int(st.num_matches_with_depth),
                    int(st.num_pnp_inliers),
                    len(dbg.new_left) if dbg is not None else 0,
                    float(left_raw.mean()),
                    float(right_raw.mean()),
                    float(left_raw.std()),
                    float(right_raw.std()),
                    float(st.total_ms),
                )
            )
            try:
                self._publish(left_msg.header.stamp, pose, bool(result.stats.tracking_ok))
                self._publish_debug(left_msg.header.stamp, index, left, right, result, stamp)
            except Exception:
                if self.context.ok():
                    raise
                return  # Ctrl+C already shut ROS down: the frame is in the trajectory, stop quietly.
            index += 1
            self._processed += 1
            now = time.monotonic()
            self._processed_rate = 0.9 * self._processed_rate + 0.1 / max(now - self._last_done, 1e-6)
            self._last_done = now

    def _publish_debug(self, ros_stamp: Any, index: int, left: np.ndarray, right: np.ndarray, result: Any, stamp: float) -> None:
        """Debug image, landmarks and the wheel-odometry overlay path."""
        reference = self._reference_at(stamp)
        if reference is not None and self._reference0_inv is not None:
            anchored = self._reference0_inv @ reference
            ps = PoseStamped()
            ps.header.stamp, ps.header.frame_id = ros_stamp, self._odom_frame
            ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = (float(v) for v in anchored[:3, 3])
            o = ps.pose.orientation
            o.x, o.y, o.z, o.w = (float(v) for v in Rotation.from_matrix(anchored[:3, :3]).as_quat())
            self._reference_path.header.stamp = ros_stamp
            append_path_pose(self._reference_path, ps, self._path_step, self._path_turn, self._path_max)
            now = time.monotonic()
            if now - self._path_last[1] >= self._path_period:
                self._path_last[1] = now
                self._reference_pub.publish(self._reference_path)
            ref = Odometry()  # Same pose as an arrow, so rviz shows where the wheels think the chair is.
            ref.header = ps.header
            ref.child_frame_id = "odom_" + self._base_frame
            ref.pose.pose = ps.pose
            self._reference_odom_pub.publish(ref)

        for source, buffer in (("lidar", self._lidar_buffer), ("wheel", self._odom_buffer)):
            with self._lock:
                nearest = min(buffer, key=lambda item: abs(item[0] - stamp)) if buffer else None
            if nearest is None or abs(nearest[0] - stamp) > 0.2:
                continue
            if source in self._source_diverged:
                continue
            anchor = self._source_anchor.setdefault(source, invert_pose(nearest[1]))
            pose = anchor @ nearest[1]
            last = self._source_prev.get(source)
            self._source_prev[source] = (nearest[0], pose[:3, 3].copy())
            if last is not None and nearest[0] - last[0] > 0.02:
                speed = float(np.linalg.norm(pose[:2, 3] - last[1][:2])) / (nearest[0] - last[0])
                if speed > 3.0:  # A wheelchair cannot do that: the odometry diverged (DLIO did once). Stop drawing it.
                    self._source_diverged.add(source)
                    self.get_logger().warn(f"{source} odometry jumped at {speed:.1f} m/s: diverged, no longer drawn in rviz")
                    continue
            sp = PoseStamped()
            sp.header.stamp, sp.header.frame_id = ros_stamp, self._odom_frame
            sp.pose.position.x, sp.pose.position.y, sp.pose.position.z = (float(v) for v in pose[:3, 3])
            q = sp.pose.orientation
            q.x, q.y, q.z, q.w = (float(v) for v in Rotation.from_matrix(pose[:3, :3]).as_quat())
            publisher, path = self._source_paths[source]
            path.header.stamp = ros_stamp
            append_path_pose(path, sp, self._path_step, self._path_turn, self._path_max)
            now = time.monotonic()
            if now - self._source_last.get(source, 0.0) >= self._path_period:
                self._source_last[source] = now
                publisher.publish(path)

        dbg = result.debug
        if dbg is not None and result.stats.tracking_ok and dbg.inliers.any():  # Grow the sparse map.
            pts = dbg.points_camera[dbg.inliers]
            world = pts @ result.world_from_camera[:3, :3].T + result.world_from_camera[:3, 3]  # Camera world frame.
            body = world @ self._body_from_camera[:3, :3].T + self._body_from_camera[:3, 3]  # vo_odom frame.
            for p in body[np.isfinite(body).all(axis=1)]:
                key = tuple(np.floor(p / self._voxel).astype(int))
                self._landmarks.pop(key, None)  # Re-insert: recently seen voxels move to the end.
                self._landmarks[key] = p
            while len(self._landmarks) > self._landmark_max:  # Forget the voxels not seen for longest.
                del self._landmarks[next(iter(self._landmarks))]
            now = time.monotonic()
            if self._landmark_pub.get_subscription_count() > 0 and now - self._landmark_last >= self._landmark_period and self._landmarks:
                self._landmark_last = now
                header = Header(stamp=ros_stamp, frame_id=self._odom_frame)
                self._landmark_pub.publish(point_cloud2.create_cloud_xyz32(header, np.array(list(self._landmarks.values()), dtype=np.float32)))

        with self._viz_lock:
            self._viz.record(index, left, result, reference)  # Trajectory history (cheap).
        if self._save_video or self._debug_pub.get_subscription_count() > 0:  # Hand the newest frame to the debug worker.
            with self._debug_cv:
                self._debug_job = (ros_stamp, index, left, right, result)
                self._debug_cv.notify()

    def _debug_loop(self) -> None:
        """Render and publish ``/vo/debug_image`` for the newest frame, at most ``debug_rate_hz``."""
        last = 0.0
        while self._running:
            with self._debug_cv:
                while self._debug_job is None and self._running:
                    self._debug_cv.wait(0.2)
                if not self._running:
                    return
                stamp, index, left, right, result = self._debug_job
                self._debug_job = None
            subscribed = self._debug_pub.get_subscription_count() > 0
            period = (
                min(self._debug_period, self._video_period)
                if subscribed and self._save_video
                else (self._debug_period if subscribed else self._video_period)
            )
            wait = period - (time.monotonic() - last)
            if wait > 0:
                time.sleep(wait)
            last = time.monotonic()
            with self._viz_lock:
                canvas = self._viz.draw(index, left, right, result, self._processed_rate)
            if self._debug_scale != 1.0:
                canvas = cv2.resize(canvas, None, fx=self._debug_scale, fy=self._debug_scale, interpolation=cv2.INTER_AREA)
            if self._save_video and last - self._video_last >= 0.9 * self._video_period:
                self._video_last = last
                self._write_video(canvas)
            if not subscribed:
                continue
            msg = Image()
            msg.header.stamp, msg.header.frame_id = stamp, "vo_debug"
            msg.height, msg.width = canvas.shape[:2]
            msg.encoding, msg.step = "bgr8", canvas.shape[1] * 3
            msg.data = array.array("B", canvas.tobytes())  # Fast path: a bytes value is range-checked per byte in Python.
            try:
                self._debug_pub.publish(msg)
            except Exception:
                if self.context.ok():
                    raise
                return  # ROS is shutting down.

    def _on_save(self, request: Any, response: Any) -> Any:
        """``/vo/save`` (std_srvs/Trigger): write the run folder and its full report now, without stopping the VO."""
        if not self._output_dir or len(self._trajectory) == 0:
            response.success, response.message = False, "nothing to save yet (no output_dir or no VO pose)"
            return response
        if not self._save_lock.acquire(blocking=False):
            response.success, response.message = False, "a save is already running"
            return response

        def work() -> None:
            try:
                folder = self._save_data()
                metrics = write_live_report(folder)
                approaches = ", ".join(
                    f"{name}: end diff {m.get('end_error_m') or float('nan'):.2f} m" for name, m in (metrics.get("approaches") or {}).items()
                )
                self.get_logger().info(f"saved report: {folder / 'report.html'} ({approaches})")
            except Exception as exc:  # noqa: BLE001 - a failed snapshot must not stop the VO.
                self.get_logger().error(f"save failed: {exc}")
            finally:
                self._save_lock.release()

        threading.Thread(target=work, daemon=True).start()
        response.success, response.message = True, f"saving {Path(self._output_dir).resolve() / 'report.html'}"
        return response

    def _write_video(self, canvas: np.ndarray) -> None:
        """Append the debug canvas to ``video.webm`` (VP8: plays in any browser, so report.html can embed it)."""
        if self._video is None:
            path = Path(self._output_dir).resolve() / "video.webm"
            path.parent.mkdir(parents=True, exist_ok=True)
            self._video_size = (canvas.shape[1], canvas.shape[0])
            self._video = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"VP80"), 1.0 / self._video_period, self._video_size)  # type: ignore[attr-defined]
            if not self._video.isOpened():
                self.get_logger().warn("cannot write video.webm (no VP8 encoder in this OpenCV); video disabled")
                self._save_video, self._video = False, None
                return
        if (canvas.shape[1], canvas.shape[0]) != self._video_size:
            canvas = cv2.resize(canvas, self._video_size)
        self._video.write(canvas)

    def _publish(self, stamp: Any, pose: np.ndarray, ok: bool) -> None:
        """Publish odometry, path and (optionally) TF for one pose."""
        q = Rotation.from_matrix(pose[:3, :3]).as_quat()
        msg = Odometry()
        msg.header.stamp = stamp
        msg.header.frame_id = self._odom_frame
        msg.child_frame_id = self._base_frame
        msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z = (float(v) for v in pose[:3, 3])
        o = msg.pose.pose.orientation
        o.x, o.y, o.z, o.w = (float(v) for v in q)
        cov = 1e-3 if ok else 1.0  # Inflate while coasting through a tracking failure.
        msg.pose.covariance = list(np.diag([cov] * 6).ravel())
        self._odom_pub.publish(msg)

        ps = PoseStamped()
        ps.header = msg.header
        ps.pose = msg.pose.pose
        self._path.header.stamp = stamp
        append_path_pose(self._path, ps, self._path_step, self._path_turn, self._path_max)
        now = time.monotonic()
        if now - self._path_last[0] >= self._path_period:
            self._path_last[0] = now
            self._path_pub.publish(self._path)

        if self._tf is not None:
            tf = TransformStamped()
            tf.header = msg.header
            tf.child_frame_id = f"vo_{self._base_frame}"
            tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = (float(v) for v in pose[:3, 3])
            r = tf.transform.rotation
            r.x, r.y, r.z, r.w = (float(v) for v in q)
            self._tf.sendTransform(tf)

    def _report(self) -> None:
        """Log throughput every few seconds and, while no stereo pair forms, why."""
        if self._output_dir and len(self._trajectory) and time.monotonic() - self._checkpoint_last >= self._checkpoint_s:
            self._checkpoint_last = time.monotonic()
            try:
                self._save_data()
            except Exception as exc:  # noqa: BLE001 - never let saving stop the VO.
                self.get_logger().warn(f"checkpoint failed: {exc}")
        left, right = self._image_counts
        log = self.get_logger()
        log.info(f"images left {left} right {right}, pairs received {self._received}, processed {self._processed}, dropped {self._dropped}")
        recent = self._frames[self._frames_reported :]
        self._frames_reported = len(self._frames)
        if recent:
            a = np.array(recent)
            ok = int(a[:, 1].sum())
            line = (
                f"tracking ok {ok}/{len(a)} frames | median tracked {np.median(a[:, 2]):.0f}, with stereo depth {np.median(a[:, 3]):.0f}, "
                f"PnP inliers {np.median(a[:, 4]):.0f}, new detections {np.median(a[:, 5]):.0f} | brightness L {a[:, 6].mean():.0f} "
                f"R {a[:, 7].mean():.0f}, contrast L {a[:, 8].mean():.0f} R {a[:, 9].mean():.0f} | {np.median(a[:, 10]):.0f} ms/frame"
            )
            if ok < 0.5 * len(a):
                hints = []
                if min(a[:, 6].mean(), a[:, 7].mean()) < 25 or min(a[:, 8].mean(), a[:, 9].mean()) < 12:
                    dark = "left" if a[:, 6].mean() + a[:, 8].mean() < a[:, 7].mean() + a[:, 9].mean() else "right"
                    hints.append(f"the {dark} image is dark or flat (lens covered? lights off?)")
                if self._selfcal_enabled and not self._selfcal_pending and time.monotonic() - self._selfcal_last_run > 20.0:
                    self._selfcal_pending = True  # The worker re-checks the stereo calibration on the next frames.
                    hints.append("checking the stereo calibration now")
                elif not hints:
                    hints.append("stereo calibration was checked; likely the view (textureless, blur, something close) - see /vo/debug_image")
                log.warn(line + " -> TRACKING FAILING: " + "; ".join(hints))
            else:
                log.info(line)
        if self._received:
            return
        for side, topic in enumerate(self._topics):
            if self._image_counts[side]:
                continue
            publishers = self.get_publishers_info_by_topic(topic)
            if not publishers:
                log.warn(f"no publisher on {topic}: check `ros2 topic list`, ROS_DOMAIN_ID and the topic name")
            elif self._reliable and all(p.qos_profile.reliability == ReliabilityPolicy.BEST_EFFORT for p in publishers):
                log.warn(f"{topic} is published BEST_EFFORT but this node asks RELIABLE: relaunch with reliable_qos:=false")
            else:
                log.warn(
                    f"{topic} has {len(publishers)} publisher(s) but no image arrived: discovery works, data does not "
                    "(container/host shared memory or network; see README troubleshooting)"
                )
        if left and right:
            gap = abs(self._image_stamps[0] - self._image_stamps[1])
            log.warn(
                f"both cameras arrive but never pair: latest stamps differ by {gap * 1e3:.0f} ms > slop {self._slop * 1e3:.0f} ms "
                "(different clocks? raise max_pair_dt_s or fix the camera timestamps)"
            )

    def shutdown(self) -> None:
        """Stop the worker and save the trajectory."""
        self._running = False
        with self._lock:
            self._lock.notify_all()
        with self._debug_cv:
            self._debug_cv.notify_all()
        self._worker.join(timeout=5.0)
        for approach in self._parallel:
            approach.stop()
        self._debug_worker.join(timeout=5.0)
        if self._video is not None:
            self._video.release()
            self._video = None
        if not self._output_dir or len(self._trajectory) == 0:
            return
        # print, not the logger: ROS is already shut down here and rosout would complain.
        folder = self._save_data(final=True)
        print(f"[metricvision_vo] saved {folder} ({len(self._trajectory)} VO poses); writing the report ...", flush=True)
        try:
            metrics = write_live_report(folder)
        except Exception as exc:  # noqa: BLE001 - the trajectories are saved; the report can be rerun.
            print(f"[metricvision_vo] comparison failed ({exc}); rerun: python -m metricvision.evaluation.live_report {folder}", flush=True)
            return
        summary = ", ".join(
            f"{k} {metrics[k]:.3f}" for k in ("end_error_m", "drift_pct", "length_ratio", "yaw_error_deg", "ate_se3_m") if k in metrics
        )
        print(f"[metricvision_vo] VO vs {metrics.get('reference', 'reference')}: {summary or metrics.get('note', '')}", flush=True)
        print(f"[metricvision_vo] open {folder / 'report.html'}", flush=True)

    def _save_data(self, final: bool = False) -> Path:
        """Write the run folder's data (atomically; also every ``checkpoint_s`` while running, so a killed run keeps it).

        Layout: ``data/`` trajectories (TUM) + per-frame stats, ``calibration/`` rig files, ``run_info.yaml``.
        """
        folder = Path(self._output_dir).resolve()
        data = folder / "data"
        data.mkdir(parents=True, exist_ok=True)
        with self._lock:
            odom = Trajectory(list(self._odom_log.timestamps), list(self._odom_log.poses))
            lidar = Trajectory(list(self._lidar_log.timestamps), list(self._lidar_log.poses))
        vo = Trajectory(list(self._trajectory.timestamps), list(self._trajectory.poses))
        frames = list(self._frames)

        def atomic(path: Path, write: Any) -> None:
            tmp = path.with_name(path.name + ".part")
            write(tmp)
            tmp.replace(path)

        outputs = [("vo.tum", vo), ("odom.tum", odom), ("lidar.tum", lidar)]
        for approach in self._parallel:
            outputs.append((f"vo_{approach.name}.tum", approach.snapshot()))
            rows_p = "".join(f"{t:.6f},{ok},{inl},{ms:.1f}\n" for t, ok, inl, ms in list(approach.frames))
            atomic(data / f"frames_{approach.name}.csv", lambda tmp, r=rows_p: tmp.write_text("stamp,tracking_ok,pnp_inliers,total_ms\n" + r))
        for name, trajectory in outputs:
            if len(trajectory):
                atomic(data / name, lambda tmp, tr=trajectory: write_tum(tr, tmp))
        header = (
            "stamp,tracking_ok,tracked,with_depth,pnp_inliers,new_detections,brightness_left,brightness_right,contrast_left,contrast_right,total_ms"
        )
        rows = "".join(",".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in row) + "\n" for row in frames)
        atomic(data / "frames.csv", lambda tmp: tmp.write_text(header + "\n" + rows))
        ok = sum(row[1] for row in frames)
        info = dict(self._run_info)
        info.update(
            status="finished" if final else "running (checkpoint; the report is written on Ctrl+C)",
            saved=time.strftime("%Y-%m-%d %H:%M:%S"),
            duration_s=round(vo.timestamps[-1] - vo.timestamps[0], 2) if len(vo) > 1 else 0.0,
            frames={
                "stereo_pairs_received": self._received,
                "processed": self._processed,
                "dropped_to_stay_live": self._dropped,
                "tracking_ok": int(ok),
                "tracking_ok_pct": round(100.0 * ok / len(frames), 1) if frames else 0.0,
            },
            images={"left": self._image_counts[0], "right": self._image_counts[1], "sizes": self._image_sizes},
            references={"lidar_messages": len(lidar), "wheel_messages": len(odom), "rviz_overlay": self._overlay_source},
            stereo_selfcal=self._selfcal_log,
            parallel_approaches={a.name: {"processed": a.processed, "dropped_to_stay_live": a.dropped, "status": a.status} for a in self._parallel},
        )
        atomic(folder / "run_info.yaml", lambda tmp: tmp.write_text(yaml.safe_dump(info, sort_keys=False)))
        return folder


def main(args: list[str] | None = None) -> None:
    """Entry point."""
    rclpy.init(args=args)
    node = StereoVONode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception:
        if rclpy.ok():
            raise
        # Ctrl+C shut the context down while a message was being taken (rclpy race): a normal stop.
    finally:
        # Ctrl+C under `ros2 launch` arrives twice (terminal + launch); do not let the second one cut the save short.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
