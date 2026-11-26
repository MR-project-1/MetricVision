"""Stereo sequences recorded as ROS 2 bags (sqlite3 ``.db3``), e.g. the wheelchair rig.

The two colour cameras of the wheelchair are separate RealSense devices without hardware
sync, so this loader

1. pairs left/right images by header stamp (mutual nearest neighbours within ``max_pair_dt_s``),
2. undistorts and stereo-rectifies them with a rig calibration file (:class:`RigCalibration`),
3. reads wheel odometry (``nav_msgs/Odometry``) as the reference trajectory and interpolates it
   at the image timestamps, so every frame has a reference pose.

The bag is read directly with sqlite3 and the pure-Python ``rosbags`` deserialiser, so no ROS
installation is needed offline. :class:`StereoRectifier` and :func:`image_message_to_gray` are
shared with the live ROS 2 node (``metricvision.ros.vo_node``).
"""

from __future__ import annotations  # Postponed evaluation of type hints.

import sqlite3  # Random access to rosbag2 sqlite storage.
from dataclasses import dataclass  # Containers.
from pathlib import Path  # Filesystem paths.
from typing import Any  # Loosely typed messages.

import cv2  # Rectification and colour conversion.
import numpy as np  # Matrix maths.
import yaml  # Rig calibration files.
from scipy.spatial.transform import Rotation, Slerp  # Odometry interpolation.

from metricvision.core.geometry import make_pose  # Pose helpers.
from metricvision.core.types import FloatArray, ImageArray, StereoCamera, StereoFrame, Trajectory  # Shared types.
from metricvision.datasets.base import DATASETS, StereoDataset  # Base class and registry.

# --------------------------------------------------------------------------- calibration


@dataclass(frozen=True)
class RigCalibration:
    """Raw (distorted, unrectified) calibration of a two-camera rig.

    Attributes:
        k_left: 3x3 intrinsics of the left camera.
        d_left: Distortion coefficients (OpenCV plumb_bob) of the left camera.
        k_right: 3x3 intrinsics of the right camera.
        d_right: Distortion coefficients of the right camera.
        width: Image width in pixels.
        height: Image height in pixels.
        right_from_left: T_right_left, maps left-camera points into the right camera (metres).
        body_from_left: T_body_left, the raw left optical frame in the odometry body frame.
    """

    k_left: FloatArray  # Left intrinsics.
    d_left: FloatArray  # Left distortion.
    k_right: FloatArray  # Right intrinsics.
    d_right: FloatArray  # Right distortion.
    width: int  # Image width.
    height: int  # Image height.
    right_from_left: FloatArray  # Stereo extrinsic.
    body_from_left: FloatArray  # Camera-to-body extrinsic.

    @classmethod
    def load(cls, path: str | Path) -> RigCalibration:
        """Read a rig calibration YAML (format written by ``scripts/calibrate_rig_from_bags.py``).

        Args:
            path: YAML file.

        Returns:
            The calibration.
        """
        data: dict[str, Any] = yaml.safe_load(Path(path).expanduser().read_text())  # Plain YAML mapping.

        def matrix(key: str, shape: tuple[int, ...]) -> FloatArray:
            return np.asarray(data[key], dtype=np.float64).reshape(shape)  # Nested list -> array.

        return cls(
            k_left=matrix("k_left", (3, 3)),
            d_left=matrix("d_left", (-1,)),
            k_right=matrix("k_right", (3, 3)),
            d_right=matrix("d_right", (-1,)),
            width=int(data["width"]),
            height=int(data["height"]),
            right_from_left=matrix("right_from_left", (4, 4)),
            body_from_left=matrix("body_from_left", (4, 4)) if "body_from_left" in data else np.eye(4),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the calibration as plain lists for YAML output.

        Returns:
            Serialisable mapping.
        """
        return {
            "width": self.width,
            "height": self.height,
            "k_left": self.k_left.tolist(),
            "d_left": self.d_left.tolist(),
            "k_right": self.k_right.tolist(),
            "d_right": self.d_right.tolist(),
            "right_from_left": self.right_from_left.tolist(),
            "body_from_left": self.body_from_left.tolist(),
        }


class StereoRectifier:
    """Undistorts and rectifies raw left/right images of a :class:`RigCalibration`."""

    def __init__(self, calibration: RigCalibration, alpha: float = 0.0, clahe: bool = False) -> None:
        """Compute rectification maps.

        Args:
            calibration: Raw rig calibration.
            alpha: OpenCV free-scaling parameter: 0 keeps only valid pixels, 1 keeps every source pixel.
            clahe: Apply CLAHE to both rectified images. The two RealSense units expose and white-balance
                independently; equalising local contrast raises the left/right match yield by 60-230% on the
                wheelchair bags.
        """
        size = (calibration.width, calibration.height)  # OpenCV (w, h).
        extrinsic = calibration.right_from_left  # Left -> right.
        rect_l, rect_r, proj_l, proj_r, *_ = cv2.stereoRectify(
            calibration.k_left,
            calibration.d_left,
            calibration.k_right,
            calibration.d_right,
            size,
            extrinsic[:3, :3],
            extrinsic[:3, 3].reshape(3, 1),
            flags=cv2.CALIB_ZERO_DISPARITY,  # Same principal point: disparity is purely horizontal.
            alpha=alpha,
        )
        self._maps = [
            cv2.initUndistortRectifyMap(k, d, r, p, size, cv2.CV_32FC1)
            for k, d, r, p in (
                (calibration.k_left, calibration.d_left, rect_l, proj_l),
                (calibration.k_right, calibration.d_right, rect_r, proj_r),
            )
        ]  # (map_x, map_y) per camera.
        self.camera = StereoCamera(
            fx=float(proj_l[0, 0]),
            fy=float(proj_l[1, 1]),
            cx=float(proj_l[0, 2]),
            cy=float(proj_l[1, 2]),
            baseline=float(-proj_r[0, 3] / proj_r[0, 0]),  # P_r stores -fx * B.
            width=calibration.width,
            height=calibration.height,
        )  # Rectified model.
        raw_from_rect = make_pose(np.asarray(rect_l, dtype=np.float64).T, np.zeros(3))  # Rectification is a pure rotation.
        self.body_from_camera: FloatArray = calibration.body_from_left @ raw_from_rect  # Rectified left camera in the body frame.
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if clahe else None  # Optional equalisation.

    def rectify(self, left: ImageArray, right: ImageArray) -> tuple[ImageArray, ImageArray]:
        """Rectify one raw grayscale pair.

        Args:
            left: Raw left image.
            right: Raw right image.

        Returns:
            Rectified (left, right).
        """
        out = [cv2.remap(image, mx, my, interpolation=cv2.INTER_LINEAR) for image, (mx, my) in zip((left, right), self._maps, strict=True)]
        if self._clahe is not None:  # Local contrast equalisation, same for both cameras.
            out = [self._clahe.apply(image) for image in out]
        return np.asarray(out[0], dtype=np.uint8), np.asarray(out[1], dtype=np.uint8)


# --------------------------------------------------------------------------- messages


def stamp_seconds(message: Any) -> float:
    """Return a message's header stamp in seconds.

    Args:
        message: Any ROS message with a ``header``.

    Returns:
        Seconds since the epoch.
    """
    return float(message.header.stamp.sec) + float(message.header.stamp.nanosec) * 1e-9  # sec + nsec.


def image_message_to_gray(message: Any) -> ImageArray:
    """Convert a ``sensor_msgs/Image`` (rgb8, bgr8, mono8 or rgba8/bgra8) to 8-bit grayscale.

    Works for both ``rosbags`` and ``rclpy`` message objects.

    Args:
        message: Image message.

    Returns:
        (H, W) uint8 image.

    Raises:
        ValueError: For unsupported encodings.
    """
    height, width, step = int(message.height), int(message.width), int(message.step)  # Geometry.
    buffer = np.frombuffer(bytes(message.data) if not isinstance(message.data, np.ndarray) else message.data, dtype=np.uint8)
    channels = {"mono8": 1, "rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4}.get(message.encoding)  # Bytes per pixel.
    if channels is None:  # 16-bit or Bayer images are not used here.
        raise ValueError(f"Unsupported image encoding '{message.encoding}'")
    image = buffer.reshape(height, step)[:, : width * channels].reshape(height, width, channels)  # Drop row padding.
    if channels == 1:  # Already grayscale.
        return np.ascontiguousarray(image[:, :, 0])
    code = {
        "rgb8": cv2.COLOR_RGB2GRAY,
        "bgr8": cv2.COLOR_BGR2GRAY,
        "rgba8": cv2.COLOR_RGBA2GRAY,
        "bgra8": cv2.COLOR_BGRA2GRAY,
    }[message.encoding]  # Colour conversion.
    return np.asarray(cv2.cvtColor(image, code), dtype=np.uint8)


def resolution_adapter(
    k_source: FloatArray,
    d_source: FloatArray,
    k_target: FloatArray,
    d_target: FloatArray,
    tolerance_px: float = 2.0,
    tolerance_distortion: float = 2e-3,
) -> FloatArray | None:
    """Pixel map from another streaming mode of the same camera to the calibrated one, if it is a pure scale + shift.

    RealSense colour modes are scaled windows of one sensor (e.g. 640x480 = 0.6 * 1280x720 shifted by (-64, +24)),
    with the same distortion in normalised coordinates. Then ``K_target K_source^-1`` maps raw pixels exactly and
    the calibrated rig can be used without recalibrating.

    Args:
        k_source: 3x3 intrinsics of the incoming mode (from its camera_info).
        d_source: Distortion of the incoming mode.
        k_target: 3x3 intrinsics the rig calibration was made at.
        d_target: Distortion of the calibration.
        tolerance_px: Largest allowed skew / non-axis-aligned term of the map, in target pixels over the image.
        tolerance_distortion: Largest allowed difference between the distortion coefficients.

    Returns:
        2x3 affine (target <- source pixels), or None when the two modes are not related that way.
    """
    k_source = np.asarray(k_source, dtype=np.float64).reshape(3, 3)
    k_target = np.asarray(k_target, dtype=np.float64).reshape(3, 3)
    n = min(len(d_source), len(d_target))
    if n and np.max(np.abs(np.asarray(d_source[:n], dtype=np.float64) - np.asarray(d_target[:n], dtype=np.float64))) > tolerance_distortion:
        return None  # Different lens model: this is a different camera or a different calibration.
    affine = (k_target @ np.linalg.inv(k_source))[:2]
    if abs(affine[0, 1]) * 1000.0 > tolerance_px or abs(affine[1, 0]) * 1000.0 > tolerance_px:
        return None  # Skew / rotation: not a crop-and-scale of the same sensor.
    if abs(affine[0, 0] - affine[1, 1]) * 1000.0 > tolerance_px:
        return None  # Anisotropic scaling: not the same pixels.
    return affine


def apply_resolution_adapter(image: ImageArray, affine: FloatArray, size: tuple[int, int]) -> ImageArray:
    """Resample an image with :func:`resolution_adapter`'s map (area filter for the downscale).

    Rows or columns the source mode does not cover (e.g. the top and bottom 24 rows when 1280x720 feeds a
    640x480 calibration) are filled by replicating the border, so they create no artificial corners.

    Args:
        image: (H, W) or (H, W, C) uint8 image in the source mode.
        affine: 2x3 target <- source pixel map (diagonal scale + shift).
        size: Target (width, height).

    Returns:
        Image in the target mode.
    """
    sx, sy = float(affine[0, 0]), float(affine[1, 1])
    # fx/fy rather than a size: a size would round the scale to size/width (0.6 instead of e.g. 0.60033).
    resized = cv2.resize(image, (0, 0), fx=sx, fy=sy, interpolation=cv2.INTER_AREA)
    # cv2.resize maps pixel centres as x' = s x + (s - 1) / 2; the remaining shift turns that into the wanted map.
    shift = np.array([[1.0, 0.0, affine[0, 2] - (sx - 1.0) / 2.0], [0.0, 1.0, affine[1, 2] - (sy - 1.0) / 2.0]], dtype=np.float64)
    adapted = cv2.warpAffine(resized, shift, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return np.asarray(adapted, dtype=np.uint8)


def odometry_pose(message: Any) -> FloatArray:
    """Return T_odom_base from a ``nav_msgs/Odometry`` message.

    Args:
        message: Odometry message.

    Returns:
        4x4 pose.
    """
    p, q = message.pose.pose.position, message.pose.pose.orientation  # Position and quaternion.
    return make_pose(Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix(), np.array([p.x, p.y, p.z]))


class Rosbag2Sqlite:
    """Minimal random-access reader for rosbag2 sqlite3 storage."""

    def __init__(self, bag: str | Path) -> None:
        """Open a bag folder (containing ``metadata.yaml`` and one ``.db3``) or a ``.db3`` file.

        Args:
            bag: Bag folder or database file.

        Raises:
            FileNotFoundError: If no database is found.
        """
        from rosbags.typesys import Stores, get_typestore  # Pure-Python CDR deserialiser (dependency of evo).

        path = Path(bag).expanduser()  # Resolve "~".
        databases = sorted(path.glob("*.db3")) if path.is_dir() else [path]  # Folder or file.
        if not databases or not databases[0].is_file():
            raise FileNotFoundError(f"No .db3 file in {path}")
        if len(databases) > 1:  # Split bags are not needed for our recordings.
            raise ValueError(f"Bag {path} is split into {len(databases)} files; only single-file bags are supported")
        self.path = path  # For tags and messages.
        self._db = sqlite3.connect(f"file:{databases[0]}?mode=ro", uri=True, check_same_thread=False)  # Read-only.
        self._topics = {name: (tid, typ) for tid, name, typ in self._db.execute("SELECT id, name, type FROM topics")}
        self._typestore = get_typestore(Stores.ROS2_HUMBLE)  # Message definitions.

    @property
    def topics(self) -> dict[str, str]:
        """Return topic name -> message type.

        Returns:
            Mapping of all topics in the bag.
        """
        return {name: typ for name, (_, typ) in self._topics.items()}

    def message_ids(self, topic: str) -> list[int]:
        """Return the row ids of a topic's messages in receive order.

        Args:
            topic: Topic name.

        Returns:
            Message row ids (empty if the topic is absent).
        """
        if topic not in self._topics:
            return []
        rows = self._db.execute("SELECT id FROM messages WHERE topic_id=? ORDER BY timestamp", (self._topics[topic][0],))
        return [int(r[0]) for r in rows]

    def read(self, topic: str, message_id: int) -> Any:
        """Deserialise one message.

        Args:
            topic: Topic the message belongs to (sets its type).
            message_id: Row id from :meth:`message_ids`.

        Returns:
            The message object.
        """
        (data,) = self._db.execute("SELECT data FROM messages WHERE id=?", (message_id,)).fetchone()  # Raw CDR.
        return self._typestore.deserialize_cdr(data, self._topics[topic][1])

    def messages(self, topic: str) -> list[Any]:
        """Deserialise all messages of a (small) topic.

        Args:
            topic: Topic name.

        Returns:
            Messages in receive order.
        """
        if topic not in self._topics:
            return []
        tid, typ = self._topics[topic]  # Id and type.
        rows = self._db.execute("SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp", (tid,))
        return [self._typestore.deserialize_cdr(data, typ) for (data,) in rows]

    def header_stamps(self, topic: str) -> tuple[list[int], FloatArray]:
        """Return message ids and header stamps of a topic without decoding the payload twice later.

        Args:
            topic: Topic name.

        Returns:
            (ids, stamps in seconds).
        """
        ids = self.message_ids(topic)  # Receive order.
        stamps = np.array([stamp_seconds(self.read(topic, i)) for i in ids], dtype=np.float64)  # Capture times.
        return ids, stamps


# --------------------------------------------------------------------------- pairing and interpolation


def pair_by_timestamp(left: FloatArray, right: FloatArray, max_dt_s: float) -> list[tuple[int, int]]:
    """Pair two timestamp lists by mutual nearest neighbour within a tolerance.

    Args:
        left: Sorted left timestamps (s).
        right: Sorted right timestamps (s).
        max_dt_s: Largest accepted |t_left - t_right|.

    Returns:
        (left index, right index) pairs in time order, each index used at most once.
    """
    if len(left) == 0 or len(right) == 0:
        return []

    def nearest(a: FloatArray, b: FloatArray) -> np.ndarray:
        pos = np.clip(np.searchsorted(b, a), 1, len(b) - 1)  # Insertion points.
        before, after = b[pos - 1], b[pos]  # Neighbours.
        return np.where(np.abs(a - before) <= np.abs(after - a), pos - 1, pos) if len(b) > 1 else np.zeros(len(a), int)

    l2r, r2l = nearest(left, right), nearest(right, left)  # Nearest in each direction.
    pairs = [
        (i, int(j))
        for i, j in enumerate(l2r)
        if r2l[j] == i and abs(left[i] - right[j]) <= max_dt_s  # Mutual and close enough.
    ]
    return pairs


def interpolate_poses(times: FloatArray, poses: list[FloatArray], query: FloatArray) -> tuple[FloatArray, np.ndarray]:
    """Interpolate a pose trajectory (linear translation, slerp rotation).

    Args:
        times: Sorted pose times (s).
        poses: 4x4 poses at ``times``.
        query: Times to interpolate at.

    Returns:
        ((M, 4, 4) poses for the queries inside [times[0], times[-1]], boolean mask of those queries).
    """
    stack = np.asarray(poses, dtype=np.float64)  # (N, 4, 4).
    inside = (query >= times[0]) & (query <= times[-1])  # No extrapolation.
    q = query[inside]  # Valid queries.
    slerp = Slerp(times, Rotation.from_matrix(stack[:, :3, :3]))  # Rotation interpolator.
    out = np.repeat(np.eye(4)[None], len(q), axis=0)  # Output poses.
    out[:, :3, :3] = slerp(q).as_matrix()  # Rotations.
    for axis in range(3):  # Translations.
        out[:, axis, 3] = np.interp(q, times, stack[:, axis, 3])
    return out, inside


# --------------------------------------------------------------------------- dataset


@DATASETS.register("rosbag_stereo")
class RosbagStereoDataset(StereoDataset):
    """Rectified stereo pairs and wheel-odometry reference from a ROS 2 bag."""

    def __init__(
        self,
        root: str | Path,
        calibration: str | Path,
        left_topic: str = "/cam1/cam1/color/image_raw",
        right_topic: str = "/cam2/cam2/color/image_raw",
        odom_topic: str = "/odom",
        max_pair_dt_s: float = 0.035,
        alpha: float = 0.0,
        clahe: bool = False,
        require_reference: bool = True,
    ) -> None:
        """Index and pair the images, prepare rectification and the reference trajectory.

        Args:
            root: Bag folder.
            calibration: Rig calibration YAML (see :class:`RigCalibration`).
            left_topic: Left image topic.
            right_topic: Right image topic.
            odom_topic: Wheel-odometry topic used as reference ("" to disable).
            max_pair_dt_s: Largest left/right stamp difference accepted for a stereo pair.
            alpha: Rectification free-scaling parameter (0 = valid pixels only).
            clahe: Equalise local contrast of both rectified images (see :class:`StereoRectifier`).
            require_reference: Drop frames outside the odometry time span, so every frame has a reference pose.
        """
        self._bag = Rosbag2Sqlite(root)  # Reader.
        self._left_topic, self._right_topic = left_topic, right_topic  # Topics.
        self._rectifier = StereoRectifier(RigCalibration.load(calibration), alpha=alpha, clahe=clahe)  # Calibration.
        left_ids, left_t = self._bag.header_stamps(left_topic)  # Left index.
        right_ids, right_t = self._bag.header_stamps(right_topic)  # Right index.
        lo, ro = np.argsort(left_t, kind="stable"), np.argsort(right_t, kind="stable")  # Capture order.
        left_ids, left_t = [left_ids[i] for i in lo], left_t[lo]
        right_ids, right_t = [right_ids[i] for i in ro], right_t[ro]
        pairs = pair_by_timestamp(left_t, right_t, max_pair_dt_s)  # Stereo pairs.

        frame_t = np.array([left_t[i] for i, _ in pairs], dtype=np.float64)  # Frame times (left stamps).
        self._reference: Trajectory | None = None  # Wheel odometry at frame times.
        keep = np.ones(len(pairs), dtype=bool)  # Frames to use.
        odometry = self._bag.messages(odom_topic) if odom_topic else []  # Reference messages.
        if odometry and len(frame_t) > 0:
            odom_t = np.array([stamp_seconds(m) for m in odometry])  # Odometry stamps.
            order = np.argsort(odom_t, kind="stable")  # Time order.
            odom_t = odom_t[order]
            odom_poses = [odometry_pose(odometry[i]) for i in order]
            unique = np.concatenate([[True], np.diff(odom_t) > 0])  # Slerp needs strictly increasing times.
            odom_poses = [p for p, u in zip(odom_poses, unique, strict=True) if u]
            reference_poses, inside = interpolate_poses(odom_t[unique], odom_poses, frame_t)  # Poses at frame times.
            self._reference = Trajectory()
            for t, pose in zip(frame_t[inside], reference_poses, strict=True):
                self._reference.append(float(t), pose)
            if require_reference:
                keep = inside  # Only frames with a reference pose.

        self._pairs = [(left_ids[i], right_ids[j]) for (i, j), k in zip(pairs, keep, strict=True) if k]  # Message ids per frame.
        self._timestamps = [float(t) for t, k in zip(frame_t, keep, strict=True) if k]  # Left stamps.
        self._pair_dt = [float(abs(left_t[i] - right_t[j])) for (i, j), k in zip(pairs, keep, strict=True) if k]  # Sync quality.
        self.num_left, self.num_right = len(left_ids), len(right_ids)  # For reporting.

    @property
    def camera(self) -> StereoCamera:
        """Return the rectified stereo camera.

        Returns:
            Rectified intrinsics and baseline.
        """
        return self._rectifier.camera

    @property
    def tag(self) -> str:
        """Return ``wheelchair_<bag name>``.

        Returns:
            Filesystem-safe tag.
        """
        name = self._bag.path.name if self._bag.path.is_dir() else self._bag.path.stem  # Bag folder name.
        return f"wheelchair_{name}"

    @property
    def body_from_camera(self) -> FloatArray:
        """Return T_base_camera of the rectified left camera.

        Returns:
            4x4 transform from the calibration file.
        """
        return self._rectifier.body_from_camera

    @property
    def pair_dt_s(self) -> list[float]:
        """Return |t_left - t_right| of every stereo pair.

        Returns:
            Seconds, one per frame.
        """
        return list(self._pair_dt)

    def __len__(self) -> int:
        """Return the number of stereo pairs.

        Returns:
            Frame count.
        """
        return len(self._pairs)

    def load_frame(self, index: int) -> StereoFrame:
        """Decode, convert and rectify one pair.

        Args:
            index: Frame index.

        Returns:
            Rectified stereo frame stamped with the left image time.
        """
        left_id, right_id = self._pairs[index]  # Message ids.
        left = image_message_to_gray(self._bag.read(self._left_topic, left_id))  # Raw left.
        right = image_message_to_gray(self._bag.read(self._right_topic, right_id))  # Raw right.
        left, right = self._rectifier.rectify(left, right)  # Undistort + rectify.
        return StereoFrame(index=index, timestamp=self._timestamps[index], left=left, right=right)

    def ground_truth(self) -> Trajectory | None:
        """Return wheel odometry (T_odom_base) interpolated at the frame times.

        Returns:
            Reference trajectory, or None without an odometry topic.
        """
        return self._reference
