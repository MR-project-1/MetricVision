"""D455 tooling: rosbag extraction (ROS 1 and ROS 2), factory calibration, the stereo_folder loader end to end."""

from __future__ import annotations  # Postponed evaluation of type hints.

import importlib.util  # Load the script as a module.
from pathlib import Path  # Paths.

import cv2  # Image writing.
import numpy as np  # Arrays.
import pytest  # Fixtures.
from scipy.spatial.transform import Rotation  # Quaternions.

from metricvision.calibration.camera import RawStereoCalib  # Calibration format.
from metricvision.config import ComponentConfig, ExperimentConfig  # Configs.
from metricvision.datasets.stereo_folder import StereoFolderDataset  # Loader under test.
from metricvision.datasets.synthetic import SyntheticPlaneDataset  # Image source.
from metricvision.pipeline import ExperimentRunner  # End to end.

N = 6  # Frames in the bag test.


def _load_extractor():  # noqa: ANN202
    """Import scripts/extract_rosbag.py."""
    path = Path(__file__).resolve().parents[1] / "scripts" / "extract_rosbag.py"
    spec = importlib.util.spec_from_file_location("extract_rosbag", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_folder(root: Path, dataset: SyntheticPlaneDataset, frames: int, distortion: bool = False) -> RawStereoCalib:
    """Write a dataset as a stereo folder with calib.yaml and groundtruth.tum.

    Args:
        root: Destination.
        dataset: Image source.
        frames: Frame count.
        distortion: Unused hook (rectified source images, zero distortion).

    Returns:
        The written calibration.
    """
    cam = dataset.camera
    k = cam.intrinsic_matrix
    calib = RawStereoCalib(k, np.zeros(5), k, np.zeros(5), np.eye(3), np.array([-cam.baseline, 0.0, 0.0]), cam.width, cam.height)
    (root / "left").mkdir(parents=True)
    (root / "right").mkdir(parents=True)
    calib.to_yaml(str(root / "calib.yaml"))
    gt = dataset.ground_truth()
    rows = []
    for i in range(frames):
        frame = dataset.load_frame(i)
        stamp = 1_700_000_000_000_000_000 + i * 50_000_000
        cv2.imwrite(str(root / "left" / f"{stamp}.png"), frame.left)
        cv2.imwrite(str(root / "right" / f"{stamp}.png"), frame.right)
        pose = gt.poses[i]
        rows.append([stamp * 1e-9, *pose[:3, 3], *Rotation.from_matrix(pose[:3, :3]).as_quat()])
    np.savetxt(root / "groundtruth.tum", np.array(rows), fmt="%.9f")
    return calib


def test_stereo_folder_runs_end_to_end(tmp_path: Path) -> None:
    """A stereo folder (rectified on load, identity rectification) must give an accurate sparse trajectory.

    Args:
        tmp_path: Temporary folder.
    """
    source = SyntheticPlaneDataset(num_frames=25)
    _write_folder(tmp_path / "seq", source, 25)
    dataset = StereoFolderDataset(tmp_path / "seq")
    assert len(dataset) == 25 and abs(dataset.camera.baseline - source.camera.baseline) < 1e-9
    config = ExperimentConfig(
        dataset=ComponentConfig("stereo_folder", {"root": str(tmp_path / "seq")}),
        matcher=ComponentConfig("orb"),
        depth=ComponentConfig("sgbm"),
        output_dir=tmp_path / "out",
        pipeline="sparse",
    )
    summary = ExperimentRunner(config).run(show_progress=False)
    assert summary.num_tracking_failures == 0
    assert summary.errors is not None and summary.errors.ate_rmse_m < 0.02
    assert summary.loop_closure_error_m > 0.0  # Ground-truth-free drift is reported.


def _header(store, stamp: int, frame: str):  # noqa: ANN001, ANN202
    """ROS header for either message store."""
    header = store.types["std_msgs/msg/Header"]
    time_t = store.types["builtin_interfaces/msg/Time"]
    extra = {"seq": 0} if "seq" in header.__dataclass_fields__ else {}
    return header(stamp=time_t(sec=stamp // 10**9, nanosec=stamp % 10**9), frame_id=frame, **extra)


def _messages(store, source: SyntheticPlaneDataset) -> list:  # noqa: ANN001
    """D455-like stream: two IR images, IMU at 200 Hz and camera_info."""
    image_t = store.types["sensor_msgs/msg/Image"]
    imu_t = store.types["sensor_msgs/msg/Imu"]
    vec = store.types["geometry_msgs/msg/Vector3"]
    quat = store.types["geometry_msgs/msg/Quaternion"]
    info_t = store.types["sensor_msgs/msg/CameraInfo"]
    roi_t = store.types["sensor_msgs/msg/RegionOfInterest"]
    k = source.camera.intrinsic_matrix
    out = []
    for i in range(N):
        frame = source.load_frame(i)
        stamp = 1_700_000_000_000_000_000 + i * 33_333_333
        for topic, img in (("/camera/infra1/image_rect_raw", frame.left), ("/camera/infra2/image_rect_raw", frame.right)):
            msg = image_t(
                header=_header(store, stamp, "cam"),
                height=img.shape[0],
                width=img.shape[1],
                encoding="mono8",
                is_bigendian=0,
                step=img.shape[1],
                data=np.ascontiguousarray(img).ravel(),
            )
            out.append((topic, "sensor_msgs/msg/Image", stamp, msg))
        for j in range(7):
            t = stamp + j * 5_000_000
            msg = imu_t(
                header=_header(store, t, "imu"),
                orientation=quat(x=0.0, y=0.0, z=0.0, w=1.0),
                orientation_covariance=np.zeros(9),
                angular_velocity=vec(x=0.01, y=0.0, z=0.0),
                angular_velocity_covariance=np.zeros(9),
                linear_acceleration=vec(x=0.0, y=-9.81, z=0.0),
                linear_acceleration_covariance=np.zeros(9),
            )
            out.append(("/camera/imu", "sensor_msgs/msg/Imu", t, msg))
        upper = "K" in info_t.__dataclass_fields__
        for topic, tx in (("/camera/infra1/camera_info", 0.0), ("/camera/infra2/camera_info", -k[0, 0] * source.camera.baseline)):
            p = np.hstack([k, [[tx], [0.0], [0.0]]])
            fields = {"d": np.zeros(5), "k": k.ravel(), "r": np.eye(3).ravel(), "p": p.ravel()}
            fields = {(key.upper() if upper else key): v for key, v in fields.items()}
            msg = info_t(
                header=_header(store, stamp, "cam"),
                height=source.camera.height,
                width=source.camera.width,
                distortion_model="plumb_bob",
                binning_x=0,
                binning_y=0,
                roi=roi_t(x_offset=0, y_offset=0, height=0, width=0, do_rectify=False),
                **fields,
            )
            out.append((topic, "sensor_msgs/msg/CameraInfo", stamp, msg))
    return sorted(out, key=lambda m: m[2])


@pytest.mark.parametrize("fmt", ["ros2", "ros1"])
def test_extract_rosbag(tmp_path: Path, fmt: str) -> None:
    """Extraction must pair the IR images, log the IMU and recover the factory calibration.

    Args:
        tmp_path: Temporary folder.
        fmt: Bag format.
    """
    pytest.importorskip("rosbags")
    from rosbags.typesys import Stores, get_typestore

    source = SyntheticPlaneDataset(num_frames=N)
    if fmt == "ros2":
        from rosbags.rosbag2 import Writer

        store = get_typestore(Stores.ROS2_HUMBLE)
        bag = tmp_path / "bag2"
        with Writer(bag, version=8) as writer:
            conns: dict = {}
            for topic, mtype, t, msg in _messages(store, source):
                if topic not in conns:
                    conns[topic] = writer.add_connection(topic, mtype, typestore=store)
                writer.write(conns[topic], t, store.serialize_cdr(msg, mtype))
    else:
        from rosbags.rosbag1 import Writer

        store = get_typestore(Stores.ROS1_NOETIC)
        bag = tmp_path / "data.bag"
        with Writer(bag) as writer:
            conns = {}
            for topic, mtype, t, msg in _messages(store, source):
                if topic not in conns:
                    conns[topic] = writer.add_connection(topic, mtype, typestore=store)
                writer.write(conns[topic], t, store.serialize_ros1(msg, mtype))
    out = tmp_path / "seq"
    _load_extractor().main(["--bag", str(bag), "--out", str(out)])
    factory = RawStereoCalib.from_yaml(str(out / "factory_calib.yaml"))
    assert abs(np.linalg.norm(factory.T) - source.camera.baseline) < 1e-9
    assert np.allclose(factory.K_L, source.camera.intrinsic_matrix)
    imu = np.loadtxt(out / "imu.csv", delimiter=",", skiprows=1)
    assert imu.shape == (7 * N, 7)
    dataset = StereoFolderDataset(out, calib=out / "factory_calib.yaml", already_rectified=True)
    assert len(dataset) == N and dataset.load_frame(0).left.shape == (source.camera.height, source.camera.width)
