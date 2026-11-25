"""Estimate the rotation of the (rectified left) camera in the wheelchair base frame from VO runs + wheel odometry.

The nominal TF chain (base_link -> camera mount -> RealSense optical convention) leaves the VO path rotated
by ~15 deg against wheel odometry on a straight drive, so the cameras do not point where the TF says.
For a planar, non-holonomic base the rotation is observable from the data:

* while the chair turns, the camera's relative rotation axis must be the base z axis (up),
* while it drives straight, the camera's translation direction must be the base x axis (forward; the sign
  comes from the odometry).

The two sets of vector pairs are solved with Wahba/Kabsch for R_base_camera. The lever arm (translation)
is kept from tf_static. The result is written back into the rig YAML as ``body_from_left``.

Usage:
    uv run python scripts/estimate_camera_to_base.py --run classical --results results/simulation --calibration configs/wheelchair/rig_640x480.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from metricvision.core.geometry import invert_pose, make_pose
from metricvision.datasets.rosbag_stereo import RigCalibration, RosbagStereoDataset, StereoRectifier


def load_tum_poses(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read TUM poses as 4x4 matrices.

    Args:
        path: trajectory.tum.

    Returns:
        (times, (N, 4, 4) poses).
    """
    data = np.loadtxt(path)
    poses = np.repeat(np.eye(4)[None], len(data), axis=0)
    poses[:, :3, :3] = Rotation.from_quat(data[:, 4:8]).as_matrix()
    poses[:, :3, 3] = data[:, 1:4]
    return data[:, 0], poses


def main() -> None:
    """Estimate and write R_base_camera."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=Path("results/simulation"))
    parser.add_argument("--bags", type=Path, default=Path("ros_bags"))
    parser.add_argument("--run", default="classical", help="Run folder whose trajectories are used (scripts/run_simulation.sh).")
    parser.add_argument("--calibration", type=Path, default=Path("configs/wheelchair/rig_640x480.yaml"))
    parser.add_argument("--gap", type=int, default=15, help="Frames between the two poses of a relative motion (1 s at 15 Hz).")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    calib = RigCalibration.load(args.calibration)
    rect = StereoRectifier(calib)
    body_from_cam_old = rect.body_from_camera  # What the runs were conjugated with.
    raw_from_rect = invert_pose(calib.body_from_left) @ body_from_cam_old  # Fixed rectification rotation.

    axes_cam, axes_base, dirs_cam, dirs_base = [], [], [], []
    for bag in sorted(d for d in args.bags.iterdir() if (d / "metadata.yaml").is_file()):
        dataset = RosbagStereoDataset(bag, args.calibration)
        ref = dataset.ground_truth()
        tum = args.results / dataset.tag / args.run / "trajectory.tum"
        if ref is None or not tum.is_file():
            continue
        times, body_poses = load_tum_poses(tum)
        cam = np.array([invert_pose(body_from_cam_old) @ p @ body_from_cam_old for p in body_poses])  # Undo conjugation.
        ref_t = np.array(ref.timestamps)
        nearest = np.clip(np.searchsorted(ref_t, times), 0, len(ref_t) - 1)  # TUM rounds stamps: match by time.
        nearest = np.where(np.abs(ref_t[nearest - 1] - times) < np.abs(ref_t[nearest] - times), nearest - 1, nearest)
        keep = np.abs(ref_t[nearest] - times) < 1e-3
        times, cam, nearest = times[keep], cam[keep], nearest[keep]
        odo = np.array([ref.poses[i] for i in nearest])
        used_axes = used_dirs = 0
        for k in range(len(times) - args.gap):
            rc = invert_pose(cam[k]) @ cam[k + args.gap]  # Camera relative motion.
            rb = invert_pose(odo[k]) @ odo[k + args.gap]  # Base relative motion.
            ang_c = np.linalg.norm(Rotation.from_matrix(rc[:3, :3]).as_rotvec())
            ang_b = np.linalg.norm(Rotation.from_matrix(rb[:3, :3]).as_rotvec())
            if ang_c > np.radians(8) and ang_b > np.radians(8):  # Turning: axis_cam -> +/- z_base.
                axis = Rotation.from_matrix(rc[:3, :3]).as_rotvec() / ang_c
                sign = np.sign(Rotation.from_matrix(rb[:3, :3]).as_rotvec()[2])
                axes_cam.append(axis)
                axes_base.append(np.array([0.0, 0.0, sign]))
                used_axes += 1
            elif ang_c < np.radians(1.5) and ang_b < np.radians(1.5):  # Straight: t_cam -> +/- x_base.
                tc, tb = rc[:3, 3], rb[:3, 3]
                if np.linalg.norm(tc) > 0.2 and abs(tb[0]) > 0.2:
                    dirs_cam.append(tc / np.linalg.norm(tc))
                    dirs_base.append(np.array([np.sign(tb[0]), 0.0, 0.0]))
                    used_dirs += 1
        print(f"{bag.name}: {used_axes} turning and {used_dirs} straight relative motions")

    a = np.array(axes_cam + dirs_cam)
    b = np.array(axes_base + dirs_base)
    w = np.r_[np.full(len(axes_cam), 1.0 / max(len(axes_cam), 1)), np.full(len(dirs_cam), 1.0 / max(len(dirs_cam), 1))]
    u, _, vt = np.linalg.svd((b * w[:, None]).T @ a)  # Wahba: R a ~ b.
    rot = u @ np.diag([1, 1, np.sign(np.linalg.det(u @ vt))]) @ vt
    res_axes = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", (rot @ np.array(axes_cam).T).T, np.array(axes_base)), -1, 1)))
    res_dirs = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", (rot @ np.array(dirs_cam).T).T, np.array(dirs_base)), -1, 1)))
    print(f"residuals: turn axis median {np.median(res_axes):.2f} deg, forward direction median {np.median(res_dirs):.2f} deg")
    delta = rot @ body_from_cam_old[:3, :3].T
    ypr = Rotation.from_matrix(delta).as_euler("ZYX", degrees=True)
    print(f"correction to the TF-based camera rotation: yaw {ypr[0]:.2f}, pitch {ypr[1]:.2f}, roll {ypr[2]:.2f} deg")

    body_from_cam_new = make_pose(rot, body_from_cam_old[:3, 3])  # Keep the TF lever arm.
    body_from_left_new = body_from_cam_new @ invert_pose(raw_from_rect)
    if args.dry_run:
        return
    text = args.calibration.read_text()
    header = "".join(line + "\n" for line in text.splitlines() if line.startswith("#"))
    header += (
        f"# body_from_left rotation re-estimated by scripts/estimate_camera_to_base.py from '{args.run}' runs vs /odom\n"
        f"#   (turn-axis residual {np.median(res_axes):.2f} deg, forward residual {np.median(res_dirs):.2f} deg; "
        f"yaw correction {ypr[0]:.2f} deg); lever arm from tf_static.\n"
    )
    data = yaml.safe_load(text)
    data["body_from_left"] = body_from_left_new.tolist()
    args.calibration.write_text(header + yaml.safe_dump(data, sort_keys=False, default_flow_style=None))
    print(f"updated {args.calibration}")


if __name__ == "__main__":
    main()
