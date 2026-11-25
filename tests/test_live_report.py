"""VO vs wheel-odometry comparison written for every live run."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from metricvision.core.geometry import make_pose
from metricvision.core.types import Trajectory
from metricvision.evaluation.live_report import read_tum, write_live_report
from metricvision.evaluation.writers import write_tum


def _yaw_pose(x: float, y: float, yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return make_pose(np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]), np.array([x, y, 0.0]))


def test_live_report_compares_vo_with_odom(tmp_path: Path) -> None:
    """Odometry in its own frame at 10 Hz, VO from identity at 15 Hz, 10% too long: the report sees the scale."""
    odom, vo = Trajectory(), Trajectory()
    start = _yaw_pose(5.0, -2.0, 0.7)  # /odom does not start at the origin.
    for t in np.arange(0.0, 10.01, 0.1):  # Quarter circle of radius 2 m.
        a = t / 10.0 * np.pi / 2
        odom.append(100.0 + t, start @ _yaw_pose(2 * np.sin(a), 2 * (1 - np.cos(a)), a))
    for t in np.arange(0.05, 9.9, 1 / 15):
        a = t / 10.0 * np.pi / 2
        vo.append(100.0 + t, _yaw_pose(2.2 * np.sin(a), 2.2 * (1 - np.cos(a)), a))
    write_tum(vo, tmp_path / "vo.tum")
    write_tum(odom, tmp_path / "odom.tum")

    metrics = write_live_report(tmp_path)

    assert metrics["frames"] == len(vo)
    assert abs(metrics["length_ratio"] - 1.1) < 0.01
    assert metrics["yaw_error_deg"] < 0.5
    assert 0.2 < metrics["end_error_m"] < 0.35  # 10% of a ~3 m arc.
    assert "ate_se3_m" in metrics
    for name in ("odom_at_vo.tum", "summary.md", "plots/trajectory_topdown.png", "plots/position_heading.png", "plots/difference.png", "report.html"):
        assert (tmp_path / name).is_file()
    assert json.loads((tmp_path / "metrics.json").read_text())["frames"] == metrics["frames"]
    anchored = read_tum(tmp_path / "odom_at_vo.tum")
    np.testing.assert_allclose(anchored.poses[0], np.eye(4), atol=1e-9)  # Same start frame as the VO.


def test_live_report_without_odom(tmp_path: Path) -> None:
    """No /odom (e.g. reference_topic:=""): still a plot and a clear note, no crash."""
    vo = Trajectory()
    for i in range(5):
        vo.append(float(i), np.eye(4))
    write_tum(vo, tmp_path / "vo.tum")
    (tmp_path / "odom.tum").write_text("")
    metrics = write_live_report(tmp_path)
    assert metrics["note"] == "no reference odometry received"
    assert (tmp_path / "plots" / "trajectory_topdown.png").is_file()


def test_lidar_is_the_reference_and_flags_a_pushed_chair(tmp_path: Path) -> None:
    """Lidar saw a 1.5 m drive, the wheels nothing (pushed by hand): lidar is used and the wheels are flagged."""
    vo, lidar, wheel = Trajectory(), Trajectory(), Trajectory()
    for t in np.arange(0.0, 10.01, 0.05):
        lidar.append(t, _yaw_pose(0.15 * t, 0.0, 0.0))
        wheel.append(t, np.eye(4))
    for t in np.arange(0.1, 9.9, 1 / 15):
        vo.append(t, _yaw_pose(0.15 * t * 1.02, 0.0, 0.0))
    for name, trajectory in (("vo.tum", vo), ("lidar.tum", lidar), ("odom.tum", wheel)):
        write_tum(trajectory, tmp_path / name)

    metrics = write_live_report(tmp_path)

    assert metrics["reference"] == "lidar odometry /Odometry"
    assert abs(metrics["length_ratio"] - 1.02) < 0.01
    assert any("pushed by hand" in note for note in metrics["notes"])
    assert set(metrics["by_reference"]) == {"lidar", "wheel"}
    assert (tmp_path / "lidar_at_vo.tum").is_file() and (tmp_path / "report.html").is_file()


def test_diverged_lidar_is_cut_where_it_breaks(tmp_path: Path) -> None:
    """DLIO jumps 7 m at t=5 s and runs away: only the first 5 s are used, and the report says so."""
    vo, lidar = Trajectory(), Trajectory()
    for t in np.arange(0.0, 10.01, 0.01):
        x = 0.2 * t if t < 5.0 else 0.2 * t + 7.0 + 50.0 * (t - 5.0)
        lidar.append(t, _yaw_pose(x, 0.0, 0.0))
    for t in np.arange(0.1, 9.9, 1 / 15):
        vo.append(t, _yaw_pose(0.2 * t, 0.0, 0.0))
    write_tum(vo, tmp_path / "vo.tum")
    write_tum(lidar, tmp_path / "lidar.tum")

    metrics = write_live_report(tmp_path)

    import re

    cut = [float(m.group(1)) for n in metrics["notes"] if (m := re.search(r"became invalid ([0-9.]+) s", n))]
    assert cut and 4.6 <= cut[0] <= 4.9, metrics["notes"]  # Jump at 5 s minus the VO start, cut up to 0.2 s early.
    assert metrics["end_error_m"] < 0.01  # Compared on the valid part only, where VO and lidar agree.
    assert metrics["duration_s"] < 5.0


def test_run_folder_layout(tmp_path: Path) -> None:
    """The node's layout (data/ ...) is read, and the report links the video and plots."""
    data = tmp_path / "data"
    data.mkdir()
    vo, wheel = Trajectory(), Trajectory()
    for t in np.arange(0.0, 5.01, 0.05):
        wheel.append(t, _yaw_pose(0.3 * t, 0.0, 0.0))
    for t in np.arange(0.1, 4.9, 1 / 15):
        vo.append(t, _yaw_pose(0.3 * t, 0.0, 0.0))
    write_tum(vo, data / "vo.tum")
    write_tum(wheel, data / "odom.tum")
    (data / "frames.csv").write_text(
        "stamp,tracking_ok,tracked,with_depth,pnp_inliers,new_detections,bl,br,cl,cr,ms\n"
        + "".join(f"{t:.3f},1,300,300,290,30,100,100,50,50,40\n" for t in vo.timestamps)
    )
    (tmp_path / "video.webm").write_bytes(b"")

    metrics = write_live_report(tmp_path)

    assert metrics["tracking_ok_pct"] == 100.0
    assert (data / "metrics.json").is_file() and (data / "odom_at_vo.tum").is_file()
    assert (tmp_path / "plots" / "tracking.png").is_file()
    page = (tmp_path / "report.html").read_text()
    assert 'src="video.webm"' in page and "plots/tracking.png" in page
