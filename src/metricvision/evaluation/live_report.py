"""Compare a VO run with reference odometry and write the run's report.

A run folder (written by the live node, see ``metricvision.ros.vo_node``)::

    results/live/<YYYYmmdd_HHMMSS>/
        report.html          <- open this: summary, notes, video, interactive charts (works offline)
        summary.md           headline numbers and notes as text
        video.webm           the VO debug view (keypoints, stereo matches, map, numbers), ~5 fps
        run_info.yaml        settings, calibration, topics, frame counts, git commit, stereo self-calibration
        plots/               trajectory_topdown.png, position_heading.png, difference.png, tracking.png
        data/                vo.tum, lidar.tum, odom.tum (raw), lidar_at_vo.tum, odom_at_vo.tum (at the VO stamps,
                             anchored at the first VO frame: same frame as vo.tum), frames.csv, metrics.json
        calibration/         rig_at_start.yaml (+ rig_selfcal.yaml if the node re-estimated it)
        bag/                 only with record:=true

References, best first: DLIO lidar-inertial odometry (``/Odometry``) and the wheels (``/odom``). Neither is perfect
ground truth (wheels slip and see nothing when the chair is pushed by hand, DLIO can diverge), so the numbers
measure agreement; a reference is cut where it becomes physically impossible (see :func:`valid_until`).

Usage (recompute for an existing run):
    uv run python -m metricvision.evaluation.live_report results/live/20261124_173000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from metricvision.core.geometry import invert_pose, make_pose
from metricvision.core.types import FloatArray, Trajectory
from metricvision.datasets.rosbag_stereo import interpolate_poses
from metricvision.evaluation.trajectory_metrics import evaluate_trajectory
from metricvision.evaluation.writers import write_tum


def read_tum(path: Path) -> Trajectory:
    """Read a TUM file (``t tx ty tz qx qy qz qw`` per line)."""
    from scipy.spatial.transform import Rotation

    trajectory = Trajectory()
    rows = np.loadtxt(path, ndmin=2)
    for row in rows:
        trajectory.append(float(row[0]), make_pose(Rotation.from_quat(row[4:8]).as_matrix(), row[1:4]))
    return trajectory


def _path_length(xy: FloatArray, min_step_m: float = 0.02) -> float:
    """Distance travelled, counting a step only once the position moved ``min_step_m`` from the last counted point.

    Summing every sample would turn pose jitter into distance: a standing chair's 100 Hz lidar odometry
    "travelled" 0.11 m in 13 s that way.
    """
    if len(xy) < 2:
        return 0.0
    total, anchor = 0.0, xy[0]
    for point in xy[1:]:
        step = float(np.linalg.norm(point - anchor))
        if step >= min_step_m:
            total += step
            anchor = point
    return total + float(np.linalg.norm(xy[-1] - anchor))  # The last partial step.


def _yaw(pose: FloatArray) -> float:
    return float(np.arctan2(pose[1, 0], pose[0, 0]))


def valid_until(reference: Trajectory, max_speed_mps: float = 3.0, window_s: float = 0.2) -> tuple[float, float] | None:
    """First time a reference odometry becomes physically impossible (a wheelchair moves < 3 m/s).

    Lidar odometry can diverge (DLIO jumped 6.9 m between two messages and ended km away in one run); everything
    after that is garbage, so the comparison has to stop there.

    Returns:
        (time, jump in metres) of the first violation, or None if the whole trajectory is plausible.
    """
    if len(reference) < 3:
        return None
    t = np.asarray(reference.timestamps)
    xyz = np.array([p[:3, 3] for p in reference.poses])
    j = np.searchsorted(t, t + window_s)  # Compare each pose with the one ~window_s later (robust to stamp jitter).
    j = np.minimum(j, len(t) - 1)
    dt = t[j] - t
    ok = dt > 0.5 * window_s
    speed = np.zeros(len(t))
    speed[ok] = np.linalg.norm(xyz[j[ok]] - xyz[ok], axis=1) / dt[ok]
    bad = np.nonzero(speed > max_speed_mps)[0]
    if len(bad) == 0:
        return None
    i = int(bad[0])
    return float(t[i]), float(np.linalg.norm(xyz[j[i]] - xyz[i]))


def _truncate(trajectory: Trajectory, until: float) -> Trajectory:
    out = Trajectory()
    for stamp, pose in zip(trajectory.timestamps, trajectory.poses, strict=True):
        if stamp < until:
            out.append(stamp, pose)
    return out


def odometry_at(vo: Trajectory, odom: Trajectory) -> Trajectory:
    """/odom interpolated at the VO timestamps it covers, anchored so its first pose is the identity."""
    if len(odom) < 2 or len(vo) == 0:
        return Trajectory()
    order = np.argsort(odom.timestamps)
    times = np.asarray(odom.timestamps)[order]
    poses = [odom.poses[i] for i in order]
    keep = np.concatenate([[True], np.diff(times) > 0])  # Slerp needs strictly increasing times.
    times, poses = times[keep], [p for p, k in zip(poses, keep, strict=True) if k]
    query = np.asarray(vo.timestamps)
    interpolated, inside = interpolate_poses(times, poses, query)
    out = Trajectory()
    if len(interpolated) == 0:
        return out
    origin_inv = invert_pose(interpolated[0])
    for t, pose in zip(query[inside], interpolated, strict=True):
        out.append(float(t), origin_inv @ pose)
    return out


def compare_with_odometry(vo: Trajectory, odom_at_vo: Trajectory, label: str = "wheel odometry /odom") -> dict[str, Any]:
    """Agreement of VO with a reference odometry (``label``) on the frames both have.

    Both trajectories are anchored at the first common frame (no fitting) for the end-point, length and
    heading numbers; the SE(3) ATE/RPE use the usual Umeyama alignment.

    Returns:
        ``frames``, ``duration_s``, ``odom_length_m``, ``vo_length_m``, ``length_ratio`` (metric-scale check),
        ``end_error_m``, ``drift_pct`` (end error / odometry length), ``yaw_error_deg`` (final heading),
        ``max_error_m`` (anchored), ``ate_se3_m``, ``rpe_trans_m``, ``rpe_rot_deg``.
    """
    stamps = {round(t, 6): i for i, t in enumerate(vo.timestamps)}
    pairs = [(stamps[round(t, 6)], j) for j, t in enumerate(odom_at_vo.timestamps) if round(t, 6) in stamps]
    if len(pairs) < 2:
        return {"frames": len(pairs), "note": f"no overlap between VO and {label}", "notes": [f"No overlap between VO and {label}."]}
    first_vo_inv = invert_pose(vo.poses[pairs[0][0]])
    est = np.array([(first_vo_inv @ vo.poses[i]) for i, _ in pairs])
    ref = np.array([odom_at_vo.poses[j] for _, j in pairs])
    err = np.linalg.norm(est[:, :2, 3] - ref[:, :2, 3], axis=1)
    odom_len = _path_length(ref[:, :2, 3])
    vo_len = _path_length(est[:, :2, 3])
    yaw_err = np.degrees(np.angle(np.exp(1j * (_yaw(est[-1]) - _yaw(ref[-1])))))
    notes: list[str] = []
    if odom_len <= 0.05 and vo_len > 0.05:
        notes.append(
            f"The reference ({label}) did not move ({odom_len:.2f} m) but the VO did ({vo_len:.2f} m): the chair was pushed "
            "by hand (wheels record nothing then), the reference was frozen, or moving people pulled the VO."
        )
    elif odom_len <= 0.05:
        notes.append(f"Neither {label} nor the VO moved: a standstill run (VO drift check only).")
    if odom_len <= 0.05:
        notes.append("Length ratio and drift % need a path of more than 5 cm, so they are not computed.")
    metrics: dict[str, Any] = {
        "reference": label,
        "notes": notes,
        "frames": len(pairs),
        "duration_s": float(odom_at_vo.timestamps[pairs[-1][1]] - odom_at_vo.timestamps[pairs[0][1]]),
        "odom_length_m": odom_len,
        "vo_length_m": vo_len,
        "length_ratio": vo_len / odom_len if odom_len > 0.05 else float("nan"),
        "end_error_m": float(err[-1]),
        "drift_pct": 100.0 * float(err[-1]) / odom_len if odom_len > 0.05 else float("nan"),
        "yaw_error_deg": float(abs(yaw_err)),
        "max_error_m": float(err.max()),
    }
    estimate, reference = Trajectory(), Trajectory()
    for (i, j), e, r in zip(pairs, est, ref, strict=True):
        estimate.append(vo.timestamps[i], e)
        reference.append(odom_at_vo.timestamps[j], r)
    try:  # Needs a non-degenerate path; a chair standing still has none.
        errors = evaluate_trajectory(estimate, reference, max_time_diff_s=1e-3)
        metrics.update(ate_se3_m=errors.ate_rmse_m, rpe_trans_m=errors.rpe_trans_rmse_m, rpe_rot_deg=errors.rpe_rot_rmse_deg)
    except Exception as exc:  # noqa: BLE001 - report, do not lose the other numbers.
        metrics["ate_note"] = f"SE(3) ATE not computed: {exc}"
    return metrics


def _data_dir(folder: Path) -> Path:
    """``folder/data`` (current layout) or ``folder`` itself (runs saved before the layout existed)."""
    return folder / "data" if (folder / "data" / "vo.tum").is_file() else folder


def _frames(data: Path) -> FloatArray | None:
    file = data / "frames.csv"
    if not file.is_file() or not file.stat().st_size:
        return None
    rows = np.loadtxt(file, delimiter=",", skiprows=1, ndmin=2)
    return rows if rows.size else None


_COLOURS = {"vo": "#1f7ae0", "vo2": "#8e44ad", "lidar": "#e07b00", "wheel": "#1a9b55", "err": "#d6453d"}


def _plots(
    folder: Path,
    vo: Trajectory,
    refs: list[tuple[str, str, Trajectory, Trajectory]],
    metrics: dict[str, Any],
    frames: FloatArray | None,
    extra: list[tuple[str, Trajectory]] | None = None,
    main_name: str = "classical",
) -> None:
    """Static figures for documents and slides (``plots/``); report.html has the interactive versions."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = folder / "plots"
    out.mkdir(exist_ok=True)
    t0 = vo.timestamps[0]
    first_inv = invert_pose(vo.poses[0])
    vo_p = np.array([first_inv @ p for p in vo.poses])
    vo_t = np.asarray(vo.timestamps) - t0
    series = [(f"Stereo VO {main_name}", "vo", vo_t, vo_p)]
    for name, trajectory in extra or []:
        if len(trajectory):
            inv = invert_pose(trajectory.poses[0])
            series.append((f"Stereo VO {name}", "vo2", np.asarray(trajectory.timestamps) - t0, np.array([inv @ p for p in trajectory.poses])))
    n_vo = len(series)
    series += [(label, key, np.asarray(at.timestamps) - t0, np.array(at.poses)) for key, label, _, at in refs if len(at)]
    title = f"Run {folder.name}"

    def yaw(poses: FloatArray) -> FloatArray:
        return np.asarray(np.degrees(np.unwrap(np.arctan2(poses[:, 1, 0], poses[:, 0, 0]))))

    # 1. Top-down trajectory.
    fig, ax = plt.subplots(figsize=(8, 7))
    for name, key, _, poses in series[::-1]:
        ax.plot(
            poses[:, 0, 3],
            poses[:, 1, 3],
            color=_COLOURS[key],
            lw=1.6 if key.startswith("vo") else 2.4,
            label=f"{name} ({_path_length(poses[:, :2, 3]):.2f} m)",
        )
        ax.plot(poses[-1, 0, 3], poses[-1, 1, 3], "o", mfc="none", color=_COLOURS[key])
    ax.plot(0, 0, "ko", label="start")
    ax.set(title=f"{title}: top-down path (all anchored at the first VO frame)", xlabel="x [m]", ylabel="y [m]")
    ax.axis("equal")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "trajectory_topdown.png", dpi=130)
    plt.close(fig)

    # 2. x, y, heading over time.
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for name, key, t, poses in series[::-1]:
        lw = 1.4 if key.startswith("vo") else 2.2
        axes[0].plot(t, poses[:, 0, 3], color=_COLOURS[key], lw=lw, label=name)
        axes[1].plot(t, poses[:, 1, 3], color=_COLOURS[key], lw=lw)
        axes[2].plot(t, yaw(poses), color=_COLOURS[key], lw=lw)
    for a, lab in zip(axes, ("x [m]", "y [m]", "heading [deg]"), strict=True):
        a.set_ylabel(lab)
        a.grid(alpha=0.3)
    axes[0].legend(fontsize=9)
    axes[0].set_title(f"{title}: position and heading over time")
    axes[2].set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(out / "position_heading.png", dpi=130)
    plt.close(fig)

    # 3. Difference to the primary reference.
    if len(series) > n_vo:
        name, key, t_ref, ref = series[n_vo]
        index = {round(t, 6): i for i, t in enumerate(vo.timestamps)}
        pairs = [(index[round(t + t0, 6)], j) for j, t in enumerate(t_ref) if round(t + t0, 6) in index]
        if pairs:
            i, j = np.array(pairs).T
            dpos = np.linalg.norm(vo_p[i, :2, 3] - ref[j, :2, 3], axis=1)
            dyaw = (yaw(vo_p)[i] - yaw(ref)[j] + 180.0) % 360.0 - 180.0
            fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
            axes[0].plot(t_ref[j], dpos, color=_COLOURS["err"])
            axes[0].set_ylabel("|xy difference| [m]")
            axes[1].plot(t_ref[j], dyaw, color=_COLOURS["err"])
            axes[1].set_ylabel("heading difference [deg]")
            axes[1].set_xlabel("time [s]")
            axes[0].set_title(f"{title}: stereo VO minus {name}")
            text = (
                f"end difference {metrics.get('end_error_m', float('nan')):.3f} m ({metrics.get('drift_pct', float('nan')):.1f} % of path)\n"
                f"length ratio VO/reference {metrics.get('length_ratio', float('nan')):.3f}\n"
                f"final heading difference {metrics.get('yaw_error_deg', float('nan')):.1f} deg"
            )
            axes[0].text(0.01, 0.95, text, transform=axes[0].transAxes, va="top", fontsize=9, family="monospace")
            for a in axes:
                a.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(out / "difference.png", dpi=130)
            plt.close(fig)

    # 4. Tracking health.
    if frames is not None:
        t = frames[:, 0] - t0
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        lost = frames[:, 1] < 0.5
        for k in np.nonzero(lost)[0]:
            axes[0].axvspan(t[k] - 0.05, t[k] + 0.05, color=_COLOURS["err"], alpha=0.15, lw=0)
        axes[0].plot(t, frames[:, 4], color=_COLOURS["vo"], label="PnP inliers")
        axes[0].plot(t, frames[:, 3], color=_COLOURS["wheel"], lw=1, label="points with stereo depth")
        axes[0].plot(t, frames[:, 5], color="grey", lw=0.8, label="new detections")
        axes[0].set_ylabel("points")
        axes[0].legend(fontsize=8)
        axes[0].set_title(f"{title}: tracking health (red = tracking lost)")
        axes[1].plot(t, frames[:, 6], color=_COLOURS["vo"], label="left")
        axes[1].plot(t, frames[:, 7], color=_COLOURS["err"], label="right")
        axes[1].set_ylabel("image brightness")
        axes[1].set_xlabel("time [s]")
        axes[1].legend(fontsize=8)
        for a in axes:
            a.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "tracking.png", dpi=130)
        plt.close(fig)


def _summary_md(folder: Path, metrics: dict[str, Any]) -> str:
    def fmt(value: Any, unit: str = "", digits: int = 3) -> str:
        if isinstance(value, float):
            return "—" if not np.isfinite(value) else f"{value:.{digits}f}{unit}"
        return f"{value}{unit}" if value is not None else "—"

    lines = [f"# Run {folder.name}: stereo VO vs {metrics.get('reference', 'reference')}", ""]
    lines += [f"> {n}" for n in metrics.get("notes", [])] + ([""] if metrics.get("notes") else [])
    rows = [
        ("Tracking OK", fmt(metrics.get("tracking_ok_pct"), " %", 1)),
        ("Duration compared", fmt(metrics.get("duration_s"), " s", 1)),
        ("Stereo VO path", fmt(metrics.get("vo_length_m"), " m", 2)),
        *[(f"{k} reference path", fmt(v.get("odom_length_m"), " m", 2)) for k, v in metrics.get("by_reference", {}).items()],
        ("Length ratio VO / reference", fmt(metrics.get("length_ratio"))),
        ("End position difference", fmt(metrics.get("end_error_m"), " m")),
        ("Drift (end difference / path)", fmt(metrics.get("drift_pct"), " %", 1)),
        ("Final heading difference", fmt(metrics.get("yaw_error_deg"), " deg", 1)),
        ("ATE (SE3-aligned RMSE)", fmt(metrics.get("ate_se3_m"), " m")),
        ("RPE translation / rotation per frame", f"{fmt(metrics.get('rpe_trans_m'), ' m', 4)} / {fmt(metrics.get('rpe_rot_deg'), ' deg', 3)}"),
    ]
    lines += ["| Metric | Value |", "|---|---|"] + [f"| {k} | {v} |" for k, v in rows]
    lines += [
        "",
        "Files: `report.html` (interactive), `video.webm`, `plots/`, `data/` (TUM trajectories, frames.csv, metrics.json), "
        "`run_info.yaml`, `calibration/`.",
        "",
    ]
    return "\n".join(lines)


def write_live_report(folder: Path) -> dict[str, Any]:
    """Compare ``data/vo.tum`` with the reference odometries and write the report files (see module docstring)."""
    folder = Path(folder)
    data = _data_dir(folder)
    vo = read_tum(data / "vo.tum")
    refs: list[tuple[str, str, Trajectory, Trajectory]] = []
    validity_notes: list[str] = []
    for key, label, name, at_name in (
        ("lidar", "lidar odometry /Odometry", "lidar.tum", "lidar_at_vo.tum"),
        ("wheel", "wheel odometry /odom", "odom.tum", "odom_at_vo.tum"),
    ):
        file = data / name
        raw = read_tum(file) if file.is_file() and file.stat().st_size else Trajectory()
        broken = valid_until(raw)
        if broken is not None and len(vo):
            validity_notes.append(
                f"{label} became invalid {broken[0] - vo.timestamps[0]:.1f} s into the run (jumped {broken[1]:.1f} m in 0.2 s: "
                "the odometry diverged). It is compared with the VO only before that."
            )
            raw = _truncate(raw, broken[0])
        at_vo = odometry_at(vo, raw)
        if len(at_vo):
            write_tum(at_vo, data / at_name)
            refs.append((key, label, raw, at_vo))
    by_reference = {key: compare_with_odometry(vo, at_vo, label) for key, label, _, at_vo in refs}
    metrics: dict[str, Any] = (
        dict(by_reference[refs[0][0]])
        if refs
        else {"frames": 0, "note": "no reference odometry received", "notes": ["No /odom or /Odometry received."]}
    )
    metrics["by_reference"] = {k: {kk: vv for kk, vv in m.items() if kk != "notes"} for k, m in by_reference.items()}
    metrics["notes"] = validity_notes + list(metrics.get("notes", []))
    if "lidar" in by_reference and "wheel" in by_reference:
        lidar_len, wheel_len = by_reference["lidar"].get("odom_length_m", 0.0), by_reference["wheel"].get("odom_length_m", 0.0)
        if lidar_len > 0.1 and wheel_len < 0.5 * lidar_len:
            metrics["notes"].append(
                f"The lidar saw the chair move {lidar_len:.2f} m but the wheels only {wheel_len:.2f} m: it was pushed by hand "
                "(motors disengaged), so the wheel odometry is not a valid reference for this run."
            )
    metrics["vo_frames"] = len(vo)
    metrics["reference_messages"] = {key: len(raw) for key, _, raw, _ in refs}
    # Approaches that ran in parallel on the same frames (data/vo_<name>.tum, e.g. the learned front end).
    main_name = "classical"
    info_file = folder / "run_info.yaml"
    if info_file.is_file():
        import yaml

        main_name = ((yaml.safe_load(info_file.read_text()) or {}).get("approaches") or {}).get("main", main_name)
    extra = [(f.stem[3:], read_tum(f)) for f in sorted(data.glob("vo_*.tum")) if f.stat().st_size]
    keys = ("end_error_m", "drift_pct", "length_ratio", "yaw_error_deg", "ate_se3_m", "rpe_trans_m", "vo_length_m", "frames")
    metrics["approaches"] = {main_name: {k: metrics.get(k) for k in keys}}
    for name, trajectory in extra:
        if refs:
            at = odometry_at(trajectory, refs[0][2])
            m = compare_with_odometry(trajectory, at, refs[0][1]) if len(at) else {}
            metrics["approaches"][name] = {k: m.get(k) for k in keys}
        frames_x = data / f"frames_{name}.csv"
        if frames_x.is_file() and frames_x.stat().st_size:
            rows_x = np.loadtxt(frames_x, delimiter=",", skiprows=1, ndmin=2)
            if rows_x.size:
                metrics["approaches"].setdefault(name, {})["tracking_ok_pct"] = 100.0 * float(rows_x[:, 1].mean())
    frames = _frames(data)
    if frames is not None:
        ok_pct = 100.0 * float(frames[:, 1].mean())
        metrics["tracking_ok_pct"] = ok_pct
        metrics["median_pnp_inliers"] = float(np.median(frames[:, 4]))
        metrics["approaches"][main_name]["tracking_ok_pct"] = ok_pct
        if ok_pct < 50:
            metrics["notes"].append(
                f"Tracking failed in {100 - ok_pct:.0f} % of the frames: while lost the VO coasts on its last velocity, "
                "so its path is meaningless there. See the tracking panels (inliers, detections, brightness)."
            )
    (data / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float) + "\n")
    (folder / "summary.md").write_text(_summary_md(folder, metrics))
    for old in ("metrics.md", "trajectories.png"):  # Files of the earlier flat layout.
        (folder / old).unlink(missing_ok=True)
    if len(vo):
        _plots(folder, vo, refs, metrics, frames, extra, main_name)
        from metricvision.evaluation.live_report_html import write_index, write_report_html

        write_report_html(folder, vo, refs, metrics, extra, main_name)
        if folder.parent.name == "live":
            write_index(folder.parent)
    return metrics


def main() -> None:
    """CLI: recompute the report for one or more run folders."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folders", nargs="+", type=Path)
    for folder in parser.parse_args().folders:
        metrics = write_live_report(folder)
        print(f"{folder}: {metrics.get('reference', 'no reference')}; notes: {len(metrics.get('notes', []))}; open {folder / 'report.html'}")


if __name__ == "__main__":
    main()
