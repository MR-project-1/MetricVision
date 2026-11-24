"""Demo figures for the windowed-BA + monocular-depth-prior POC (final report).

Usage:
    uv run python scripts/make_poc_figures.py --out docs/poc/figures \
        --kitti ~/datasets/kitti --euroc ~/datasets/euroc/MH_01_easy

Writes (PDF for LaTeX, PNG for Markdown):
  poc_depth_<tag>        camera image | Depth Anything V2 | SGBM stereo | network/stereo ratio
  poc_gating_<tag>       which landmarks get a depth prior (stereo trusted / accepted / rejected)
  poc_scale_drift        per-keyframe network/stereo scale over both sequences (from results/)
  poc_ba_effect          KITTI drift accumulation (no BA / BA / BA+priors) and per-window BA statistics
  poc_trajectory_zoom    KITTI 00 trajectories with a zoom on the loop where BA corrects drift (from results/)
Needs the ``learned`` extra and the result folders written by the POC runs (see docs/poc/README.md).
"""

from __future__ import annotations  # Postponed evaluation of type hints.

import argparse  # Command-line arguments.
import copy  # Deep copies before alignment.
import csv  # Per-frame statistics.
from pathlib import Path  # Paths.
from typing import Any  # Loose typing for plot handles.

import matplotlib  # Plotting backend selection.

matplotlib.use("Agg")  # Headless rendering.
import matplotlib.pyplot as plt  # noqa: E402  # Plotting.
import numpy as np  # noqa: E402  # Arrays.
from evo.core import sync  # noqa: E402  # Timestamp association.
from evo.core.trajectory import PoseTrajectory3D  # noqa: E402  # evo trajectory type.
from evo.tools import file_interface  # noqa: E402  # TUM reader.

import metricvision.backend.keyframe_window as keyframe_window  # noqa: E402  # Patched to record one BA window.
from metricvision.backend.keyframe_window import DepthPriorConfig, KeyframeWindow  # noqa: E402  # Prior gating.
from metricvision.backend.stereo_odometry import SparseStereoOdometry  # noqa: E402  # Sparse VO.
from metricvision.backend.window_ba import WindowBAConfig, _linearise  # noqa: E402  # Residuals for the histogram.
from metricvision.config import ExperimentConfig  # noqa: E402  # Config parsing.
from metricvision.core.geometry import invert_pose  # noqa: E402  # Rigid inverse.
from metricvision.core.stereo import triangulate_rectified, triangulation_covariance  # noqa: E402  # Stereo depth + covariance.
from metricvision.datasets import DATASETS  # noqa: E402  # Loaders.
from metricvision.depth import DEPTH_ESTIMATORS  # noqa: E402  # SGBM and Depth Anything V2.
from metricvision.evaluation.kitti_metrics import load_kitti_poses  # noqa: E402  # Line-aligned KITTI poses.
from metricvision.frontend import STEREO_FRONTENDS  # noqa: E402  # Sparse front end.
from metricvision.frontend.stereo_klt import StereoTracks  # noqa: E402  # Track container.

GREEN, RED, GREY, BLUE, ORANGE, NAVY = "#2ca02c", "#d62728", "#9e9e9e", "#1f77b4", "#ff7f0e", "#000080"  # Shared palette.
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "figure.dpi": 100})  # Report-friendly defaults.


def _save(fig: Any, out: Path, name: str) -> None:
    """Save a figure as PDF and PNG.

    Args:
        fig: Matplotlib figure.
        out: Output folder.
        name: File stem.
    """
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")  # Vector.
    fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight")  # Raster.
    plt.close(fig)  # Free memory.
    print(f"wrote {out / name}.png")  # Progress.


def _load(config_path: str, root: str) -> tuple[ExperimentConfig, Any]:
    """Parse a config and build its dataset.

    Args:
        config_path: Experiment YAML.
        root: Dataset root.

    Returns:
        (config, dataset).
    """
    config = ExperimentConfig.from_yaml(config_path).with_overrides(dataset_root=root)  # Same settings as the runs.
    return config, DATASETS.create(config.dataset.name, **config.dataset.params)  # Loader.


def depth_figure(config: ExperimentConfig, dataset: Any, index: int, network: Any, out: Path) -> None:
    """Camera image, network depth, stereo depth and their ratio for one frame.

    Args:
        config: Experiment config.
        dataset: Loader.
        index: Frame to show.
        network: Depth Anything V2 estimator.
        out: Output folder.
    """
    frame = dataset.load_frame(index)  # Stereo pair.
    mono = network.compute(frame.left, frame.right, dataset.camera)  # Network depth.
    stereo = DEPTH_ESTIMATORS.create(config.depth.name, **config.depth.params).compute(frame.left, frame.right, dataset.camera)  # SGBM.
    ratio = np.log2(mono / stereo)  # >0: network deeper than stereo.
    vmax = float(np.nanpercentile(np.concatenate([mono[np.isfinite(mono)], stereo[np.isfinite(stereo)]]), 97))  # Shared depth scale.
    height, width = frame.left.shape  # Image size.
    fig, axes = plt.subplots(2, 2, figsize=(14, 14 * height / width + 1.4))  # 2x2 grid.
    panels = [
        (frame.left, "gray", None, None, "Left camera image"),
        (mono, "turbo", 0, vmax, "Depth Anything V2 metric depth (monocular)"),
        (stereo, "turbo", 0, vmax, "SGBM stereo depth (baseline-metric)"),
        (ratio, "RdBu_r", -1, 1, "log2(network / stereo): red = network too deep"),
    ]
    for axis, (image, cmap, vmin, vmax_i, title) in zip(axes.ravel(), panels, strict=True):  # Fill panels.
        shown = axis.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax_i, interpolation="nearest")  # Image.
        axis.set_title(title)  # Caption.
        axis.axis("off")  # No ticks.
        if cmap != "gray":  # Colour bar for data panels.
            fig.colorbar(shown, ax=axis, fraction=0.025, pad=0.01, label="log2 ratio" if cmap == "RdBu_r" else "depth [m]")  # Legend.
    valid = np.isfinite(ratio) & (stereo < 20)  # Where stereo is a trustworthy reference.
    fig.suptitle(
        f"{dataset.tag}, frame {index}: median network/stereo depth ratio {np.median(2 ** ratio[valid]):.2f} (stereo < 20 m)", y=1.0
    )  # Headline.
    fig.tight_layout()  # Compact.
    _save(fig, out, f"poc_depth_{dataset.tag}")  # Write.


def gating_figure(config: ExperimentConfig, dataset: Any, index: int, network: Any, out: Path) -> None:
    """Show which landmarks the RoMeO-style gate gives a depth prior.

    Args:
        config: Experiment config.
        dataset: Loader.
        index: Frame to show.
        network: Depth Anything V2 estimator.
        out: Output folder.
    """
    camera = dataset.camera  # Rig.
    frame = dataset.load_frame(index)  # Stereo pair.
    frontend = STEREO_FRONTENDS.create(config.stereo_frontend.name, **config.stereo_frontend.params)  # Front end.
    pairs = frontend.detect(frame.left, frame.right, StereoTracks.empty())  # Stereo landmarks.
    points, valid = triangulate_rectified(pairs.left, pairs.disparity, camera)  # Camera = world for this demo.
    pairs, points = pairs.subset(valid), points[valid]  # Keep triangulated ones.
    cov = triangulation_covariance(pairs.left, pairs.disparity, camera, config.odometry.stereo_sigma_px)  # Stereo covariance.
    depth_map = network.compute(frame.left, frame.right, camera)  # Network depth.
    prior_cfg = DepthPriorConfig(
        rel_sigma=config.odometry.depth_prior_rel_sigma, scale_align=config.odometry.depth_prior_scale_align
    )  # Same gate as the runs.
    window = KeyframeWindow(5, prior_config=prior_cfg)  # Fresh window.
    pixels = np.column_stack([pairs.left, pairs.right[:, 0]]).astype(np.float64)  # (u_l, v_l, u_r).
    stats = window.add_keyframe(index, np.eye(4), np.arange(len(pairs)), pixels, points, cov, depth_map)  # Gate.
    prior = window._keyframes[0].prior_depth  # Gated priors (NaN = none).
    z = points[:, 2]  # Stereo depth.
    sigma_z = np.sqrt(cov[:, 2, 2])  # Stereo depth std.
    trusted = sigma_z <= prior_cfg.min_stereo_rel_sigma * z  # Stereo precise enough on its own.
    accepted = np.isfinite(prior)  # Prior kept.
    rejected = ~trusted & ~accepted  # Weak stereo but the prior disagreed.
    raw = depth_map[
        np.clip(np.rint(pairs.left[:, 1]).astype(int), 0, camera.height - 1), np.clip(np.rint(pairs.left[:, 0]).astype(int), 0, camera.width - 1)
    ]  # Network.
    aligned = raw * (stats.prior_scale if np.isfinite(stats.prior_scale) else 1.0)  # After scale alignment.
    fig = plt.figure(figsize=(15, 4.6))  # Wide layout.
    grid = fig.add_gridspec(1, 2, width_ratios=[2.6, 1])  # Image | scatter.
    ax_img, ax_sc = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])  # Axes.
    ax_img.imshow(frame.left, cmap="gray")  # Camera image.
    for mask, colour, label in (
        (trusted, GREY, "stereo trusted (no prior needed)"),
        (accepted, GREEN, "prior accepted"),
        (rejected, RED, "prior rejected (inconsistent)"),
    ):  # Classes.
        ax_img.scatter(pairs.left[mask, 0], pairs.left[mask, 1], s=9, color=colour, label=f"{label}: {int(mask.sum())}")  # Landmarks.
    ax_img.legend(loc="lower right", fontsize=8, framealpha=0.85)  # Legend.
    ax_img.set_title(f"{dataset.tag}, frame {index}: depth-prior gate (scale aligned by {stats.prior_scale:.2f})")  # Title.
    ax_img.axis("off")  # No ticks.
    weak = ~trusted  # Only weak-stereo landmarks are candidates.
    ax_sc.scatter(z[weak & accepted], aligned[weak & accepted], s=8, color=GREEN, label="accepted")  # Candidates kept.
    ax_sc.scatter(z[rejected], aligned[rejected], s=8, color=RED, label="rejected")  # Candidates dropped.
    ax_sc.scatter(z[trusted], aligned[trusted], s=5, color=GREY, alpha=0.5, label="stereo trusted")  # Not candidates.
    top = float(np.nanpercentile(z, 98))  # Plot range.
    ax_sc.plot([0, top], [0, top], "k--", lw=1)  # Perfect agreement.
    ax_sc.set_xlim(0, top)  # Range.
    ax_sc.set_ylim(0, max(top, float(np.nanmax(aligned[np.isfinite(aligned)]))) * 1.08)  # Range (data-driven).
    ax_sc.set_xlabel("stereo depth [m]")  # Axis.
    ax_sc.set_ylabel("scale-aligned network depth [m]")  # Axis.
    ax_sc.set_title("Network vs. stereo depth per landmark")  # Title.
    ax_sc.legend(fontsize=8)  # Legend.
    fig.tight_layout()  # Compact.
    _save(fig, out, f"poc_gating_{dataset.tag}")  # Write.


def scale_drift_figure(results: Path, out: Path) -> None:
    """Per-keyframe network/stereo scale ratio over both sequences.

    Args:
        results: Results root.
        out: Output folder.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.6))  # KITTI | EuRoC.
    for axis, tag in zip(axes, ("kitti_00", "euroc_MH_01_easy"), strict=True):  # Both sequences.
        rows = list(csv.DictReader((results / tag / "sparse_fused_lm_ba5_prior" / "frame_stats.csv").open()))  # Per-frame stats.
        pts = [
            (float(r["timestamp"]), float(r["prior_scale"]))
            for r in rows
            if r["prior_scale"] not in ("", "nan") and np.isfinite(float(r["prior_scale"]))
        ]  # Keyframes with a scale.
        t, scale = np.array(pts).T  # Arrays.
        t = t - t[0]  # Seconds from start.
        axis.scatter(t, scale, s=3, color=BLUE, alpha=0.35, label="per keyframe")  # Raw ratios.
        window = 25  # Smoothing window (keyframes).
        smooth = np.array([np.median(scale[max(0, i - window) : i + window + 1]) for i in range(scale.size)])  # Running median.
        axis.plot(t, smooth, color=NAVY, lw=1.8, label="running median")  # Trend.
        axis.axhline(1.0, color="k", ls="--", lw=1, label="network = stereo")  # Ideal.
        axis.set_title(
            f"{tag}: stereo / network depth ratio (median {np.median(scale):.2f}, 5-95% {np.percentile(scale, 5):.2f}-{np.percentile(scale, 95):.2f})"
        )  # Summary.
        axis.set_xlabel("time [s]")  # Axis.
        axis.set_ylabel("scale ratio")  # Axis.
        axis.legend(fontsize=8, loc="upper right")  # Legend.
    fig.tight_layout()  # Compact.
    _save(fig, out, "poc_scale_drift")  # Write.


RUNS = [  # Sparse variants compared in the drift figures: (folder, label, colour).
    ("sparse_fused_lm", "no BA (fused anchors, rig LM)", ORANGE),
    ("sparse_fused_lm_ba5", "+ windowed BA (5 KF)", GREEN),
    ("sparse_fused_lm_ba5_prior", "+ windowed BA + depth priors", NAVY),
]


def _record_windows(config: ExperimentConfig, dataset: Any, frames: int) -> list[tuple[Any, Any, list[int]]]:
    """Run the sparse pipeline with windowed BA and record every BA window (input, output, keyframe frames).

    Args:
        config: Experiment config.
        dataset: Loader.
        frames: Number of frames to process.

    Returns:
        (problem, result, frame indices) for every window that was optimised.
    """
    recorded: list[tuple[Any, Any, list[int]]] = []  # Recorded windows.
    original_ba = keyframe_window.windowed_ba  # Real optimiser.
    original_optimise = KeyframeWindow.optimise  # Real window method.
    current_frames: list[int] = []  # Frames of the window being optimised.

    def recorder(problem: Any, camera_matrix: Any, baseline: float, ba_config: Any = None) -> Any:
        """Call the optimiser and keep its input, output and keyframe frames."""
        result = original_ba(problem, camera_matrix, baseline, ba_config)  # Optimise.
        recorded.append((problem, result, list(current_frames)))  # Keep.
        return result  # Unchanged behaviour.

    def optimise(self: KeyframeWindow, *args: Any) -> Any:
        """Note the window's keyframe frames, then run the real method."""
        current_frames[:] = [kf.frame_index for kf in self._keyframes]  # Frames in this window.
        return original_optimise(self, *args)  # Unchanged behaviour.

    keyframe_window.windowed_ba = recorder  # type: ignore[assignment]  # Record without changing the pipeline.
    KeyframeWindow.optimise = optimise  # type: ignore[method-assign]  # Record window frames.
    try:
        frontend = STEREO_FRONTENDS.create(config.stereo_frontend.name, **config.stereo_frontend.params)  # Front end.
        odometry_config = config.odometry.__class__(**{**config.odometry.__dict__, "window_size": 5})  # Windowed BA on.
        odometry = SparseStereoOdometry(dataset.camera, frontend, odometry_config)  # Pipeline.
        for index in range(frames):  # Track.
            odometry.process(dataset.load_frame(index))
    finally:
        keyframe_window.windowed_ba = original_ba  # Restore.
        KeyframeWindow.optimise = original_optimise  # type: ignore[method-assign]  # Restore.
    return [r for r in recorded if len(r[0].poses) == odometry_config.window_size]  # Full windows only.


def ba_effect_figure(config: ExperimentConfig, dataset: Any, results: Path, frames: int, out: Path) -> None:
    """How windowed BA acts: drift accumulation over the sequence, and what happens inside the windows.

    Args:
        config: Experiment config.
        dataset: KITTI loader (ground truth needed).
        results: Results root (full-sequence runs).
        frames: Frames tracked for the per-window statistics.
        out: Output folder.
    """
    gt = dataset.ground_truth().poses  # T_world,cam per frame; the estimate and GT share frame 0 as origin.
    centres = np.array([g[:3, 3] for g in gt])  # GT positions.
    distance = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(centres, axis=0), axis=1))]) / 1000.0  # km travelled.
    fig, (ax_rot, ax_pos, ax_win) = plt.subplots(1, 3, figsize=(17, 4.6))  # Three panels.
    for run, label, colour in RUNS:  # Drift curves (origin-aligned: no fitting, pure accumulation).
        estimate = load_kitti_poses(results / dataset.tag / run / "trajectory.kitti")  # Line-aligned with GT.
        n = min(len(estimate), len(gt))  # Common frames.
        heading = [
            np.degrees(np.arccos(np.clip(0.5 * (np.trace(gt[i][:3, :3].T @ estimate[i][:3, :3]) - 1.0), -1.0, 1.0))) for i in range(n)
        ]  # Orientation error.
        position = [np.linalg.norm(estimate[i][:3, 3] - gt[i][:3, 3]) for i in range(n)]  # Position error.
        ax_rot.plot(distance[:n], heading, color=colour, lw=1.4, label=label)  # Heading drift.
        ax_pos.plot(distance[:n], position, color=colour, lw=1.4, label=label)  # Position drift.
    ax_rot.set_title("Accumulated orientation error (start-aligned)")  # Title.
    ax_rot.set_xlabel("distance travelled [km]")  # Axis.
    ax_rot.set_ylabel("orientation error vs. ground truth [deg]")  # Axis.
    ax_rot.legend(fontsize=8)  # Legend.
    ax_pos.set_title("Accumulated position error (start-aligned)")  # Title.
    ax_pos.set_xlabel("distance travelled [km]")  # Axis.
    ax_pos.set_ylabel("position error vs. ground truth [m]")  # Axis.
    ax_pos.legend(fontsize=8)  # Legend.
    windows = _record_windows(config, dataset, frames)  # Per-window statistics.
    camera_matrix, baseline = dataset.camera.intrinsic_matrix, dataset.camera.baseline  # Rig.
    pure = WindowBAConfig(huber_delta=1e9, pixel_sigma=1.0)  # Raw pixel residuals.
    before = np.concatenate([np.sqrt(_linearise(p, p.poses, p.points, camera_matrix, baseline, pure)[2]) for p, _, _ in windows])  # Pixels.
    after = np.concatenate([np.sqrt(_linearise(p, r.poses, r.points, camera_matrix, baseline, pure)[2]) for p, r, _ in windows])  # Pixels.

    def newest_error(poses: Any, frame_ids: list[int]) -> float:
        """Newest keyframe's position error relative to the window's fixed first keyframe, vs ground truth."""
        est = poses[0] @ invert_pose(poses[-1])  # T_kf0,kfN (poses are T_cam,world).
        ref = invert_pose(gt[frame_ids[0]]) @ gt[frame_ids[-1]]  # Ground-truth equivalent.
        return float(np.linalg.norm(est[:3, 3] - ref[:3, 3]))  # Metres.

    err_b = np.array([newest_error(p.poses, f) for p, _, f in windows])  # Before BA.
    err_a = np.array([newest_error(r.poses, f) for _, r, f in windows])  # After BA.
    bins = np.logspace(-2, 1.3, 50)  # Pixel-error bins.
    ax_win.hist(
        before, bins=bins, color=ORANGE, alpha=0.6, label=f"before BA: mean {before.mean():.3f} px, p95 {np.percentile(before, 95):.2f}"
    )  # Before.
    ax_win.hist(after, bins=bins, color=GREEN, alpha=0.6, label=f"after BA: mean {after.mean():.3f} px, p95 {np.percentile(after, 95):.2f}")  # After.
    ax_win.set_xscale("log")  # Wide range.
    ax_win.set_xlabel("stereo reprojection error (u_l, v_l, u_r) [px]")  # Axis.
    ax_win.set_ylabel("observations")  # Axis.
    ax_win.set_title(f"Inside the windows: {len(windows)} BA windows, frames 0-{frames}")  # Title.
    ax_win.legend(fontsize=8, loc="upper left")  # Legend.
    note = (
        f"newest-keyframe error vs. GT within a window:\nmedian {100 * np.median(err_b):.1f} -> {100 * np.median(err_a):.1f} cm, "
        f"better in {100 * np.mean(err_a < err_b):.0f}% of windows\n(gain is long-term: refined landmarks feed tracking)"
    )  # Honest per-window finding.
    ax_win.text(
        0.02, 0.62, note, transform=ax_win.transAxes, fontsize=8, va="top", bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"}
    )  # Annotation.
    fig.suptitle("KITTI 00: effect of the sliding-window rigid-rig bundle adjustment", y=1.02)  # Headline.
    fig.tight_layout()  # Compact.
    _save(fig, out, "poc_ba_effect")  # Write.


def trajectory_zoom_figure(config: ExperimentConfig, dataset: Any, results: Path, out: Path) -> None:
    """KITTI trajectories (left) and a zoom on the loop where windowed BA reduces drift (right).

    Args:
        config: Experiment config.
        dataset: KITTI loader.
        results: Results root.
        out: Output folder.
    """
    reference = dataset.ground_truth()  # Ground truth.
    gt = PoseTrajectory3D(poses_se3=list(reference.poses), timestamps=np.asarray(reference.timestamps))  # evo form.
    fig, (ax, zoom) = plt.subplots(1, 2, figsize=(15, 6.6), gridspec_kw={"width_ratios": [1.25, 1]})  # Full | zoom.
    gt_xyz = gt.positions_xyz  # GT positions.
    for axis in (ax, zoom):  # GT in both.
        axis.plot(gt_xyz[:, 0], gt_xyz[:, 2], "k--", lw=1.6, label="ground truth")
    for run, label, colour in RUNS:  # Estimates.
        estimate = file_interface.read_tum_trajectory_file(str(results / dataset.tag / run / "trajectory.tum"))  # Trajectory.
        ref, est = sync.associate_trajectories(gt, estimate, max_diff=0.01)  # Associate.
        aligned = copy.deepcopy(est)  # Keep original.
        aligned.align(ref, correct_scale=False)  # SE(3), as in the metric.
        xyz = aligned.positions_xyz  # Positions.
        for axis in (ax, zoom):  # Both panels.
            axis.plot(xyz[:, 0], xyz[:, 2], color=colour, lw=1.4, label=label)
    box = (-205, -65, 195, 395)  # The left loop (x0, x1, z0, z1).
    zoom.set_xlim(box[0], box[1])  # Zoom window.
    zoom.set_ylim(box[2], box[3])  # Zoom window.
    zoom.set_aspect("equal")  # Metric.
    zoom.set_title("Zoom: the left loop")  # Title.
    zoom.set_xlabel("x [m]")  # Axis.
    ax.add_patch(plt.Rectangle((box[0], box[2]), box[1] - box[0], box[3] - box[2], fill=False, ec="0.35", lw=1.2, ls=":"))  # Zoom outline.
    ax.set_aspect("equal", adjustable="datalim")  # Metric.
    ax.set_xlabel("x [m]")  # Axis.
    ax.set_ylabel("z [m]")  # Axis.
    ax.set_title("KITTI 00 (3.7 km), SE(3)-aligned, no loop closure")  # Title.
    ax.legend(loc="upper left", fontsize=9)  # Legend.
    fig.tight_layout()  # Compact.
    _save(fig, out, "poc_trajectory_zoom")  # Write.


def main() -> None:
    """Generate all POC demo figures."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])  # CLI.
    parser.add_argument("--out", type=Path, default=Path("docs/poc/figures"))  # Output folder.
    parser.add_argument("--results", type=Path, default=Path("results"))  # Results root.
    parser.add_argument("--kitti", required=True)  # KITTI root.
    parser.add_argument("--euroc", required=True)  # EuRoC scene root.
    parser.add_argument("--ba-frames", type=int, default=1500)  # Frames tracked for the per-window BA statistics.
    args = parser.parse_args()  # Parse.
    args.out.mkdir(parents=True, exist_ok=True)  # Ensure the folder exists.
    kitti_cfg, kitti = _load("configs/kitti.yaml", str(Path(args.kitti).expanduser()))  # KITTI.
    euroc_cfg, euroc = _load("configs/euroc.yaml", str(Path(args.euroc).expanduser()))  # EuRoC.
    for config, dataset, index in ((kitti_cfg, kitti, 300), (euroc_cfg, euroc, 1200)):  # Per-dataset qualitative figures.
        network = DEPTH_ESTIMATORS.create(config.prior_depth.name, **config.prior_depth.params)  # Indoor/outdoor model.
        depth_figure(config, dataset, index, network, args.out)
        gating_figure(config, dataset, index, network, args.out)
    scale_drift_figure(args.results, args.out)  # From the POC runs.
    trajectory_zoom_figure(kitti_cfg, kitti, args.results, args.out)  # From the POC runs.
    ba_effect_figure(kitti_cfg, kitti, args.results, args.ba_frames, args.out)  # Runs the pipeline for the window statistics.


if __name__ == "__main__":  # Script entry.
    main()  # Run.
