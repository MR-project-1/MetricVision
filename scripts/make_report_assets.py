"""Turn ``results/`` into the tables and figures used by the report (docs/).

Usage:
    uv run python scripts/make_report_assets.py --results results --out docs \
        --config configs/kitti.yaml:/data/kitti --config configs/euroc.yaml:/data/euroc/MH_01_easy

For every ``results/<dataset>/<run>/`` it reads ``summary.json`` (ATE/RPE, timings),
adds the KITTI devkit segment errors for KITTI sequences, and writes
``docs/tables/<dataset>.{md,tex}`` plus trajectory, error-vs-length and per-frame figures
to ``docs/figures/``.
"""

from __future__ import annotations  # Postponed evaluation of type hints.

import argparse  # Command-line arguments.
import copy  # Deep copies before alignment.
import csv  # Per-frame statistics.
import json  # Summaries.
from pathlib import Path  # Paths.
from typing import Any  # Loosely typed JSON.

import matplotlib  # Plotting backend selection.

matplotlib.use("Agg")  # Headless rendering.
import matplotlib.pyplot as plt  # noqa: E402  # Plotting (after the backend is chosen).
import numpy as np  # noqa: E402  # Arrays.
from evo.core import sync  # noqa: E402  # Timestamp association.
from evo.core.trajectory import PoseTrajectory3D  # noqa: E402  # evo trajectory type.
from evo.tools import file_interface  # noqa: E402  # TUM reader.

from metricvision.config import ExperimentConfig  # noqa: E402  # Dataset construction from configs.
from metricvision.datasets import DATASETS  # noqa: E402  # Loaders (for ground truth).
from metricvision.evaluation.kitti_metrics import SEGMENT_LENGTHS_M, kitti_segment_errors, load_kitti_poses  # noqa: E402  # Devkit metric.

LABELS = {  # Human-readable run names (fallback: the folder name).
    "dense_orb_opencv": "Dense ORB+SGBM, OpenCV-LM",
    "dense_orb_lm": "Dense ORB+SGBM, our LM (left only)",
    "dense_sift_lm": "Dense SIFT+SGBM, our LM (left only)",
    "sparse_f2f_none": "Sparse f2f, PnP only",
    "sparse_f2f_opencv": "Sparse f2f, OpenCV-LM",
    "sparse_f2f_lm": "Sparse f2f, rig-LM",
    "sparse_birth_lm": "Sparse birth anchors, rig-LM",
    "sparse_fused_opencv": "Sparse fused anchors, OpenCV-LM",
    "sparse_fused_lm": "Sparse fused anchors, rig-LM (proposed)",
    "abl_ref": "Ablation ref: fused, rig-LM, shared noise",
    "abl_mono": "- right-image residuals",
    "abl_nocov": "- covariance weighting",
    "abl_nocov_gate40": "- covariance weighting + 40B depth gate",
    "abl_nohuber": "- Huber kernel",
    "abl_sigma1": "LM noise 1 px (mid-report)",
    "abl_independent": "Independent L/R poses (12 DoF)",
    "abl_averaged": "Averaged L/R poses",
    "abl_ransac_stereo": "+ stereo AP3P-MSAC RANSAC",
    "abl_invdepth": "+ inverse-depth anchors",
    "abl_inflate": "+ pose-uncertainty inflation",
    "abl_invdepth_inflate": "+ inverse depth + inflation",
    "abl_adaptive": "+ adaptive noise model",
    "abl_realign5": "+ birth-template re-alignment",
    "abl_klt_stereo": "+ KLT stereo matching",
    "full": "Full proposal",
    "full_recalib": "Full proposal, recalibrated KITTI",
    "abl_f2f": "Frame-to-frame (no anchors), rig-LM",
    "abl_ba5": "+ sliding-window BA (5 KF)",
    "abl_ba7": "+ sliding-window BA (7 KF)",
    "abl_umeyama3d": "3D-3D Umeyama (same inliers)",
    "abl_mahalanobis3d": "3D-3D Mahalanobis (same inliers)",
    "abl_soft": "SOFT-style R/t decoupling (same inliers)",
    "abl_replenish050": "Replenish below 50 % of tracks",
    "abl_replenish035": "Replenish below 35 % of tracks",
    "final": "Final configuration",
    "ext_libviso2": "libviso2 (external)",
    "ext_orbslam3": "ORB-SLAM3 stereo VO (external)",
    "ext_basalt": "Basalt VO (external)",
}
ORDER = list(LABELS)  # Table/legend order.
KEY_RUNS = ("dense_orb_lm", "sparse_f2f_lm", "sparse_fused_lm")  # Uncluttered comparison panel.
_PALETTE = ["#9e9e9e", "#1f77b4", "#17becf", "#ffbb78", "#ff7f0e", "#d62728", "#9467bd", "#8c564b", "#2ca02c"]  # Mid-report runs.
COLOURS = {run: (_PALETTE[i] if i < len(_PALETTE) else plt.get_cmap("tab20")(i - len(_PALETTE))) for i, run in enumerate(ORDER)}  # noqa: E501  # One colour per method in every plot.


def _colour(run: str) -> str | None:
    """Return the plot colour of a run (None lets Matplotlib choose).

    Args:
        run: Folder name.

    Returns:
        Colour string or None.
    """
    return COLOURS.get(run)  # Fixed palette for known runs.


def _smooth(values: np.ndarray, window: int = 25) -> np.ndarray:
    """Centred moving average for readable per-frame curves.

    Args:
        values: Per-frame values.
        window: Window length in frames.

    Returns:
        Smoothed values (same length).
    """
    if len(values) < window:  # Too short to smooth.
        return values  # Unchanged.
    kernel = np.ones(window) / window  # Box filter.
    return np.convolve(np.pad(values, (window // 2, window - 1 - window // 2), mode="edge"), kernel, mode="valid")  # Same-length average.


def _label(run: str) -> str:
    """Return the display label of a run.

    Args:
        run: Folder name.

    Returns:
        Label.
    """
    return LABELS.get(run, run)  # Fallback to the folder name.


def _runs(dataset_dir: Path) -> list[Path]:
    """List run folders in report order.

    Args:
        dataset_dir: ``results/<dataset>``.

    Returns:
        Run folders that contain a summary.
    """
    runs = [p for p in dataset_dir.iterdir() if (p / "summary.json").is_file()]  # Completed runs only.
    return sorted(runs, key=lambda p: (ORDER.index(p.name) if p.name in ORDER else len(ORDER), p.name))  # Known runs first.


def _ground_truth(config_path: str, root: str) -> tuple[str, Any, Any]:
    """Build the dataset for a config and return its tag, ground truth and loader.

    Args:
        config_path: Experiment YAML.
        root: Dataset root.

    Returns:
        (tag, ground-truth Trajectory, dataset).
    """
    config = ExperimentConfig.from_yaml(config_path).with_overrides(
        dataset_root=root or None
    )  # Same loader settings as the runs (empty root: keep the config).
    dataset = DATASETS.create(config.dataset.name, **config.dataset.params)  # Loader.
    return dataset.tag, dataset.ground_truth(), dataset  # Everything needed for plots.


def _to_evo(trajectory: Any) -> PoseTrajectory3D:
    """Convert a metricvision Trajectory to evo.

    Args:
        trajectory: metricvision Trajectory.

    Returns:
        evo trajectory.
    """
    return PoseTrajectory3D(poses_se3=list(trajectory.poses), timestamps=np.asarray(trajectory.timestamps))  # Direct conversion.


def _fmt(value: Any, digits: int) -> str:
    """Format a number for tables (``-`` for missing).

    Args:
        value: Number or None.
        digits: Decimal places.

    Returns:
        Formatted string.
    """
    return "-" if value is None or (isinstance(value, float) and not np.isfinite(value)) else f"{value:.{digits}f}"  # Missing-safe.


def _table(rows: list[dict[str, Any]], kitti: bool) -> tuple[str, str]:
    """Render the results table as Markdown and LaTeX.

    Args:
        rows: One dict per run.
        kitti: Whether to include devkit columns.

    Returns:
        (markdown, latex).
    """
    head = ["Method", "ATE RMSE [m]", "RPE trans [m]", "RPE rot [deg]", "RPE 1s trans [m]", "ANEES"]  # Common columns.
    if kitti:  # Devkit columns.
        head += ["t_rel [%]", "r_rel [deg/100m]"]
    head += ["Fail", "Inliers", "Time [ms]", "FPS"]  # Robustness and runtime.
    body = []  # Table rows.
    for r in rows:  # One run per row.
        cells = [_label(r["run"]), _fmt(r["ate"], 3), _fmt(r["rpe_t"], 4), _fmt(r["rpe_r"], 3), _fmt(r.get("rpe_1s"), 4), _fmt(r.get("anees"), 2)]  # Accuracy, consistency.
        if kitti:
            cells += [_fmt(r.get("t_rel"), 2), _fmt(r.get("r_rel"), 3)]  # Devkit.
        cells += [str(r["fail"]), _fmt(r["inliers"], 0), _fmt(r["ms"], 1), _fmt(r["fps"], 1)]  # Robustness and runtime.
        body.append(cells)  # Store.
    md = "| " + " | ".join(head) + " |\n|" + "---|" * len(head) + "\n" + "".join("| " + " | ".join(c) + " |\n" for c in body)  # Markdown.
    tex_head = [h.replace("%", r"\%").replace("_", r"\_") for h in head]  # Escape LaTeX specials.
    tex = (  # booktabs table body (wrapped in a table environment by the report).
        r"\begin{tabular}{l"
        + "r" * (len(head) - 1)
        + "}\n\\toprule\n"
        + " & ".join(tex_head)
        + r" \\"
        + "\n\\midrule\n"
        + "".join(" & ".join(c) + r" \\" + "\n" for c in body)
        + "\\bottomrule\n\\end{tabular}\n"
    )
    return md, tex  # Both formats.


def _combined(all_rows: dict[str, list[dict[str, Any]]]) -> tuple[str, str]:
    """Render one compact table with KITTI and EuRoC columns side by side.

    Args:
        all_rows: Rows per dataset tag (one KITTI and one EuRoC tag).

    Returns:
        (markdown, latex).
    """
    kitti_tag = next(t for t in all_rows if t.startswith("kitti"))  # KITTI tag.
    euroc_tag = next(t for t in all_rows if not t.startswith("kitti"))  # EuRoC tag.
    euroc = {r["run"]: r for r in all_rows[euroc_tag]}  # Lookup by run.
    head = ["Method", "ATE [m]", "t_rel [%]", "r_rel [deg/100m]", "RPE_t [cm]", "ms", "ATE [cm]", "RPE_t [mm]", "ms"]  # Columns.
    body = []  # Rows.
    for r in all_rows[kitti_tag]:  # KITTI order.
        e = euroc.get(r["run"], {})  # Matching EuRoC run.
        body.append(
            [
                _label(r["run"]),
                _fmt(r["ate"], 2),
                _fmt(r.get("t_rel"), 2),
                _fmt(r.get("r_rel"), 2),
                _fmt(100 * r["rpe_t"], 2),
                _fmt(r["ms"], 0),
                _fmt(100 * e["ate"], 1) if e else "-",
                _fmt(1000 * e["rpe_t"], 1) if e else "-",
                _fmt(e.get("ms"), 0),
            ]
        )
    md = (
        f"| | {kitti_tag} | | | | | {euroc_tag} | | |\n"
        + "| "
        + " | ".join(head)
        + " |\n|"
        + "---|" * len(head)
        + "\n"
        + "".join("| " + " | ".join(c) + " |\n" for c in body)
    )
    tex_head = [h.replace("%", r"\%").replace("_", r"\_") for h in head]  # Escape.
    tex = (
        r"\begin{tabular}{lrrrrr|rrr}"
        + "\n\\toprule\n"
        + r" & \multicolumn{5}{c|}{KITTI 00 (4541 frames, 3.7 km)} & \multicolumn{3}{c}{EuRoC MH\_01 (3682 frames)} \\"
        + "\n"
        + " & ".join(tex_head)
        + r" \\"
        + "\n\\midrule\n"
        + "".join(" & ".join(c) + r" \\" + "\n" for c in body)
        + "\\bottomrule\n\\end{tabular}\n"
    )
    return md, tex  # Both formats.


def main() -> None:
    """Generate all report assets."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])  # CLI.
    parser.add_argument("--results", type=Path, default=Path("results"))  # Results root.
    parser.add_argument("--out", type=Path, default=Path("docs"))  # Output root.
    parser.add_argument("--config", action="append", default=[], help="<config.yaml>:<dataset root>, once per dataset.")  # Datasets.
    args = parser.parse_args()  # Parse.
    (args.out / "figures").mkdir(parents=True, exist_ok=True)  # Figure folder.
    (args.out / "tables").mkdir(parents=True, exist_ok=True)  # Table folder.
    all_rows: dict[str, list[dict[str, Any]]] = {}  # For results.json.

    for spec in args.config:  # One dataset per --config.
        config_path, root = spec.split(":", 1)  # Split spec.
        tag, gt, dataset = _ground_truth(config_path, root)  # Ground truth.
        dataset_dir = args.results / tag  # Runs of this dataset.
        if gt is None or not dataset_dir.is_dir():  # Nothing to report.
            print(f"skipping {tag}: no ground truth or no results")  # Inform.
            continue
        kitti = tag.startswith("kitti")  # Devkit applies.
        gt_poses = load_kitti_poses(Path(root) / "poses" / f"{tag.split('_')[1]}.txt") if kitti else None  # Line-aligned GT.
        rows: list[dict[str, Any]] = []  # Table rows.
        fig_traj, (ax_traj, ax_key) = plt.subplots(1, 2, figsize=(12, 5.4))  # Top view: all methods | key methods.
        fig_height, ax_height = plt.subplots(figsize=(8, 3.0))  # Vertical coordinate over time.
        up = (1, -1.0) if kitti else (2, 1.0)  # Vertical axis: KITTI camera y points down; EuRoC world z points up.
        gt_xyz = np.array([p[:3, 3] for p in gt.poses])  # GT positions.
        plane = (0, 2) if kitti else (0, 1)  # KITTI: x-z (camera frame); EuRoC: x-y (world).
        for axis in (ax_traj, ax_key):  # Ground truth in both panels.
            axis.plot(gt_xyz[:, plane[0]], gt_xyz[:, plane[1]], "k--", lw=1.5, label="Ground truth")  # GT.
        ax_height.plot(np.asarray(gt.timestamps) - gt.timestamps[0], up[1] * gt_xyz[:, up[0]], "k--", lw=1.5, label="Ground truth")  # GT height.
        fig_len, ax_len = plt.subplots(1, 2, figsize=(9, 3.4)) if kitti else (None, None)  # Devkit per length.
        fig_frames, ax_frames = plt.subplots(2, 1, figsize=(8, 5), sharex=True)  # Per-frame stats.
        fig_ape, ax_ape = plt.subplots(figsize=(8, 3.4))  # Position error along the trajectory.
        timing: list[tuple[str, float, float, float, float]] = []  # (run, match, pose, depth, other) mean ms.
        for run_dir in _runs(dataset_dir):  # Every run.
            summary = json.loads((run_dir / "summary.json").read_text())  # Aggregates.
            errors = summary.get("errors") or {}  # Accuracy.
            row = {  # Table row.
                "run": run_dir.name,
                "ate": errors.get("ate_rmse_m"),
                "rpe_t": errors.get("rpe_trans_rmse_m"),
                "rpe_r": errors.get("rpe_rot_rmse_deg"),
                "fail": summary["num_tracking_failures"],
                "inliers": summary["mean_pnp_inliers"],
                "ms": summary["mean_total_ms"],
                "fps": summary["fps"],
                "frames": summary["num_frames"],
                "rpe_1s": errors.get("rpe_1s_trans_rmse_m"),
                "anees": (summary.get("consistency") or {}).get("anees"),
            }
            estimate = file_interface.read_tum_trajectory_file(str(run_dir / "trajectory.tum"))  # Body-frame estimate.
            ref, est = sync.associate_trajectories(_to_evo(gt), estimate, max_diff=0.01)  # Associate.
            aligned = copy.deepcopy(est)  # Keep the original.
            aligned.align(ref, correct_scale=False)  # SE(3) alignment, as in the metric.
            xyz = aligned.positions_xyz  # Aligned positions.
            colour = _colour(run_dir.name)  # Method colour.
            ax_traj.plot(xyz[:, plane[0]], xyz[:, plane[1]], lw=1.0, color=colour, label=_label(run_dir.name))  # Estimate.
            if run_dir.name in KEY_RUNS:  # Key methods also in the uncluttered panel and the height plot.
                ax_key.plot(xyz[:, plane[0]], xyz[:, plane[1]], lw=1.6, color=colour, label=_label(run_dir.name))  # Estimate.
                ax_height.plot(ref.timestamps - gt.timestamps[0], up[1] * xyz[:, up[0]], lw=1.2, color=colour, label=_label(run_dir.name))  # Height.
            travelled = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(ref.positions_xyz, axis=0), axis=1))])  # Ground-truth distance.
            ape = np.linalg.norm(xyz - ref.positions_xyz, axis=1)  # Position error after alignment.
            ax_ape.plot(travelled, ape, lw=1.0, color=colour, label=_label(run_dir.name))  # Error vs distance.
            if kitti and gt_poses is not None:  # Devkit metric on line-aligned poses.
                est_poses = load_kitti_poses(run_dir / "trajectory.kitti")  # Estimated poses.
                gt_used = gt_poses[: len(est_poses)]  # Same frames (runs may be capped).
                devkit = kitti_segment_errors(est_poses, gt_used)  # Average over 100-800 m.
                row["t_rel"], row["r_rel"] = devkit.translation_percent, devkit.rotation_deg_per_100m  # Store.
                per_len = [kitti_segment_errors(est_poses, gt_used, [length]) for length in SEGMENT_LENGTHS_M]  # Per length.
                ax_len[0].plot(SEGMENT_LENGTHS_M, [e.translation_percent for e in per_len], "o-", color=colour, label=_label(run_dir.name))  # t_rel.
                ax_len[1].plot(
                    SEGMENT_LENGTHS_M, [e.rotation_deg_per_100m for e in per_len], "o-", color=colour, label=_label(run_dir.name)
                )  # r_rel.
            with (run_dir / "frame_stats.csv").open() as handle:  # Per-frame stats.
                stats = list(csv.DictReader(handle))  # Rows.
            index = np.array([int(s["frame_index"]) for s in stats])  # Frames.
            ax_frames[0].plot(
                index, _smooth(np.array([float(s["num_pnp_inliers"]) for s in stats])), lw=0.9, color=colour, label=_label(run_dir.name)
            )  # Inliers.
            ax_frames[1].plot(
                index, _smooth(np.array([float(s["total_ms"]) for s in stats])), lw=0.9, color=colour, label=_label(run_dir.name)
            )  # Runtime.
            tracked = stats[1:]  # Exclude initialisation.
            means = [float(np.mean([float(s[key]) for s in tracked])) for key in ("match_ms", "pose_ms", "depth_ms", "total_ms")]  # Stage means.
            timing.append((run_dir.name, means[0], means[1], means[2], max(means[3] - sum(means[:3]), 0.0)))  # Remainder = bookkeeping.
            rows.append(row)  # Collect.
        for axis, title in ((ax_traj, "all methods"), (ax_key, "key methods")):  # Shared styling.
            axis.set_aspect("equal", adjustable="datalim")  # Metric aspect.
            axis.set_xlabel("x [m]")
            axis.set_ylabel("z [m]" if kitti else "y [m]")
            axis.legend(fontsize=6 if axis is ax_traj else 8)
            axis.set_title(f"{tag}: top view, {title} (SE(3)-aligned)", fontsize=10)
        fig_traj.tight_layout()
        fig_traj.savefig(args.out / "figures" / f"{tag}_trajectory.pdf")  # Vector for LaTeX.
        fig_traj.savefig(args.out / "figures" / f"{tag}_trajectory.png", dpi=150)  # Raster for Markdown.
        ax_height.set_xlabel("time [s]")
        ax_height.set_ylabel("height [m]")
        ax_height.set_title(f"{tag}: vertical position over time (SE(3)-aligned)", fontsize=10)
        ax_height.legend(fontsize=7)
        fig_height.tight_layout()
        fig_height.savefig(args.out / "figures" / f"{tag}_height.pdf")
        fig_height.savefig(args.out / "figures" / f"{tag}_height.png", dpi=150)
        ax_frames[0].set_ylabel("PnP inliers (25-frame mean)")
        ax_frames[1].set_ylabel("time / frame [ms] (25-frame mean)")
        ax_frames[1].set_xlabel("frame")
        ax_frames[1].set_yscale("log")
        ax_frames[0].legend(fontsize=7, ncol=2)
        fig_frames.tight_layout()
        fig_frames.savefig(args.out / "figures" / f"{tag}_per_frame.pdf")
        fig_frames.savefig(args.out / "figures" / f"{tag}_per_frame.png", dpi=150)
        ax_ape.set_xlabel("distance travelled [m]")
        ax_ape.set_ylabel("position error [m]")
        ax_ape.set_title(f"{tag}: absolute position error along the trajectory (SE(3)-aligned)")
        ax_ape.legend(fontsize=7, ncol=2)
        fig_ape.tight_layout()
        fig_ape.savefig(args.out / "figures" / f"{tag}_ape_vs_distance.pdf")
        fig_ape.savefig(args.out / "figures" / f"{tag}_ape_vs_distance.png", dpi=150)
        fig_time, ax_time = plt.subplots(figsize=(8, 0.45 * len(timing) + 1.8))  # Runtime breakdown (room for the legend below).
        names = [_label(t[0]) for t in timing]  # Bar labels.
        left = np.zeros(len(timing))  # Stack offsets.
        for k, (stage, colour) in enumerate(
            (
                ("front end (match/track)", "#4c72b0"),
                ("pose (PnP + refine)", "#dd8452"),
                ("stereo (SGBM / detection)", "#55a868"),
                ("other", "#c7c7c7"),
            )
        ):  # Stages.
            widths = np.array([t[k + 1] for t in timing])  # Stage times.
            ax_time.barh(names, widths, left=left, color=colour, label=stage)  # Stacked bars.
            left += widths  # Advance.
        ax_time.invert_yaxis()  # Table order top to bottom.
        ax_time.set_xlabel("mean time per frame [ms]")
        ax_time.set_title(f"{tag}: runtime breakdown")
        ax_time.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=4, frameon=False)  # Below the axes, never over bars.
        fig_time.tight_layout()
        fig_time.savefig(args.out / "figures" / f"{tag}_runtime.pdf")
        fig_time.savefig(args.out / "figures" / f"{tag}_runtime.png", dpi=150)
        if kitti and fig_len is not None:  # Devkit plot.
            ax_len[0].set_xlabel("segment length [m]")
            ax_len[0].set_ylabel("translation error [%]")
            ax_len[1].set_xlabel("segment length [m]")
            ax_len[1].set_ylabel("rotation error [deg/100 m]")
            ax_len[0].legend(fontsize=6)
            fig_len.tight_layout()
            fig_len.savefig(args.out / "figures" / f"{tag}_devkit.pdf")
            fig_len.savefig(args.out / "figures" / f"{tag}_devkit.png", dpi=150)
        plt.close("all")  # Free memory.
        md, tex = _table(rows, kitti)  # Render tables.
        (args.out / "tables" / f"{tag}.md").write_text(md)  # Markdown.
        (args.out / "tables" / f"{tag}.tex").write_text(tex)  # LaTeX.
        all_rows[tag] = rows  # Keep.
        print(md)  # Show.
    if len(all_rows) == 2:  # Combined KITTI + EuRoC table for the report.
        (args.out / "tables" / "combined.md").write_text(_combined(all_rows)[0])  # Markdown.
        (args.out / "tables" / "combined.tex").write_text(_combined(all_rows)[1])  # LaTeX.
    (args.out / "tables" / "results.json").write_text(json.dumps(all_rows, indent=2, default=float) + "\n")  # Machine-readable.


if __name__ == "__main__":  # Script entry.
    main()  # Run.
