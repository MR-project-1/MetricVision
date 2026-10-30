"""Paired statistical comparison of runs against a reference run (Wilcoxon signed-rank + block-bootstrap CI).

KITTI: paired samples are the devkit segments (start frame, length), errors in % (translation) and deg/100 m
(rotation). Other datasets: paired samples are 1-second relative translation errors at each ground-truth-associated
frame (blocks of 1 s for the bootstrap).

Usage:
    uv run python scripts/compare_runs.py --results results_abl --reference abl_ref \
        --config "configs/kitti.yaml:$KITTI_ROOT" --config "configs/euroc.yaml:$EUROC_ROOT" --out docs_abl/tables
Writes stats_<tag>.md and stats_<tag>.json (negative mean difference = better than the reference).
"""

from __future__ import annotations  # Postponed evaluation of type hints.

import argparse  # CLI.
import json  # Output.
from functools import partial  # Bound sample collectors.
from pathlib import Path  # Paths.

import numpy as np  # Arrays.
from evo.core import sync  # Association.
from evo.tools import file_interface  # TUM reader.

from metricvision.config import ExperimentConfig  # Dataset construction.
from metricvision.core.geometry import invert_pose  # Poses.
from metricvision.datasets import DATASETS  # Registry.
from metricvision.evaluation.kitti_metrics import kitti_segment_error_list, load_kitti_poses  # Devkit.
from metricvision.evaluation.statistics import paired_comparison  # Tests.
from metricvision.evaluation.trajectory_metrics import _to_evo  # Trajectory conversion.


def _kitti_samples(run_dir: Path, gt_poses: list) -> dict[tuple[int, int], tuple[float, float]]:
    """Per-segment devkit errors keyed by (first frame, length)."""
    est = load_kitti_poses(run_dir / "trajectory.kitti")
    segs = kitti_segment_error_list(est, gt_poses[: len(est)])
    return {(f, length): (100.0 * t, 100.0 * np.degrees(r)) for f, length, t, r in segs}


def _rpe1s_samples(run_dir: Path, gt) -> dict[int, tuple[float, float]]:  # noqa: ANN001
    """1-second relative errors keyed by the ground-truth timestamp (in ms)."""
    est = file_interface.read_tum_trajectory_file(str(run_dir / "trajectory.tum"))
    ref, est = sync.associate_trajectories(_to_evo(gt), est, max_diff=0.01)
    n = max(1, int(round(1.0 / float(np.median(np.diff(ref.timestamps))))))
    out = {}
    for i in range(ref.num_poses - n):
        d_gt = invert_pose(ref.poses_se3[i]) @ ref.poses_se3[i + n]
        d_est = invert_pose(est.poses_se3[i]) @ est.poses_se3[i + n]
        e = invert_pose(d_est) @ d_gt
        angle = np.degrees(np.arccos(np.clip(0.5 * (np.trace(e[:3, :3]) - 1.0), -1.0, 1.0)))
        out[int(round(ref.timestamps[i] * 1000))] = (float(np.linalg.norm(e[:3, 3])), float(angle))
    return out


def main() -> None:
    """Compare every run with the reference."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--reference", required=True, help="reference run folder name")
    parser.add_argument("--config", action="append", default=[], help="<config.yaml>:<dataset root>")
    parser.add_argument("--out", type=Path, default=Path("docs/tables"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for spec in args.config:
        config_path, _, root = spec.partition(":")
        config = ExperimentConfig.from_yaml(config_path).with_overrides(dataset_root=root or None)
        dataset = DATASETS.create(config.dataset.name, **config.dataset.params)
        tag, gt = dataset.tag, dataset.ground_truth()
        folder = args.results / tag
        if gt is None or not (folder / args.reference).is_dir():
            print(f"skipping {tag}: no ground truth or no reference run")
            continue
        kitti = tag.startswith("kitti")
        gt_poses = load_kitti_poses(Path(root) / "poses" / f"{tag.split('_')[1]}.txt") if kitti else None
        collect = partial(_kitti_samples, gt_poses=gt_poses) if kitti else partial(_rpe1s_samples, gt=gt)
        reference = collect(folder / args.reference)
        unit = ("t_rel [%]", "r_rel [deg/100m]") if kitti else ("RPE_1s trans [m]", "RPE_1s rot [deg]")
        rows = []
        for run_dir in sorted(p for p in folder.iterdir() if (p / "summary.json").is_file() and p.name != args.reference):
            samples = collect(run_dir)
            keys = sorted(set(reference) & set(samples))
            if not keys:
                continue
            blocks = np.array([k[0] if kitti else k // 1000 for k in keys])
            row = {"run": run_dir.name, "samples": len(keys)}
            for col, name in enumerate(("translation", "rotation")):
                a = np.array([reference[k][col] for k in keys])
                b = np.array([samples[k][col] for k in keys])
                row[name] = paired_comparison(a, b, blocks).to_dict()
            rows.append(row)
        (args.out / f"stats_{tag}.json").write_text(json.dumps({"reference": args.reference, "units": unit, "rows": rows}, indent=2))
        lines = [
            f"### {tag}: paired comparison against `{args.reference}` (negative = better; 95 % block-bootstrap CI; Wilcoxon p)",
            "",
            f"| Run | {unit[0]} ref | run | diff [95 % CI] | p | win | {unit[1]} diff [95 % CI] | p |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in rows:
            t, q = r["translation"], r["rotation"]
            lines.append(
                f"| {r['run']} | {t['mean_a']:.4g} | {t['mean_b']:.4g} | {t['mean_diff']:+.3g} [{t['ci_low']:+.3g}, {t['ci_high']:+.3g}] | "
                f"{t['p_value']:.2g} | {100 * t['win_rate']:.0f} % | {q['mean_diff']:+.3g} [{q['ci_low']:+.3g}, {q['ci_high']:+.3g}] | {q['p_value']:.2g} |"
            )
        (args.out / f"stats_{tag}.md").write_text("\n".join(lines) + "\n")
        print("\n".join(lines))


if __name__ == "__main__":
    main()
