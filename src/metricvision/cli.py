"""Command-line entry point: ``metricvision run <config>`` and ``metricvision list``."""

from __future__ import annotations  # Postponed evaluation of type hints.

import argparse  # Argument parsing from the standard library.
import json  # Pretty-printing the summary.
from collections.abc import Sequence  # argv type hint.
from pathlib import Path  # Path arguments.

from metricvision.config import ExperimentConfig  # Experiment settings.
from metricvision.datasets import DATASETS  # Dataset registry.
from metricvision.depth import DEPTH_ESTIMATORS  # Depth registry.
from metricvision.evaluation.writers import sanitise_for_json  # NaN-safe JSON conversion.
from metricvision.frontend import MATCHERS, STEREO_FRONTENDS  # Front-end registries.
from metricvision.pipeline import ExperimentRunner  # Experiment runner.


def build_parser() -> argparse.ArgumentParser:
    """Define the command-line interface.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(prog="metricvision", description="Benchmark VO front ends with a fixed stereo back end.")  # Top-level parser.
    commands = parser.add_subparsers(dest="command", required=True)  # Sub-commands.

    run = commands.add_parser("run", help="Run one experiment from a YAML config.")  # `metricvision run`.
    run.add_argument("config", type=Path, help="Path to the experiment YAML file.")  # Required config path.
    run.add_argument("--matcher", help="Override the front end (e.g. orb, sift).")  # Swap extractor without editing YAML.
    run.add_argument("--calibration", help="Rig calibration YAML for the wheelchair bags (default: from the config).")
    run.add_argument("--stereo-frontend", help="Sparse pipeline front end: fast_orb_klt (classical) or superpoint_lightglue (learned).")
    run.add_argument("--dataset-root", help="Override the dataset root folder.")  # Point at data elsewhere.
    run.add_argument("--max-frames", type=int, help="Process at most this many frames.")  # Quick tests.
    run.add_argument("--output-dir", type=Path, help="Override the results root folder.")  # Redirect outputs.
    run.add_argument("--pipeline", choices=["dense", "sparse"], help="Override the VO pipeline.")  # Dense SGBM or sparse stereo.
    run.add_argument("--refine", choices=["none", "opencv", "lm"], help="Override the PnP refinement method.")  # LM ablation.
    run.add_argument(
        "--sequence", help="Override the dataset sequence (KITTI: 00-21; EuRoC scenes are chosen with --dataset-root)."
    )  # Other sequences.
    run.add_argument("--depth", help="Override the dense pipeline depth source (sgbm, depth_anything_v2 = monocular metric VO).")  # Mono VO.
    run.add_argument("--window", type=int, help="Sparse pipeline: windowed BA size in keyframes (0 disables).")  # BA.
    run.add_argument(
        "--depth-prior", action=argparse.BooleanOptionalAction, default=None, help="Sparse pipeline: gated monocular depth priors in the BA."
    )  # Priors.
    run.add_argument(
        "--dynamic-mask", action=argparse.BooleanOptionalAction, default=None, help="Mask people and vehicles (SegFormer, Cityscapes)."
    )  # Dynamics.
    run.add_argument(
        "--dynamic-gate",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Sparse pipeline: drop tracks inconsistent with their landmark (motion/occlusion).",
    )  # Dynamics.
    run.add_argument("--anchor", choices=["previous", "birth", "fused"], help="Override sparse-pipeline point anchoring.")  # Anchoring ablation.
    run.add_argument("--run-name", help="Output folder name (defaults to the front-end name).")  # Keep ablations apart.
    run.add_argument("--no-progress", action="store_true", help="Hide the progress bar.")  # For logs/CI.

    commands.add_parser("list", help="List registered datasets, matchers and depth estimators.")  # `metricvision list`.
    return parser  # Ready to parse.


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface.

    Args:
        argv: Arguments (defaults to sys.argv[1:]).

    Returns:
        Process exit code (0 on success).
    """
    args = build_parser().parse_args(argv)  # Parse arguments.
    if args.command == "list":  # Show what can be selected in configs.
        print("datasets:         ", ", ".join(DATASETS.names()))  # Dataset keys.
        print("matchers:         ", ", ".join(MATCHERS.names()))  # Pairwise front-end keys (dense pipeline).
        print("stereo front ends:", ", ".join(STEREO_FRONTENDS.names()))  # Sparse front-end keys (sparse pipeline).
        print("depth estimators: ", ", ".join(DEPTH_ESTIMATORS.names()))  # Depth keys.
        return 0  # Success.
    config = ExperimentConfig.from_yaml(args.config).with_overrides(  # Load YAML, then apply CLI overrides.
        matcher=args.matcher,  # Optional extractor swap.
        dataset_root=args.dataset_root,  # Optional data location.
        max_frames=args.max_frames,  # Optional frame cap.
        output_dir=args.output_dir,  # Optional results folder.
        pipeline=args.pipeline,  # Optional pipeline swap.
        refine=args.refine,  # Optional refinement swap.
        run_name=args.run_name,  # Optional folder name.
        anchor=args.anchor,  # Optional anchoring swap.
        sequence=args.sequence,  # Optional sequence swap.
        depth=args.depth,  # Optional depth source.
        window=args.window,  # Optional BA size.
        depth_prior=args.depth_prior,  # Optional priors.
        stereo_frontend=args.stereo_frontend,  # Optional classical/learned front end.
        calibration=args.calibration,  # Optional rig file (wheelchair bags).
        dynamic_mask=args.dynamic_mask,  # Optional masking.
        dynamic_gate=args.dynamic_gate,  # Optional gate.
    )
    runner = ExperimentRunner(config)  # Build all components.
    summary = runner.run(show_progress=not args.no_progress)  # Run the experiment.
    print(json.dumps(sanitise_for_json(summary.to_dict()), indent=2))  # Show the summary.
    print(f"Results written to {runner.output_dir}")  # Tell the user where to look.
    return 0  # Success.


if __name__ == "__main__":  # Allow `python -m metricvision.cli`.
    raise SystemExit(main())  # Propagate the exit code.
