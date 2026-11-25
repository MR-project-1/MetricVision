"""Runs one experiment end to end: load data, run VO, evaluate, write results."""

from __future__ import annotations  # Postponed evaluation of type hints.

import math  # Finite-value checks.
from collections.abc import Sequence  # Input type hints.
from dataclasses import asdict, dataclass  # Summary container.
from pathlib import Path  # Output paths.
from typing import Any  # Loosely typed summary values.

import numpy as np  # Aggregate statistics.
from tqdm import tqdm  # Progress bar.

from metricvision.backend.odometry import StereoVisualOdometry  # Dense keyframe VO pipeline.
from metricvision.backend.stats import FrameStats  # Per-frame measurements.
from metricvision.backend.stereo_odometry import SparseStereoOdometry  # Sparse frame-to-frame VO pipeline.
from metricvision.config import ExperimentConfig  # Experiment settings.
from metricvision.core.types import StereoCamera  # Rectified camera model.
from metricvision.datasets import DATASETS, StereoDataset  # Dataset registry (import registers loaders).
from metricvision.depth import DEPTH_ESTIMATORS, DepthEstimator  # Depth registry (import registers estimators).
from metricvision.evaluation.trajectory_metrics import TrajectoryErrors, evaluate_trajectory  # Accuracy metrics.
from metricvision.evaluation.writers import write_json, write_kitti, write_rows_csv, write_tum  # Output writers.
from metricvision.frontend import MATCHERS, STEREO_FRONTENDS  # Front-end registries (import registers them).


def _finite_mean(values: Sequence[float]) -> float:
    """Mean of the finite values, or NaN if there are none.

    Args:
        values: Numbers that may include NaN.

    Returns:
        Mean of finite entries.
    """
    finite = [v for v in values if math.isfinite(v)]  # Drop NaN and inf.
    return float(np.mean(finite)) if finite else float("nan")  # Avoid NumPy's empty-mean warning.


@dataclass(frozen=True)
class RunSummary:
    """Aggregate results of one experiment, written to summary.json.

    Attributes:
        dataset: Dataset sequence tag.
        matcher: Front-end name.
        run: Output folder name of the run.
        pipeline: ``dense`` or ``sparse``.
        refine: PnP refinement method.
        num_frames: Frames processed.
        num_keyframes: Keyframes created.
        num_tracking_failures: Frames where PnP failed.
        failure_rate: Failures divided by tracked frames (all but the first).
        mean_matches: Mean front-end matches per frame.
        mean_matches_with_depth: Mean matches with valid depth per frame.
        mean_pnp_inliers: Mean PnP inliers per frame.
        mean_essential_inlier_ratio: Mean 2D-2D RANSAC inlier ratio.
        mean_match_ms: Mean front-end time per frame.
        mean_pose_ms: Mean pose-estimation time per frame.
        mean_depth_ms: Mean depth time per frame (amortised over keyframes).
        mean_total_ms: Mean end-to-end time per frame.
        fps: Throughput implied by mean_total_ms.
        errors: Trajectory errors, or None without ground truth.
    """

    dataset: str  # Sequence tag.
    matcher: str  # Front end.
    run: str  # Run name.
    pipeline: str  # Pipeline.
    refine: str  # Refinement method.
    num_frames: int  # Frame count.
    num_keyframes: int  # Keyframe count.
    num_tracking_failures: int  # Failure count.
    failure_rate: float  # Failure fraction.
    mean_matches: float  # Matching volume.
    mean_matches_with_depth: float  # Usable matches.
    mean_pnp_inliers: float  # Verified matches.
    mean_essential_inlier_ratio: float  # Match quality.
    mean_match_ms: float  # Front-end cost.
    mean_pose_ms: float  # PnP cost.
    mean_depth_ms: float  # Depth cost.
    mean_total_ms: float  # Total cost.
    fps: float  # Throughput.
    errors: TrajectoryErrors | None  # Accuracy.

    @classmethod
    def from_stats(
        cls,
        dataset: str,
        matcher: str,
        run: str,
        pipeline: str,
        refine: str,
        stats: Sequence[FrameStats],
        num_keyframes: int,
        errors: TrajectoryErrors | None,
    ) -> RunSummary:
        """Aggregate per-frame statistics.

        Args:
            dataset: Sequence tag.
            matcher: Front-end name.
            run: Run name.
            pipeline: Pipeline name.
            refine: Refinement method.
            stats: Per-frame measurements; the first entry is the initialisation frame.
            num_keyframes: Keyframes created.
            errors: Trajectory errors, if ground truth was available.

        Returns:
            The run summary.
        """
        tracked = list(stats[1:])  # Exclude the initialisation frame, which has no matching.
        failures = sum(not s.tracking_ok for s in tracked)  # Count failures.
        mean_total = _finite_mean([s.total_ms for s in stats])  # Mean per-frame time.
        return cls(  # Build the summary.
            dataset=dataset,  # Sequence tag.
            matcher=matcher,  # Front end.
            run=run,  # Run name.
            pipeline=pipeline,  # Pipeline.
            refine=refine,  # Refinement.
            num_frames=len(stats),  # All frames.
            num_keyframes=num_keyframes,  # Keyframes.
            num_tracking_failures=failures,  # Failures.
            failure_rate=failures / len(tracked) if tracked else float("nan"),  # Fraction of tracked frames.
            mean_matches=_finite_mean([s.num_matches for s in tracked]),  # Matches.
            mean_matches_with_depth=_finite_mean([s.num_matches_with_depth for s in tracked]),  # Usable matches.
            mean_pnp_inliers=_finite_mean([s.num_pnp_inliers for s in tracked]),  # Inliers.
            mean_essential_inlier_ratio=_finite_mean([s.essential_inlier_ratio for s in tracked]),  # Quality.
            mean_match_ms=_finite_mean([s.match_ms for s in tracked]),  # Front-end time.
            mean_pose_ms=_finite_mean([s.pose_ms for s in tracked]),  # PnP time.
            mean_depth_ms=_finite_mean([s.depth_ms for s in stats]),  # Depth time, amortised over all frames.
            mean_total_ms=mean_total,  # Total time.
            fps=1000.0 / mean_total if mean_total > 0 else float("nan"),  # Frames per second.
            errors=errors,  # Accuracy.
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain nested dictionary for JSON output.

        Returns:
            Field name -> value mapping.
        """
        return asdict(self)  # Recursively converts the nested TrajectoryErrors too.


def build_odometry(config: ExperimentConfig, camera: StereoCamera) -> StereoVisualOdometry | SparseStereoOdometry:
    """Instantiate the VO pipeline named in a config (shared by offline runs and the ROS 2 node).

    Args:
        config: Experiment configuration.
        camera: Rectified stereo camera.

    Returns:
        The dense or sparse VO pipeline.
    """
    if config.pipeline == "sparse":  # Sparse stereo front end, frame-to-frame tracking.
        frontend = STEREO_FRONTENDS.create(config.stereo_frontend.name, **config.stereo_frontend.params)  # Front end under test.
        prior = DEPTH_ESTIMATORS.create(config.prior_depth.name, **config.prior_depth.params) if config.odometry.depth_prior else None  # BA priors.
        return SparseStereoOdometry(camera, frontend, config.odometry, prior)
    matcher = MATCHERS.create(config.matcher.name, **config.matcher.params)  # Pairwise matcher + dense SGBM depth.
    depth: DepthEstimator = DEPTH_ESTIMATORS.create(config.depth.name, **config.depth.params)  # Fixed depth source.
    return StereoVisualOdometry(camera, matcher, depth, config.odometry)


class ExperimentRunner:
    """Builds the components named in a config and runs one sequence through them."""

    def __init__(self, config: ExperimentConfig) -> None:
        """Instantiate the dataset, front end, depth estimator and VO pipeline.

        Args:
            config: Experiment configuration.
        """
        self._config = config  # Settings.
        self._dataset: StereoDataset = DATASETS.create(config.dataset.name, **config.dataset.params)  # Data source.
        self._odometry = build_odometry(config, self._dataset.camera)  # VO pipeline.

    @property
    def output_dir(self) -> Path:
        """Return the folder this run writes to: ``<output_dir>/<dataset tag>/<matcher>``.

        Returns:
            Output folder path.
        """
        return self._config.output_dir / self._dataset.tag / self._config.name  # One folder per (sequence, run).

    def run(self, show_progress: bool = True) -> RunSummary:
        """Process the sequence, evaluate against ground truth and write all outputs.

        Outputs written to ``output_dir``:
        ``trajectory.tum`` (body frame, for evo), ``trajectory.kitti`` (body frame, KITTI format),
        ``frame_stats.csv`` (per-frame measurements) and ``summary.json`` (aggregates).

        Args:
            show_progress: Whether to display a progress bar.

        Returns:
            The run summary.
        """
        self._odometry.reset()  # Start from a clean state.
        num_frames = len(self._dataset)  # Sequence length.
        if self._config.max_frames is not None:  # Optional cap.
            num_frames = min(num_frames, self._config.max_frames)  # Apply it.
        stats: list[FrameStats] = []  # Per-frame measurements.
        description = f"{self._dataset.tag} | {self._config.name}"  # Progress-bar label.
        for index in tqdm(range(num_frames), desc=description, disable=not show_progress):  # Main loop.
            result = self._odometry.process(self._dataset.load_frame(index))  # Track one frame.
            stats.append(result.stats)  # Keep its measurements.

        estimate = self._odometry.trajectory.conjugated(self._dataset.body_from_camera)  # Express poses in the ground-truth body frame.
        reference = self._dataset.ground_truth()  # May be None.
        errors = evaluate_trajectory(estimate, reference) if reference is not None and len(estimate) > 2 else None  # Accuracy, when possible.
        summary = RunSummary.from_stats(  # Aggregate.
            self._dataset.tag,
            self._config.frontend_name,
            self._config.name,
            self._config.pipeline,
            self._config.odometry.pnp.refine_method,
            stats,
            self._odometry.num_keyframes,
            errors,
        )

        output_dir = self.output_dir  # Resolve once.
        output_dir.mkdir(parents=True, exist_ok=True)  # Create folders as needed.
        write_tum(estimate, output_dir / "trajectory.tum")  # For evo (TUM/EuRoC ground truth).
        write_kitti(estimate, output_dir / "trajectory.kitti")  # For evo/devkit (KITTI ground truth).
        if reference is not None:  # Ground truth next to the estimate (same body frame), for plots and comparisons.
            write_tum(reference, output_dir / "reference.tum")
        write_rows_csv([s.to_dict() for s in stats], output_dir / "frame_stats.csv")  # Per-frame data for plots.
        write_json(summary.to_dict(), output_dir / "summary.json")  # Aggregates.
        return summary  # Also returned for programmatic use (tests, sweeps).
