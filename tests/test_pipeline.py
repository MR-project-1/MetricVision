"""End-to-end tests on the synthetic sequence: the whole pipeline must recover the true motion."""

from __future__ import annotations  # Postponed evaluation of type hints.

import json  # Reading summary.json.
from dataclasses import replace  # Config variants.
from pathlib import Path  # Temporary output folder.

import pytest  # Parametrised tests.

from metricvision.backend.odometry import OdometryConfig  # Back-end defaults.
from metricvision.backend.pose import PnPConfig  # Refinement selection.
from metricvision.config import ComponentConfig, ExperimentConfig  # Programmatic configs.
from metricvision.pipeline import ExperimentRunner  # Unit under test.


def _config(matcher: str, output_dir: Path) -> ExperimentConfig:
    """Build a small synthetic experiment.

    Args:
        matcher: Front-end name.
        output_dir: Where results are written.

    Returns:
        Experiment configuration.
    """
    return ExperimentConfig(  # Mirror configs/synthetic.yaml, but shorter.
        dataset=ComponentConfig("synthetic", {"num_frames": 30}),  # 30 frames keeps the test fast.
        matcher=ComponentConfig(matcher, {"max_keypoints": 1500}),  # Front end under test.
        depth=ComponentConfig("sgbm", {"num_disparities": 32, "min_depth": 0.5, "max_depth": 20.0}),  # Depth suited to the scene.
        odometry=OdometryConfig(),  # Default back end.
        output_dir=output_dir,  # Temporary folder.
    )


@pytest.mark.parametrize("matcher", ["orb", "sift"])  # Run for every classical front end.
def test_synthetic_sequence_is_tracked_accurately(matcher: str, tmp_path: Path) -> None:
    """Every frame must track and the trajectory error must be small.

    Args:
        matcher: Front-end name (from parametrize).
        tmp_path: Pytest-provided temporary folder.
    """
    runner = ExperimentRunner(_config(matcher, tmp_path))  # Build the pipeline.
    summary = runner.run(show_progress=False)  # Run it.
    assert summary.num_tracking_failures == 0  # Easy synthetic data must never lose track.
    assert summary.errors is not None  # Synthetic data always has ground truth.
    assert summary.errors.ate_rmse_m < 0.02  # Under 2 cm over ~0.6 m of motion.
    for name in (
        "trajectory.tum",
        "trajectory.kitti",
        "frame_stats.csv",
        "summary.json",
    ):  # Every output file.
        assert (runner.output_dir / name).is_file()  # Must exist.
    saved = json.loads((runner.output_dir / "summary.json").read_text())  # Parse the written summary.
    assert saved["matcher"] == matcher  # Summary refers to the right front end.


@pytest.mark.parametrize(
    ("pipeline", "refine", "anchor"),
    [
        ("dense", "lm", "previous"),
        ("sparse", "opencv", "previous"),
        ("sparse", "lm", "previous"),
        ("sparse", "lm", "birth"),
        ("sparse", "lm", "fused"),
    ],
)  # New pipeline/refinement/anchoring combinations.
def test_sparse_pipeline_and_rig_lm_track_accurately(pipeline: str, refine: str, anchor: str, tmp_path: Path) -> None:
    """The sparse stereo front end and the covariance-weighted LM must also recover the synthetic motion.

    Args:
        pipeline: ``dense`` or ``sparse``.
        refine: PnP refinement method.
        anchor: Sparse point anchoring.
        tmp_path: Pytest-provided temporary folder.
    """
    config = replace(  # Same experiment as above with the variant under test.
        _config("orb", tmp_path),
        pipeline=pipeline,
        odometry=OdometryConfig(pnp=PnPConfig(refine=refine), sparse_point_anchor=anchor),
        run_name=f"{pipeline}_{refine}",
    )
    runner = ExperimentRunner(config)  # Build the pipeline.
    summary = runner.run(show_progress=False)  # Run it.
    assert summary.num_tracking_failures == 0  # Never lose track.
    assert summary.errors is not None and summary.errors.ate_rmse_m < 0.02  # Accurate.
    assert summary.pipeline == pipeline and summary.refine == refine  # Recorded in the summary.
    assert runner.output_dir.name == f"{pipeline}_{refine}"  # Run name used for the folder.


def test_sparse_pipeline_with_windowed_ba_tracks_accurately(tmp_path: Path) -> None:
    """The sparse pipeline with a 5-keyframe windowed BA must still track the synthetic sequence accurately.

    Args:
        tmp_path: Pytest-provided temporary folder.
    """
    config = replace(  # Sparse, fused anchors, rig LM, windowed BA.
        _config("orb", tmp_path),
        pipeline="sparse",
        odometry=OdometryConfig(pnp=PnPConfig(refine="lm"), sparse_point_anchor="fused", window_size=5),
        run_name="sparse_ba",
    )
    summary = ExperimentRunner(config).run(show_progress=False)  # Run it.
    assert summary.num_tracking_failures == 0  # Never lose track.
    assert summary.errors is not None and summary.errors.ate_rmse_m < 0.02  # Accurate.
