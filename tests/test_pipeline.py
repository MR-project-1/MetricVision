"""End-to-end tests on the synthetic sequence: the whole pipeline must recover the true motion."""

from __future__ import annotations  # Postponed evaluation of type hints.

import json  # Reading summary.json.
from pathlib import Path  # Temporary output folder.

import pytest  # Parametrised tests.

from metricvision.backend.odometry import OdometryConfig  # Back-end defaults.
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
