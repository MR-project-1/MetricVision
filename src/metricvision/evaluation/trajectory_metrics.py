"""Trajectory error metrics computed with the evo library."""

from __future__ import annotations  # Postponed evaluation of type hints.

import copy  # Deep copies so alignment does not modify the caller's data.
from dataclasses import asdict, dataclass  # Result container and dict conversion.

import numpy as np  # Array conversion.
from evo.core import metrics, sync  # APE/RPE metrics and timestamp association.
from evo.core.geometry import GeometryException  # Raised when Umeyama alignment is degenerate.
from evo.core.trajectory import PoseTrajectory3D  # evo's trajectory type.

from metricvision.core.types import Trajectory  # Our trajectory type.


@dataclass(frozen=True)
class TrajectoryErrors:
    """Summary of trajectory accuracy.

    Attributes:
        ate_rmse_m: Absolute trajectory error (translation), RMSE after SE(3) alignment, metres.
        ate_mean_m: Mean absolute trajectory error, metres.
        ate_median_m: Median absolute trajectory error, metres.
        ate_max_m: Maximum absolute trajectory error, metres.
        alignment: ``"se3"`` for Umeyama alignment, ``"origin"`` when the path was too degenerate for it.
        rpe_trans_rmse_m: Frame-to-frame relative translation error, RMSE, metres.
        rpe_rot_rmse_deg: Frame-to-frame relative rotation error, RMSE, degrees.
        num_pairs: Number of estimate/ground-truth poses matched by timestamp.
        rpe_1s_trans_rmse_m: Relative translation error over 1 s, RMSE, metres.
        rpe_1s_rot_rmse_deg: Relative rotation error over 1 s, RMSE, degrees.
        path_length_m: Ground-truth path length.
        final_position_error_m: Position error of the last pose after SE(3) alignment of the first pose only.
    """

    ate_rmse_m: float  # Headline accuracy number.
    ate_mean_m: float  # Mean error.
    ate_median_m: float  # Median error (robust to a few bad frames).
    ate_max_m: float  # Worst-case error.
    alignment: str  # Which alignment was used.
    rpe_trans_rmse_m: float  # Local translational drift.
    rpe_rot_rmse_deg: float  # Local rotational drift.
    num_pairs: int  # Poses used in the evaluation.
    rpe_1s_trans_rmse_m: float = float("nan")  # Drift over one second.
    rpe_1s_rot_rmse_deg: float = float("nan")  # Rotation drift over one second.
    path_length_m: float = float("nan")  # Distance travelled.
    final_position_error_m: float = float("nan")  # Open-loop end-point error.

    def to_dict(self) -> dict[str, float | int | str]:
        """Convert to a plain dictionary for JSON output.

        Returns:
            Field name -> value mapping.
        """
        return asdict(self)  # Dataclass fields as a dict.


def _to_evo(trajectory: Trajectory) -> PoseTrajectory3D:
    """Convert our Trajectory into evo's PoseTrajectory3D.

    Args:
        trajectory: Time-stamped poses.

    Returns:
        Equivalent evo trajectory.
    """
    return PoseTrajectory3D(  # evo stores poses as 4x4 SE(3) matrices.
        poses_se3=[np.asarray(pose, dtype=np.float64) for pose in trajectory.poses],  # Pose list.
        timestamps=np.asarray(trajectory.timestamps, dtype=np.float64),  # Matching timestamps.
    )


def evaluate_trajectory(estimate: Trajectory, reference: Trajectory, max_time_diff_s: float = 0.01) -> TrajectoryErrors:
    """Compute ATE and RPE of an estimate against ground truth.

    Poses are associated by timestamp, the estimate is aligned to the reference
    with an SE(3) (rotation + translation, no scale) Umeyama fit because stereo
    odometry is metric, and translation errors are summarised. A perfectly
    straight path (common in short robot runs) makes Umeyama degenerate; in that
    case the first poses are aligned instead and this is reported.

    Args:
        estimate: Estimated trajectory, in the same body frame as the reference.
        reference: Ground-truth trajectory.
        max_time_diff_s: Maximum timestamp difference for two poses to be associated.

    Returns:
        Summary error statistics.
    """
    reference_evo, estimate_evo = sync.associate_trajectories(_to_evo(reference), _to_evo(estimate), max_diff=max_time_diff_s)  # Pair poses by time.
    aligned = copy.deepcopy(estimate_evo)  # Keep the unaligned estimate for RPE.
    try:  # Preferred: best-fit alignment over the whole trajectory.
        aligned.align(reference_evo, correct_scale=False)  # SE(3) Umeyama alignment.
        alignment = "se3"  # Record which alignment was used.
    except GeometryException:  # Collinear path: rotation about the line is unobservable.
        aligned = copy.deepcopy(estimate_evo)  # Restart from the unaligned estimate.
        aligned.align_origin(reference_evo)  # Align the first poses instead.
        alignment = "origin"  # Record the fallback so results are interpreted correctly.
    ape = metrics.APE(metrics.PoseRelation.translation_part)  # Absolute position error metric.
    ape.process_data((reference_evo, aligned))  # Compute per-pose errors.
    rpe_t = metrics.RPE(
        metrics.PoseRelation.translation_part, delta=1, delta_unit=metrics.Unit.frames, all_pairs=False
    )  # Relative translation over 1 frame.
    rpe_t.process_data((reference_evo, estimate_evo))  # RPE is invariant to global alignment.
    rpe_r = metrics.RPE(
        metrics.PoseRelation.rotation_angle_deg, delta=1, delta_unit=metrics.Unit.frames, all_pairs=False
    )  # Relative rotation over 1 frame.
    rpe_r.process_data((reference_evo, estimate_evo))  # Compute per-pair rotation errors.
    rate = 1.0 / float(np.median(np.diff(reference_evo.timestamps))) if reference_evo.num_poses > 1 else 1.0  # Frames per second.
    one_second = max(1, int(round(rate)))  # Frames in one second.
    rpe_1s_t = rpe_1s_r = float("nan")
    if reference_evo.num_poses > one_second + 1:  # Long enough for 1 s pairs.
        rel_t = metrics.RPE(metrics.PoseRelation.translation_part, delta=one_second, delta_unit=metrics.Unit.frames, all_pairs=True)
        rel_t.process_data((reference_evo, estimate_evo))
        rel_r = metrics.RPE(metrics.PoseRelation.rotation_angle_deg, delta=one_second, delta_unit=metrics.Unit.frames, all_pairs=True)
        rel_r.process_data((reference_evo, estimate_evo))
        rpe_1s_t = float(rel_t.get_statistic(metrics.StatisticsType.rmse))
        rpe_1s_r = float(rel_r.get_statistic(metrics.StatisticsType.rmse))
    ref_xyz = reference_evo.positions_xyz
    path_length = float(np.sum(np.linalg.norm(np.diff(ref_xyz, axis=0), axis=1)))  # Distance travelled.
    origin_aligned = copy.deepcopy(estimate_evo)
    origin_aligned.align_origin(reference_evo)  # Open-loop: only the first pose is aligned.
    final_error = float(np.linalg.norm(origin_aligned.positions_xyz[-1] - ref_xyz[-1]))
    return TrajectoryErrors(  # Collect the statistics.
        ate_rmse_m=float(ape.get_statistic(metrics.StatisticsType.rmse)),  # ATE RMSE.
        ate_mean_m=float(ape.get_statistic(metrics.StatisticsType.mean)),  # ATE mean.
        ate_median_m=float(ape.get_statistic(metrics.StatisticsType.median)),  # ATE median.
        ate_max_m=float(ape.get_statistic(metrics.StatisticsType.max)),  # ATE max.
        alignment=alignment,  # Alignment method.
        rpe_trans_rmse_m=float(rpe_t.get_statistic(metrics.StatisticsType.rmse)),  # RPE translation RMSE.
        rpe_rot_rmse_deg=float(rpe_r.get_statistic(metrics.StatisticsType.rmse)),  # RPE rotation RMSE.
        num_pairs=int(reference_evo.num_poses),  # Number of associated poses.
        rpe_1s_trans_rmse_m=rpe_1s_t,  # One-second drift.
        rpe_1s_rot_rmse_deg=rpe_1s_r,
        path_length_m=path_length,
        final_position_error_m=final_error,
    )
