"""Statistical consistency of the predicted pose uncertainty (RQ2): NEES / ANEES against ground truth.

The LM covariance describes the uncertainty of the current pose *given* the landmarks, i.e. of the motion relative to
the previous frame. The consistency test therefore uses the relative motion: with the left perturbation
T_cur_world <- exp(delta^) T_cur_world and the previous pose taken as known, the frame-to-frame motion
T_cur_prev = T_cur_world T_world_prev is perturbed by the same delta, so

    delta_k = log( T_cur_prev^est (T_cur_prev^gt)^-1 ),   NEES_k = delta_k^T Sigma_k^-1 delta_k  ~ chi^2(6).

A consistent estimator has mean NEES ~ 6 (ANEES ~ 1 per degree of freedom); larger means over-confident.
"""

from __future__ import annotations  # Postponed evaluation of type hints.

from collections.abc import Sequence  # Inputs.
from dataclasses import asdict, dataclass  # Result container.

import cv2  # Rotation vectors.
import numpy as np  # Maths.

from metricvision.core.geometry import invert_pose  # Rigid inverse.
from metricvision.core.types import FloatArray, Trajectory  # Shared types.

CHI2_6_95 = 12.592  # 95 % quantile of chi^2(6).


@dataclass(frozen=True)
class Consistency:
    """NEES summary.

    Attributes:
        mean_nees: Mean NEES over evaluated frames (ideal 6).
        median_nees: Median NEES (ideal 5.35).
        anees: Mean NEES per degree of freedom (ideal 1).
        frac_above_95: Fraction of frames above the chi^2(6) 95 % quantile (ideal 0.05).
        num_frames: Frames evaluated.
    """

    mean_nees: float
    median_nees: float
    anees: float
    frac_above_95: float
    num_frames: int

    def to_dict(self) -> dict[str, float | int]:
        """Return a plain dictionary.

        Returns:
            Fields.
        """
        return asdict(self)


def relative_nees(
    estimate: Trajectory, reference: Trajectory, covariances: Sequence[FloatArray | None], max_time_diff_s: float = 0.01
) -> tuple[Consistency, FloatArray]:
    """NEES of each frame-to-frame motion.

    Args:
        estimate: Estimated T_world_cam (camera frame), one pose per frame.
        reference: Ground truth T_world_cam in the same camera frame.
        covariances: Per-frame 6x6 covariance (translation first) of the left perturbation of T_cam_world, or None.
        max_time_diff_s: Timestamp association tolerance.

    Returns:
        Summary and the per-frame NEES (NaN where not evaluated).
    """
    ref_t = np.asarray(reference.timestamps)
    nees = np.full(len(estimate), np.nan)
    if ref_t.size == 0:
        return Consistency(float("nan"), float("nan"), float("nan"), float("nan"), 0), nees
    order = np.argsort(ref_t)
    ref_t = ref_t[order]

    def lookup(t: float) -> FloatArray | None:
        """Ground-truth pose closest in time."""
        i = int(np.clip(np.searchsorted(ref_t, t), 1, len(ref_t) - 1)) if len(ref_t) > 1 else 0
        j = i - 1 if len(ref_t) > 1 and abs(ref_t[i - 1] - t) < abs(ref_t[i] - t) else i
        return reference.poses[order[j]] if abs(ref_t[j] - t) <= max_time_diff_s else None

    for k in range(1, len(estimate)):
        cov = covariances[k] if k < len(covariances) else None
        if cov is None or not np.isfinite(cov).all():
            continue
        gt_prev, gt_cur = lookup(estimate.timestamps[k - 1]), lookup(estimate.timestamps[k])
        if gt_prev is None or gt_cur is None:
            continue
        rel_est = invert_pose(estimate.poses[k]) @ estimate.poses[k - 1]  # T_cur_prev.
        rel_gt = invert_pose(gt_cur) @ gt_prev
        delta = rel_est @ invert_pose(rel_gt)  # exp(delta^) ~ this.
        rvec, _ = cv2.Rodrigues(delta[:3, :3])
        err = np.concatenate([delta[:3, 3], rvec.ravel()])  # (rho, phi) to first order.
        try:
            nees[k] = float(err @ np.linalg.solve(cov, err))
        except np.linalg.LinAlgError:
            continue
    valid = nees[np.isfinite(nees)]
    if valid.size == 0:
        return Consistency(float("nan"), float("nan"), float("nan"), float("nan"), 0), nees
    return (
        Consistency(
            mean_nees=float(np.mean(valid)),
            median_nees=float(np.median(valid)),
            anees=float(np.mean(valid) / 6.0),
            frac_above_95=float(np.mean(valid > CHI2_6_95)),
            num_frames=int(valid.size),
        ),
        nees,
    )
