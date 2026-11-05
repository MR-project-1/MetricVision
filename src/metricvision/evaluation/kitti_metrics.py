"""KITTI odometry devkit metric: relative translation (%) and rotation (deg/100 m) errors over 100-800 m segments.

A direct port of ``evaluate_odometry.cpp`` from the official KITTI devkit, so numbers
are comparable with the KITTI leaderboard (which reports the average over all
segment lengths and start frames spaced 10 frames apart).
"""

from __future__ import annotations  # Postponed evaluation of type hints.

from collections.abc import Sequence  # Input type hints.
from dataclasses import dataclass  # Result container.
from pathlib import Path  # Pose file paths.

import numpy as np  # Pose maths.

from metricvision.core.geometry import invert_pose  # Rigid inverse.
from metricvision.core.types import FloatArray  # Shared types.

SEGMENT_LENGTHS_M = (100, 200, 300, 400, 500, 600, 700, 800)  # Devkit segment lengths.
STEP_SIZE_FRAMES = 10  # Devkit spacing between segment start frames.


@dataclass(frozen=True)
class KittiErrors:
    """Average devkit errors.

    Attributes:
        translation_percent: Mean relative translation error over all segments, in percent.
        rotation_deg_per_100m: Mean relative rotation error, degrees per 100 m.
        num_segments: Number of evaluated segments.
    """

    translation_percent: float  # t_rel.
    rotation_deg_per_100m: float  # r_rel.
    num_segments: int  # Segment count.


def load_kitti_poses(path: Path) -> list[FloatArray]:
    """Read a KITTI pose file (12 numbers per line, row-major 3x4).

    Args:
        path: Pose file.

    Returns:
        4x4 poses, one per line.
    """
    poses = []  # Output.
    for line in path.read_text().splitlines():  # One pose per line.
        if line.strip():  # Skip blank lines.
            pose = np.eye(4, dtype=np.float64)  # Homogeneous.
            pose[:3, :] = np.array(line.split(), dtype=np.float64).reshape(3, 4)  # Top 3x4 block.
            poses.append(pose)  # Store.
    return poses  # All poses.


def kitti_segment_error_list(
    estimate: Sequence[FloatArray], ground_truth: Sequence[FloatArray], lengths: Sequence[int] = SEGMENT_LENGTHS_M
) -> list[tuple[int, int, float, float]]:
    """Per-segment devkit errors (for paired statistics between methods).

    Args:
        estimate: Estimated T_world_cam per frame.
        ground_truth: Ground-truth T_world_cam per frame (same frames).
        lengths: Segment lengths in metres.

    Returns:
        (first frame, length, translation error [m/m], rotation error [rad/m]) per segment.

    Raises:
        ValueError: If the sequences differ in length.
    """
    if len(estimate) != len(ground_truth):  # Devkit requires line-by-line correspondence.
        raise ValueError(f"estimate has {len(estimate)} poses but ground truth has {len(ground_truth)}")  # Clear error.
    positions = np.array([pose[:3, 3] for pose in ground_truth])  # Ground-truth positions.
    distances = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(positions, axis=0), axis=1))])  # Distance travelled.
    segments: list[tuple[int, int, float, float]] = []
    for first in range(0, len(ground_truth), STEP_SIZE_FRAMES):  # Segment starts.
        for length in lengths:  # Segment lengths.
            beyond = np.nonzero(distances[first:] > distances[first] + length)[0]  # Frames past the segment end.
            if beyond.size == 0:  # Sequence too short from here.
                continue  # Skip.
            last = first + int(beyond[0])  # Last frame of the segment.
            delta_gt = invert_pose(ground_truth[first]) @ ground_truth[last]  # True relative motion.
            delta_est = invert_pose(estimate[first]) @ estimate[last]  # Estimated relative motion.
            error = invert_pose(delta_est) @ delta_gt  # Residual motion.
            cos_angle = np.clip(0.5 * (np.trace(error[:3, :3]) - 1.0), -1.0, 1.0)  # Rotation angle cosine.
            segments.append((first, int(length), float(np.linalg.norm(error[:3, 3])) / length, float(np.arccos(cos_angle)) / length))
    return segments


def kitti_segment_errors(
    estimate: Sequence[FloatArray], ground_truth: Sequence[FloatArray], lengths: Sequence[int] = SEGMENT_LENGTHS_M
) -> KittiErrors:
    """Compute the devkit's segment errors (frames must correspond one-to-one).

    Args:
        estimate: Estimated T_world_cam per frame.
        ground_truth: Ground-truth T_world_cam per frame.
        lengths: Segment lengths in metres (the devkit uses 100-800 m; pass one length for per-length plots).

    Returns:
        Average translation and rotation errors.
    """
    segments = kitti_segment_error_list(estimate, ground_truth, lengths)  # Per segment.
    if not segments:  # Sequence shorter than 100 m.
        return KittiErrors(float("nan"), float("nan"), 0)  # Undefined.
    t_errors = [seg[2] for seg in segments]
    r_errors = [seg[3] for seg in segments]
    return KittiErrors(  # Devkit averaging over all segments.
        translation_percent=100.0 * float(np.mean(t_errors)),  # Percent.
        rotation_deg_per_100m=100.0 * float(np.degrees(np.mean(r_errors))),  # Degrees per 100 m.
        num_segments=len(t_errors),  # Count.
    )
