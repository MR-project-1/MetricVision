"""Writers for trajectories, per-frame statistics and run summaries."""

from __future__ import annotations  # Postponed evaluation of type hints.

import csv  # Per-frame statistics as CSV.
import json  # Run summaries as JSON.
import math  # NaN detection for JSON sanitising.
from collections.abc import Mapping, Sequence  # Input type hints.
from pathlib import Path  # Output paths.
from typing import Any  # Loosely typed summary values.

from metricvision.core.geometry import quaternion_xyzw_from_rotation  # TUM quaternion order.
from metricvision.core.types import Trajectory  # Trajectory container.


def write_tum(trajectory: Trajectory, path: Path) -> None:
    """Write a trajectory in TUM format: ``timestamp tx ty tz qx qy qz qw`` per line.

    evo reads this with ``evo_ape tum`` and, against EuRoC ground truth, with ``evo_ape euroc``.

    Args:
        trajectory: Poses to write.
        path: Output file.
    """
    with path.open("w") as handle:  # Overwrite any previous file.
        for timestamp, pose in zip(trajectory.timestamps, trajectory.poses, strict=True):  # One line per pose.
            qx, qy, qz, qw = quaternion_xyzw_from_rotation(pose[:3, :3])  # Orientation as a quaternion.
            tx, ty, tz = pose[:3, 3]  # Position.
            handle.write(f"{timestamp:.9f} {tx:.9f} {ty:.9f} {tz:.9f} {qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}\n")  # TUM line.


def write_kitti(trajectory: Trajectory, path: Path) -> None:
    """Write a trajectory in KITTI format: the top 3x4 of each pose, row-major, per line.

    evo reads this with ``evo_ape kitti``; the official KITTI devkit uses the same format.

    Args:
        trajectory: Poses to write (one per frame, no gaps).
        path: Output file.
    """
    with path.open("w") as handle:  # Overwrite any previous file.
        for pose in trajectory.poses:  # One line per pose; line number = frame index.
            handle.write(" ".join(f"{value:.9e}" for value in pose[:3, :].reshape(-1)) + "\n")  # 12 numbers.


def write_rows_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    """Write a list of dictionaries as a CSV file with a header row.

    Args:
        rows: Records with identical keys.
        path: Output file.
    """
    if not rows:  # Nothing to write.
        path.write_text("")  # Create an empty file so downstream scripts do not crash.
        return  # Done.
    with path.open("w", newline="") as handle:  # newline="" is required by the csv module.
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))  # Header from the first record.
        writer.writeheader()  # Column names.
        writer.writerows(rows)  # All records.


def sanitise_for_json(value: Any) -> Any:
    """Replace NaN/inf floats with None so the output is valid JSON.

    Args:
        value: Any JSON-like value.

    Returns:
        The value with non-finite floats replaced, recursively.
    """
    if isinstance(value, float) and not math.isfinite(value):  # NaN and inf are not valid JSON.
        return None  # Encode as null.
    if isinstance(value, dict):  # Recurse into mappings.
        return {key: sanitise_for_json(item) for key, item in value.items()}  # Sanitise every value.
    if isinstance(value, list):  # Recurse into lists.
        return [sanitise_for_json(item) for item in value]  # Sanitise every element.
    return value  # Everything else is already JSON-safe.


def write_json(data: Mapping[str, Any], path: Path) -> None:
    """Write a mapping as pretty-printed, strictly valid JSON.

    Args:
        data: Mapping to serialise.
        path: Output file.
    """
    path.write_text(json.dumps(sanitise_for_json(dict(data)), indent=2, allow_nan=False) + "\n")  # Human-readable output.
