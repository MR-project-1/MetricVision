"""Core data containers shared by every part of the benchmark.

Keeping these types in one module means datasets, front ends, back ends and
evaluation code all agree on array shapes, units and coordinate conventions.

Pose convention used throughout: ``T_a_b`` (written ``a_from_b`` in code) is a
4x4 homogeneous matrix that maps a point expressed in frame ``b`` into frame
``a``, i.e. ``p_a = T_a_b @ p_b``.
"""

from __future__ import annotations

from dataclasses import dataclass, field  # Typed, boilerplate-free containers.
from typing import Any  # Used for descriptor arrays whose dtype depends on the extractor.

import numpy as np  # Numerical arrays for images, points and poses.
import numpy.typing as npt  # Generic NumPy array type hints.

FloatArray = npt.NDArray[np.float64]  # Double-precision array (poses, 3D points).
Float32Array = npt.NDArray[np.float32]  # Single-precision array (keypoints, depth maps).
ImageArray = npt.NDArray[np.uint8]  # 8-bit grayscale image.
BoolArray = npt.NDArray[np.bool_]  # Boolean mask array.
RealArray = npt.NDArray[np.floating[Any]]  # Float array of any precision (pixel coordinates from OpenCV or NumPy).


@dataclass(frozen=True)
class StereoCamera:
    """Pinhole intrinsics of a rectified stereo pair; the left camera is the reference.

    Attributes:
        fx: Focal length along the image x axis, in pixels.
        fy: Focal length along the image y axis, in pixels.
        cx: Principal point x coordinate, in pixels.
        cy: Principal point y coordinate, in pixels.
        baseline: Distance between the two optical centres, in metres.
        width: Image width in pixels.
        height: Image height in pixels.
    """

    fx: float  # Horizontal focal length in pixels.
    fy: float  # Vertical focal length in pixels.
    cx: float  # Principal point, x.
    cy: float  # Principal point, y.
    baseline: float  # Stereo baseline in metres.
    width: int  # Image width in pixels.
    height: int  # Image height in pixels.

    @property
    def intrinsic_matrix(self) -> FloatArray:
        """Return the 3x3 camera matrix K used by OpenCV geometry routines.

        Returns:
            The intrinsic matrix as a float64 array of shape (3, 3).
        """
        return np.array(  # Build K row by row.
            [
                [self.fx, 0.0, self.cx],  # Row 1: fx, zero skew, cx.
                [0.0, self.fy, self.cy],  # Row 2: 0, fy, cy.
                [0.0, 0.0, 1.0],  # Row 3: homogeneous normalisation.
            ],
            dtype=np.float64,  # OpenCV geometry functions expect float64.
        )


@dataclass(frozen=True)
class StereoFrame:
    """One synchronised, rectified stereo image pair.

    Attributes:
        index: Zero-based position of the frame in its sequence.
        timestamp: Capture time in seconds.
        left: Rectified left grayscale image of shape (H, W).
        right: Rectified right grayscale image of shape (H, W).
    """

    index: int  # Position of the frame in the sequence.
    timestamp: float  # Capture time in seconds.
    left: ImageArray  # Rectified left image.
    right: ImageArray  # Rectified right image.


@dataclass(frozen=True)
class ImageView:
    """A single image plus a stable identifier: the unit that front ends operate on.

    The identifier lets sparse matchers cache features, so a keyframe's
    features are extracted once no matter how many frames match against it.

    Attributes:
        frame_id: Unique identifier of the image within a run (the frame index).
        image: Grayscale pixels of shape (H, W).
    """

    frame_id: int  # Unique id of the image within a run.
    image: ImageArray  # Grayscale pixels.


@dataclass(frozen=True)
class Features:
    """Keypoints and descriptors produced by a sparse extractor.

    Attributes:
        keypoints: Pixel coordinates (x, y), float32 array of shape (N, 2).
        descriptors: Descriptor matrix of shape (N, D); uint8 for binary, float32 for float descriptors.
        scores: Detector response per keypoint, float32 array of shape (N,).
    """

    keypoints: Float32Array  # (N, 2) keypoint locations.
    descriptors: npt.NDArray[Any]  # (N, D) descriptors.
    scores: Float32Array  # (N,) detector responses.

    def __len__(self) -> int:
        """Return the number of keypoints.

        Returns:
            Number of detected keypoints N.
        """
        return int(self.keypoints.shape[0])  # Row count of the keypoint array.


@dataclass(frozen=True)
class Correspondences:
    """2D-2D matches between an image A and an image B.

    Row ``i`` of ``points_a`` and row ``i`` of ``points_b`` observe the same scene point.

    Attributes:
        points_a: Pixel coordinates in image A, float32 array of shape (N, 2).
        points_b: Pixel coordinates in image B, float32 array of shape (N, 2).
        scores: Match confidence (higher is better), float32 array of shape (N,).
    """

    points_a: Float32Array  # (N, 2) points in image A.
    points_b: Float32Array  # (N, 2) points in image B.
    scores: Float32Array  # (N,) confidences.

    def __post_init__(self) -> None:
        """Validate that the three arrays describe the same number of matches.

        Raises:
            ValueError: If the array shapes are inconsistent.
        """
        n = self.points_a.shape[0]  # Number of matches implied by points_a.
        if self.points_a.shape != (n, 2) or self.points_b.shape != (
            n,
            2,
        ):  # Both point arrays must be (N, 2).
            raise ValueError("points_a and points_b must both have shape (N, 2)")  # Reject malformed input early.
        if self.scores.shape != (n,):  # Exactly one score per match.
            raise ValueError("scores must have shape (N,)")  # Reject malformed input early.

    def __len__(self) -> int:
        """Return the number of matches.

        Returns:
            Number of correspondences N.
        """
        return int(self.points_a.shape[0])  # Row count of the point arrays.

    @classmethod
    def empty(cls) -> Correspondences:
        """Create a correspondence set that contains no matches.

        Returns:
            An empty Correspondences instance with correctly shaped arrays.
        """
        no_points = np.zeros((0, 2), dtype=np.float32)  # Zero rows, two columns.
        return cls(no_points, no_points.copy(), np.zeros((0,), dtype=np.float32))  # Empty but well-formed.

    def subset(self, mask: BoolArray) -> Correspondences:
        """Keep only the matches selected by a boolean mask.

        Args:
            mask: Boolean array of shape (N,), True for matches to keep.

        Returns:
            A new Correspondences with the selected rows.
        """
        return Correspondences(self.points_a[mask], self.points_b[mask], self.scores[mask])  # Row-wise filtering.

    def top_k(self, k: int) -> Correspondences:
        """Keep the ``k`` most confident matches, used to enforce equal match budgets.

        Args:
            k: Maximum number of matches to keep.

        Returns:
            A new Correspondences with at most ``k`` rows, sorted by descending score.
        """
        if len(self) <= k:  # Nothing to trim.
            return self  # Return unchanged (instances are immutable, so sharing is safe).
        order = np.argsort(-self.scores)[:k]  # Indices of the k highest scores.
        return Correspondences(self.points_a[order], self.points_b[order], self.scores[order])  # Trimmed copy.


@dataclass
class Trajectory:
    """Time-stamped sequence of 4x4 poses ``T_world_frame``.

    Attributes:
        timestamps: Pose times in seconds.
        poses: 4x4 float64 homogeneous transforms, one per timestamp.
    """

    timestamps: list[float] = field(default_factory=list)  # Times in seconds.
    poses: list[FloatArray] = field(default_factory=list)  # Matching 4x4 poses.

    def append(self, timestamp: float, pose: FloatArray) -> None:
        """Add one pose to the end of the trajectory.

        Args:
            timestamp: Time of the pose in seconds.
            pose: 4x4 homogeneous transform T_world_frame.
        """
        self.timestamps.append(float(timestamp))  # Store time as a plain float.
        self.poses.append(np.asarray(pose, dtype=np.float64).copy())  # Copy so later edits cannot alias it.

    def __len__(self) -> int:
        """Return the number of poses.

        Returns:
            Number of stored poses.
        """
        return len(self.poses)  # One pose per timestamp.

    def conjugated(self, new_from_old: FloatArray) -> Trajectory:
        """Re-express every pose in a different rigidly attached frame.

        Given poses of frame ``old`` relative to its own start and the fixed
        transform ``T_new_old`` between two rigidly attached frames, this returns
        the poses of frame ``new`` relative to its own start:
        ``T_new0_newk = T_new_old @ T_old0_oldk @ T_new_old^-1``.

        Args:
            new_from_old: Constant 4x4 transform T_new_old (e.g. T_body_camera).

        Returns:
            A new Trajectory with the same timestamps and converted poses.
        """
        old_from_new = np.linalg.inv(new_from_old)  # Inverse of the rigid offset.
        converted = Trajectory()  # Output container.
        for timestamp, pose in zip(self.timestamps, self.poses, strict=True):  # Walk the poses in order.
            converted.append(timestamp, new_from_old @ pose @ old_from_new)  # Apply the conjugation.
        return converted  # Poses of the new frame.
