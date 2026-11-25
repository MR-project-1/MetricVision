"""Sparse stereo front end: FAST + ORB stereo matching and KLT tracking with a circular L/R check.

Unlike the pairwise ``FeatureMatcher`` front ends (which rely on dense SGBM
depth), this front end produces its own sparse stereo pairs and tracks them
frame to frame, yielding 3D-2D (and 3D-2D-right) correspondences per frame:

1. Detection (on replenishment): FAST corners in the left image, bucketed on a
   grid for even coverage; FAST corners in the right image; ORB descriptors on
   both; Hamming matching restricted to the epipolar band (same row +/- tol,
   positive disparity) with ratio and mutual checks; sub-pixel disparity via a
   single-level KLT refinement left -> right.
2. Tracking: for every stereo pair (l0, r0) of the previous frame,
       l0 --KLT--> l1 --KLT--> r1 --KLT--> r0'   (prev L -> cur L -> cur R -> prev R)
   and the track is kept only if the circle closes (|r0' - r0| < tol) and the
   current pair (l1, r1) satisfies the epipolar constraint.
"""

from __future__ import annotations  # Postponed evaluation of type hints.

from abc import ABC, abstractmethod  # Abstract base class machinery.
from dataclasses import dataclass  # Track container.
from typing import ClassVar  # Class-level constants.

import cv2  # FAST, ORB, brute-force matching and pyramidal KLT.
import numpy as np  # Array maths.

from metricvision.core.registry import Registry  # Name -> class lookup for configs.
from metricvision.core.types import BoolArray, Float32Array, ImageArray  # Shared types.


@dataclass(frozen=True)
class StereoTracks:
    """Stereo feature pairs in one frame: row i of ``left`` and ``right`` observe the same point.

    Attributes:
        left: (N, 2) float32 pixel coordinates in the left image.
        right: (N, 2) float32 pixel coordinates in the right image.
        confidence: Optional (N,) learned match confidence in (0, 1] (None for classical front ends).
    """

    left: Float32Array  # Left observations.
    right: Float32Array  # Right observations.
    confidence: Float32Array | None = None  # Learned confidence (LightGlue), if any.

    def __len__(self) -> int:
        """Return the number of stereo pairs.

        Returns:
            Pair count N.
        """
        return int(self.left.shape[0])  # Rows.

    @property
    def disparity(self) -> Float32Array:
        """Return u_left - u_right for every pair.

        Returns:
            (N,) disparities in pixels.
        """
        return self.left[:, 0] - self.right[:, 0]  # Horizontal offset.

    @classmethod
    def empty(cls) -> StereoTracks:
        """Create a container with no pairs.

        Returns:
            Empty tracks.
        """
        return cls(np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32))  # Well-formed empty arrays.

    def subset(self, mask: BoolArray) -> StereoTracks:
        """Keep only the pairs selected by ``mask``.

        Args:
            mask: (N,) boolean selection.

        Returns:
            Filtered tracks.
        """
        conf = None if self.confidence is None else self.confidence[mask]  # Same rows.
        return StereoTracks(self.left[mask], self.right[mask], conf)  # Same rows from both images.

    def concatenate(self, other: StereoTracks) -> StereoTracks:
        """Append another set of pairs.

        Args:
            other: Pairs to append.

        Returns:
            Combined tracks.
        """
        conf = None  # Classical tracks carry no confidence.
        if self.confidence is not None or other.confidence is not None:  # Mixed: missing confidences count as 1.
            ones = [np.ones(len(t), dtype=np.float32) if t.confidence is None else t.confidence for t in (self, other)]  # Fill.
            conf = np.concatenate(ones).astype(np.float32)  # Stack.
        return StereoTracks(np.vstack([self.left, other.left]), np.vstack([self.right, other.right]), conf)  # Stack rows.


class StereoFrontend(ABC):
    """Abstract sparse stereo front end: detects stereo pairs and tracks them across frames."""

    name: ClassVar[str] = "abstract"  # Output folder name; subclasses override.

    @property
    @abstractmethod
    def min_tracks(self) -> int:
        """Return the track count below which new features should be detected.

        Returns:
            Replenishment threshold.
        """

    @abstractmethod
    def detect(self, left: ImageArray, right: ImageArray, existing: StereoTracks) -> StereoTracks:
        """Detect new stereo pairs away from existing tracks.

        Args:
            left: Rectified left image.
            right: Rectified right image.
            existing: Pairs already tracked in this frame (new ones avoid them).

        Returns:
            Newly detected pairs only.
        """

    @abstractmethod
    def track(
        self,
        previous: tuple[ImageArray, ImageArray],
        current: tuple[ImageArray, ImageArray],
        tracks: StereoTracks,
        predicted_left: Float32Array | None = None,
    ) -> tuple[StereoTracks, BoolArray]:
        """Track stereo pairs from the previous frame into the current one.

        Args:
            previous: (left, right) images of the previous frame.
            current: (left, right) images of the current frame.
            tracks: Pairs observed in the previous frame.
            predicted_left: Optional (N, 2) predicted left positions in the current frame (initial KLT flow).

        Returns:
            The current-frame pairs of the surviving tracks, and an (N,) mask of which input tracks survived.
        """

    def synchronize(self) -> None:
        """Wait for asynchronous work (none for CPU front ends)."""
        return None  # CPU only.

    def reset(self) -> None:
        """Clear internal state before a new sequence."""
        return None  # Stateless by default.


STEREO_FRONTENDS: Registry[StereoFrontend] = Registry("stereo front end")  # Global registry.


@STEREO_FRONTENDS.register("fast_orb_klt")
class FastOrbKltFrontend(StereoFrontend):
    """FAST + ORB stereo matching with KLT temporal tracking and a circular left/right check."""

    name = "fast_orb_klt"  # Output folder name.

    def __init__(
        self,
        max_features: int = 1500,
        fast_threshold: int = 15,
        grid_cols: int = 12,
        grid_rows: int = 4,
        min_distance_px: int = 8,
        replenish_fraction: float = 0.7,
        epipolar_tolerance_px: float = 1.5,
        min_disparity: float = 1.0,
        max_disparity: float = 160.0,
        max_hamming: int = 60,
        ratio: float = 0.9,
        subpixel: bool = True,
        klt_window: int = 21,
        klt_levels: int = 4,
        circular_tolerance_px: float = 1.0,
    ) -> None:
        """Configure detection, stereo matching and tracking.

        Args:
            max_features: Target number of tracked stereo pairs.
            fast_threshold: FAST intensity threshold.
            grid_cols: Detection grid columns (features are bucketed for even coverage).
            grid_rows: Detection grid rows.
            min_distance_px: New features keep at least this distance from existing tracks.
            replenish_fraction: Detect new features when tracks drop below this fraction of max_features.
            epipolar_tolerance_px: Maximum row difference between left and right observations.
            min_disparity: Minimum disparity (sets the maximum depth fx B / d).
            max_disparity: Maximum disparity searched (sets the minimum depth).
            max_hamming: Maximum ORB Hamming distance for a stereo match.
            ratio: Lowe ratio test among epipolar candidates.
            subpixel: Refine the right position with single-level KLT for sub-pixel disparity.
            klt_window: KLT window size (pixels).
            klt_levels: KLT pyramid levels for temporal tracking.
            circular_tolerance_px: Maximum closing error of the circular L/R check.
        """
        self._max_features = max_features  # Track budget.
        self._fast = cv2.FastFeatureDetector.create(threshold=fast_threshold, nonmaxSuppression=True)  # Corner detector.
        self._orb = cv2.ORB.create(nfeatures=max_features * 4, edgeThreshold=19, patchSize=31)  # Descriptor extractor.
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)  # Binary descriptor matcher.
        self._grid = (grid_cols, grid_rows)  # Bucketing grid.
        self._min_distance = min_distance_px  # Spacing from existing tracks.
        self._replenish_fraction = replenish_fraction  # Replenishment trigger.
        self._epi_tol = epipolar_tolerance_px  # Row tolerance.
        self._disparity_range = (min_disparity, max_disparity)  # Valid disparities.
        self._max_hamming = max_hamming  # Descriptor distance gate.
        self._ratio = ratio  # Ratio test.
        self._subpixel = subpixel  # Sub-pixel refinement on/off.
        self._klt_window = (klt_window, klt_window)  # KLT window.
        self._klt_levels = klt_levels  # KLT pyramid depth.
        self._klt_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)  # KLT stopping rule.
        self._circular_tol = circular_tolerance_px  # Circular check tolerance.

    @property
    def min_tracks(self) -> int:
        """Return the replenishment threshold.

        Returns:
            Track count below which detection runs.
        """
        return int(self._replenish_fraction * self._max_features)  # Fraction of the budget.

    def _klt(
        self, image_a: ImageArray, image_b: ImageArray, points: Float32Array, initial: Float32Array | None, levels: int
    ) -> tuple[Float32Array, BoolArray]:
        """Run pyramidal Lucas-Kanade from image A to image B.

        Args:
            image_a: Source image.
            image_b: Target image.
            points: (N, 2) points in image A.
            initial: Optional (N, 2) initial guesses in image B.
            levels: Pyramid levels (0 = single level).

        Returns:
            Tracked (N, 2) points and an (N,) success mask.
        """
        if len(points) == 0:  # Nothing to track.
            return np.zeros((0, 2), dtype=np.float32), np.zeros(0, dtype=bool)  # Empty result.
        source = np.ascontiguousarray(points, dtype=np.float32).reshape(-1, 1, 2)  # OpenCV layout.
        guess = None if initial is None else np.ascontiguousarray(initial, dtype=np.float32).reshape(-1, 1, 2).copy()  # Initial flow.
        flags = cv2.OPTFLOW_USE_INITIAL_FLOW if guess is not None else 0  # Use the guess when given.
        tracked, status, _ = cv2.calcOpticalFlowPyrLK(  # Pyramidal KLT.
            image_a,
            image_b,
            source,
            guess,  # type: ignore[arg-type]  # None is valid without OPTFLOW_USE_INITIAL_FLOW.
            winSize=self._klt_window,  # Window size.
            maxLevel=levels,  # Pyramid depth.
            criteria=self._klt_criteria,  # Stopping rule.
            flags=flags,  # Use the initial guess when given.
        )
        tracked = np.asarray(tracked, dtype=np.float32).reshape(-1, 2)  # Back to (N, 2).
        height, width = image_b.shape[:2]  # Bounds.
        inside = (tracked[:, 0] >= 0) & (tracked[:, 0] < width - 1) & (tracked[:, 1] >= 0) & (tracked[:, 1] < height - 1)  # Within the image.
        return tracked, status.reshape(-1).astype(bool) & inside  # Points and validity.

    def _epipolar_ok(self, left: Float32Array, right: Float32Array) -> BoolArray:
        """Check the rectified epipolar constraint and the disparity range.

        Args:
            left: (N, 2) left points.
            right: (N, 2) right points.

        Returns:
            (N,) mask of geometrically valid pairs.
        """
        disparity = left[:, 0] - right[:, 0]  # Horizontal offset.
        same_row = np.abs(left[:, 1] - right[:, 1]) <= self._epi_tol  # Rectified: same row.
        return same_row & (disparity >= self._disparity_range[0]) & (disparity <= self._disparity_range[1])  # Valid pairs.

    def _bucketed_fast(self, image: ImageArray, mask: ImageArray, existing: Float32Array) -> list[cv2.KeyPoint]:
        """Detect FAST corners and keep the strongest per grid cell, topping up cells around existing tracks.

        Args:
            image: Left image.
            mask: Detection mask (0 near existing tracks).
            existing: (M, 2) existing track positions.

        Returns:
            Selected keypoints.
        """
        keypoints = self._fast.detect(image, mask)  # All corners.
        if not keypoints:  # Textureless image.
            return []  # Nothing to add.
        cols, rows = self._grid  # Grid size.
        height, width = image.shape[:2]  # Image size.
        quota = int(np.ceil(self._max_features / (cols * rows)))  # Target features per cell.

        def cell_of(x: Float32Array, y: Float32Array) -> np.ndarray:
            """Linear cell index of pixel positions."""
            return np.minimum((y * rows / height).astype(int), rows - 1) * cols + np.minimum((x * cols / width).astype(int), cols - 1)  # Row-major.

        occupied = (
            np.bincount(cell_of(existing[:, 0], existing[:, 1]), minlength=cols * rows) if len(existing) else np.zeros(cols * rows, dtype=int)
        )  # Existing per cell.
        points = np.array([kp.pt for kp in keypoints], dtype=np.float32)  # Corner positions.
        responses = np.array([kp.response for kp in keypoints], dtype=np.float32)  # Corner strengths.
        cells = cell_of(points[:, 0], points[:, 1])  # Cell of each corner.
        order = np.lexsort((-responses, cells))  # Group by cell, strongest first.
        selected: list[cv2.KeyPoint] = []  # Output.
        rank = 0  # Position within the current cell.
        for position, index in enumerate(order):  # Walk corners cell by cell.
            if position > 0 and cells[index] != cells[order[position - 1]]:  # New cell.
                rank = 0  # Restart the count.
            if rank < quota - occupied[cells[index]]:  # Cell still has room.
                selected.append(keypoints[index])  # Keep this corner.
            rank += 1  # Next rank.
        return selected  # Bucketed corners.

    def detect(self, left: ImageArray, right: ImageArray, existing: StereoTracks) -> StereoTracks:
        """Detect FAST corners, describe with ORB and match along epipolar lines.

        Args:
            left: Rectified left image.
            right: Rectified right image.
            existing: Pairs already tracked in this frame.

        Returns:
            Newly detected stereo pairs.
        """
        mask = np.full(left.shape[:2], 255, dtype=np.uint8)  # Allow detection everywhere...
        for x, y in existing.left:  # ...except near existing tracks.
            cv2.circle(mask, (int(round(float(x))), int(round(float(y)))), self._min_distance, 0, -1)  # Block a disc.
        left_kps = self._bucketed_fast(left, mask, existing.left)  # Bucketed left corners.
        right_kps = list(self._fast.detect(right, None))  # All right corners (candidates).
        if not left_kps or len(right_kps) < 2:  # Nothing to match.
            return StereoTracks.empty()  # No new pairs.
        left_described, left_desc = self._orb.compute(left, left_kps)  # ORB descriptors (drops border corners).
        right_described, right_desc = self._orb.compute(right, right_kps)  # Same on the right.
        if left_desc is None or right_desc is None or len(left_described) == 0 or len(right_described) < 2:  # Descriptors failed.
            return StereoTracks.empty()  # No new pairs.
        pts_l = np.array([kp.pt for kp in left_described], dtype=np.float32)  # Left positions.
        pts_r = np.array([kp.pt for kp in right_described], dtype=np.float32)  # Right positions.
        disparity = pts_l[:, None, 0] - pts_r[None, :, 0]  # (N, M) candidate disparities.
        allowed = (
            (np.abs(pts_l[:, None, 1] - pts_r[None, :, 1]) <= self._epi_tol)
            & (disparity >= self._disparity_range[0])
            & (disparity <= self._disparity_range[1])
        )  # Epipolar band.
        allowed_u8 = allowed.astype(np.uint8)  # OpenCV mask format.
        forward: dict[int, int] = {}  # Left index -> right index.
        for candidates in self._bf.knnMatch(left_desc, right_desc, k=2, mask=allowed_u8):  # Best two candidates per left corner.
            if not candidates or candidates[0].distance > self._max_hamming:  # No acceptable candidate.
                continue  # Skip.
            if len(candidates) == 2 and candidates[0].distance >= self._ratio * candidates[1].distance:  # Ambiguous.
                continue  # Skip.
            forward[candidates[0].queryIdx] = candidates[0].trainIdx  # Keep.
        backward = {
            m.queryIdx: m.trainIdx for m in self._bf.match(right_desc, left_desc, mask=np.ascontiguousarray(allowed_u8.T))
        }  # Right -> left best.
        pairs = [(i, j) for i, j in forward.items() if backward.get(j) == i]  # Mutual matches only.
        if not pairs:  # No stereo matches.
            return StereoTracks.empty()  # No new pairs.
        index_l, index_r = (np.array(v, dtype=np.int64) for v in zip(*pairs, strict=True))  # Split indices.
        new_left, new_right = pts_l[index_l], pts_r[index_r]  # Matched positions.
        if self._subpixel:  # Refine the right position with single-level KLT seeded at the FAST match.
            refined, ok = self._klt(left, right, new_left, new_right, levels=0)  # Sub-pixel right position.
            ok &= np.linalg.norm(refined - new_right, axis=1) < 2.0  # Reject large jumps.
            new_left, new_right = new_left[ok], refined[ok]  # Keep refined pairs.
        valid = self._epipolar_ok(new_left, new_right)  # Re-check geometry after refinement.
        return StereoTracks(new_left[valid], new_right[valid])  # New pairs.

    def track(
        self,
        previous: tuple[ImageArray, ImageArray],
        current: tuple[ImageArray, ImageArray],
        tracks: StereoTracks,
        predicted_left: Float32Array | None = None,
    ) -> tuple[StereoTracks, BoolArray]:
        """Track pairs with KLT around the circle prev L -> cur L -> cur R -> prev R.

        Args:
            previous: (left, right) images of the previous frame.
            current: (left, right) images of the current frame.
            tracks: Pairs of the previous frame.
            predicted_left: Optional predicted positions in the current left image.

        Returns:
            Surviving current-frame pairs and the (N,) survival mask over ``tracks``.
        """
        if len(tracks) == 0:  # Nothing to track.
            return StereoTracks.empty(), np.zeros(0, dtype=bool)  # Empty result.
        (prev_l, prev_r), (cur_l, cur_r) = previous, current  # Unpack images.
        left_1, ok = self._klt(prev_l, cur_l, tracks.left, predicted_left, self._klt_levels)  # prev L -> cur L.
        right_guess = left_1 - np.column_stack([tracks.disparity, np.zeros(len(tracks), dtype=np.float32)])  # Assume similar disparity.
        right_1, ok_r = self._klt(cur_l, cur_r, left_1, right_guess, 2)  # cur L -> cur R (stereo).
        back_guess = right_1 + (tracks.left - left_1)  # Undo the temporal motion as the guess.
        right_0, ok_b = self._klt(cur_r, prev_r, right_1, back_guess, self._klt_levels)  # cur R -> prev R.
        closes = np.linalg.norm(right_0 - tracks.right, axis=1) < self._circular_tol  # Circle closes at the stored prev R.
        keep = ok & ok_r & ok_b & closes & self._epipolar_ok(left_1, right_1)  # All checks.
        return StereoTracks(left_1[keep], right_1[keep]), keep  # Survivors.
