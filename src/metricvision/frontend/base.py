"""The front-end interface: the only thing that changes between experiments.

Every front end answers one question: "which pixels in image A correspond to
which pixels in image B?". Framing it pairwise (rather than as keypoint tracks)
lets sparse methods (ORB, SIFT, SuperPoint, ALIKED) and dense matchers
(MASt3R) plug into exactly the same back end.

Class hierarchy::

    FeatureMatcher                 (ABC: match two images)
    └── SparseFeatureMatcher       (ABC: extract per image, cache, then match features)
        └── OpenCVFeatureMatcher   (ABC: OpenCV detector + brute-force kNN matching)
            ├── OrbMatcher
            └── SiftMatcher
"""

from __future__ import annotations  # Postponed evaluation of type hints.

from abc import ABC, abstractmethod  # Abstract base class machinery.
from collections import OrderedDict  # Insertion-ordered dict used as an LRU cache.
from typing import Any, ClassVar  # Loosely typed OpenCV objects and class-level constants.

import cv2  # Brute-force descriptor matching.
import numpy as np  # Array construction.

from vobench.core.registry import Registry  # Name -> class lookup for configs.
from vobench.core.types import Correspondences, Features, ImageArray, ImageView  # Shared types.


class FeatureMatcher(ABC):
    """Abstract front end that produces 2D-2D correspondences between two images."""

    name: ClassVar[str] = "abstract"  # Short identifier used in output folder names; subclasses override.

    @abstractmethod
    def match(self, view_a: ImageView, view_b: ImageView) -> Correspondences:
        """Find correspondences between two images.

        Args:
            view_a: Reference image (typically the keyframe).
            view_b: Query image (typically the current frame).

        Returns:
            Matched pixel coordinates in both images, with confidences.
        """

    def synchronize(self) -> None:
        """Block until all asynchronous (GPU) work has finished.

        CPU matchers need nothing here. GPU matchers must override this with
        ``torch.cuda.synchronize()`` so runtime measurements are correct.
        """
        return None  # CPU work is synchronous, so there is nothing to wait for.

    def reset(self) -> None:
        """Clear any internal state (caches) before processing a new sequence."""
        return None  # Stateless by default.


class SparseFeatureMatcher(FeatureMatcher, ABC):
    """Front end that extracts features per image, then matches feature sets.

    Features are cached by ``frame_id`` in a small LRU cache, so a keyframe is
    processed once even though every following frame is matched against it.
    """

    def __init__(self, cache_size: int = 8) -> None:
        """Set up the feature cache.

        Args:
            cache_size: Maximum number of images whose features are kept in memory.
        """
        self._cache: OrderedDict[int, Features] = OrderedDict()  # frame_id -> features, oldest first.
        self._cache_size = cache_size  # Upper bound on cached entries.

    @abstractmethod
    def extract(self, image: ImageArray) -> Features:
        """Detect keypoints and compute descriptors for one image.

        Args:
            image: Grayscale image.

        Returns:
            Keypoints, descriptors and detector scores.
        """

    @abstractmethod
    def match_features(self, features_a: Features, features_b: Features) -> Correspondences:
        """Match two precomputed feature sets.

        Args:
            features_a: Features of the reference image.
            features_b: Features of the query image.

        Returns:
            Matched keypoint coordinates with confidences.
        """

    def features_for(self, view: ImageView) -> Features:
        """Return features for an image, extracting them only on a cache miss.

        Args:
            view: Image with a stable identifier.

        Returns:
            The image's features.
        """
        if view.frame_id in self._cache:  # Cache hit.
            self._cache.move_to_end(view.frame_id)  # Mark as most recently used.
            return self._cache[view.frame_id]  # Reuse previous extraction.
        features = self.extract(view.image)  # Cache miss: run the extractor.
        self._cache[view.frame_id] = features  # Store the result.
        if len(self._cache) > self._cache_size:  # Cache over capacity.
            self._cache.popitem(last=False)  # Evict the least recently used entry.
        return features  # Freshly extracted features.

    def match(self, view_a: ImageView, view_b: ImageView) -> Correspondences:
        """Extract (or reuse) features for both images, then match them.

        Args:
            view_a: Reference image.
            view_b: Query image.

        Returns:
            Correspondences between the two images.
        """
        return self.match_features(self.features_for(view_a), self.features_for(view_b))  # Extract-then-match.

    def reset(self) -> None:
        """Empty the feature cache."""
        self._cache.clear()  # Drop all cached features.


class OpenCVFeatureMatcher(SparseFeatureMatcher, ABC):
    """Shared logic for OpenCV detectors matched with brute force, a ratio test and a mutual check.

    Subclasses only provide the detector and the descriptor distance norm, so
    ORB and SIFT are matched with identical logic and thresholds.
    """

    def __init__(
        self,
        detector: Any,
        norm_type: int,
        ratio: float = 0.8,
        mutual_check: bool = True,
        cache_size: int = 8,
    ) -> None:
        """Configure detection and matching.

        Args:
            detector: An OpenCV Feature2D object (e.g. from ``cv2.ORB.create``).
            norm_type: Descriptor distance, ``cv2.NORM_HAMMING`` or ``cv2.NORM_L2``.
            ratio: Lowe ratio-test threshold; a match survives if best < ratio * second best.
            mutual_check: Keep only matches that are also nearest neighbours in the reverse direction.
            cache_size: Feature cache capacity.
        """
        super().__init__(cache_size=cache_size)  # Initialise the LRU cache.
        self._detector = detector  # Keypoint detector and descriptor extractor.
        self._bf = cv2.BFMatcher(norm_type, crossCheck=False)  # Brute force; crossCheck off because we need kNN.
        self._ratio = ratio  # Ratio-test threshold.
        self._mutual_check = mutual_check  # Whether to enforce mutual nearest neighbours.

    def extract(self, image: ImageArray) -> Features:
        """Detect keypoints and compute descriptors with the OpenCV detector.

        Args:
            image: Grayscale image.

        Returns:
            Keypoints, descriptors and detector responses.
        """
        keypoints, descriptors = self._detector.detectAndCompute(image, None)  # No mask: use the whole image.
        if descriptors is None or len(keypoints) == 0:  # Nothing detected (e.g. a blank wall).
            return Features(  # Return an empty but well-formed feature set.
                keypoints=np.zeros((0, 2), dtype=np.float32),  # No locations.
                descriptors=np.zeros((0, 1), dtype=np.uint8),  # No descriptors.
                scores=np.zeros((0,), dtype=np.float32),  # No responses.
            )
        return Features(  # Convert OpenCV objects to plain arrays.
            keypoints=np.array([kp.pt for kp in keypoints], dtype=np.float32).reshape(-1, 2),  # (x, y) per keypoint.
            descriptors=descriptors,  # Already an (N, D) array.
            scores=np.array([kp.response for kp in keypoints], dtype=np.float32),  # Detector strength.
        )

    def match_features(self, features_a: Features, features_b: Features) -> Correspondences:
        """Brute-force kNN matching with Lowe's ratio test and an optional mutual check.

        Args:
            features_a: Features of the reference image.
            features_b: Features of the query image.

        Returns:
            Surviving matches; confidence is 1 / (1 + descriptor distance).
        """
        if len(features_a) < 2 or len(features_b) < 2:  # kNN with k=2 needs at least two candidates.
            return Correspondences.empty()  # Not enough features to match.
        knn = self._bf.knnMatch(features_a.descriptors, features_b.descriptors, k=2)  # Two nearest neighbours per feature.
        forward: dict[int, tuple[int, float]] = {}  # index in A -> (index in B, distance).
        for pair in knn:  # One entry per feature in A.
            if len(pair) < 2:  # OpenCV can return fewer than k neighbours.
                continue  # Cannot apply the ratio test.
            best, second = pair  # Nearest and second-nearest neighbours.
            if best.distance < self._ratio * second.distance:  # Distinctive enough to trust.
                forward[best.queryIdx] = (best.trainIdx, float(best.distance))  # Keep the candidate.
        if self._mutual_check and forward:  # Enforce that B's best match points back to A.
            backward = {m.queryIdx: m.trainIdx for m in self._bf.match(features_b.descriptors, features_a.descriptors)}  # B -> A.
            forward = {a: (b, d) for a, (b, d) in forward.items() if backward.get(b) == a}  # Keep mutual pairs only.
        if not forward:  # Nothing survived filtering.
            return Correspondences.empty()  # Report zero matches.
        index_a = np.fromiter(forward.keys(), dtype=np.int64)  # Indices into A.
        index_b = np.array([b for b, _ in forward.values()], dtype=np.int64)  # Indices into B.
        distances = np.array([d for _, d in forward.values()], dtype=np.float32)  # Descriptor distances.
        return Correspondences(  # Package the result.
            points_a=features_a.keypoints[index_a],  # Matched locations in A.
            points_b=features_b.keypoints[index_b],  # Matched locations in B.
            scores=(1.0 / (1.0 + distances)).astype(np.float32),  # Smaller distance -> higher confidence.
        )


MATCHERS: Registry[FeatureMatcher] = Registry("matcher")  # Global registry of front ends.