"""Learned pairwise matchers for the dense pipeline: SuperPoint + LightGlue and ALIKED + LightGlue (cvg/LightGlue).

These are the front ends of the dense-pipeline benchmark (``--matcher superpoint_lightglue`` / ``aliked_lightglue``).
The hybrid SuperPoint + LightGlue *stereo* front end of the sparse pipeline and the ROS 2 node lives in
``frontend/learned.py`` (Hugging Face SuperPoint + Kornia LightGlue); its pairwise variant is ``superpoint_lightglue_hf``.

Both plug into the same ``SparseFeatureMatcher`` interface as ORB and SIFT, so they run in the
dense pipeline (keyframe matching + fixed SGBM depth) under identical back-end settings.
Images are processed at native resolution (no resizing), keeping the benchmark's fairness rule.

Requires the optional ``learned`` extra (``uv sync --extra learned``). Network weights are
downloaded by the LightGlue package on first use. SuperPoint's weights are licensed for
non-commercial research use only.
"""

from __future__ import annotations  # Postponed evaluation of type hints.

from typing import Any, ClassVar  # Loosely typed torch objects; class-level constants.

import numpy as np  # Array conversion.

from metricvision.core.types import Correspondences, Features, ImageArray  # Shared types.
from metricvision.frontend.base import MATCHERS, SparseFeatureMatcher  # Registry and caching base class.


class LightGlueMatcher(SparseFeatureMatcher):
    """A learned local-feature extractor matched with LightGlue; subclasses pick the feature type."""

    feature: ClassVar[str] = "superpoint"  # LightGlue feature type (and extractor) to use.

    def __init__(
        self,
        max_keypoints: int = 2000,
        ratio: float = 0.8,
        mutual_check: bool = True,
        detection_threshold: float | None = None,
        filter_threshold: float = 0.1,
        device: str = "cuda",
        cache_size: int = 8,
    ) -> None:
        """Load the extractor and LightGlue.

        Args:
            max_keypoints: Keypoint budget per image (kept equal across extractors for fairness).
            ratio: Accepted for CLI compatibility with ``--matcher`` swaps; LightGlue does its own filtering.
            mutual_check: Accepted for compatibility; LightGlue's assignment is already mutual.
            detection_threshold: Detector score threshold (None keeps the extractor's default).
            filter_threshold: LightGlue match-confidence threshold.
            device: Torch device.
            cache_size: Feature cache capacity.
        """
        super().__init__(cache_size=cache_size)  # Feature cache.
        import torch  # Optional dependency, imported lazily.
        from lightglue import ALIKED, LightGlue, SuperPoint  # Optional dependency.

        self._torch = torch  # Keep a handle.
        self._device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")  # Fall back to CPU.
        extractor_cls: Any = {"superpoint": SuperPoint, "aliked": ALIKED}[self.feature]  # Extractor class.
        conf: dict[str, Any] = {"max_num_keypoints": max_keypoints}  # Keypoint budget.
        if detection_threshold is not None:  # Optional override.
            conf["detection_threshold"] = detection_threshold
        self._extractor = extractor_cls(**conf).eval().to(self._device)  # Detector + descriptor.
        self._matcher = LightGlue(features=self.feature, filter_threshold=filter_threshold).eval().to(self._device)  # Matcher.
        self._image_size: Any = None  # (1, 2) tensor (w, h) of the last image, needed by LightGlue.

    def extract(self, image: ImageArray) -> Features:
        """Detect and describe keypoints at native resolution.

        Args:
            image: Grayscale image.

        Returns:
            Keypoints, descriptors (float32) and detector scores.
        """
        torch = self._torch  # Shorthand.
        with torch.inference_mode():  # No autograd.
            tensor = torch.from_numpy(np.ascontiguousarray(image)).to(self._device).float().div_(255.0)[None]  # (1, H, W).
            feats = self._extractor.extract(tensor, resize=None)  # Native resolution.
        self._image_size = feats["image_size"]  # Same for every frame of a sequence.
        return Features(  # Back to NumPy so the cache and the base class stay framework-agnostic.
            keypoints=feats["keypoints"][0].cpu().numpy().astype(np.float32),
            descriptors=feats["descriptors"][0].cpu().numpy().astype(np.float32),
            scores=feats["keypoint_scores"][0].cpu().numpy().astype(np.float32),
        )

    def match_features(self, features_a: Features, features_b: Features) -> Correspondences:
        """Match two feature sets with LightGlue.

        Args:
            features_a: Reference image features.
            features_b: Query image features.

        Returns:
            Matched keypoints with LightGlue confidences.
        """
        if len(features_a) == 0 or len(features_b) == 0:  # Nothing to match.
            return Correspondences.empty()
        torch = self._torch  # Shorthand.

        def side(features: Features) -> dict[str, Any]:
            """LightGlue input for one image."""
            return {
                "keypoints": torch.from_numpy(features.keypoints).to(self._device)[None],
                "descriptors": torch.from_numpy(features.descriptors).to(self._device)[None],
                "image_size": self._image_size,
            }

        with torch.inference_mode():  # No autograd.
            out = self._matcher({"image0": side(features_a), "image1": side(features_b)})  # Match.
            matches = out["matches"][0].cpu().numpy()  # (K, 2) index pairs.
            scores = out["scores"][0].cpu().numpy().astype(np.float32)  # (K,) confidences.
        if matches.size == 0:  # No matches.
            return Correspondences.empty()
        return Correspondences(features_a.keypoints[matches[:, 0]], features_b.keypoints[matches[:, 1]], scores)  # Matched points.

    def synchronize(self) -> None:
        """Wait for queued GPU work so the front-end timer is honest."""
        if self._device.type == "cuda":  # Only meaningful on GPU.
            self._torch.cuda.synchronize(self._device)


@MATCHERS.register("superpoint_lightglue")
class SuperPointLightGlueMatcher(LightGlueMatcher):
    """SuperPoint keypoints and descriptors matched with LightGlue."""

    name = "superpoint_lightglue"  # Output folder name.
    feature = "superpoint"  # LightGlue feature type.


@MATCHERS.register("aliked_lightglue")
class AlikedLightGlueMatcher(LightGlueMatcher):
    """ALIKED keypoints and descriptors matched with LightGlue."""

    name = "aliked_lightglue"  # Output folder name.
    feature = "aliked"  # LightGlue feature type.
