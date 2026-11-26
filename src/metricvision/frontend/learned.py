"""Learned front ends: SuperPoint keypoints + LightGlue matching, inside the rigid-rig stereo geometry.

Two registrations:

* ``MATCHERS["superpoint_lightglue_hf"]``: a pairwise matcher for the dense pipeline (keyframe -> frame) using these
  same networks. (``MATCHERS["superpoint_lightglue"]`` is the official cvg/LightGlue implementation in
  ``frontend/lightglue_matchers.py``, used for the front-end benchmark.)
* ``STEREO_FRONTENDS["superpoint_lightglue"]``: the hybrid stereo front end for the sparse pipeline.
  Per frame: SuperPoint on both images, LightGlue left<->right (stereo) and previous-left<->current-left
  (temporal). The network proposes correspondences; the rig decides:
    - stereo matches must satisfy the rectified epipolar constraint (same row, positive disparity
      within range): LightGlue knows nothing about the rig, so this rejects wrong depths;
    - temporal matches must land near the constant-velocity prediction (replaces the circular check);
    - SuperPoint keypoints are only pixel-accurate (NMS on an 1/8-resolution grid), but depth error
      grows with disparity error, so positions are refined to sub-pixel with KLT seeded at the match.
  Each track carries ``min(stereo confidence, temporal confidence)`` so the rig LM can weight it.

SuperPoint weights come from the Hugging Face hub (``transformers``), LightGlue weights from Kornia.
"""

from __future__ import annotations  # Postponed evaluation of type hints.

from typing import Any, ClassVar  # Loosely typed torch objects.

import numpy as np  # Array maths.
from scipy.spatial import cKDTree  # Nearest-neighbour lookups of track positions.

from metricvision.core.types import BoolArray, Correspondences, Features, Float32Array, ImageArray  # Shared types.
from metricvision.dynamic import pixels_in_mask  # Forbidden-region lookups.
from metricvision.frontend.base import MATCHERS, SparseFeatureMatcher  # Pairwise registry and caching base.
from metricvision.frontend.stereo_klt import STEREO_FRONTENDS, FastOrbKltFrontend, StereoTracks  # Stereo registry and KLT/epipolar helpers.


class LearnedModels:
    """Lazily loads SuperPoint and LightGlue once per process and runs them on the GPU."""

    _instance: ClassVar[LearnedModels | None] = None  # Process-wide singleton (models are large).

    def __init__(self, max_keypoints: int, keypoint_threshold: float, device: str | None) -> None:
        """Load the pretrained networks.

        Args:
            max_keypoints: SuperPoint keypoint budget per image.
            keypoint_threshold: SuperPoint detection threshold.
            device: Torch device; defaults to CUDA when available.
        """
        import kornia.feature as kf  # LightGlue (optional dependency, extra "learned").
        import torch  # Optional dependency.
        from transformers import SuperPointForKeypointDetection  # SuperPoint.

        self.torch = torch  # Handle.
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")  # Prefer the GPU.
        self.superpoint: Any = (
            SuperPointForKeypointDetection.from_pretrained(  # Pretrained SuperPoint (MagicLeap weights).
                "magic-leap-community/superpoint", keypoint_threshold=keypoint_threshold, max_keypoints=max_keypoints
            )
            .to(self.device)
            .eval()
        )
        self.lightglue: Any = kf.LightGlue(features="superpoint").to(self.device).eval()  # Pretrained LightGlue.

    @classmethod
    def get(cls, max_keypoints: int = 2048, keypoint_threshold: float = 0.005, device: str | None = None) -> LearnedModels:
        """Return the shared instance, loading it on first use.

        Args:
            max_keypoints: SuperPoint keypoint budget (first call only).
            keypoint_threshold: SuperPoint threshold (first call only).
            device: Torch device (first call only).

        Returns:
            The loaded models.
        """
        if cls._instance is None:  # First use.
            cls._instance = cls(max_keypoints, keypoint_threshold, device)  # Load.
        return cls._instance  # Shared.

    def extract(self, image: ImageArray) -> tuple[Float32Array, Float32Array, Float32Array]:
        """Run SuperPoint on one grayscale image.

        Args:
            image: Grayscale uint8 image.

        Returns:
            (N, 2) pixel keypoints, (N, 256) descriptors and (N,) scores.
        """
        torch = self.torch  # Alias.
        height, width = image.shape[:2]  # Size.
        pad_h, pad_w = (-height) % 8, (-width) % 8  # SuperPoint works on an 8 px cell grid.
        padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode="edge")  # Pad to multiples of 8.
        with torch.inference_mode():  # No gradients.
            x = torch.from_numpy(padded).to(self.device).float().div(255.0)  # [0, 1].
            x = x.view(1, 1, *padded.shape).expand(1, 3, *padded.shape)  # The HF model reads one channel of RGB input.
            out = self.superpoint(pixel_values=x)  # Detect + describe.
            mask = out.mask[0].bool()  # Valid keypoints.
            kps = out.keypoints[0][mask].float()  # Relative (x, y) in [0, 1].
            kps = kps * torch.tensor([padded.shape[1], padded.shape[0]], device=self.device, dtype=kps.dtype)  # To pixels.
            desc = out.descriptors[0][mask].float()  # (N, 256).
            scores = out.scores[0][mask].float()  # (N,).
        inside = (kps[:, 0] < width - 1) & (kps[:, 1] < height - 1)  # Drop keypoints in the padding.
        return kps[inside].cpu().numpy(), desc[inside].cpu().numpy(), scores[inside].cpu().numpy()  # NumPy.

    def match(
        self, kps_a: Float32Array, desc_a: Float32Array, size_a: tuple[int, int], kps_b: Float32Array, desc_b: Float32Array, size_b: tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray, Float32Array]:
        """Match two SuperPoint feature sets with LightGlue.

        Args:
            kps_a: (N, 2) keypoints of image A (pixels).
            desc_a: (N, 256) descriptors of A.
            size_a: (width, height) of A.
            kps_b: (M, 2) keypoints of image B.
            desc_b: (M, 256) descriptors of B.
            size_b: (width, height) of B.

        Returns:
            Matched indices into A and B, and the (K,) match confidences.
        """
        torch = self.torch  # Alias.
        if len(kps_a) < 2 or len(kps_b) < 2:  # Nothing to match.
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)  # Empty.
        with torch.inference_mode():  # No gradients.

            def pack(kps: Float32Array, desc: Float32Array, size: tuple[int, int]) -> dict[str, Any]:
                """LightGlue input for one image."""
                return {
                    "keypoints": torch.from_numpy(kps).to(self.device)[None],  # (1, N, 2).
                    "descriptors": torch.from_numpy(desc).to(self.device)[None],  # (1, N, 256).
                    "image_size": torch.tensor([size], device=self.device, dtype=torch.float32),  # (1, 2) as (W, H).
                }

            out = self.lightglue({"image0": pack(kps_a, desc_a, size_a), "image1": pack(kps_b, desc_b, size_b)})  # Attention matching.
            matches = out["matches"][0].cpu().numpy()  # (K, 2) index pairs.
            scores = out["scores"][0].float().cpu().numpy()  # (K,) confidences.
        return matches[:, 0].astype(np.int64), matches[:, 1].astype(np.int64), scores.astype(np.float32)  # Split.

    def synchronize(self) -> None:
        """Wait for queued GPU work (honest timing)."""
        if self.device.startswith("cuda"):  # GPU only.
            self.torch.cuda.synchronize()  # Block.


@MATCHERS.register("superpoint_lightglue_hf")
class SuperPointLightGlueMatcher(SparseFeatureMatcher):
    """Pairwise learned matcher for the dense pipeline: SuperPoint features, LightGlue matching."""

    name = "superpoint_lightglue_hf"  # Output folder name.

    def __init__(self, max_keypoints: int = 2000, ratio: float = 0.8, mutual_check: bool = True, min_confidence: float = 0.0) -> None:
        """Configure the matcher.

        Args:
            max_keypoints: SuperPoint budget (kept equal to ORB/SIFT for fairness).
            ratio: Accepted for config compatibility (LightGlue has no ratio test).
            mutual_check: Accepted for config compatibility (LightGlue matches are mutual by construction).
            min_confidence: Drop matches below this LightGlue confidence.
        """
        super().__init__()  # Feature cache.
        self._models = LearnedModels.get(max_keypoints=max_keypoints)  # Shared networks.
        self._min_confidence = min_confidence  # Gate.
        self._sizes: dict[int, tuple[int, int]] = {}  # Image size per descriptor-array identity.

    def extract(self, image: ImageArray) -> Features:
        """Run SuperPoint.

        Args:
            image: Grayscale image.

        Returns:
            Keypoints, descriptors and scores.
        """
        kps, desc, scores = self._models.extract(image)  # Network.
        self._sizes[id(desc)] = (image.shape[1], image.shape[0])  # Remember the size for LightGlue normalisation.
        return Features(keypoints=kps, descriptors=desc, scores=scores)  # Package.

    def match_features(self, features_a: Features, features_b: Features) -> Correspondences:
        """Match with LightGlue.

        Args:
            features_a: Reference features.
            features_b: Query features.

        Returns:
            Correspondences with LightGlue confidences as scores.
        """
        size_a = self._sizes.get(id(features_a.descriptors), (1, 1))  # Width, height.
        size_b = self._sizes.get(id(features_b.descriptors), (1, 1))  # Width, height.
        ia, ib, conf = self._models.match(
            features_a.keypoints, features_a.descriptors, size_a, features_b.keypoints, features_b.descriptors, size_b
        )  # Match.
        keep = conf >= self._min_confidence  # Gate.
        return Correspondences(features_a.keypoints[ia[keep]], features_b.keypoints[ib[keep]], conf[keep])  # Package.

    def synchronize(self) -> None:
        """Wait for the GPU so the front-end timer is honest."""
        self._models.synchronize()  # torch.cuda.synchronize().

    def reset(self) -> None:
        """Clear caches."""
        super().reset()  # Features.
        self._sizes.clear()  # Sizes.


class _FrameFeatures:
    """SuperPoint features of one stereo frame plus its LightGlue stereo matches (rig-filtered, sub-pixel)."""

    def __init__(
        self, left: ImageArray, kps: Float32Array, desc: Float32Array, scores: Float32Array, right_of: dict[int, tuple[Float32Array, float]]
    ) -> None:
        """Store the features.

        Args:
            left: Left image (identity used as the cache key).
            kps: (N, 2) left keypoints.
            desc: (N, 256) left descriptors.
            scores: (N,) SuperPoint scores.
            right_of: Left keypoint index -> (sub-pixel right position, stereo confidence).
        """
        self.left = left  # Cache key.
        self.kps, self.desc, self.scores = kps, desc, scores  # Left features.
        self.right_of = right_of  # Stereo partners.
        self.position_of: dict[int, Float32Array] = {}  # Keypoint index -> (refined) left position handed out as a track.
        self._tree: cKDTree | None = None  # Lookup of handed-out positions.
        self._tree_index: np.ndarray = np.zeros(0, dtype=np.int64)  # Keypoint index per tree row.

    def assign(self, index: int, position: Float32Array) -> None:
        """Record that keypoint ``index`` is tracked at ``position`` in this frame.

        Args:
            index: Left keypoint index.
            position: Track position (possibly sub-pixel refined).
        """
        self.position_of[index] = position  # Store.
        self._tree = None  # Invalidate the lookup.

    def lookup(self, positions: Float32Array) -> np.ndarray:
        """Map track positions back to keypoint indices (-1 if unknown).

        Args:
            positions: (N, 2) track positions in this frame.

        Returns:
            (N,) keypoint indices.
        """
        if not self.position_of:  # Nothing handed out.
            return np.full(len(positions), -1, dtype=np.int64)  # Unknown.
        if self._tree is None:  # Build lazily.
            self._tree_index = np.fromiter(self.position_of.keys(), dtype=np.int64)  # Indices.
            self._tree = cKDTree(np.array([self.position_of[i] for i in self._tree_index]))  # Positions.
        dist, row = self._tree.query(positions, k=1)  # Nearest handed-out position.
        return np.where(dist < 1e-3, self._tree_index[np.minimum(row, len(self._tree_index) - 1)], -1)  # Exact matches only.


@STEREO_FRONTENDS.register("superpoint_lightglue")
class SuperPointLightGlueStereoFrontend(FastOrbKltFrontend):
    """Hybrid stereo front end: SuperPoint + LightGlue correspondences, rigid-rig checks, KLT sub-pixel refinement."""

    name = "superpoint_lightglue"  # Output folder name.

    def __init__(self, max_keypoints: int = 2048, min_confidence: float = 0.2, prediction_gate_px: float = 40.0, **kwargs: Any) -> None:
        """Configure the learned front end (rig/KLT settings are shared with ``fast_orb_klt``).

        Args:
            max_keypoints: SuperPoint budget per image.
            min_confidence: LightGlue matches below this confidence are dropped.
            prediction_gate_px: Temporal matches must land within this distance of the motion prediction.
            **kwargs: ``FastOrbKltFrontend`` settings (max_features, epipolar_tolerance_px, disparity range, KLT window...).
        """
        kwargs.pop("fast_threshold", None)  # Classical-only option: ignore if the config carries it.
        kwargs.pop("circular_tolerance_px", None)  # Replaced by the prediction gate.
        super().__init__(**kwargs)  # Rig, KLT and budget settings.
        self._models = LearnedModels.get(max_keypoints=max_keypoints)  # Shared networks.
        self._min_conf = min_confidence  # Gate.
        self._gate = prediction_gate_px  # Temporal gate.
        self._current: _FrameFeatures | None = None  # Features of the latest frame.

    def reset(self) -> None:
        """Forget cached features."""
        self._current = None  # Clear.

    def synchronize(self) -> None:
        """Wait for the GPU so the front-end timer is honest."""
        self._models.synchronize()  # torch.cuda.synchronize().

    def _features(self, left: ImageArray, right: ImageArray) -> _FrameFeatures:
        """SuperPoint on both images, LightGlue stereo matching, rig filtering and sub-pixel refinement (cached per frame).

        Args:
            left: Rectified left image.
            right: Rectified right image.

        Returns:
            The frame's features.
        """
        if self._current is not None and self._current.left is left:  # Already computed for this frame.
            return self._current  # Reuse.
        kps_l, desc_l, scores_l = self._models.extract(left)  # Left SuperPoint.
        kps_r, desc_r, _ = self._models.extract(right)  # Right SuperPoint.
        size = (left.shape[1], left.shape[0])  # (W, H).
        il, ir, conf = self._models.match(kps_l, desc_l, size, kps_r, desc_r, size)  # LightGlue stereo.
        keep = conf >= self._min_conf  # Confidence gate.
        il, ir, conf = il[keep], ir[keep], conf[keep]  # Confident matches.
        pts_l, pts_r = kps_l[il], kps_r[ir]  # Matched positions.
        if self._subpixel and len(il):  # Depth needs sub-pixel disparity: refine the right point with KLT.
            refined, ok = self._klt(left, right, pts_l, pts_r, levels=0)  # Seeded at the LightGlue match.
            ok &= np.linalg.norm(refined - pts_r, axis=1) < 2.0  # Reject large jumps.
            il, pts_l, pts_r, conf = il[ok], pts_l[ok], refined[ok], conf[ok]  # Refined pairs.
        rig = self._epipolar_ok(pts_l, pts_r)  # Rigid-rig constraint: same row, disparity in range.
        right_of = {int(i): (p, float(c)) for i, p, c in zip(il[rig], pts_r[rig], conf[rig], strict=True)}  # Stereo partners.
        self._current = _FrameFeatures(left, kps_l, desc_l, scores_l, right_of)  # Cache.
        return self._current  # Done.

    def detect(self, left: ImageArray, right: ImageArray, existing: StereoTracks, forbidden: BoolArray | None = None) -> StereoTracks:
        """New stereo tracks from this frame's rig-consistent LightGlue stereo matches, away from existing tracks.

        Args:
            left: Rectified left image.
            right: Rectified right image.
            existing: Tracks already present in this frame.
            forbidden: Optional (H, W) mask of regions where no new tracks may start (e.g. moving objects).

        Returns:
            New tracks only.
        """
        frame = self._features(left, right)  # Cached features.
        candidates = np.array([i for i in frame.right_of if i not in frame.position_of], dtype=np.int64)  # Unused stereo keypoints.
        if candidates.size and forbidden is not None:  # E.g. the dynamic-object mask.
            candidates = candidates[~pixels_in_mask(forbidden, frame.kps[candidates])]  # No tracks on moving objects.
        if candidates.size == 0:  # Nothing new.
            return StereoTracks.empty()  # Empty.
        if len(existing):  # Keep new tracks away from existing ones.
            dist, _ = cKDTree(existing.left).query(frame.kps[candidates], k=1)  # Distance to nearest track.
            candidates = candidates[dist >= self._min_distance]  # Spaced out.
        budget = max(self._max_features - len(existing), 0)  # Room left.
        quality = np.array([frame.scores[i] * frame.right_of[i][1] for i in candidates])  # Detector score x stereo confidence.
        candidates = candidates[np.argsort(-quality)][:budget] if candidates.size else candidates  # Best first.
        left_pts = frame.kps[candidates].astype(np.float32)  # Left positions.
        right_pts = np.array([frame.right_of[i][0] for i in candidates], dtype=np.float32).reshape(-1, 2)  # Right positions.
        conf = np.array([frame.right_of[i][1] for i in candidates], dtype=np.float32)  # Stereo confidence.
        for i, p in zip(candidates.tolist(), left_pts, strict=True):  # Hand them out.
            frame.assign(i, p)  # Remember for the next frame's temporal lookup.
        return StereoTracks(left_pts, right_pts, conf)  # New tracks.

    def track(
        self,
        previous: tuple[ImageArray, ImageArray],
        current: tuple[ImageArray, ImageArray],
        tracks: StereoTracks,
        predicted_left: Float32Array | None = None,
    ) -> tuple[StereoTracks, BoolArray]:
        """Track by LightGlue temporal matching (previous left -> current left), then the current stereo match.

        Args:
            previous: Previous (left, right) images.
            current: Current (left, right) images.
            tracks: Tracks in the previous frame.
            predicted_left: Constant-velocity predictions in the current left image.

        Returns:
            Surviving current-frame tracks and the survival mask.
        """
        n = len(tracks)  # Input size.
        keep = np.zeros(n, dtype=bool)  # Survival mask.
        if n == 0:  # Nothing to track.
            return StereoTracks.empty(), keep  # Empty.
        prev = self._current if self._current is not None and self._current.left is previous[0] else self._features(*previous)  # Previous features.
        cur = self._features(*current)  # Current features (becomes the cache).
        prev_index = prev.lookup(tracks.left)  # Which previous keypoint each track is.
        size = (current[0].shape[1], current[0].shape[0])  # (W, H).
        ip, ic, conf_t = self._models.match(prev.kps, prev.desc, size, cur.kps, cur.desc, size)  # LightGlue temporal.
        temporal = {int(p): (int(c), float(s)) for p, c, s in zip(ip, ic, conf_t, strict=True) if s >= self._min_conf}  # prev -> (cur, conf).
        left_out = np.zeros((n, 2), dtype=np.float32)  # Current left positions.
        right_out = np.zeros((n, 2), dtype=np.float32)  # Current right positions.
        conf_out = np.zeros(n, dtype=np.float32)  # Track confidence.
        index_out = np.full(n, -1, dtype=np.int64)  # Current keypoint per surviving track.
        used: set[int] = set(cur.position_of)  # Current keypoints may be used once.
        for row, pi in enumerate(prev_index.tolist()):  # Each track.
            hit = temporal.get(pi) if pi >= 0 else None  # Temporal match.
            if hit is None or hit[0] not in cur.right_of or hit[0] in used:  # Needs a stereo partner, used once.
                continue  # Lost.
            ci, ct = hit  # Current keypoint and temporal confidence.
            position = cur.kps[ci]  # Current left keypoint.
            if predicted_left is not None and np.linalg.norm(position - predicted_left[row]) > self._gate:  # Far from the motion prediction.
                continue  # Reject (wrong temporal match).
            right_pos, cs = cur.right_of[ci]  # Stereo partner.
            left_out[row], right_out[row], conf_out[row] = position, right_pos, min(ct, cs)  # Track state.
            index_out[row] = ci  # Remember the keypoint.
            keep[row] = True  # Survives.
            used.add(ci)  # Taken.
        rows = np.flatnonzero(keep)  # Surviving rows.
        if self._subpixel and rows.size:  # LightGlue proposes, KLT refines: sub-pixel left (temporal) then right (stereo).
            seed_l = left_out[rows]  # LightGlue positions (pixel-quantised keypoints).
            refined_l, ok = self._klt(previous[0], current[0], tracks.left[rows], seed_l, levels=2)  # Previous track -> current left.
            ok &= np.linalg.norm(refined_l - seed_l, axis=1) < 2.0  # Stay close to the learned match.
            seed_r = right_out[rows] + (refined_l - seed_l)  # Shift the stereo seed by the left refinement.
            refined_r, ok_r = self._klt(current[0], current[1], refined_l, seed_r, levels=0)  # Sub-pixel disparity.
            ok &= ok_r & (np.linalg.norm(refined_r - seed_r, axis=1) < 2.0) & self._epipolar_ok(refined_l, refined_r)  # Rig check again.
            left_out[rows], right_out[rows] = refined_l, refined_r  # Refined state.
            keep[rows[~ok]] = False  # Drop failed refinements (their keypoints stay free for detection).
        for row in np.flatnonzero(keep).tolist():  # Hand out survivors for the next frame's lookup.
            cur.assign(int(index_out[row]), left_out[row])  # Refined position -> keypoint.
        return StereoTracks(left_out[keep], right_out[keep], conf_out[keep]), keep  # Survivors.
