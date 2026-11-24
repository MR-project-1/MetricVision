"""Keyframe window for the sparse pipeline: stores stereo observations, gates monocular depth priors, runs windowed BA.

Depth-prior gating follows the idea of RoMeO (Cheng et al., 2024) of only trusting a monocular
depth prior when it is consistent with the geometry, adapted to a stereo rig:

1. *Keyframe scale check.* On landmarks whose stereo depth is already precise
   (sigma_Z / Z below ``reliable_rel_sigma``), compute the median ratio r = Z_stereo / d*.
   With ``scale_align`` the network depth is rescaled by r (stereo fixes the metric scale,
   the network contributes relative depth where stereo is weak); without it, the whole
   keyframe's priors are dropped when |r - 1| > ``scale_gate``.
2. *Per-landmark consistency.* A prior survives only if |Z_stereo - d*| is within
   ``gate_sigmas`` combined standard deviations (stereo sigma_Z and prior sigma).
3. *Only where stereo is weak.* Priors are kept only for landmarks whose stereo depth is
   uncertain (sigma_Z / Z above ``min_stereo_rel_sigma``); near landmarks keep pure stereo.
4. *One prior per landmark per window.* Network errors are strongly correlated across nearby
   keyframes, so only the newest keyframe's prior enters the BA (independent priors from every
   keyframe would make the network look far more certain than it is).
5. *Trust region.* If the BA moves the newest keyframe further than ``max_correction_m`` /
   ``max_correction_deg`` from the tracking estimate, the whole update is rejected.
"""

from __future__ import annotations  # Postponed evaluation of type hints.

from dataclasses import dataclass  # Containers.

import numpy as np  # Array maths.

from metricvision.backend.window_ba import IndexArray, WindowBAConfig, WindowBAProblem, windowed_ba  # Optimiser.
from metricvision.core.geometry import invert_pose  # Rigid inverse.
from metricvision.core.types import Float32Array, FloatArray  # Shared types.


@dataclass(frozen=True)
class DepthPriorConfig:
    """Gating and weighting of monocular depth priors.

    Attributes:
        rel_sigma: Prior standard deviation as a fraction of the prior depth.
        gate_sigmas: Per-landmark consistency gate, in combined standard deviations.
        scale_align: Rescale the network depth to the stereo scale per keyframe.
        scale_gate: Without alignment, drop a keyframe's priors when its scale ratio deviates more than this.
        reliable_rel_sigma: Stereo sigma_Z / Z below which a landmark is trusted for the scale check.
        min_scale_points: Minimum trusted landmarks needed for the scale check (else the keyframe gets no priors).
        min_stereo_rel_sigma: Priors are only kept where stereo sigma_Z / Z exceeds this (weak stereo).
        max_correction_m: Reject a BA that moves the newest keyframe further than this (metres).
        max_correction_deg: Reject a BA that rotates the newest keyframe more than this (degrees).
    """

    rel_sigma: float = 0.2  # 20% relative prior noise (measured raw error of DA-V2 Small is ~25%).
    gate_sigmas: float = 3.0  # 3-sigma consistency.
    scale_align: bool = True  # Stereo-anchored scale.
    scale_gate: float = 0.2  # 20% scale tolerance without alignment.
    reliable_rel_sigma: float = 0.03  # Trusted stereo depths.
    min_scale_points: int = 20  # Enough evidence for a scale estimate.
    min_stereo_rel_sigma: float = 0.05  # Priors only where stereo sigma_Z / Z exceeds this.
    max_correction_m: float = 0.5  # Trust region on the newest keyframe (translation).
    max_correction_deg: float = 2.0  # Trust region on the newest keyframe (rotation).


@dataclass
class WindowKeyframe:
    """Observations stored for one keyframe.

    Attributes:
        frame_index: Frame number in the sequence.
        camera_from_world: Pose T_cam,world (refined by BA while in the window).
        ids: (n,) landmark ids observed.
        pixels: (n, 3) observations (u_l, v_l, u_r).
        prior_depth: (n,) gated prior depth d* per observation (NaN = no prior).
    """

    frame_index: int  # Frame number.
    camera_from_world: FloatArray  # Pose.
    ids: IndexArray  # Landmark ids.
    pixels: FloatArray  # Stereo observations.
    prior_depth: FloatArray  # Gated priors.


@dataclass(frozen=True)
class WindowStats:
    """What happened at one keyframe.

    Attributes:
        priors: Depth priors that passed the gates at this keyframe.
        prior_scale: Median stereo/network depth ratio (NaN when not computed).
        landmarks: Landmarks in the BA (0 when BA did not run).
        observations: Stereo observations in the BA.
        prior_factors: Prior factors in the BA.
        initial_cost: BA cost before optimisation.
        final_cost: BA cost after optimisation.
        rejected: The BA update was rejected by the trust region.
    """

    priors: int = 0  # Gated priors added.
    prior_scale: float = float("nan")  # Scale ratio.
    landmarks: int = 0  # BA size.
    observations: int = 0  # BA size.
    prior_factors: int = 0  # BA size.
    initial_cost: float = float("nan")  # BA cost.
    final_cost: float = float("nan")  # BA cost.
    rejected: bool = False  # Trust-region rejection.


class KeyframeWindow:
    """Sliding window of keyframes plus a landmark store, optimised by rigid-rig BA."""

    def __init__(self, size: int, ba_config: WindowBAConfig | None = None, prior_config: DepthPriorConfig | None = None) -> None:
        """Create an empty window.

        Args:
            size: Number of keyframes kept (>= 2).
            ba_config: BA settings.
            prior_config: Depth-prior gating settings.

        Raises:
            ValueError: If size < 2.
        """
        if size < 2:  # BA needs at least one free pose.
            raise ValueError("window size must be at least 2")
        self._size = size  # Capacity.
        self._ba_config = ba_config or WindowBAConfig()  # BA settings.
        self._prior_config = prior_config or DepthPriorConfig()  # Gating settings.
        self._keyframes: list[WindowKeyframe] = []  # Oldest first.
        self._points: dict[int, FloatArray] = {}  # Landmark id -> world position.

    @property
    def num_keyframes(self) -> int:
        """Return the number of keyframes currently in the window.

        Returns:
            Keyframe count.
        """
        return len(self._keyframes)  # Current size.

    def point(self, landmark_id: int) -> FloatArray | None:
        """Return the stored world position of a landmark.

        Args:
            landmark_id: Track id.

        Returns:
            Position, or None if unknown.
        """
        return self._points.get(landmark_id)  # May be None.

    def _gate_priors(
        self, camera_from_world: FloatArray, pixels: FloatArray, points: FloatArray, covariances: FloatArray, depth_map: Float32Array
    ) -> tuple[FloatArray, float]:
        """Sample and gate network depths for one keyframe's observations.

        Args:
            camera_from_world: Keyframe pose T_cam,world.
            pixels: (n, 3) observations (u_l, v_l, u_r).
            points: (n, 3) world landmarks (NaN rows allowed).
            covariances: (n, 3, 3) world covariances.
            depth_map: (H, W) network depth for the keyframe's left image.

        Returns:
            (n,) gated prior depths (NaN where rejected) and the scale ratio (NaN if not computed).
        """
        cfg = self._prior_config  # Shorthand.
        height, width = depth_map.shape  # Map size.
        cols = np.clip(np.rint(pixels[:, 0]).astype(int), 0, width - 1)  # Nearest pixel.
        rows = np.clip(np.rint(pixels[:, 1]).astype(int), 0, height - 1)  # Nearest pixel.
        prior = depth_map[rows, cols].astype(np.float64)  # Network depth at each observation.
        r3 = camera_from_world[2, :3]  # Third row of the rotation (depth axis).
        z_stereo = points @ r3 + camera_from_world[2, 3]  # Landmark depth in this camera.
        sigma_stereo = np.sqrt(np.maximum(np.einsum("i,nij,j->n", r3, covariances, r3), 0.0))  # Stereo depth std.
        valid = np.isfinite(prior) & np.isfinite(z_stereo) & np.isfinite(sigma_stereo) & (z_stereo > 0)  # Comparable.
        reliable = valid & (sigma_stereo < cfg.reliable_rel_sigma * z_stereo)  # Precise stereo depths.
        scale = float("nan")  # Not computed yet.
        if reliable.sum() >= cfg.min_scale_points:  # Enough evidence for a scale check.
            scale = float(np.median(z_stereo[reliable] / prior[reliable]))  # Stereo / network.
            if cfg.scale_align:  # Stereo fixes the scale; the network keeps its relative depth.
                prior = prior * scale  # Rescale.
            elif abs(scale - 1.0) > cfg.scale_gate:  # Network scale is off for this keyframe.
                return np.full(len(pixels), np.nan), scale  # Reject all priors.
        elif cfg.scale_align:  # Cannot align without trusted stereo points.
            return np.full(len(pixels), np.nan), scale  # Reject all priors.
        tolerance = cfg.gate_sigmas * np.sqrt(
            sigma_stereo**2 + (cfg.rel_sigma * z_stereo) ** 2
        )  # Combined uncertainty, relative to the stereo reference (a wrong prior must not widen its own gate).
        accepted = valid & (np.abs(z_stereo - prior) < tolerance) & (sigma_stereo > cfg.min_stereo_rel_sigma * z_stereo)  # Consistent and useful.
        return np.where(accepted, prior, np.nan), scale  # Gated priors.

    def add_keyframe(
        self,
        frame_index: int,
        world_from_camera: FloatArray,
        ids: IndexArray,
        pixels: FloatArray,
        points: FloatArray,
        covariances: FloatArray,
        depth_map: Float32Array | None,
    ) -> WindowStats:
        """Insert a keyframe, update the landmark store and gate its depth priors.

        Args:
            frame_index: Frame number.
            world_from_camera: Current pose T_world,cam.
            ids: (n,) ids of the tracks observed in this frame.
            pixels: (n, 3) their (u_l, v_l, u_r).
            points: (n, 3) their current world anchors (NaN rows allowed).
            covariances: (n, 3, 3) anchor covariances.
            depth_map: Network depth for the left image, or None for no priors.

        Returns:
            Prior statistics for this keyframe.
        """
        camera_from_world = invert_pose(world_from_camera)  # BA works with T_cam,world.
        finite = np.isfinite(points).all(axis=1)  # Usable anchors.
        for landmark_id, point in zip(ids[finite].tolist(), points[finite], strict=True):  # Latest anchors into the store.
            self._points[landmark_id] = point.copy()  # Overwrites older estimates.
        prior, scale = (
            (np.full(len(ids), np.nan), float("nan"))
            if depth_map is None
            else self._gate_priors(camera_from_world, pixels, points, covariances, depth_map)
        )  # Priors.
        keep = finite  # Only observations of known landmarks enter the window.
        self._keyframes.append(
            WindowKeyframe(frame_index, camera_from_world, ids[keep].copy(), pixels[keep].astype(np.float64), prior[keep])
        )  # Store.
        if len(self._keyframes) > self._size:  # Slide.
            self._keyframes.pop(0)  # Drop the oldest.
            referenced = set(np.concatenate([kf.ids for kf in self._keyframes]).tolist())  # Ids still observed in the window.
            self._points = {k: v for k, v in self._points.items() if k in referenced}  # Forget the rest.
        return WindowStats(priors=int(np.isfinite(prior[keep]).sum()), prior_scale=scale)  # Report.

    def optimise(self, camera_matrix: FloatArray, baseline: float, stats: WindowStats) -> tuple[FloatArray | None, WindowStats]:
        """Run windowed BA over the current keyframes.

        Landmarks observed by at least two keyframes are optimised; the oldest keyframe is fixed.

        Args:
            camera_matrix: Rectified intrinsics.
            baseline: Stereo baseline.
            stats: Statistics from ``add_keyframe`` to extend.

        Returns:
            The newest keyframe's refined T_world,cam (None if BA did not run) and updated statistics.
        """
        if len(self._keyframes) < 2:  # Nothing to optimise yet.
            return None, stats
        all_ids = np.concatenate([kf.ids for kf in self._keyframes])  # Every observation's landmark.
        unique, counts = np.unique(all_ids, return_counts=True)  # Observation counts.
        selected = unique[counts >= 2]  # Landmarks linking keyframes.
        if selected.size < 10:  # Too weak to bother.
            return None, stats
        index = {int(landmark_id): i for i, landmark_id in enumerate(selected.tolist())}  # Id -> BA index.
        obs_k: list[IndexArray] = []  # Factor lists: keyframe ids...
        obs_j: list[IndexArray] = []  # ...landmark ids...
        obs_px: list[FloatArray] = []  # ...and pixels.
        newest_prior: dict[int, tuple[int, float]] = {}  # Landmark index -> (keyframe, depth) of its newest prior.
        for k, kf in enumerate(self._keyframes):  # Collect factors.
            mask = np.isin(kf.ids, selected)  # Observations of selected landmarks.
            js = np.array([index[int(i)] for i in kf.ids[mask]], dtype=np.int64)  # BA indices.
            obs_k.append(np.full(js.size, k, dtype=np.int64))  # Keyframe ids.
            obs_j.append(js)  # Landmark ids.
            obs_px.append(kf.pixels[mask])  # Pixels.
            has_prior = np.isfinite(kf.prior_depth[mask])  # Gated priors among them.
            for landmark, depth in zip(js[has_prior].tolist(), kf.prior_depth[mask][has_prior].tolist(), strict=True):  # Later keyframes overwrite.
                newest_prior[landmark] = (k, depth)  # Keep only the newest prior per landmark.
        pri_j = list(newest_prior)  # Landmarks with a prior.
        pri_k = [newest_prior[j][0] for j in pri_j]  # Their keyframes.
        prior_depth = np.array([newest_prior[j][1] for j in pri_j], dtype=np.float64)  # (P,).
        problem = WindowBAProblem(
            poses=np.array([kf.camera_from_world for kf in self._keyframes]),
            fixed=np.arange(len(self._keyframes)) == 0,  # Oldest keyframe fixes the gauge.
            points=np.array([self._points[int(i)] for i in selected.tolist()]),
            obs_keyframe=np.concatenate(obs_k),
            obs_point=np.concatenate(obs_j),
            obs_pixels=np.vstack(obs_px),
            prior_keyframe=np.array(pri_k, dtype=np.int64),
            prior_point=np.array(pri_j, dtype=np.int64),
            prior_depth=prior_depth,
            prior_sigma=self._prior_config.rel_sigma * prior_depth,
        )
        result = windowed_ba(problem, camera_matrix, baseline, self._ba_config)  # Optimise.
        before, after = invert_pose(problem.poses[-1]), invert_pose(result.poses[-1])  # Newest pose, T_world,cam.
        correction = invert_pose(before) @ after  # How far BA moved it.
        cos_angle = np.clip(0.5 * (np.trace(correction[:3, :3]) - 1.0), -1.0, 1.0)  # Rotation of the correction.
        if (
            np.linalg.norm(correction[:3, 3]) > self._prior_config.max_correction_m
            or np.degrees(np.arccos(cos_angle)) > self._prior_config.max_correction_deg
        ):
            return None, WindowStats(priors=stats.priors, prior_scale=stats.prior_scale, rejected=True)  # Trust region: keep tracking's estimate.
        for kf, pose in zip(self._keyframes, result.poses, strict=True):  # Write back poses.
            kf.camera_from_world = pose  # Refined.
        for landmark_id, point in zip(selected.tolist(), result.points, strict=True):  # Write back landmarks.
            self._points[int(landmark_id)] = point  # Refined.
        stats = WindowStats(
            priors=stats.priors,
            prior_scale=stats.prior_scale,
            landmarks=int(selected.size),
            observations=int(problem.obs_point.size),
            prior_factors=int(prior_depth.size),
            initial_cost=result.initial_cost,
            final_cost=result.final_cost,
        )
        return invert_pose(self._keyframes[-1].camera_from_world), stats  # Newest pose as T_world,cam.
