"""Keyframe sliding-window bundle adjustment with the rigid-rig stereo residual.

The window holds the last ``W`` keyframes (poses T_cam_world) and every landmark observed in at least two of them.
The cost is the same robust 3-D residual (u_L, v_L, u_R) as the per-frame LM (``backend.lm``), with one pixel noise
sigma (the odometry's shared noise model) and a Huber kernel on the whitened residual norm:

    min_{T_1..T_W, X_1..X_L}  sum_{(k, j)} rho_H( || r_kj(T_k, X_j) ||^2 / sigma^2 ).

The oldest keyframe is held fixed (gauge); the stereo baseline fixes scale. Each Levenberg-Marquardt step eliminates
the landmarks with the Schur complement: every landmark has a 3x3 block, so the reduced camera system is only
(6 W') x (6 W') for the W' free poses. Landmarks are XYZ in the world; far points are handled by the robust kernel and
by never fixing them (they are constrained by their observations only).

Compared with per-frame pose estimation against fused anchors, this
(i) revises recent poses jointly with the landmarks (pose-landmark correlation is no longer ignored), and
(ii) re-estimates each landmark from all its observations in the window instead of from an information-form fusion
that assumes perfectly known poses.
"""

from __future__ import annotations  # Postponed evaluation of type hints.

from collections import deque  # Bounded keyframe window.
from dataclasses import dataclass  # Containers.

import numpy as np  # Array maths.

from metricvision.backend.lm import stereo_residuals  # Shared residual and analytic Jacobians.
from metricvision.core.geometry import se3_exp  # Pose update.
from metricvision.core.types import FloatArray  # Shared types.

CHI2_STEREO_99 = 11.34  # chi^2(3 dof, 99 %): post-BA outlier threshold.


@dataclass(frozen=True)
class WindowBAConfig:
    """Bundle-adjustment settings.

    Attributes:
        window: Number of keyframes in the window (0 disables bundle adjustment).
        keyframe_every: Insert a keyframe every N tracked frames.
        min_translation_m: ...or as soon as the camera moved this far since the last keyframe.
        iterations: Maximum Levenberg-Marquardt iterations per call.
        huber_delta: Huber threshold on the whitened residual norm (sigma units).
        min_observations: Landmarks need this many keyframe observations to enter the problem.
    """

    window: int = 0  # 0 = off.
    keyframe_every: int = 3  # Keyframe period in frames.
    min_translation_m: float = 1.0  # Keyframe on motion.
    iterations: int = 6  # LM iterations.
    huber_delta: float = 2.0  # Robust threshold.
    min_observations: int = 2  # Multi-view landmarks only.


@dataclass
class Keyframe:
    """One keyframe of the window.

    Attributes:
        frame_index: Source frame.
        camera_from_world: T_cam_world (updated by the optimisation).
        track_ids: (N,) identifiers of the observed landmarks.
        observations: (N, 3) observations (u_L, v_L, u_R).
    """

    frame_index: int
    camera_from_world: FloatArray
    track_ids: np.ndarray
    observations: FloatArray


@dataclass(frozen=True)
class BAResult:
    """Outcome of one optimisation.

    Attributes:
        initial_cost: Robust cost before.
        final_cost: Robust cost after.
        iterations: Accepted iterations.
        num_landmarks: Landmarks in the problem.
        num_observations: Observations in the problem.
        outlier_ids: Landmarks with a chi^2 outlier observation in the newest keyframe.
    """

    initial_cost: float
    final_cost: float
    iterations: int
    num_landmarks: int
    num_observations: int
    outlier_ids: np.ndarray


class SlidingWindowBA:
    """Keyframe window, landmark store and the Schur-complement LM solver."""

    def __init__(self, config: WindowBAConfig, camera_matrix: FloatArray, baseline: float) -> None:
        """Create an empty window.

        Args:
            config: Settings.
            camera_matrix: 3x3 rectified intrinsics.
            baseline: Stereo baseline in metres.
        """
        self._config = config  # Settings.
        self._camera_matrix = camera_matrix  # K.
        self._baseline = baseline  # B.
        self._keyframes: deque[Keyframe] = deque(maxlen=max(config.window, 1))  # Window.
        self.landmarks: dict[int, FloatArray] = {}  # track id -> world point.
        self._frames_since_keyframe = 0  # Keyframe policy state.
        self.newest_covariance: FloatArray | None = None  # Marginal covariance of the newest keyframe pose (ours: rho, phi).

    @property
    def keyframes(self) -> list[Keyframe]:
        """Return the keyframes in the window, oldest first.

        Returns:
            Keyframes.
        """
        return list(self._keyframes)

    def reset(self) -> None:
        """Forget everything."""
        self._keyframes.clear()
        self.landmarks.clear()
        self._frames_since_keyframe = 0

    def wants_keyframe(self, camera_from_world: FloatArray) -> bool:
        """Apply the keyframe policy to the current frame.

        Args:
            camera_from_world: Current T_cam_world.

        Returns:
            Whether the frame becomes a keyframe.
        """
        self._frames_since_keyframe += 1
        if not self._keyframes:  # First keyframe.
            return True
        last = self._keyframes[-1].camera_from_world
        moved = np.linalg.norm(camera_from_world[:3, :3].T @ camera_from_world[:3, 3] - last[:3, :3].T @ last[:3, 3])  # Camera centres.
        return self._frames_since_keyframe >= self._config.keyframe_every or moved >= self._config.min_translation_m

    def add_keyframe(
        self, frame_index: int, camera_from_world: FloatArray, track_ids: np.ndarray, observations: FloatArray, points: FloatArray
    ) -> None:
        """Insert a keyframe; new landmarks are initialised from the given world points.

        Args:
            frame_index: Source frame.
            camera_from_world: Its pose.
            track_ids: (N,) identifiers.
            observations: (N, 3) (u_L, v_L, u_R).
            points: (N, 3) current world estimates (anchors) of those landmarks; NaN rows are skipped.
        """
        ok = (track_ids >= 0) & np.isfinite(points).all(axis=1) & np.isfinite(observations).all(axis=1)
        for tid, point in zip(track_ids[ok].tolist(), points[ok], strict=True):  # Initialise unseen landmarks.
            if tid not in self.landmarks:
                self.landmarks[tid] = point.astype(np.float64).copy()
        self._keyframes.append(Keyframe(frame_index, camera_from_world.copy(), track_ids[ok].copy(), observations[ok].astype(np.float64)))
        self._frames_since_keyframe = 0
        alive = set(np.concatenate([kf.track_ids for kf in self._keyframes]).tolist())  # Landmarks still in the window.
        for tid in [t for t in self.landmarks if t not in alive]:
            del self.landmarks[tid]

    # ------------------------------------------------------------------ optimisation
    def _problem(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Collect the observations of multi-view landmarks.

        Returns:
            Landmark ids (L,), observation keyframe slot (M,), observation landmark index (M,), observations (M, 3).
        """
        ids = np.concatenate([kf.track_ids for kf in self._keyframes]) if self._keyframes else np.zeros(0, dtype=np.int64)
        unique, counts = np.unique(ids, return_counts=True)
        landmark_ids = unique[counts >= self._config.min_observations]  # Multi-view only.
        slots, lms, obs = [], [], []
        for slot, kf in enumerate(self._keyframes):
            pos = np.searchsorted(landmark_ids, kf.track_ids)
            pos = np.clip(pos, 0, max(len(landmark_ids) - 1, 0))
            hit = (landmark_ids.size > 0) & (landmark_ids[pos] == kf.track_ids) if landmark_ids.size else np.zeros(len(kf.track_ids), dtype=bool)
            slots.append(np.full(int(hit.sum()), slot))
            lms.append(pos[hit])
            obs.append(kf.observations[hit])
        if not slots:
            return landmark_ids, np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros((0, 3))
        return landmark_ids, np.concatenate(slots), np.concatenate(lms), np.concatenate(obs)

    def _linearise(
        self, poses: list[FloatArray], points: FloatArray, slots: np.ndarray, lms: np.ndarray, obs: FloatArray, sigma: float
    ) -> tuple[float, FloatArray, FloatArray, FloatArray, np.ndarray]:
        """Whitened residuals and Jacobians of every observation, with IRLS Huber weights.

        Returns:
            Robust cost, weighted whitened residuals (M, 3), pose Jacobians (M, 3, 6), point Jacobians (M, 3, 3),
            squared whitened norms (M,) (inf where the point is behind the camera).
        """
        m = len(slots)
        res = np.zeros((m, 3))
        jp = np.zeros((m, 3, 6))
        jx = np.zeros((m, 3, 3))
        sq = np.full(m, np.inf)
        for slot, pose in enumerate(poses):
            rows = np.flatnonzero(slots == slot)
            if rows.size == 0:
                continue
            right = np.column_stack([obs[rows, 2], obs[rows, 1]])  # stereo_residuals reads u_R from column 0.
            r, j_pose, j_point, valid = stereo_residuals(pose, points[lms[rows]], obs[rows, :2], right, self._camera_matrix, self._baseline)
            r, j_pose, j_point = r / sigma, j_pose / sigma, j_point / sigma  # Whitening (isotropic pixel noise).
            r[~valid], j_pose[~valid], j_point[~valid] = 0.0, 0.0, 0.0  # Points behind the camera do not count.
            res[rows], jp[rows], jx[rows] = r, j_pose, j_point
            sq[rows] = np.where(valid, np.sum(r**2, axis=1), np.inf)
        delta = self._config.huber_delta
        norm = np.sqrt(np.where(np.isfinite(sq), sq, 0.0))
        weight = np.where(norm <= delta, 1.0, delta / np.maximum(norm, 1e-12))  # IRLS weights.
        cost = float(np.sum(np.where(norm <= delta, norm**2, 2 * delta * norm - delta**2)))  # Huber cost.
        sw = np.sqrt(weight)[:, None]
        return cost, res * sw, jp * sw[:, :, None], jx * sw[:, :, None], sq

    @staticmethod
    def _reduced_system(
        jp: FloatArray, jx: FloatArray, r: FloatArray, slots: np.ndarray, lms: np.ndarray, n_pose: int, n_lm: int, lam: float
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
        """Normal equations with the landmarks eliminated (Schur complement), with LM damping ``lam``.

        Returns:
            Reduced camera matrix (W, W, 6, 6), reduced gradient (W, 6), H_ll^-1 (L, 3, 3), H_pl flattened (6W, 3L), g_l (L, 3).
        """
        h_pp = np.zeros((n_pose, n_pose, 6, 6))
        jp_t = jp.transpose(0, 2, 1)  # Batched matmuls are much faster than einsum here.
        jx_t = jx.transpose(0, 2, 1)
        np.add.at(h_pp, (slots, slots), jp_t @ jp)
        h_pl = jp_t @ jx  # (M, 6, 3), one per observation.
        h_ll = np.zeros((n_lm, 3, 3))
        np.add.at(h_ll, lms, jx_t @ jx)
        g_p = np.zeros((n_pose, 6))
        np.add.at(g_p, slots, (jp_t @ r[..., None])[..., 0])
        g_l = np.zeros((n_lm, 3))
        np.add.at(g_l, lms, (jx_t @ r[..., None])[..., 0])
        for i in range(n_pose):  # LM damping (multiplicative) plus a tiny absolute term for unobservable directions.
            h_pp[i, i] += lam * np.diag(np.diag(h_pp[i, i])) + 1e-9 * np.eye(6)
        h_ll_inv = np.linalg.inv(h_ll + lam * h_ll * np.eye(3) + 1e-9 * np.eye(3))
        dense = np.zeros((n_lm, n_pose, 6, 3))  # H_pl arranged per (landmark, keyframe slot).
        dense[lms, slots] = h_pl
        tmp_flat = (dense @ h_ll_inv[:, None]).transpose(1, 2, 0, 3).reshape(6 * n_pose, 3 * n_lm)  # H_pl H_ll^-1.
        dense_flat = dense.transpose(1, 2, 0, 3).reshape(6 * n_pose, 3 * n_lm)
        schur = h_pp - (tmp_flat @ dense_flat.T).reshape(n_pose, 6, n_pose, 6).transpose(0, 2, 1, 3)
        rhs = g_p - (tmp_flat @ g_l.reshape(-1)).reshape(n_pose, 6)
        return schur, rhs, h_ll_inv, dense_flat, g_l

    def optimise(self, sigma: float) -> BAResult | None:
        """Run Levenberg-Marquardt on the window (oldest keyframe fixed).

        Args:
            sigma: Pixel noise (shared noise model).

        Returns:
            The result, or None when the window is too small.
        """
        self.newest_covariance = None
        if len(self._keyframes) < 2:
            return None
        landmark_ids, slots, lms, obs = self._problem()
        if landmark_ids.size < 6:
            return None
        poses = [kf.camera_from_world.copy() for kf in self._keyframes]
        points = np.stack([self.landmarks[int(t)] for t in landmark_ids])
        n_pose, n_lm = len(poses), len(landmark_ids)
        free = np.arange(1, n_pose)  # Oldest keyframe fixed.
        cost, r, jp, jx, _ = self._linearise(poses, points, slots, lms, obs, sigma)
        initial_cost, lam, accepted = cost, 1e-4, 0
        for _ in range(self._config.iterations):
            schur, rhs, h_ll_inv, dense_flat, g_l = self._reduced_system(jp, jx, r, slots, lms, n_pose, n_lm, lam)
            s_free = schur[np.ix_(free, free)].transpose(0, 2, 1, 3).reshape(6 * len(free), 6 * len(free))
            try:
                d_pose_free = np.linalg.solve(s_free, -rhs[free].reshape(-1)).reshape(-1, 6)
            except np.linalg.LinAlgError:
                return None
            d_pose = np.zeros((n_pose, 6))
            d_pose[free] = d_pose_free
            coupling = (dense_flat.T @ d_pose.reshape(-1)).reshape(n_lm, 3)  # sum_w H_lp dp_w.
            d_point = -(h_ll_inv @ (g_l + coupling)[..., None])[..., 0]  # Back-substitution.
            cand_poses = [se3_exp(d_pose[i]) @ poses[i] for i in range(n_pose)]
            cand_points = points + d_point
            cand_cost, cand_r, cand_jp, cand_jx, _ = self._linearise(cand_poses, cand_points, slots, lms, obs, sigma)
            if cand_cost < cost:  # Accept.
                poses, points, cost, r, jp, jx = cand_poses, cand_points, cand_cost, cand_r, cand_jp, cand_jx
                lam = max(lam / 10.0, 1e-9)
                accepted += 1
                if np.max(np.abs(d_pose)) < 1e-7 and np.max(np.abs(d_point)) < 1e-6:
                    break
            else:  # Reject and damp more.
                lam *= 10.0
                if lam > 1e6:
                    break
        for kf, pose in zip(self._keyframes, poses, strict=True):  # Write back.
            kf.camera_from_world = pose
        for tid, point in zip(landmark_ids.tolist(), points, strict=True):
            self.landmarks[tid] = point
        _, r, jp, jx, sq = self._linearise(poses, points, slots, lms, obs, sigma)
        schur, _, _, _, _ = self._reduced_system(jp, jx, r, slots, lms, n_pose, n_lm, 0.0)  # Undamped information.
        try:  # Marginal covariance of the newest pose: block of the inverse reduced camera system.
            cov_free = np.linalg.inv(schur[np.ix_(free, free)].transpose(0, 2, 1, 3).reshape(6 * len(free), 6 * len(free)))
            self.newest_covariance = cov_free[-6:, -6:]
        except np.linalg.LinAlgError:
            self.newest_covariance = None
        newest = slots == n_pose - 1
        outliers = np.unique(landmark_ids[lms[newest & (sq > CHI2_STEREO_99)]])
        return BAResult(initial_cost, cost, accepted, n_lm, len(slots), outliers)
