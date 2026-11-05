"""Sliding-window bundle adjustment with GTSAM (industrial-standard factor-graph back end).

Same window, keyframe policy and landmark store as ``backend.window_ba.SlidingWindowBA`` (it subclasses it), but the
optimisation is a GTSAM factor graph:

* ``GenericStereoFactor3D``: its measurement is ``StereoPoint2(u_L, u_R, v)``, exactly our 3-D rigid-rig residual
  (u_L, v_L, u_R) with a single row coordinate, so the model is identical to the per-frame LM and the NumPy BA;
* a Huber robust noise model with the shared pixel sigma;
* gauge: the oldest keyframe is either held fixed (``gauge: fix``) or tied to its previous marginal covariance by a
  prior (``gauge: prior``), which approximates marginalising the keyframes that left the window instead of treating
  the oldest pose as perfect;
* ``gtsam.Marginals`` gives the covariance of the newest keyframe, so the pose covariance used for consistency (NEES)
  and pose inflation is the one of the bundle-adjusted solution.

GTSAM is optional (``uv sync --extra industrial``); the NumPy solver remains the default and the reference.
"""

from __future__ import annotations  # Postponed evaluation of type hints.

import numpy as np  # Arrays.

from metricvision.backend.window_ba import CHI2_STEREO_99, BAResult, SlidingWindowBA, WindowBAConfig  # Window logic.
from metricvision.core.geometry import invert_pose  # Poses.
from metricvision.core.types import FloatArray  # Shared types.


def _require_gtsam():  # noqa: ANN202
    """Import GTSAM with a helpful error.

    Returns:
        The gtsam module.

    Raises:
        ImportError: If GTSAM is not installed.
    """
    try:
        import gtsam  # noqa: PLC0415  # Optional dependency.
    except ImportError as error:  # pragma: no cover - depends on the environment.
        raise ImportError("odometry.ba_solver=gtsam needs GTSAM: `uv sync --extra industrial` (or pip install gtsam)") from error
    return gtsam


def gtsam_to_ours(covariance: FloatArray) -> FloatArray:
    """Convert a GTSAM Pose3 marginal (T_world_cam, right perturbation, order [rot, trans]) to our convention.

    Ours: left perturbation of T_cam_world, order [trans, rot]. Since T_cw' = exp(d) T_cw  <=>  T_wc' = T_wc exp(-d),
    the GTSAM tangent vector is -(phi, rho): the covariance is the same matrix with its blocks swapped.

    Args:
        covariance: 6x6 GTSAM marginal.

    Returns:
        6x6 covariance of (rho, phi).
    """
    order = [3, 4, 5, 0, 1, 2]
    return covariance[np.ix_(order, order)]


def ours_to_gtsam(covariance: FloatArray) -> FloatArray:
    """Inverse of ``gtsam_to_ours`` (the block swap is its own inverse).

    Args:
        covariance: 6x6 covariance of (rho, phi).

    Returns:
        6x6 GTSAM marginal.
    """
    return gtsam_to_ours(covariance)


class GtsamWindowBA(SlidingWindowBA):
    """Sliding-window BA solved with GTSAM (Levenberg-Marquardt on a stereo factor graph)."""

    def __init__(self, config: WindowBAConfig, camera_matrix: FloatArray, baseline: float, gauge: str = "fix") -> None:
        """Create an empty window.

        Args:
            config: Settings.
            camera_matrix: 3x3 rectified intrinsics.
            baseline: Stereo baseline in metres.
            gauge: ``fix`` (oldest keyframe constant) or ``prior`` (oldest keyframe tied to its previous marginal).

        Raises:
            ValueError: If the gauge mode is unknown.
        """
        super().__init__(config, camera_matrix, baseline)
        if gauge not in ("fix", "prior"):
            raise ValueError(f"gauge must be 'fix' or 'prior', got '{gauge}'")
        self._gtsam = _require_gtsam()
        self._gauge = gauge
        k = camera_matrix
        self._calibration = self._gtsam.Cal3_S2Stereo(float(k[0, 0]), float(k[1, 1]), 0.0, float(k[0, 2]), float(k[1, 2]), float(baseline))
        self._marginals: dict[int, FloatArray] = {}  # frame index -> GTSAM marginal of that keyframe's pose.

    def reset(self) -> None:
        """Forget everything."""
        super().reset()
        if hasattr(self, "_marginals"):
            self._marginals.clear()

    def optimise(self, sigma: float) -> BAResult | None:
        """Build the factor graph of the window and solve it.

        Args:
            sigma: Pixel noise (shared noise model).

        Returns:
            The result (``newest_covariance`` is set as a side effect), or None when the window is too small.
        """
        gtsam = self._gtsam
        self.newest_covariance = None
        if len(self._keyframes) < 2:
            return None
        landmark_ids, slots, lms, obs = self._problem()
        if landmark_ids.size < 6:
            return None
        pose_key = [gtsam.symbol("x", i) for i in range(len(self._keyframes))]
        point_key = [gtsam.symbol("l", i) for i in range(len(landmark_ids))]
        huber = gtsam.noiseModel.mEstimator.Huber.Create(self._config.huber_delta)
        noise = gtsam.noiseModel.Robust.Create(huber, gtsam.noiseModel.Isotropic.Sigma(3, sigma))
        graph, values = gtsam.NonlinearFactorGraph(), gtsam.Values()
        for slot, landmark, (u_l, v_l, u_r) in zip(slots.tolist(), lms.tolist(), obs, strict=True):
            graph.add(gtsam.GenericStereoFactor3D(gtsam.StereoPoint2(u_l, u_r, v_l), noise, pose_key[slot], point_key[landmark], self._calibration))
        for slot, kf in enumerate(self._keyframes):
            values.insert(pose_key[slot], gtsam.Pose3(invert_pose(kf.camera_from_world)))  # GTSAM poses are T_world_cam.
        for index, tid in enumerate(landmark_ids.tolist()):
            values.insert(point_key[index], gtsam.Point3(*self.landmarks[int(tid)]))
        oldest = self._keyframes[0]
        oldest_pose = gtsam.Pose3(invert_pose(oldest.camera_from_world))
        prior_cov = self._marginals.get(oldest.frame_index) if self._gauge == "prior" else None
        if prior_cov is not None and np.isfinite(prior_cov).all():  # Approximate marginalisation.
            graph.add(gtsam.PriorFactorPose3(pose_key[0], oldest_pose, gtsam.noiseModel.Gaussian.Covariance(prior_cov + 1e-12 * np.eye(6))))
        else:  # Classic gauge: oldest keyframe fixed.
            graph.add(gtsam.NonlinearEqualityPose3(pose_key[0], oldest_pose))
        params = gtsam.LevenbergMarquardtParams()
        params.setMaxIterations(self._config.iterations)
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, values, params)
        initial_cost = float(graph.error(values))
        try:
            result = optimizer.optimize()
        except RuntimeError:  # Indeterminant system (e.g. a degenerate landmark).
            return None
        final_cost = float(graph.error(result))
        for slot, kf in enumerate(self._keyframes):  # Write back.
            kf.camera_from_world = invert_pose(result.atPose3(pose_key[slot]).matrix())
        points = np.stack([np.asarray(result.atPoint3(point_key[i])) for i in range(len(landmark_ids))])
        for tid, point in zip(landmark_ids.tolist(), points, strict=True):
            self.landmarks[tid] = point
        try:  # Marginal covariances of the keyframes (gauge prior next time; NEES of the newest).
            marginals = gtsam.Marginals(graph, result)
            for slot, kf in enumerate(self._keyframes):
                self._marginals[kf.frame_index] = np.asarray(marginals.marginalCovariance(pose_key[slot]))
            self.newest_covariance = gtsam_to_ours(self._marginals[self._keyframes[-1].frame_index])
        except RuntimeError:
            self.newest_covariance = None
        live = {kf.frame_index for kf in self._keyframes}
        for frame in [f for f in self._marginals if f not in live]:
            del self._marginals[frame]
        _, _, _, _, sq = self._linearise([kf.camera_from_world for kf in self._keyframes], points, slots, lms, obs, sigma)
        newest = slots == len(self._keyframes) - 1
        outliers = np.unique(landmark_ids[lms[newest & (sq > CHI2_STEREO_99)]])
        return BAResult(initial_cost, final_cost, int(optimizer.iterations()), len(landmark_ids), len(slots), outliers)
