"""Warped-patch re-alignment of long feature tracks (drift correction for anchored landmarks).

Chained frame-to-frame KLT estimates only a translation per frame, so the patch's rotation and scale change
leave a small error (about 1 % of the flow) that accumulates along the track: after tens of frames the
pixel position has slid off the physical point. With birth/fused anchors that drift becomes a systematic
mismatch between a landmark and its observation. Re-aligning each long track against a *fixed* template
from the frame where it was born removes the accumulation (cf. the feature alignment of SVO, Forster et al.).

The template is warped into the current view with the affine map predicted from the poses and the landmark
depth (fronto-parallel local surface), and only a 2-D translation plus an affine brightness change is
estimated by Gauss-Newton, for all tracks at once.
"""

from __future__ import annotations  # Postponed evaluation of type hints.

import cv2  # Image sampling and single-level KLT.
import numpy as np  # Vectorised maths.

from metricvision.core.types import BoolArray, FloatArray, ImageArray  # Shared types.


def predict_affine(
    camera_matrix: FloatArray, pixels_ref: FloatArray, depth_ref: FloatArray, current_from_reference: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Predict where reference pixels land in the current image and the local affine map.

    Args:
        camera_matrix: 3x3 intrinsics (both views share them after rectification).
        pixels_ref: (N, 2) pixels in the reference (birth) image.
        depth_ref: (N,) depth of each point in the reference camera, metres.
        current_from_reference: T_cur_ref.

    Returns:
        (N, 2, 2) Jacobians d u_cur / d u_ref and (N, 2) predicted current pixels.
    """
    fx, fy, cx, cy = camera_matrix[0, 0], camera_matrix[1, 1], camera_matrix[0, 2], camera_matrix[1, 2]  # Intrinsics.
    rotation, translation = current_from_reference[:3, :3], current_from_reference[:3, 3]  # Pose parts.

    def project(pixels: FloatArray) -> FloatArray:
        """Back-project at the reference depth, transform and project into the current view."""
        points = np.column_stack([(pixels[:, 0] - cx) / fx * depth_ref, (pixels[:, 1] - cy) / fy * depth_ref, depth_ref])  # Ref frame.
        p = points @ rotation.T + translation  # Current frame.
        z = np.where(np.abs(p[:, 2]) > 1e-9, p[:, 2], 1e-9)  # Safe depth.
        return np.column_stack([fx * p[:, 0] / z + cx, fy * p[:, 1] / z + cy])  # Pinhole.

    centre = project(pixels_ref)  # Predicted positions.
    du = project(pixels_ref + np.array([1.0, 0.0])) - centre  # Image of a unit step in u.
    dv = project(pixels_ref + np.array([0.0, 1.0])) - centre  # Image of a unit step in v.
    return np.stack([du, dv], axis=2), centre  # Columns are the images of the unit steps.


def align_warped(
    image_ref: ImageArray,
    image_cur: ImageArray,
    pixels_ref: FloatArray,
    affine: FloatArray,
    pixels_init: FloatArray,
    half_window: int = 6,
    iterations: int = 15,
    max_shift_px: float = 2.0,
) -> tuple[FloatArray, BoolArray]:
    """Align warped reference templates in the current image (translation + affine brightness, Gauss-Newton).

    Args:
        image_ref: Reference (birth) image.
        image_cur: Current image.
        pixels_ref: (N, 2) template centres in the reference image.
        affine: (N, 2, 2) predicted d u_cur / d u_ref.
        pixels_init: (N, 2) initial current positions (the KLT-tracked ones).
        half_window: Template half size in current-image pixels.
        iterations: Gauss-Newton iterations.
        max_shift_px: Corrections larger than this are rejected (occlusion or wrong track).

    Returns:
        (N, 2) aligned positions and an (N,) success mask.
    """
    n = pixels_ref.shape[0]  # Track count.
    if n == 0:  # Nothing to do.
        return pixels_init.copy(), np.zeros(0, dtype=bool)  # Empty result.
    grid = np.arange(-half_window, half_window + 1, dtype=np.float64)  # Offsets.
    gx, gy = np.meshgrid(grid, grid)  # Template grid in current-image pixels.
    offsets = np.stack([gx.ravel(), gy.ravel()], axis=1)  # (P, 2).
    det = affine[:, 0, 0] * affine[:, 1, 1] - affine[:, 0, 1] * affine[:, 1, 0]  # Area change.
    valid = np.isfinite(det) & (np.abs(det) > 0.2) & (np.abs(det) < 5.0)  # Reject extreme warps.
    inverse = np.linalg.inv(np.where(valid[:, None, None], affine, np.eye(2)))  # Current offsets -> reference offsets.
    ref_points = pixels_ref[:, None, :] + offsets @ inverse.transpose(0, 2, 1)  # (N, P, 2) sampling positions.
    template = cv2.remap(
        image_ref.astype(np.float32), ref_points[..., 0].astype(np.float32), ref_points[..., 1].astype(np.float32), cv2.INTER_LINEAR
    ).astype(np.float64)  # (N, P) warped templates.
    side = 2 * half_window + 1  # Template side.
    grad_x = np.gradient(template.reshape(n, side, side), axis=2).reshape(n, -1)  # Template gradients.
    grad_y = np.gradient(template.reshape(n, side, side), axis=1).reshape(n, -1)
    uv = pixels_init.astype(np.float64).copy()  # Current estimate.
    gain, bias = np.ones(n), np.zeros(n)  # Affine brightness: I = gain * T + bias.
    current = image_cur.astype(np.float32)  # Sampling source.
    for _ in range(iterations):  # Gauss-Newton.
        points = uv[:, None, :] + offsets[None]  # Sampling positions.
        sampled = cv2.remap(current, points[..., 0].astype(np.float32), points[..., 1].astype(np.float32), cv2.INTER_LINEAR).astype(np.float64)
        error = sampled - gain[:, None] * template - bias[:, None]  # Photometric residual.
        jac = np.stack([gain[:, None] * grad_x, gain[:, None] * grad_y, -template, -np.ones_like(template)], axis=2)  # (N, P, 4).
        jac_t = jac.transpose(0, 2, 1)  # (N, 4, P); batched matmul is far faster than einsum here.
        hessian = jac_t @ jac + 1e-6 * np.eye(4)  # Normal equations.
        step = -np.linalg.solve(hessian, jac_t @ error[..., None])[..., 0]  # (N, 4).
        uv += step[:, :2]  # Position update.
        gain += step[:, 2]  # Brightness gain.
        bias += step[:, 3]  # Brightness offset.
        if np.max(np.abs(step[:, :2])) < 1e-3:  # Converged.
            break
    height, width = image_cur.shape[:2]  # Bounds.
    ok = (
        valid
        & np.isfinite(uv).all(axis=1)
        & (np.linalg.norm(uv - pixels_init, axis=1) < max_shift_px)
        & (uv[:, 0] > half_window)
        & (uv[:, 0] < width - half_window - 1)
        & (uv[:, 1] > half_window)
        & (uv[:, 1] < height - half_window - 1)
    )  # Accept small, in-image corrections only.
    return uv, ok  # Aligned positions.


def refine_right(
    left: ImageArray, right: ImageArray, pixels_left: FloatArray, pixels_right_init: FloatArray, window: int = 21
) -> tuple[FloatArray, BoolArray]:
    """Re-match right observations from corrected left positions with single-level KLT.

    Args:
        left: Current left image.
        right: Current right image.
        pixels_left: (N, 2) corrected left positions.
        pixels_right_init: (N, 2) initial right positions.
        window: KLT window size.

    Returns:
        (N, 2) right positions and an (N,) success mask (moves of more than 2 px are rejected).
    """
    if len(pixels_left) == 0:  # Nothing to do.
        return pixels_right_init.copy(), np.zeros(0, dtype=bool)
    tracked, status, _ = cv2.calcOpticalFlowPyrLK(
        left,
        right,
        np.ascontiguousarray(pixels_left, dtype=np.float32).reshape(-1, 1, 2),
        np.ascontiguousarray(pixels_right_init, dtype=np.float32).reshape(-1, 1, 2).copy(),
        winSize=(window, window),
        maxLevel=0,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
    )  # Single-level refinement around the previous right position.
    tracked = np.asarray(tracked, dtype=np.float64).reshape(-1, 2)  # Back to (N, 2).
    ok = (status.reshape(-1) == 1) & (np.linalg.norm(tracked - pixels_right_init, axis=1) < 2.0)  # Small corrections only.
    return tracked, ok  # Refined right positions.
