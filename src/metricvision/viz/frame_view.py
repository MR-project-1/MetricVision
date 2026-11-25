"""One-canvas view of a VO step: which keypoints were matched, which PnP kept, and the trajectory it adds up to.

Layout (BGR image, OpenCV drawing only, so it works with ``opencv-python-headless``):

    +--------------------------------------+--------------------------------------+
    | current LEFT image                   | sparse: current RIGHT image          |
    |  green line: reference -> current    |   stereo partner of every inlier,    |
    |  dot colour: depth (red near, blue   |   lines join a sample of L/R pairs   |
    |  far), red x: PnP outlier, grey: no  | dense: REFERENCE keyframe image      |
    |  depth, cyan o: new detection        |   lines join keyframe -> current     |
    +--------------------------------------+--------------------------------------+
    | top-down map (base frame, metres):   | numbers for this frame + inlier      |
    |  VO path vs wheel odometry           | history                              |
    +--------------------------------------+--------------------------------------+

The VO pose is shown in the body frame, ``T_body = B @ T_world_cam @ B^-1`` with ``B = body_from_camera``,
and the reference (wheel odometry) is anchored at its first pose, so both start at the origin.
"""

from __future__ import annotations

from collections import OrderedDict, deque

import cv2
import numpy as np

from metricvision.backend.odometry import TrackingResult
from metricvision.core.geometry import invert_pose
from metricvision.core.types import FloatArray, ImageArray

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_GREEN, _RED, _GREY, _CYAN, _WHITE, _YELLOW = (60, 220, 60), (40, 40, 235), (150, 150, 150), (255, 220, 0), (240, 240, 240), (0, 215, 255)
_VO_COLOUR, _REF_COLOUR, _BG = (255, 160, 40), (230, 230, 230), (28, 28, 30)


def depth_colours(depth: np.ndarray, near: float = 0.5, far: float = 15.0) -> np.ndarray:
    """Map metric depth to BGR colours (TURBO: red near, blue far; log-spaced so indoor ranges spread out).

    Args:
        depth: (N,) depth in metres (NaN allowed).
        near: Depth drawn fully red.
        far: Depth drawn fully blue.

    Returns:
        (N, 3) uint8 BGR colours (grey for NaN).
    """
    d = np.clip(np.nan_to_num(depth, nan=far), near, far)
    t = 1.0 - (np.log(d) - np.log(near)) / (np.log(far) - np.log(near))  # 1 = near.
    lut = cv2.applyColorMap(np.arange(256, dtype=np.uint8)[:, None], cv2.COLORMAP_TURBO)[:, 0]
    out = lut[np.round(t * 255).astype(int)]
    out[~np.isfinite(depth)] = _GREY
    return out


def _text(img: np.ndarray, text: str, org: tuple[int, int], scale: float = 0.5, colour: tuple[int, int, int] = _WHITE, thick: int = 1) -> None:
    """Draw text on a dark box so it reads on any image."""
    (tw, th), base = cv2.getTextSize(text, _FONT, scale, thick)
    x, y = org
    cv2.rectangle(img, (x - 3, y - th - 3), (x + tw + 3, y + base + 1), (0, 0, 0), -1)
    cv2.putText(img, text, org, _FONT, scale, colour, thick, cv2.LINE_AA)


def _pt(p: np.ndarray) -> tuple[int, int]:
    """Pixel tuple for OpenCV drawing."""
    return int(round(float(p[0]))), int(round(float(p[1])))


class VOVisualizer:
    """Accumulates the trajectory and renders one debug canvas per processed frame."""

    def __init__(self, body_from_camera: FloatArray, title: str = "", max_stereo_lines: int = 40, history: int = 300) -> None:
        """Set up the state.

        Args:
            body_from_camera: T_body_cam (rectified left camera in the odometry body frame).
            title: Shown in the info panel (bag / pipeline name).
            max_stereo_lines: How many L/R (or keyframe/current) pairs to join with lines.
            history: Frames shown in the inlier plot.
        """
        self._b = np.asarray(body_from_camera, dtype=np.float64)
        self._b_inv = invert_pose(self._b)
        self._title = title
        self._max_lines = max_stereo_lines
        self._vo: list[FloatArray] = []  # Body poses.
        self._ref: list[FloatArray] = []  # Anchored reference poses.
        self._ref0_inv: FloatArray | None = None
        self._failed: list[int] = []  # Indices into _vo where tracking failed.
        self._inlier_history: deque[int] = deque(maxlen=history)
        self._images: OrderedDict[int, ImageArray] = OrderedDict()  # Recent left images (dense keyframe lookup).
        self._distance = 0.0
        self._ref_distance = 0.0

    def _remember(self, index: int, image: ImageArray) -> None:
        """Keep recent left images so a dense keyframe can be shown."""
        self._images[index] = image
        while len(self._images) > 64:
            self._images.popitem(last=False)

    def record(self, index: int, left: ImageArray, result: TrackingResult, reference_pose: FloatArray | None = None) -> None:
        """Add one frame to the history without drawing (cheap; used while nobody is watching).

        Args:
            index: Frame index (as given to the pipeline).
            left: Rectified left image (kept briefly so a dense keyframe can be shown).
            result: What the pipeline returned for this frame.
            reference_pose: Wheel odometry T_odom_base at this frame, if available.
        """
        pose = self._b @ result.world_from_camera @ self._b_inv
        if self._vo:
            self._distance += float(np.linalg.norm(pose[:2, 3] - self._vo[-1][:2, 3]))
        self._vo.append(pose)
        if not result.stats.tracking_ok:
            self._failed.append(len(self._vo) - 1)
        if reference_pose is not None:
            if self._ref0_inv is None:
                self._ref0_inv = invert_pose(reference_pose)
            ref = self._ref0_inv @ reference_pose
            if self._ref:
                self._ref_distance += float(np.linalg.norm(ref[:2, 3] - self._ref[-1][:2, 3]))
            self._ref.append(ref)
        self._inlier_history.append(int(result.stats.num_pnp_inliers))
        self._remember(index, left)

    def draw(self, index: int, left: ImageArray, right: ImageArray, result: TrackingResult, fps: float | None = None) -> np.ndarray:
        """Render the canvas for a frame already passed to ``record`` (no state change).

        Args:
            index: Frame index.
            left: Rectified left image (grey).
            right: Rectified right image (grey).
            result: What the pipeline returned for this frame.
            fps: Measured processing rate to print.

        Returns:
            BGR canvas.
        """
        h, w = left.shape[:2]
        top = np.hstack([self._left_panel(left, result), self._right_panel(right, result)])
        self._draw_match_lines(top, result, w)
        bottom_h = max(360, h * 3 // 4)
        bottom = np.hstack([self._map_panel(w, bottom_h), self._info_panel(w, bottom_h, index, result, fps)])
        return np.vstack([top, bottom])

    def _left_panel(self, left: ImageArray, result: TrackingResult) -> np.ndarray:
        """Current left image with the temporal correspondences."""
        img = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
        dbg = result.debug
        if dbg is None:
            _text(img, "first frame: features detected, world origin set", (10, 25), 0.55, _YELLOW)
            return img
        depth = dbg.points_camera[:, 2]
        colours = depth_colours(depth)
        for i in range(len(dbg.current_left)):
            c = dbg.current_left[i]
            if not np.isfinite(depth[i]):
                cv2.circle(img, _pt(c), 2, _GREY, -1, cv2.LINE_AA)
            elif dbg.inliers[i]:
                cv2.line(img, _pt(dbg.reference_left[i]), _pt(c), _GREEN, 1, cv2.LINE_AA)
                cv2.circle(img, _pt(c), 3, tuple(int(v) for v in colours[i]), -1, cv2.LINE_AA)
            else:
                x, y = _pt(c)
                cv2.line(img, (x - 3, y - 3), (x + 3, y + 3), _RED, 1, cv2.LINE_AA)
                cv2.line(img, (x - 3, y + 3), (x + 3, y - 3), _RED, 1, cv2.LINE_AA)
        for p in dbg.new_left:
            cv2.circle(img, _pt(p), 4, _CYAN, 1, cv2.LINE_AA)
        ref = "previous frame" if dbg.reference_index == result.stats.frame_index - 1 else f"keyframe {dbg.reference_index}"
        _text(img, f"LEFT  tracks from {ref}", (10, 22), 0.55)
        legend = [("inlier (colour = depth)", _GREEN), ("PnP outlier", _RED), ("no depth", _GREY), ("new detection", _CYAN)]
        for k, (label, col) in enumerate(legend):
            _text(img, label, (10, img.shape[0] - 12 - 18 * (len(legend) - 1 - k)), 0.45, col)
        self._depth_bar(img)
        return img

    @staticmethod
    def _depth_bar(img: np.ndarray) -> None:
        """Small depth colour scale in the top-right corner."""
        x0, y0, bw, bh = img.shape[1] - 150, 10, 130, 10
        depths = np.exp(np.linspace(np.log(0.5), np.log(15.0), bw))
        bar = depth_colours(depths)[None].repeat(bh, axis=0)
        img[y0 : y0 + bh, x0 : x0 + bw] = bar
        _text(img, "0.5", (x0, y0 + bh + 14), 0.4)
        _text(img, "depth m", (x0 + 40, y0 + bh + 14), 0.4)
        _text(img, "15", (x0 + bw - 16, y0 + bh + 14), 0.4)

    def _right_panel(self, right: ImageArray, result: TrackingResult) -> np.ndarray:
        """Sparse: stereo partners in the right image. Dense: the keyframe the frame was matched to."""
        dbg = result.debug
        if dbg is not None and dbg.current_right is None:  # Dense: show the reference keyframe.
            ref_img = self._images.get(dbg.reference_index)
            img = cv2.cvtColor(ref_img if ref_img is not None else right, cv2.COLOR_GRAY2BGR)
            colours = depth_colours(dbg.points_camera[:, 2])
            for i in np.flatnonzero(dbg.inliers):
                cv2.circle(img, _pt(dbg.reference_left[i]), 3, tuple(int(v) for v in colours[i]), -1, cv2.LINE_AA)
            _text(img, f"KEYFRAME {dbg.reference_index} (ORB matched to current, depth from SGBM)", (10, 22), 0.5)
            return img
        img = cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)
        if dbg is None or dbg.current_right is None:
            _text(img, "RIGHT", (10, 22), 0.55)
            return img
        colours = depth_colours(dbg.points_camera[:, 2])
        for i in np.flatnonzero(dbg.inliers):
            cv2.circle(img, _pt(dbg.current_right[i]), 3, tuple(int(v) for v in colours[i]), -1, cv2.LINE_AA)
        for p in dbg.new_right:
            cv2.circle(img, _pt(p), 4, _CYAN, 1, cv2.LINE_AA)
        _text(img, "RIGHT  stereo partners (same row after rectification)", (10, 22), 0.5)
        return img

    def _draw_match_lines(self, canvas: np.ndarray, result: TrackingResult, width: int) -> None:
        """Join a sample of inlier pairs across the two top panels (left -> right or current -> keyframe)."""
        dbg = result.debug
        if dbg is None:
            return
        idx = np.flatnonzero(dbg.inliers)
        if idx.size == 0:
            return
        idx = idx[np.linspace(0, idx.size - 1, min(self._max_lines, idx.size)).astype(int)]
        other = dbg.current_right if dbg.current_right is not None else dbg.reference_left
        colours = depth_colours(dbg.points_camera[idx, 2])
        for c, i in zip(colours, idx, strict=True):
            a = _pt(dbg.current_left[i])
            b = _pt(other[i] + np.array([width, 0.0]))
            cv2.line(canvas, a, b, tuple(int(v) for v in c), 1, cv2.LINE_AA)

    def _map_panel(self, w: int, h: int) -> np.ndarray:
        """Top-down trajectories (x forward, y left; drawn with x right, y up)."""
        img = np.full((h, w, 3), _BG, dtype=np.uint8)
        vo = np.array([p[:2, 3] for p in self._vo])
        ref = np.array([p[:2, 3] for p in self._ref]) if self._ref else np.zeros((0, 2))
        allp = np.vstack([vo, ref, np.zeros((1, 2))])
        lo, hi = allp.min(0) - 1.0, allp.max(0) + 1.0
        margin = 30
        scale = min((w - 2 * margin) / (hi[0] - lo[0]), (h - 2 * margin - 20) / (hi[1] - lo[1]))
        centre = (lo + hi) / 2

        def px(p: np.ndarray) -> np.ndarray:
            q = (p - centre) * scale
            return np.column_stack([w / 2 + q[:, 0], (h + 20) / 2 - q[:, 1]]).astype(np.int32)

        step = 1.0 if (hi - lo).max() < 25 else 5.0  # Grid spacing in metres.
        for gx in np.arange(np.floor(lo[0] / step) * step, hi[0] + step, step):
            x = int(px(np.array([[gx, 0.0]]))[0, 0])
            cv2.line(img, (x, 28), (x, h), (50, 50, 55), 1)
        for gy in np.arange(np.floor(lo[1] / step) * step, hi[1] + step, step):
            y = int(px(np.array([[0.0, gy]]))[0, 1])
            cv2.line(img, (0, y), (w, y), (50, 50, 55), 1)
        if len(ref) > 1:
            cv2.polylines(img, [px(ref)], False, _REF_COLOUR, 2, cv2.LINE_AA)
        if len(vo) > 1:
            cv2.polylines(img, [px(vo)], False, _VO_COLOUR, 2, cv2.LINE_AA)
        for k in self._failed:
            cv2.circle(img, tuple(px(vo[k : k + 1])[0]), 3, _RED, -1)
        o = px(np.zeros((1, 2)))[0]
        cv2.circle(img, (int(o[0]), int(o[1])), 5, _WHITE, 1, cv2.LINE_AA)
        if self._vo:  # Heading arrow of the current VO pose.
            p = self._vo[-1]
            tip = p[:2, 3] + 0.6 * p[:2, 0]
            a, b = px(np.vstack([p[:2, 3], tip]))
            cv2.arrowedLine(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), _YELLOW, 2, cv2.LINE_AA, tipLength=0.35)
        _text(img, f"TOP-DOWN (base frame, grid {step:.0f} m)", (10, 20), 0.5)
        _text(img, f"VO  {self._distance:.2f} m", (10, h - 30), 0.5, _VO_COLOUR)
        if self._ref:
            err = float(np.linalg.norm(self._vo[-1][:2, 3] - self._ref[-1][:2, 3]))
            _text(img, f"wheel odom  {self._ref_distance:.2f} m   |VO - odom| now {err:.2f} m", (10, h - 10), 0.5, _REF_COLOUR)
        return img

    def _info_panel(self, w: int, h: int, index: int, result: TrackingResult, fps: float | None) -> np.ndarray:
        """Numbers for this frame and the inlier history."""
        img = np.full((h, w, 3), (20, 20, 22), dtype=np.uint8)
        s = result.stats
        y = 24
        _text(img, self._title, (12, y), 0.55, _YELLOW)
        y += 30
        if not s.tracking_ok and result.debug is not None:
            _text(img, "TRACKING LOST - coasting on the motion model", (12, y), 0.6, _RED, 2)
            y += 28
        dbg = result.debug
        with_depth = int(np.isfinite(dbg.points_camera[:, 2]).sum()) if dbg is not None else 0
        median_depth = float(np.nanmedian(dbg.points_camera[dbg.inliers, 2])) if dbg is not None and dbg.inliers.any() else float("nan")
        lines = [
            f"frame {index}   t = {s.timestamp:.2f} s" + (f"   {fps:.1f} fps" if fps else ""),
            f"tracked / matched        {s.num_matches}",
            f"  with 3D point (stereo) {with_depth}",
            f"  PnP + LM inliers       {s.num_pnp_inliers}" + (f"   ({100 * s.num_pnp_inliers / max(with_depth, 1):.0f}%)" if dbg else ""),
            f"  median inlier depth    {median_depth:.2f} m",
            f"new detections           {len(dbg.new_left) if dbg is not None else 0}" + ("   [replenish/keyframe]" if s.is_keyframe else ""),
            f"time: match {s.match_ms:.1f}  pose {s.pose_ms:.1f}  depth {s.depth_ms:.1f}  total {s.total_ms:.1f} ms",
        ]
        if self._vo:
            p = self._vo[-1]
            yaw = np.degrees(np.arctan2(p[1, 0], p[0, 0]))
            lines.append(f"VO pose  x {p[0, 3]:+.2f}  y {p[1, 3]:+.2f}  z {p[2, 3]:+.2f} m  yaw {yaw:+.1f} deg")
        for line in lines:
            _text(img, line, (12, y), 0.5)
            y += 22
        # Inlier history.
        _text(img, "PnP inliers per frame", (12, y + 8), 0.42, _GREY)
        gx0, gy0, gw, gh = 12, y + 18, w - 24, h - y - 30
        if gh > 40 and self._inlier_history:
            cv2.rectangle(img, (gx0, gy0), (gx0 + gw, gy0 + gh), (60, 60, 65), 1)
            vals = np.array(self._inlier_history, dtype=float)
            top = max(50.0, vals.max())
            xs = gx0 + np.linspace(0, gw, self._inlier_history.maxlen or len(vals))[: len(vals)]
            ys = gy0 + gh - vals / top * gh
            cv2.polylines(img, [np.column_stack([xs, ys]).astype(np.int32)], False, _GREEN, 1, cv2.LINE_AA)
            _text(img, f"{top:.0f}", (gx0 + gw - 40, gy0 + 14), 0.4, _GREY)
        return img

    def render(
        self,
        index: int,
        left: ImageArray,
        right: ImageArray,
        result: TrackingResult,
        reference_pose: FloatArray | None = None,
        fps: float | None = None,
    ) -> np.ndarray:
        """``record`` then ``draw``: add the frame and return its canvas.

        Args:
            index: Frame index.
            left: Rectified left image.
            right: Rectified right image.
            result: Pipeline output for the frame.
            reference_pose: Wheel odometry pose at this frame, if any.
            fps: Processing rate to print.

        Returns:
            BGR canvas.
        """
        self.record(index, left, result, reference_pose)
        return self.draw(index, left, right, result, fps)
