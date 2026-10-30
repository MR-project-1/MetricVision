"""ChArUco board: specification, detection and frame selection for stereo calibration.

Why ChArUco: every chessboard corner carries an ID, so a view where the board is only
partly visible (cut off by the image border, occluded, or a board with missing regions)
still gives exact 2-D/3-D correspondences. The two cameras of a stereo pair may see
*different* subsets of the board; the calibrator (calibration.calibrate_stereo_views)
uses every observation of either camera, not only corners seen by both.

Board frame: origin at the outer top-left corner of the board, x to the right, y down,
z into the board. Corner id k sits at ((k mod (nx-1)) + 1, (k div (nx-1)) + 1) * square.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np
import yaml

_NEW_API = hasattr(cv2.aruco, "CharucoDetector")


@dataclass
class BoardSpec:
    squares_x: int = 9  # squares along x (columns)
    squares_y: int = 6  # squares along y (rows)
    square_length: float = 0.040  # m - put the MEASURED printed size here
    marker_length: float = 0.030  # m
    dictionary: str = "DICT_4X4_50"
    legacy: str = "auto"  # "auto" | "true" | "false"  (OpenCV < 4.6 / calib.io layout)

    # ------------------------------------------------------------------ io
    @staticmethod
    def from_yaml(path: str) -> BoardSpec:
        with open(path) as f:
            d = yaml.safe_load(f) or {}
        d = d.get("board", d)
        spec = BoardSpec(**{k: v for k, v in d.items() if k in BoardSpec.__dataclass_fields__})
        spec.legacy = str(spec.legacy).lower()
        spec.validate()
        return spec

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> None:
        if self.marker_length >= self.square_length:
            raise ValueError("marker_length must be smaller than square_length")
        if self.squares_x < 3 or self.squares_y < 3:
            raise ValueError("board needs at least 3x3 squares")
        if not hasattr(cv2.aruco, self.dictionary):
            raise ValueError(f"unknown ArUco dictionary {self.dictionary}")
        n_markers = (self.squares_x * self.squares_y) // 2
        dict_size = int(self.dictionary.rsplit("_", 1)[-1])
        if n_markers > dict_size:
            raise ValueError(f"{n_markers} markers needed but {self.dictionary} has only {dict_size}")

    # ------------------------------------------------------------------ geometry
    @property
    def n_corners(self) -> int:
        return (self.squares_x - 1) * (self.squares_y - 1)

    @property
    def size_m(self) -> tuple[float, float]:
        return self.squares_x * self.square_length, self.squares_y * self.square_length

    def corner_points(self) -> np.ndarray:
        """(n_corners, 3) board-frame coordinates, indexed by ChArUco corner id."""
        nx = self.squares_x - 1
        k = np.arange(self.n_corners)
        return np.column_stack([(k % nx + 1) * self.square_length, (k // nx + 1) * self.square_length, np.zeros(self.n_corners)])

    def cv_board(self, legacy: bool | None = None):
        d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, self.dictionary))
        if _NEW_API:
            b = cv2.aruco.CharucoBoard((self.squares_x, self.squares_y), self.square_length, self.marker_length, d)
            if legacy is None:
                legacy = self.legacy == "true"
            b.setLegacyPattern(bool(legacy))
            return b
        return cv2.aruco.CharucoBoard_create(self.squares_x, self.squares_y, self.square_length, self.marker_length, d)

    def render(self, px_per_square: int = 100, margin_px: int = 0, legacy: bool | None = None) -> np.ndarray:
        """Board image with exactly px_per_square pixels per square (+ white margin)."""
        w = self.squares_x * px_per_square + 2 * margin_px
        h = self.squares_y * px_per_square + 2 * margin_px
        b = self.cv_board(legacy)
        if _NEW_API:
            return b.generateImage((w, h), marginSize=margin_px, borderBits=1)
        return b.draw((w, h), marginSize=margin_px, borderBits=1)


# ---------------------------------------------------------------------- detection
@dataclass
class Detection:
    ids: np.ndarray  # (n,) int corner ids
    uv: np.ndarray  # (n,2) float pixel coordinates (sub-pixel)
    n_markers: int = 0

    @property
    def n(self) -> int:
        return int(self.ids.size)


EMPTY = Detection(np.zeros(0, int), np.zeros((0, 2)), 0)


class CharucoDetector:
    def __init__(self, spec: BoardSpec, min_corners: int = 8):
        self.spec = spec
        self.min_corners = min_corners
        self.legacy = None if spec.legacy == "auto" else spec.legacy == "true"
        self._dets = {}
        for lg in [False, True] if self.legacy is None else [self.legacy]:
            self._dets[lg] = self._make(lg)

    def _make(self, legacy: bool):
        board = self.spec.cv_board(legacy)
        if _NEW_API:
            cp = cv2.aruco.CharucoParameters()
            cp.tryRefineMarkers = True
            dp = cv2.aruco.DetectorParameters()
            dp.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            return board, cv2.aruco.CharucoDetector(board, cp, dp)
        return board, None

    def _detect_one(self, img, legacy: bool) -> Detection:
        board, det = self._dets[legacy]
        if _NEW_API:
            cc, ci, mc, mi = det.detectBoard(img)
        else:  # OpenCV < 4.7 (opencv-contrib-python)
            mc, mi, _ = cv2.aruco.detectMarkers(img, board.dictionary)
            cc = ci = None
            if mi is not None and len(mi):
                _, cc, ci = cv2.aruco.interpolateCornersCharuco(mc, mi, img, board)
        if ci is None or len(ci) == 0:
            return EMPTY
        return Detection(ci.ravel().astype(int), cc.reshape(-1, 2).astype(np.float64), 0 if mi is None else len(mi))

    def detect(self, img: np.ndarray) -> Detection:
        """Detect ChArUco corners; resolves the legacy layout on the first good image."""
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if self.legacy is None:
            dets = {lg: self._detect_one(img, lg) for lg in (False, True)}
            best = max(dets, key=lambda lg: dets[lg].n)
            if dets[best].n >= self.min_corners and dets[best].n > 1.5 * dets[not best].n:
                self.legacy = best  # lock in once it's unambiguous
            d = dets[best]
        else:
            d = self._detect_one(img, self.legacy)
        return d if self.usable(d) else EMPTY

    def usable(self, d: Detection) -> bool:
        """Enough corners, and not (nearly) collinear in the board plane."""
        if d.n < self.min_corners:
            return False
        P = self.spec.corner_points()[d.ids, :2]
        s = np.linalg.svd(P - P.mean(0), compute_uv=False)
        return s[1] > 0.25 * self.spec.square_length * np.sqrt(d.n)  # >= 2 rows and 2 columns


# ---------------------------------------------------------------------- frame scoring
def sharpness(img: np.ndarray, uv: np.ndarray) -> float:
    """Variance of the Laplacian inside the detected corners' bounding box (motion blur)."""
    if uv.shape[0] == 0:
        return 0.0
    x0, y0 = np.floor(uv.min(0)).astype(int)
    x1, y1 = np.ceil(uv.max(0)).astype(int)
    roi = img[max(y0, 0) : y1 + 1, max(x0, 0) : x1 + 1]
    if roi.size < 100:
        return 0.0
    return float(cv2.Laplacian(roi, cv2.CV_64F).var())


def approx_K(width: int, height: int, hfov_deg: float) -> np.ndarray:
    f = 0.5 * width / np.tan(np.radians(hfov_deg) / 2)
    return np.array([[f, 0, (width - 1) / 2], [0, f, (height - 1) / 2], [0, 0, 1.0]])


def view_features(det: Detection, spec: BoardSpec, K: np.ndarray, size) -> np.ndarray | None:
    """Pose-diversity descriptor [u, v, tilt_x, tilt_y, log-distance] from a rough PnP."""
    if det.n < 6:
        return None
    ok, rvec, tvec = cv2.solvePnP(spec.corner_points()[det.ids], det.uv, K, None, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    n = R[:, 2]  # board normal in the camera frame
    w, h = size
    c = det.uv.mean(0)
    return np.array(
        [
            c[0] / w,
            c[1] / h,
            np.arctan2(n[0], n[2]) / np.radians(45),
            np.arctan2(n[1], n[2]) / np.radians(45),
            np.log(max(np.linalg.norm(tvec), 1e-3)) / np.log(2),
        ]
    )


def select_views(features: list, scores: list, max_views: int) -> list[int]:
    """Greedy farthest-point sampling in feature space, seeded with the best-scoring view.

    features[i] is a vector or None (unusable); scores[i] weights ties (more corners,
    seen by both cameras)."""
    idx = [i for i, f in enumerate(features) if f is not None]
    if len(idx) <= max_views:
        return idx
    F = np.stack([features[i] for i in idx])
    s = np.array([scores[i] for i in idx], float)
    s = s / max(s.max(), 1e-9)
    chosen = [int(np.argmax(s))]
    dmin = np.linalg.norm(F - F[chosen[0]], axis=1)
    while len(chosen) < max_views:
        j = int(np.argmax(dmin * (0.5 + 0.5 * s)))
        if dmin[j] <= 1e-9:
            break
        chosen.append(j)
        dmin = np.minimum(dmin, np.linalg.norm(F - F[j], axis=1))
    return sorted(idx[j] for j in chosen)
