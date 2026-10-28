"""A procedurally generated stereo sequence for smoke tests.

The camera moves in front of a textured plane. Every image is rendered with an
exact plane-induced homography, so ground truth is perfect and no dataset
download is needed. Use it to verify a fresh install, a new container or the
Jetson before touching real data.
"""

from __future__ import annotations  # Postponed evaluation of type hints.

import cv2  # Drawing and image warping.
import numpy as np  # Maths and random texture generation.

from metricvision.core.geometry import invert_pose, make_pose  # Pose helpers.
from metricvision.core.types import FloatArray, ImageArray, StereoCamera, StereoFrame, Trajectory  # Shared types.
from metricvision.datasets.base import DATASETS, StereoDataset  # Base class and registry.


def _rotation_about_y(angle_rad: float) -> FloatArray:
    """Return a rotation about the camera y axis (yaw in the optical frame).

    Args:
        angle_rad: Rotation angle in radians.

    Returns:
        3x3 rotation matrix.
    """
    c, s = np.cos(angle_rad), np.sin(angle_rad)  # Precompute trigonometric terms.
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)  # Standard R_y.


def _make_texture(width: int, height: int, seed: int) -> ImageArray:
    """Generate a corner-rich random texture.

    Args:
        width: Texture width in pixels.
        height: Texture height in pixels.
        seed: Random seed for reproducibility.

    Returns:
        8-bit grayscale texture.
    """
    rng = np.random.default_rng(seed)  # Seeded generator for deterministic output.
    texture = np.zeros((height, width), dtype=np.float32)  # Accumulator.
    for scale in (8, 32, 128):  # Sum noise at several scales for multi-scale structure.
        coarse = rng.random((max(2, height // scale), max(2, width // scale))).astype(np.float32)  # Low-resolution noise.
        texture += np.asarray(cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC), dtype=np.float32)  # Upsample and add.
    span = float(texture.max() - texture.min()) or 1.0  # Value range (guard against a constant image).
    texture = ((texture - texture.min()) / span * 255.0).astype(np.uint8)  # Min-max rescale to 8 bits.
    for _ in range((width * height) // 900):  # Density of random shapes, proportional to area.
        x, y = int(rng.integers(0, width)), int(rng.integers(0, height))  # Random anchor point.
        size = int(rng.integers(4, 24))  # Random shape size.
        colour = int(rng.integers(0, 256))  # Random intensity.
        if rng.random() < 0.5:  # Half rectangles (strong corners)...
            cv2.rectangle(texture, (x, y), (x + size, y + size), colour, thickness=-1)  # Filled rectangle.
        else:  # ...half circles (blob-like features).
            cv2.circle(texture, (x, y), size // 2, colour, thickness=-1)  # Filled circle.
    return np.asarray(cv2.GaussianBlur(texture, (3, 3), 0.8), dtype=np.uint8)  # Slight blur to avoid aliasing when warping.


@DATASETS.register("synthetic")
class SyntheticPlaneDataset(StereoDataset):
    """Stereo camera translating and yawing in front of a textured plane."""

    def __init__(
        self,
        num_frames: int = 60,
        width: int = 640,
        height: int = 480,
        focal_px: float = 450.0,
        baseline: float = 0.12,
        plane_depth: float = 4.0,
        lateral_step: float = 0.02,
        forward_step: float = 0.01,
        yaw_step_deg: float = 0.1,
        vertical_amplitude: float = 0.05,
        vertical_period_frames: int = 40,
        frame_rate_hz: float = 20.0,
        seed: int = 0,
    ) -> None:
        """Configure the synthetic sequence.

        Args:
            num_frames: Number of frames to generate.
            width: Image width in pixels.
            height: Image height in pixels.
            focal_px: Focal length in pixels (fx = fy).
            baseline: Stereo baseline in metres.
            plane_depth: Distance of the plane from the first camera, in metres.
            lateral_step: Sideways motion per frame, in metres.
            forward_step: Forward motion per frame, in metres.
            yaw_step_deg: Yaw rotation per frame, in degrees.
            vertical_amplitude: Amplitude of a sinusoidal up/down motion, in metres. Makes the
                path non-collinear so trajectory alignment is well posed.
            vertical_period_frames: Period of the up/down motion, in frames.
            frame_rate_hz: Frame rate used to create timestamps.
            seed: Random seed for the texture.
        """
        self._num_frames = num_frames  # Sequence length.
        self._camera = StereoCamera(  # Ideal pinhole stereo pair.
            fx=focal_px,
            fy=focal_px,
            cx=width / 2.0,
            cy=height / 2.0,  # Centred principal point.
            baseline=baseline,
            width=width,
            height=height,  # Geometry and image size.
        )
        self._plane_depth = plane_depth  # Plane equation: z = plane_depth in the world frame.
        self._steps = (lateral_step, forward_step, np.radians(yaw_step_deg))  # Per-frame motion increments.
        self._vertical = (vertical_amplitude, vertical_period_frames)  # Sinusoidal bob parameters.
        self._dt = 1.0 / frame_rate_hz  # Time between frames.
        self._texture = _make_texture(2 * width, 2 * height, seed)  # Twice the image size so views stay on-texture.
        self._texture_from_image0 = np.array(  # Maps first-frame pixels to texture pixels (centre crop).
            [[1.0, 0.0, width / 2.0], [0.0, 1.0, height / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64
        )

    @property
    def camera(self) -> StereoCamera:
        """Return the ideal stereo camera model.

        Returns:
            Intrinsics and baseline.
        """
        return self._camera  # Built in __init__.

    @property
    def tag(self) -> str:
        """Return the fixed tag ``synthetic``.

        Returns:
            Filesystem-safe identifier.
        """
        return "synthetic"  # Only one synthetic sequence exists.

    def __len__(self) -> int:
        """Return the number of frames.

        Returns:
            Frame count.
        """
        return self._num_frames  # Configured length.

    def world_from_camera(self, index: int) -> FloatArray:
        """Return the true pose of the left camera at a frame (world = first camera).

        Args:
            index: Zero-based frame index.

        Returns:
            4x4 transform T_world_cam.
        """
        lateral, forward, yaw = self._steps  # Per-frame increments.
        amplitude, period = self._vertical  # Up/down motion parameters.
        height = amplitude * np.sin(2.0 * np.pi * index / period)  # Sinusoidal bob along the camera y axis.
        position = np.array([lateral * index, height, forward * index])  # Lateral + forward drift with a vertical bob.
        return make_pose(_rotation_about_y(yaw * index), position)  # Constant yaw rate.

    def _render(self, camera_from_world: FloatArray) -> ImageArray:
        """Render the plane as seen from a camera pose using the plane-induced homography.

        For plane points X_w satisfying n^T X_w = d, a camera with X_c = R X_w + t sees
        them through H = K (R + t n^T / d) K^-1 relative to the first view.

        Args:
            camera_from_world: 4x4 transform T_cam_world.

        Returns:
            Rendered grayscale image.
        """
        k = self._camera.intrinsic_matrix  # Intrinsics.
        normal = np.array([[0.0, 0.0, 1.0]])  # Plane normal (row vector) in the world frame.
        rotation, translation = camera_from_world[:3, :3], camera_from_world[:3, 3:4]  # Split the pose.
        homography = k @ (rotation + translation @ normal / self._plane_depth) @ np.linalg.inv(k)  # First view -> this view.
        warp = homography @ np.linalg.inv(self._texture_from_image0)  # Texture -> this view.
        size = (self._camera.width, self._camera.height)  # Output size (width, height).
        rendered = cv2.warpPerspective(self._texture, warp, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)  # Render.
        return np.asarray(rendered, dtype=np.uint8)  # Fix the static type returned by OpenCV.

    def load_frame(self, index: int) -> StereoFrame:
        """Render one stereo pair.

        Args:
            index: Zero-based frame index.

        Returns:
            The synthetic stereo frame.
        """
        left_from_world = invert_pose(self.world_from_camera(index))  # World -> left camera.
        right_from_left = make_pose(np.eye(3), np.array([-self._camera.baseline, 0.0, 0.0]))  # Right camera sits at +baseline along x.
        return StereoFrame(  # Assemble the frame.
            index=index,  # Position in the sequence.
            timestamp=index * self._dt,  # Evenly spaced times.
            left=self._render(left_from_world),  # Left view.
            right=self._render(right_from_left @ left_from_world),  # Right view.
        )

    def ground_truth(self) -> Trajectory:
        """Return the exact trajectory of the left camera.

        Returns:
            Trajectory of T_world_cam poses.
        """
        trajectory = Trajectory()  # Output container.
        for index in range(self._num_frames):  # Every frame has a known pose.
            trajectory.append(index * self._dt, self.world_from_camera(index))  # Store T_world_cam.
        return trajectory  # Perfect ground truth.
