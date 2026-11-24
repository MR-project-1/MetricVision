"""Monocular metric depth from Depth Anything V2 (metric heads), as a drop-in ``DepthEstimator``.

Only the left image is used, so plugging this into the dense pipeline turns it into a
monocular metric VO (scale comes from the network, not the baseline). In the sparse
pipeline it supplies RoMeO-style depth priors for the windowed bundle adjustment.

Requires the optional ``learned`` extra (``uv sync --extra learned``): PyTorch + transformers.
The Small models (~25M parameters) run in fp16 well within 6 GB of VRAM.
"""

from __future__ import annotations  # Postponed evaluation of type hints.

from typing import Any  # Loosely typed torch objects (torch is an optional dependency).

import numpy as np  # Output conversion.

from metricvision.core.types import Float32Array, ImageArray, StereoCamera  # Shared types.
from metricvision.depth.base import DEPTH_ESTIMATORS, DepthEstimator  # Base class and registry.

MODEL_IDS = {  # Hugging Face checkpoints with metric heads (trained on Hypersim / Virtual KITTI 2).
    ("indoor", "small"): "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
    ("indoor", "base"): "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf",
    ("outdoor", "small"): "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
    ("outdoor", "base"): "depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf",
}
IMAGENET_MEAN = (0.485, 0.456, 0.406)  # Normalisation used in training.
IMAGENET_STD = (0.229, 0.224, 0.225)  # Normalisation used in training.


@DEPTH_ESTIMATORS.register("depth_anything_v2")
class DepthAnythingV2MetricEstimator(DepthEstimator):
    """Metric depth for the left image from a Depth Anything V2 metric model."""

    def __init__(
        self,
        variant: str = "outdoor",
        size: str = "small",
        input_size: int = 392,
        device: str = "cuda",
        half: bool = True,
        min_depth: float = 0.3,
        max_depth: float = 80.0,
        reference_focal: float | None = None,
    ) -> None:
        """Load the network.

        Args:
            variant: ``indoor`` (Hypersim-trained, up to ~20 m) or ``outdoor`` (Virtual KITTI 2, up to ~80 m).
            size: ``small`` or ``base``.
            input_size: Short side of the network input in pixels (rounded to a multiple of 14).
            device: Torch device.
            half: Run in fp16 (CUDA only).
            min_depth: Depths below this are marked invalid (NaN).
            max_depth: Depths above this are marked invalid (NaN).
            reference_focal: If given, scale depth by ``camera.fx / reference_focal`` to compensate for a
                focal length different from the training data (metric depth from a single image scales with focal).

        Raises:
            ValueError: For an unknown variant/size.
            ImportError: If the ``learned`` extra is not installed.
        """
        if (variant, size) not in MODEL_IDS:  # Typo guard.
            raise ValueError(f"Unknown Depth Anything V2 model {variant}/{size}; choose from {sorted(MODEL_IDS)}")  # Clear error.
        import torch  # Optional dependency, imported lazily.
        from transformers import AutoModelForDepthEstimation  # Optional dependency.

        self._torch = torch  # Keep a handle.
        self._device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")  # Fall back to CPU.
        self._dtype = torch.float16 if half and self._device.type == "cuda" else torch.float32  # Precision.
        self._model: Any = AutoModelForDepthEstimation.from_pretrained(MODEL_IDS[(variant, size)]).to(self._device, self._dtype).eval()  # Network.
        self._input_size = max(14, int(round(input_size / 14)) * 14)  # Multiple of the ViT patch size.
        self._mean = torch.tensor(IMAGENET_MEAN, device=self._device).view(1, 3, 1, 1)  # Normalisation.
        self._std = torch.tensor(IMAGENET_STD, device=self._device).view(1, 3, 1, 1)  # Normalisation.
        self._min_depth, self._max_depth = min_depth, max_depth  # Validity range.
        self._reference_focal = reference_focal  # Optional focal compensation.

    @property
    def peak_memory_mb(self) -> float:
        """Return the peak GPU memory allocated by PyTorch so far, in MiB (0 on CPU).

        Returns:
            Peak allocated memory.
        """
        if self._device.type != "cuda":  # CPU run.
            return 0.0
        return float(self._torch.cuda.max_memory_allocated(self._device)) / 2**20  # Bytes to MiB.

    def compute(self, left: ImageArray, right: ImageArray, camera: StereoCamera) -> Float32Array:
        """Predict metric depth for the left image (the right image is ignored).

        Args:
            left: Rectified left grayscale image.
            right: Unused.
            camera: Rectified camera (for optional focal compensation).

        Returns:
            Float32 depth map of shape (H, W) in metres; NaN marks invalid pixels.
        """
        torch = self._torch  # Shorthand.
        height, width = left.shape[:2]  # Output size.
        scale = self._input_size / min(height, width)  # Resize so the short side is input_size.
        net_h = max(14, int(round(height * scale / 14)) * 14)  # Multiple of 14.
        net_w = max(14, int(round(width * scale / 14)) * 14)  # Multiple of 14.
        with torch.inference_mode():  # No autograd.
            image = torch.from_numpy(np.ascontiguousarray(left)).to(self._device).float().div_(255.0)  # (H, W) in [0, 1].
            image = image.view(1, 1, height, width).expand(1, 3, height, width)  # Grayscale -> 3 channels.
            image = torch.nn.functional.interpolate(image, size=(net_h, net_w), mode="bilinear", align_corners=False)  # Network size.
            image = ((image - self._mean) / self._std).to(self._dtype)  # Normalise.
            predicted = self._model(pixel_values=image).predicted_depth  # (1, h, w) metric depth.
            depth = torch.nn.functional.interpolate(
                predicted.unsqueeze(1).float(), size=(height, width), mode="bilinear", align_corners=False
            )  # Full size.
            depth_np = depth[0, 0].cpu().numpy().astype(np.float32)  # To host (synchronises the GPU).
        if self._reference_focal is not None:  # Compensate for a focal length unlike the training data.
            depth_np *= np.float32(camera.fx / self._reference_focal)  # Depth scales with focal for fixed apparent size.
        depth_np[(depth_np < self._min_depth) | (depth_np > self._max_depth) | ~np.isfinite(depth_np)] = np.nan  # Validity range.
        return depth_np  # Metric depth map.
