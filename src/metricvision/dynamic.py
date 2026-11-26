"""Masks of potentially dynamic objects (people, riders, vehicles) for outdoor scenes.

A semantic segmentation network labels every pixel; pixels of movable classes are masked
(and dilated by a few pixels to cover blurry boundaries). The VO pipelines then neither
detect nor track features there, and drop matches that land on them. This complements the
geometric dynamic/occlusion gate in the sparse pipeline (``odometry.dynamic_gate``), which
catches anything moving that the network misses, and does not need to know object classes.

Requires the optional ``learned`` extra. The default SegFormer-B0 (Cityscapes) has ~4M
parameters and runs in fp16 well within 6 GB of VRAM.
"""

from __future__ import annotations  # Postponed evaluation of type hints.

from abc import ABC, abstractmethod  # Interface.
from typing import Any  # Loosely typed torch objects.

import cv2  # Mask dilation.
import numpy as np  # Arrays.

from metricvision.core.registry import Registry  # Name -> class lookup for configs.
from metricvision.core.types import BoolArray, ImageArray  # Shared types.

DYNAMIC_CLASSES = ("person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle")  # Cityscapes movable classes.
IMAGENET_MEAN = (0.485, 0.456, 0.406)  # SegFormer normalisation.
IMAGENET_STD = (0.229, 0.224, 0.225)  # SegFormer normalisation.


class DynamicMasker(ABC):
    """Predicts which pixels belong to potentially moving objects."""

    @abstractmethod
    def mask(self, image: ImageArray) -> BoolArray:
        """Return a boolean mask of dynamic pixels.

        Args:
            image: Rectified left grayscale image.

        Returns:
            (H, W) mask, True on potentially dynamic objects.
        """


MASKERS: Registry[DynamicMasker] = Registry("dynamic masker")  # Global registry.


@MASKERS.register("segformer_cityscapes")
class SegFormerDynamicMasker(DynamicMasker):
    """SegFormer trained on Cityscapes; masks people, riders and vehicles."""

    def __init__(
        self,
        model: str = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
        input_height: int = 384,
        dilate_px: int = 8,
        device: str = "cuda",
        half: bool = True,
    ) -> None:
        """Load the network.

        Args:
            model: Hugging Face checkpoint with Cityscapes labels.
            input_height: Network input height in pixels (width keeps the aspect ratio; both rounded to /32).
            dilate_px: Grow the mask by this many pixels to cover object boundaries.
            device: Torch device.
            half: Run in fp16 (CUDA only).

        Raises:
            ValueError: If the checkpoint has none of the dynamic classes.
        """
        import torch  # Optional dependency, imported lazily.
        from transformers import SegformerForSemanticSegmentation  # Optional dependency.

        self._torch = torch  # Keep a handle.
        self._device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")  # Fall back to CPU.
        self._dtype = torch.float16 if half and self._device.type == "cuda" else torch.float32  # Precision.
        self._model: Any = SegformerForSemanticSegmentation.from_pretrained(model).to(self._device, self._dtype).eval()  # Network.
        labels = {int(i): str(name).lower() for i, name in self._model.config.id2label.items()}  # Class names.
        self._dynamic_ids = torch.tensor([i for i, name in labels.items() if name in DYNAMIC_CLASSES], device=self._device)  # Movable classes.
        if self._dynamic_ids.numel() == 0:  # Wrong checkpoint.
            raise ValueError(f"{model} has none of the dynamic classes {DYNAMIC_CLASSES}")
        self._height = max(32, int(round(input_height / 32)) * 32)  # Multiple of 32.
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1)) if dilate_px > 0 else None  # Dilation.
        self._mean = torch.tensor(IMAGENET_MEAN, device=self._device).view(1, 3, 1, 1)  # Normalisation.
        self._std = torch.tensor(IMAGENET_STD, device=self._device).view(1, 3, 1, 1)  # Normalisation.

    def mask(self, image: ImageArray) -> BoolArray:
        """Segment the image and mask movable classes.

        Args:
            image: Grayscale image.

        Returns:
            (H, W) boolean mask of dynamic pixels.
        """
        torch = self._torch  # Shorthand.
        height, width = image.shape[:2]  # Output size.
        net_w = max(32, int(round(width * self._height / height / 32)) * 32)  # Keep the aspect ratio.
        with torch.inference_mode():  # No autograd.
            x = (
                torch.from_numpy(np.ascontiguousarray(image))
                .to(self._device)
                .float()
                .div_(255.0)
                .view(1, 1, height, width)
                .expand(1, 3, height, width)
            )  # Gray -> RGB.
            x = torch.nn.functional.interpolate(x, size=(self._height, net_w), mode="bilinear", align_corners=False)  # Network size.
            logits = self._model(pixel_values=((x - self._mean) / self._std).to(self._dtype)).logits  # (1, C, h/4, w/4).
            logits = torch.nn.functional.interpolate(logits.float(), size=(height, width), mode="bilinear", align_corners=False)  # Full size.
            dynamic = torch.isin(logits.argmax(dim=1)[0], self._dynamic_ids).cpu().numpy()  # Movable pixels.
        if self._kernel is not None:  # Cover boundaries.
            dynamic = cv2.dilate(dynamic.astype(np.uint8), self._kernel) > 0
        return np.asarray(dynamic, dtype=bool)  # Mask.


def pixels_in_mask(mask: BoolArray, points: np.ndarray) -> BoolArray:
    """Look up a mask at (sub-pixel) image points.

    Args:
        mask: (H, W) boolean mask.
        points: (N, 2) pixel coordinates (x, y).

    Returns:
        (N,) True where the nearest pixel is masked (points outside the image count as unmasked).
    """
    if len(points) == 0:  # Nothing to look up.
        return np.zeros(0, dtype=bool)
    cols, rows = np.rint(points[:, 0]).astype(int), np.rint(points[:, 1]).astype(int)  # Nearest pixel.
    inside = (cols >= 0) & (cols < mask.shape[1]) & (rows >= 0) & (rows < mask.shape[0])  # In the image.
    hit = np.zeros(len(points), dtype=bool)  # Default: not masked.
    hit[inside] = mask[rows[inside], cols[inside]]  # Lookup.
    return hit  # Mask values.
