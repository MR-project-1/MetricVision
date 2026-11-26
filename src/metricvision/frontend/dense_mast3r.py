"""MASt3R dense matcher (Leroy et al., ECCV 2024) as a pairwise front end: 2D matches only, never its 3D output.

MASt3R predicts dense per-pixel descriptors for an image pair; fast reciprocal nearest
neighbours on a regular grid turn them into 2D-2D correspondences. It has no per-image
keypoints, so it implements ``FeatureMatcher.match`` directly (no feature cache), and every
frame costs one full network pass on the (keyframe, current) pair.

Setup (non-commercial licence, CC BY-NC-SA 4.0):
    git clone --recursive https://github.com/naver/mast3r.git third_party/mast3r
    uv sync --extra learned     # adds roma + einops
    # Optional but faster/more robust than the on-first-use Hugging Face download (~2.75 GB):
    D=third_party/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric; mkdir -p $D
    B=https://huggingface.co/naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric/resolve/main
    curl -L -o $D/config.json $B/config.json && curl -L -C - -o $D/model.safetensors $B/model.safetensors
The ViT-Large checkpoint (~2.7 GB) runs in mixed precision within 6 GB of VRAM.
"""

from __future__ import annotations  # Postponed evaluation of type hints.

import os  # Repository location override.
import sys  # Making the research code importable.
from pathlib import Path  # Paths.
from typing import Any  # Loosely typed torch objects.

import numpy as np  # Array maths.
from PIL import Image  # Same resampling as the reference implementation.

from metricvision.core.types import Correspondences, ImageArray, ImageView  # Shared types.
from metricvision.frontend.base import MATCHERS, FeatureMatcher  # Registry and base class.

DEFAULT_REPO = Path(__file__).resolve().parents[3] / "third_party" / "mast3r"  # <repo>/third_party/mast3r.
DEFAULT_MODEL = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"  # Hugging Face checkpoint (used when no local file exists).
LOCAL_CHECKPOINT = (
    "checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"  # Local copy (config.json + model.safetensors), relative to the clone.
)


@MATCHERS.register("mast3r")
class Mast3rMatcher(FeatureMatcher):
    """Dense MASt3R descriptors matched by fast reciprocal nearest neighbours."""

    name = "mast3r"  # Output folder name.

    def __init__(
        self,
        max_keypoints: int = 2000,
        ratio: float = 0.8,
        mutual_check: bool = True,
        subsample: int = 8,
        image_size: int = 512,
        model: str = DEFAULT_MODEL,
        repo_path: str | None = None,
        device: str = "cuda",
        mixed_precision: bool = True,
    ) -> None:
        """Load the network.

        Args:
            max_keypoints: Upper bound on returned matches (the most confident are kept).
            ratio: Accepted for CLI compatibility; reciprocal matching needs no ratio test.
            mutual_check: Accepted for compatibility; matches are reciprocal by construction.
            subsample: Grid step (network pixels) of the seeds for reciprocal matching.
            image_size: Long side of the network input (the checkpoint is trained at 512).
            model: Hugging Face id or local checkpoint (folder or .pth); the default prefers ``<repo>/checkpoints/<name>/`` if present.
            repo_path: Clone of github.com/naver/mast3r (default ``third_party/mast3r``, or ``$METRICVISION_MAST3R``).
            device: Torch device.
            mixed_precision: Run the network under fp16 autocast.

        Raises:
            ImportError: If the MASt3R code is not found.
        """
        repo = Path(repo_path or os.environ.get("METRICVISION_MAST3R") or DEFAULT_REPO).expanduser()  # Research code location.
        if not (repo / "mast3r").is_dir():  # Not cloned.
            raise ImportError(f"MASt3R not found at {repo}; clone https://github.com/naver/mast3r with --recursive there")
        for path in (repo, repo / "dust3r", repo / "dust3r" / "croco"):  # Make mast3r, dust3r and croco importable.
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        import torch  # Optional dependency, imported lazily.
        from dust3r.inference import loss_of_one_batch  # One network pass on a batch of pairs.
        from dust3r.utils.device import collate_with_cat  # Batch assembly.
        from mast3r.fast_nn import fast_reciprocal_NNs  # Reciprocal dense matching.
        from mast3r.model import AsymmetricMASt3R  # Network.

        self._torch = torch  # Keep handles.
        self._forward, self._collate, self._reciprocal = loss_of_one_batch, collate_with_cat, fast_reciprocal_NNs  # Research functions.
        self._device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")  # Fall back to CPU.
        if model == DEFAULT_MODEL and (repo / LOCAL_CHECKPOINT / "model.safetensors").is_file():  # Prefer the locally downloaded checkpoint.
            model = str(repo / LOCAL_CHECKPOINT)  # Same weights, no Hugging Face download.
        load_kw = {} if model.endswith(".pth") else {"map_location": str(self._device)}  # Load weights straight to the GPU (saves ~2.7 GB of RAM).
        self._model: Any = AsymmetricMASt3R.from_pretrained(model, **load_kw).to(self._device).eval()  # ViT-L encoder + decoders.
        self._max_matches, self._subsample, self._size = max_keypoints, subsample, image_size  # Settings.
        self._amp = mixed_precision and self._device.type == "cuda"  # fp16 autocast.

    def _view(self, image: ImageArray, index: int) -> tuple[dict[str, Any], float, tuple[float, float]]:
        """Preprocess one image exactly like ``dust3r.utils.image.load_images`` (long side 512, crop to /16).

        Args:
            image: Grayscale image.
            index: View index within the pair.

        Returns:
            The view dict, the resize scale, and the crop offset (x0, y0) in resized pixels.
        """
        torch = self._torch  # Shorthand.
        pil = Image.fromarray(image).convert("RGB")  # Grayscale -> RGB.
        width, height = pil.size  # Original size.
        scale = self._size / max(width, height)  # Long side -> 512.
        resample = Image.Resampling.LANCZOS if scale < 1 else Image.Resampling.BICUBIC  # Reference resampling.
        pil = pil.resize((round(width * scale), round(height * scale)), resample)  # Resize.
        rw, rh = pil.size  # Resized size.
        cx, cy = rw // 2, rh // 2  # Centre.
        half_w, half_h = ((2 * cx) // 16) * 16 / 2, ((2 * cy) // 16) * 16 / 2  # Crop to multiples of 16.
        x0, y0 = cx - half_w, cy - half_h  # Crop offset.
        pil = pil.crop((x0, y0, cx + half_w, cy + half_h))  # Crop.
        tensor = torch.from_numpy(np.asarray(pil, dtype=np.float32) / 255.0).permute(2, 0, 1)  # (3, H, W) in [0, 1].
        tensor = (tensor - 0.5) / 0.5  # Same normalisation as ImgNorm.
        view = {"img": tensor[None], "true_shape": np.array([pil.size[::-1]], dtype=np.int32), "idx": index, "instance": str(index)}  # DUSt3R view.
        return view, scale, (x0, y0)

    def match(self, view_a: ImageView, view_b: ImageView) -> Correspondences:
        """Run MASt3R on the pair and return reciprocal descriptor matches in original pixels.

        Args:
            view_a: Reference image (keyframe).
            view_b: Query image (current frame).

        Returns:
            2D-2D correspondences with descriptor-confidence scores.
        """
        torch = self._torch  # Shorthand.
        (va, sa, oa), (vb, sb, ob) = self._view(view_a.image, 0), self._view(view_b.image, 1)  # Inputs.
        with torch.inference_mode():  # No autograd.
            out = self._forward(self._collate([(va, vb)]), self._model, None, self._device, use_amp=self._amp)  # Network pass.
            desc_a, desc_b = out["pred1"]["desc"][0], out["pred2"]["desc"][0]  # (H, W, D) dense descriptors.
            conf_a, conf_b = out["pred1"]["desc_conf"][0], out["pred2"]["desc_conf"][0]  # (H, W) confidences.
            xy_a, xy_b = self._reciprocal(
                desc_a, desc_b, subsample_or_initxy1=self._subsample, device=self._device, dist="dot", block_size=2**13
            )  # Matches.
        height, width = va["true_shape"][0]  # Network image size (same for both views).
        keep = (xy_a[:, 0] >= 3) & (xy_a[:, 0] < width - 3) & (xy_a[:, 1] >= 3) & (xy_a[:, 1] < height - 3)  # Drop borders.
        keep &= (xy_b[:, 0] >= 3) & (xy_b[:, 0] < width - 3) & (xy_b[:, 1] >= 3) & (xy_b[:, 1] < height - 3)  # In both views.
        xy_a, xy_b = xy_a[keep], xy_b[keep]  # Valid matches.
        if xy_a.size == 0:  # Nothing matched.
            return Correspondences.empty()
        ca = conf_a.float().cpu().numpy()[xy_a[:, 1], xy_a[:, 0]]  # Confidence in A.
        cb = conf_b.float().cpu().numpy()[xy_b[:, 1], xy_b[:, 0]]  # Confidence in B.
        scores = np.sqrt(ca * cb).astype(np.float32)  # Joint confidence.
        points_a = ((xy_a + np.array(oa)) / sa).astype(np.float32)  # Undo crop and resize.
        points_b = ((xy_b + np.array(ob)) / sb).astype(np.float32)  # Undo crop and resize.
        return Correspondences(points_a, points_b, scores).top_k(self._max_matches)  # Budget.

    def synchronize(self) -> None:
        """Wait for queued GPU work so the front-end timer is honest."""
        if self._device.type == "cuda":  # Only meaningful on GPU.
            self._torch.cuda.synchronize(self._device)
