#!/usr/bin/env python3
"""Minimal dense-CLIP LangSplat-style autoencoder trainer.

What this script does
---------------------
1. Reads a directory of images.
2. Extracts dense patch-level CLIP features from each image using an OpenCLIP ViT tower.
   - We follow the local `clip_main.py` path: resize shortest side, optional center crop,
     pad to a multiple of the patch size, and interpolate positional embeddings as needed.
   - We extract patch tokens with the same last-block value-path logic used there.
   - We project each patch token with the CLIP visual projection, yielding one D-D feature per patch.
3. Builds a contiguous fp16 training cache and streams batches from disk while training.
4. Exports 3-D latent patch features for each image.

Notes
-----
- This is NOT LangSplat's original SAM-mask pipeline.
- This is also NOT a literal reimplementation of MaskCLIP.
- It is a minimal dense/patched CLIP teacher suitable for a standalone scene autoencoder trainer.
- The autoencoder architecture/losses follow the released LangSplat code closely.

Outputs
-------
By default, if images live in <scene>/images, the script writes:
    <scene>/dense_clip_features/<stem>_f.npy       # [Gh, Gw, D]
    <scene>/dense_clip_features_dim3/<stem>_f.npy  # [Gh, Gw, 3]
    <scene>/dense_clip_feature_cache_fp16.npy      # [N_total_patches, D]
    <scene>/clip_autoencoder_viz/pca_comparison.png
    <scene>/clip_autoencoder_ckpt/best_ckpt.pth
    <scene>/clip_autoencoder_ckpt/config.json
    <scene>/clip_autoencoder_ckpt/run_summary.json
    <scene>/clip_autoencoder_ckpt/train_log.json

If images are elsewhere, outputs go to <images_dir>/dense_clip_outputs by default.
"""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Sequence

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.io import ImageReadMode, read_image

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Autoencoder(nn.Module):
    """Released LangSplat autoencoder.

    Encoder: input_dim -> 256 -> 128 -> 64 -> 32 -> 3
      with BatchNorm+ReLU before each Linear except the first.
    Decoder: 3 -> 16 -> 32 -> 64 -> 128 -> 256 -> 256 -> input_dim
      with ReLU before each Linear except the first.
    The latent and reconstruction are left unconstrained.
    """

    def __init__(self, encoder_dims: Sequence[int], decoder_dims: Sequence[int], input_dim: int) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")

        enc_layers: List[nn.Module] = []
        for i, out_dim in enumerate(encoder_dims):
            if i == 0:
                enc_layers.append(nn.Linear(input_dim, out_dim))
            else:
                enc_layers.append(nn.BatchNorm1d(encoder_dims[i - 1]))
                enc_layers.append(nn.ReLU(inplace=False))
                enc_layers.append(nn.Linear(encoder_dims[i - 1], out_dim))
        self.encoder = nn.ModuleList(enc_layers)

        dec_layers: List[nn.Module] = []
        for i, out_dim in enumerate(decoder_dims):
            if i == 0:
                dec_layers.append(nn.Linear(encoder_dims[-1], out_dim))
            else:
                dec_layers.append(nn.ReLU(inplace=False))
                dec_layers.append(nn.Linear(decoder_dims[i - 1], out_dim))
        self.decoder = nn.ModuleList(dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.encoder:
            x = layer(x)
        return x

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        for layer in self.decoder:
            z = layer(z)
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


@dataclass
class PCAProjection:
    mean: np.ndarray
    components: np.ndarray
    low: np.ndarray
    high: np.ndarray


def resize_shortest_side(img: Image.Image, size: int) -> Image.Image:
    width, height = img.size
    if width <= height:
        new_width = size
        new_height = max(1, int(round(height * size / width)))
    else:
        new_height = size
        new_width = max(1, int(round(width * size / height)))
    return img.resize((new_width, new_height), Image.BICUBIC)


def center_crop_square(img: Image.Image, size: int) -> Image.Image:
    width, height = img.size
    left = max((width - size) // 2, 0)
    top = max((height - size) // 2, 0)
    right = min(left + size, width)
    bottom = min(top + size, height)
    return img.crop((left, top, right, bottom))


def pad_to_multiple_bchw(batch: torch.Tensor, patch_size: int, mode: str = "constant") -> torch.Tensor:
    pad_h = (-batch.shape[-2]) % patch_size
    pad_w = (-batch.shape[-1]) % patch_size
    if pad_h == 0 and pad_w == 0:
        return batch
    if mode == "constant":
        return F.pad(batch, (0, pad_w, 0, pad_h), mode=mode, value=0.0)
    return F.pad(batch, (0, pad_w, 0, pad_h), mode=mode)


def load_and_preprocess_rgb(
    path: Path,
    load_size: int | None,
    center_crop: bool,
    patch_size: int,
    padding_mode: str,
) -> torch.Tensor:
    tensor = read_image(str(path), mode=ImageReadMode.RGB).to(torch.float32).div_(255.0)
    if load_size is not None:
        height, width = tensor.shape[-2:]
        resolved_size = abs(load_size) * min(width, height) if load_size < 0 else load_size
        resized_height, resized_width = resolve_resized_hw(width, height, int(resolved_size))
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=(resized_height, resized_width),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).squeeze(0)
        if center_crop:
            crop_height = min(resized_height, int(resolved_size))
            crop_width = min(resized_width, int(resolved_size))
            top = max((resized_height - crop_height) // 2, 0)
            left = max((resized_width - crop_width) // 2, 0)
            tensor = tensor[:, top:top + crop_height, left:left + crop_width]
    return pad_to_multiple_bchw(tensor.unsqueeze(0), patch_size, mode=padding_mode).squeeze(0)


def iter_preprocessed_images(
    paths: Sequence[Path],
    load_size: int | None,
    center_crop: bool,
    patch_size: int,
    padding_mode: str,
) -> Iterator[tuple[Path, torch.Tensor]]:
    max_workers = min(len(paths), max(1, min(8, os.cpu_count() or 1)))
    if max_workers <= 1:
        for path in paths:
            yield path, load_and_preprocess_rgb(
                path,
                load_size=load_size,
                center_crop=center_crop,
                patch_size=patch_size,
                padding_mode=padding_mode,
            )
        return

    path_iter = iter(paths)

    def submit(executor: ThreadPoolExecutor, path: Path) -> Future[torch.Tensor]:
        return executor.submit(
            load_and_preprocess_rgb,
            path,
            load_size,
            center_crop,
            patch_size,
            padding_mode,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        inflight: deque[tuple[Path, Future[torch.Tensor]]] = deque()
        for _ in range(max_workers * 2):
            try:
                path = next(path_iter)
            except StopIteration:
                break
            inflight.append((path, submit(executor, path)))

        while inflight:
            path, future = inflight.popleft()
            yield path, future.result()
            try:
                next_path = next(path_iter)
            except StopIteration:
                continue
            inflight.append((next_path, submit(executor, next_path)))


def resolve_resized_hw(width: int, height: int, size: int) -> tuple[int, int]:
    if width <= height:
        new_width = size
        new_height = max(1, int(round(height * size / width)))
    else:
        new_height = size
        new_width = max(1, int(round(width * size / height)))
    return new_height, new_width


def resolve_preprocessed_hw(
    width: int,
    height: int,
    load_size: int | None,
    center_crop: bool,
    patch_size: int,
) -> tuple[int, int]:
    if load_size is not None:
        resolved_size = abs(load_size) * min(width, height) if load_size < 0 else load_size
        height, width = resolve_resized_hw(width, height, int(resolved_size))
        if center_crop:
            height = min(height, int(resolved_size))
            width = min(width, int(resolved_size))

    padded_height = height + ((-height) % patch_size)
    padded_width = width + ((-width) % patch_size)
    return padded_height, padded_width


def interpolate_positional_embedding(
    positional_embedding: torch.Tensor,
    x: torch.Tensor,
    patch_size: int,
    height: int,
    width: int,
) -> torch.Tensor:
    if positional_embedding.ndim != 2:
        raise ValueError("Expected 2-D positional_embedding")

    num_patches = x.shape[1] - 1
    num_original_patches = positional_embedding.shape[0] - 1
    if num_patches == num_original_patches and height == width:
        return positional_embedding.to(x.dtype)

    dim = x.shape[-1]
    class_pos_embed = positional_embedding[:1]
    patch_pos_embed = positional_embedding[1:]
    grid_h = height // patch_size
    grid_w = width // patch_size
    if grid_h * grid_w != num_patches:
        raise ValueError("Number of patches does not match positional embedding interpolation target")

    grid_h_f = grid_h + 0.1
    grid_w_f = grid_w + 0.1
    patch_per_axis = int(np.sqrt(num_original_patches))
    patch_pos_embed_interp = F.interpolate(
        patch_pos_embed.reshape(1, patch_per_axis, patch_per_axis, dim).permute(0, 3, 1, 2),
        scale_factor=(grid_h_f / patch_per_axis, grid_w_f / patch_per_axis),
        mode="bicubic",
        align_corners=False,
        recompute_scale_factor=False,
    )
    if int(grid_h_f) != patch_pos_embed_interp.shape[-2] or int(grid_w_f) != patch_pos_embed_interp.shape[-1]:
        raise ValueError("Positional embedding interpolation failed")

    patch_pos_embed_interp = patch_pos_embed_interp.permute(0, 2, 3, 1).reshape(-1, dim)
    return torch.cat([class_pos_embed, patch_pos_embed_interp], dim=0).to(x.dtype)


class DenseOpenCLIPExtractor(nn.Module):
    """Dense patch-level CLIP extractor using the local clip_main-style path."""

    def __init__(
        self,
        model_name: str = "ViT-L-14-336-quickgelu",
        pretrained: str = "openai",
        device: str = "cuda",
    ) -> None:
        super().__init__()
        try:
            import open_clip
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Please install open-clip-torch") from exc

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        precision = "fp32"
        model, _, _ = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            precision=precision,
        )
        self.model = model.eval().to(self.device)

        visual = self.model.visual
        required_attrs = [
            "conv1",
            "class_embedding",
            "positional_embedding",
            "ln_pre",
            "transformer",
            "ln_post",
            "proj",
            "patch_size",
        ]
        missing = [name for name in required_attrs if not hasattr(visual, name)]
        if missing:
            raise RuntimeError(
                f"OpenCLIP visual tower missing expected attrs {missing}. "
                "This script expects a ViT-style OpenCLIP visual encoder."
            )

        mean = torch.tensor(CLIP_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(CLIP_STD, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)

    @property
    def patch_size(self) -> int:
        patch_size = self.model.visual.patch_size
        if isinstance(patch_size, int):
            return int(patch_size)
        return int(patch_size[0])

    @property
    def feature_dim(self) -> int:
        proj = self.model.visual.proj
        if proj is None:
            ln_post = self.model.visual.ln_post
            if hasattr(ln_post, "normalized_shape"):
                shape = ln_post.normalized_shape
                return int(shape[0] if isinstance(shape, (tuple, list)) else shape)
            raise RuntimeError("Unable to infer feature dimension from CLIP visual tower")
        return int(proj.shape[-1])

    def _get_patch_encodings(self, image_batch: torch.Tensor) -> torch.Tensor:
        visual = self.model.visual
        _, _, height, width = image_batch.shape
        x = visual.conv1(image_batch)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        class_token = visual.class_embedding.to(x.dtype)
        class_token = class_token + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        x = torch.cat([class_token, x], dim=1)
        x = x + interpolate_positional_embedding(
            visual.positional_embedding,
            x,
            patch_size=self.patch_size,
            height=height,
            width=width,
        )
        x = visual.ln_pre(x)
        *layers, last_resblock = visual.transformer.resblocks
        if layers:
            x = torch.nn.Sequential(*layers)(x)
        v_in_proj_weight = last_resblock.attn.in_proj_weight[-last_resblock.attn.embed_dim:]
        v_in_proj_bias = last_resblock.attn.in_proj_bias[-last_resblock.attn.embed_dim:]
        v_in = F.linear(last_resblock.ln_1(x), v_in_proj_weight, v_in_proj_bias)
        x = F.linear(v_in, last_resblock.attn.out_proj.weight, last_resblock.attn.out_proj.bias)
        x = x[:, 1:, :]
        x = visual.ln_post(x)
        if visual.proj is not None:
            x = x @ visual.proj
        return x

    @torch.inference_mode()
    def encode_dense(self, batch: torch.Tensor, return_device_tensor: bool = False) -> torch.Tensor:
        """Returns [B, Gh, Gw, D] dense CLIP features."""
        batch = batch.to(torch.float32)
        batch = (batch - self.mean) / self.std
        batch = batch.to(device=self.device, non_blocking=self.device.type == "cuda")
        if self.device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                patch_tokens = self._get_patch_encodings(batch)
        else:
            patch_tokens = self._get_patch_encodings(batch)
        gh = batch.shape[-2] // self.patch_size
        gw = batch.shape[-1] // self.patch_size
        patch_tokens = patch_tokens.reshape(batch.shape[0], gh, gw, -1).float()
        if return_device_tensor:
            return patch_tokens
        return patch_tokens.cpu()


def pca_rgb_map(
    feature_map: np.ndarray,
    projection: PCAProjection | None = None,
    q_min: float = 0.01,
    q_max: float = 0.99,
) -> tuple[np.ndarray, PCAProjection]:
    if feature_map.ndim != 3:
        raise ValueError(f"Expected [H, W, D] feature map, got {feature_map.shape}")

    flat = feature_map.reshape(-1, feature_map.shape[-1]).astype(np.float32, copy=False)
    if flat.shape[0] == 0:
        raise ValueError("Cannot run PCA on an empty feature map")

    if projection is None:
        mean = flat.mean(axis=0, keepdims=True)
        centered = flat - mean
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        num_components = min(3, vh.shape[0])
        components = vh[:num_components].T.astype(np.float32, copy=False)
        projected = centered @ components
        if num_components < 3:
            pad = np.zeros((projected.shape[0], 3 - num_components), dtype=np.float32)
            projected = np.concatenate([projected, pad], axis=1)
            comp_pad = np.zeros((components.shape[0], 3 - num_components), dtype=np.float32)
            components = np.concatenate([components, comp_pad], axis=1)
        low = np.quantile(projected, q_min, axis=0).astype(np.float32, copy=False)
        high = np.quantile(projected, q_max, axis=0).astype(np.float32, copy=False)
        projection = PCAProjection(
            mean=mean.reshape(-1).astype(np.float32, copy=False),
            components=components,
            low=low,
            high=high,
        )
    else:
        centered = flat - projection.mean.reshape(1, -1)
        projected = centered @ projection.components

    denom = np.maximum(projection.high - projection.low, 1e-6)
    rgb = np.clip((projected - projection.low) / denom, 0.0, 1.0)
    return rgb.reshape(feature_map.shape[0], feature_map.shape[1], 3), projection


def feature_map_to_image(feature_map: np.ndarray, scale: int) -> Image.Image:
    rgb = (np.clip(feature_map, 0.0, 1.0) * 255.0).astype(np.uint8)
    image = Image.fromarray(rgb)
    if scale > 1:
        image = image.resize((image.width * scale, image.height * scale), Image.NEAREST)
    return image


def save_pca_comparison_gallery(
    rows: Sequence[tuple[str, np.ndarray, np.ndarray]],
    out_path: Path,
    scale: int = 4,
    title: str = "High-D PCA vs Low-D -> Decoder -> High-D PCA",
) -> None:
    if not rows:
        return

    rendered_rows = [
        (label, feature_map_to_image(highdim, scale=scale), feature_map_to_image(recon, scale=scale))
        for label, highdim, recon in rows
    ]
    margin = 12
    gap_x = 12
    gap_y = 16
    label_height = 18
    title_height = 24 if title else 0
    col_labels = ("High-D PCA", "Decoded PCA")
    max_left = max(image_left.width for _, image_left, _ in rendered_rows)
    max_right = max(image_right.width for _, _, image_right in rendered_rows)
    max_row_height = max(max(image_left.height, image_right.height) for _, image_left, image_right in rendered_rows)
    width = margin * 2 + max_left + gap_x + max_right
    height = (
        margin * 2
        + title_height
        + label_height
        + len(rendered_rows) * label_height
        + len(rendered_rows) * max_row_height
        + max(0, len(rendered_rows) - 1) * gap_y
    )

    canvas = Image.new("RGB", (width, height), color=(24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    y = margin
    if title:
        draw.text((margin, y), title, fill=(240, 240, 240))
        y += title_height

    left_x = margin
    right_x = margin + max_left + gap_x
    draw.text((left_x, y), col_labels[0], fill=(240, 240, 240))
    draw.text((right_x, y), col_labels[1], fill=(240, 240, 240))
    y += label_height

    for label, image_left, image_right in rendered_rows:
        draw.text((margin, y), label, fill=(240, 240, 240))
        image_y = y + label_height
        canvas.paste(image_left, (left_x, image_y))
        canvas.paste(image_right, (right_x, image_y))
        y = image_y + max_row_height + gap_y

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


def safe_rate(count: int | float, seconds: float) -> float:
    if seconds <= 0.0:
        return 0.0
    return float(count) / seconds


def configure_cuda_math_mode(device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def discover_images(images_dir: Path) -> List[Path]:
    return sorted([p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES])


def resolve_output_root(images_dir: Path, output_root: Path | None) -> Path:
    if output_root is not None:
        return output_root
    if images_dir.name == "images":
        return images_dir.parent
    return images_dir / "dense_clip_outputs"


@dataclass
class FileInfo:
    stem: str
    highdim_path: Path
    lowdim_path: Path
    cache_offset: int
    grid_h: int
    grid_w: int
    count: int


@dataclass
class FeatureCacheInfo:
    path: Path
    num_rows: int
    feature_dim: int
    dtype: str


@dataclass
class ExtractResult:
    file_infos: list[FileInfo]
    feature_cache: FeatureCacheInfo | None
    device_feature_cache: torch.Tensor | None


@dataclass
class TrainSummary:
    best_epoch: int
    best_eval_mse: float


def choose_visualization_samples(file_infos: Sequence[FileInfo], num_images: int) -> list[FileInfo]:
    if num_images <= 0:
        return []
    count = min(num_images, len(file_infos))
    return random.sample(list(file_infos), k=count)


def save_selection_manifest(sample_infos: Sequence[FileInfo], out_path: Path) -> None:
    rows = [{"stem": info.stem, "highdim_path": str(info.highdim_path), "lowdim_path": str(info.lowdim_path)} for info in sample_infos]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def describe_feature_cache(
    file_infos: Sequence[FileInfo],
    feature_dim: int,
) -> tuple[int, dict[str, object]]:
    total_rows = 0
    source_files: list[dict[str, int | str]] = []
    for info in file_infos:
        stat = info.highdim_path.stat()
        total_rows += info.count
        source_files.append(
            {
                "stem": info.stem,
                "path": str(info.highdim_path),
                "grid_h": info.grid_h,
                "grid_w": info.grid_w,
                "count": info.count,
                "mtime_ns": int(stat.st_mtime_ns),
                "size_bytes": int(stat.st_size),
            }
        )

    expected_meta: dict[str, object] = {
        "cache_dtype": "float16",
        "num_rows": total_rows,
        "feature_dim": feature_dim,
        "source_files": source_files,
    }
    return total_rows, expected_meta


@torch.inference_mode()
def decode_lowdim_to_highdim(
    model: Autoencoder,
    lowdim_path: Path,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    lowdim = np.load(lowdim_path).astype(np.float32)
    flat = lowdim.reshape(-1, lowdim.shape[-1])
    chunks: list[np.ndarray] = []
    for start in range(0, flat.shape[0], batch_size):
        chunk = torch.from_numpy(flat[start:start + batch_size]).to(device)
        with cuda_autocast_context(device):
            recon = model.decode(chunk)
        recon = recon.float().cpu().numpy()
        chunks.append(recon)
    return np.concatenate(chunks, axis=0).reshape(lowdim.shape[0], lowdim.shape[1], -1)


def save_final_pca_comparison(
    model: Autoencoder,
    sample_infos: Sequence[FileInfo],
    out_path: Path,
    batch_size: int,
    device: torch.device,
    scale: int = 4,
) -> None:
    rows: list[tuple[str, np.ndarray, np.ndarray]] = []
    for info in sample_infos:
        highdim = np.load(info.highdim_path)
        highdim_rgb, projection = pca_rgb_map(highdim)
        recon = decode_lowdim_to_highdim(model, info.lowdim_path, batch_size=batch_size, device=device)
        recon_rgb, _ = pca_rgb_map(recon, projection=projection)
        rows.append((info.stem, highdim_rgb, recon_rgb))
    save_pca_comparison_gallery(rows, out_path, scale=scale)


def extract_dense_features(
    images_dir: Path,
    output_root: Path,
    model_name: str,
    pretrained: str,
    load_size: int | None,
    center_crop: bool,
    padding_mode: str,
    batch_size: int,
    reextract: bool,
    device: torch.device,
) -> ExtractResult:
    highdim_dir = output_root / "dense_clip_features"
    lowdim_dir = output_root / "dense_clip_features_dim3"
    highdim_dir.mkdir(parents=True, exist_ok=True)
    lowdim_dir.mkdir(parents=True, exist_ok=True)

    image_paths = discover_images(images_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {images_dir}")

    extractor = DenseOpenCLIPExtractor(
        model_name=model_name,
        pretrained=pretrained,
        device=str(device),
    )
    feature_dim = extractor.feature_dim
    meta_path = output_root / "dense_clip_features_meta.json"
    expected_meta = {
        "extractor": "open_clip_clip_main_style_patch_tokens_with_visual_projection",
        "clip_model_type": model_name,
        "clip_pretrained": pretrained,
        "clip_load_size": load_size,
        "clip_center_crop": center_crop,
        "clip_padding_mode": padding_mode,
        "clip_feature_normalization": "none",
        "patch_size": extractor.patch_size,
        "grid_mode": "per_image",
        "images_dir": str(images_dir),
    }
    if not reextract and meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            cached_meta = json.load(f)
        mismatch_keys = [key for key, value in expected_meta.items() if cached_meta.get(key) != value]
        if mismatch_keys:
            mismatch_str = ", ".join(mismatch_keys)
            raise ValueError(
                f"Cached dense CLIP features were produced with different settings ({mismatch_str}). "
                "Use --reextract to rebuild them."
            )

    file_infos: list[FileInfo] = []
    file_info_by_path: dict[Path, FileInfo] = {}
    cache_offsets: dict[Path, int] = {}
    total_rows = 0
    for path in image_paths:
        with Image.open(path) as image:
            width, height = image.size
        padded_height, padded_width = resolve_preprocessed_hw(
            width=width,
            height=height,
            load_size=load_size,
            center_crop=center_crop,
            patch_size=extractor.patch_size,
        )
        grid_h = padded_height // extractor.patch_size
        grid_w = padded_width // extractor.patch_size
        info = FileInfo(
            stem=path.stem,
            highdim_path=highdim_dir / f"{path.stem}_f.npy",
            lowdim_path=lowdim_dir / f"{path.stem}_f.npy",
            cache_offset=total_rows,
            grid_h=grid_h,
            grid_w=grid_w,
            count=grid_h * grid_w,
        )
        file_infos.append(info)
        file_info_by_path[path] = info
        cache_offsets[path] = info.cache_offset
        total_rows += info.count

    pending: list[Path] = []
    for path in image_paths:
        if reextract or not file_info_by_path[path].highdim_path.exists():
            pending.append(path)

    cache_path = output_root / "dense_clip_feature_cache_fp16.npy"
    cache_meta_path = output_root / "dense_clip_feature_cache_meta.json"
    cache_memmap: np.memmap | None = None
    feature_cache: FeatureCacheInfo | None = None
    device_feature_cache = maybe_allocate_device_feature_cache(total_rows, feature_dim, device=device)
    if pending:
        cache_memmap = np.lib.format.open_memmap(
            cache_path,
            mode="w+",
            dtype=np.float16,
            shape=(total_rows, feature_dim),
        )
        pending_set = set(pending)
        for path in image_paths:
            if path in pending_set:
                continue
            info = file_info_by_path[path]
            arr = np.load(info.highdim_path, mmap_mode="r")
            expected_shape = (info.grid_h, info.grid_w, feature_dim)
            if tuple(int(dim) for dim in arr.shape) != expected_shape:
                raise ValueError(
                    f"Expected {expected_shape} in {info.highdim_path}, got {arr.shape}. "
                    "Use --reextract to rebuild them."
                )
            offset = cache_offsets[path]
            flat = arr.reshape(-1, feature_dim)
            cache_memmap[offset:offset + info.count] = flat.astype(np.float16)
            if device_feature_cache is not None:
                device_feature_cache[offset:offset + info.count].copy_(
                    torch.from_numpy(flat).to(device=device, dtype=torch.float32)
                )

    batch_paths: list[Path] = []
    batch_tensors: list[torch.Tensor] = []
    batch_shape: tuple[int, int, int] | None = None
    pending_write_futures: deque[Future[None]] = deque()
    transfer_stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None

    def write_feature_batch(
        paths: Sequence[Path],
        feats_cpu: torch.Tensor,
        transfer_event: torch.cuda.Event | None,
        transfer_refs: Sequence[torch.Tensor],
    ) -> None:
        if transfer_event is not None:
            transfer_event.synchronize()
        feats = feats_cpu.numpy()
        for i, path in enumerate(paths):
            info = file_info_by_path[path]
            feat = feats[i]
            expected_shape = (info.grid_h, info.grid_w, feature_dim)
            if tuple(int(dim) for dim in feat.shape) != expected_shape:
                raise ValueError(f"Expected extracted feature shape {expected_shape} for {path}, got {feat.shape}")
            np.save(info.highdim_path, feat)
            if cache_memmap is not None:
                offset = cache_offsets[path]
                cache_memmap[offset:offset + info.count] = feat.reshape(-1, feature_dim)

    def wait_for_pending_writes(limit: int = 0) -> None:
        while len(pending_write_futures) > limit:
            pending_write_futures.popleft().result()

    def flush_batch(write_executor: ThreadPoolExecutor) -> None:
        nonlocal batch_shape
        if not batch_tensors:
            return
        batch = torch.stack(batch_tensors, dim=0)
        if device.type == "cuda":
            batch = batch.pin_memory()
        feats_device = extractor.encode_dense(batch, return_device_tensor=device_feature_cache is not None)
        paths = tuple(batch_paths)
        for i, path in enumerate(paths):
            info = file_info_by_path[path]
            offset = cache_offsets[path]
            if device_feature_cache is not None:
                device_feature_cache[offset:offset + info.count].copy_(feats_device[i].reshape(-1, feature_dim))
        if device.type == "cuda":
            feats_cpu = torch.empty(
                feats_device.shape,
                dtype=torch.float16,
                device="cpu",
                pin_memory=True,
            )
            with torch.cuda.stream(transfer_stream):
                transfer_stream.wait_stream(torch.cuda.current_stream(device))
                feats_half = feats_device.to(dtype=torch.float16)
                feats_cpu.copy_(feats_half, non_blocking=True)
                transfer_event = torch.cuda.Event()
                transfer_event.record(transfer_stream)
            transfer_refs: tuple[torch.Tensor, ...] = (feats_device, feats_half)
        else:
            feats_cpu = feats_device.to(dtype=torch.float16).cpu()
            transfer_event = None
            transfer_refs = (feats_device,)
        wait_for_pending_writes(limit=0)
        pending_write_futures.append(
            write_executor.submit(write_feature_batch, paths, feats_cpu, transfer_event, transfer_refs)
        )
        batch_paths.clear()
        batch_tensors.clear()
        batch_shape = None

    with ThreadPoolExecutor(max_workers=1) as write_executor:
        preprocessed_images = iter_preprocessed_images(
            pending,
            load_size=load_size,
            center_crop=center_crop,
            patch_size=extractor.patch_size,
            padding_mode=padding_mode,
        )
        for path, tensor in tqdm(preprocessed_images, total=len(pending), desc="Extracting dense CLIP patches"):
            shape = tuple(int(dim) for dim in tensor.shape)
            if batch_tensors and (shape != batch_shape or len(batch_tensors) >= batch_size):
                flush_batch(write_executor)
            batch_paths.append(path)
            batch_tensors.append(tensor)
            batch_shape = shape
        flush_batch(write_executor)
        wait_for_pending_writes()

    if cache_memmap is not None:
        cache_memmap.flush()
        del cache_memmap
        cache_total_rows, expected_cache_meta = describe_feature_cache(file_infos, feature_dim)
        with open(cache_meta_path, "w", encoding="utf-8") as f:
            json.dump(expected_cache_meta, f, indent=2)
        feature_cache = FeatureCacheInfo(
            path=cache_path,
            num_rows=cache_total_rows,
            feature_dim=feature_dim,
            dtype="float16",
        )
    elif cache_path.exists() and cache_meta_path.exists():
        cache_total_rows, expected_cache_meta = describe_feature_cache(file_infos, feature_dim)
        with open(cache_meta_path, "r", encoding="utf-8") as f:
            cached_meta = json.load(f)
        if cached_meta == expected_cache_meta:
            feature_cache = FeatureCacheInfo(
                path=cache_path,
                num_rows=cache_total_rows,
                feature_dim=feature_dim,
                dtype="float16",
            )

    meta = dict(expected_meta)
    meta["feature_dim"] = feature_dim
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return ExtractResult(
        file_infos=file_infos,
        feature_cache=feature_cache,
        device_feature_cache=device_feature_cache,
    )


def build_feature_cache(
    file_infos: Sequence[FileInfo],
    output_root: Path,
) -> FeatureCacheInfo:
    if not file_infos:
        raise ValueError("No feature files available to build the training cache.")

    cache_path = output_root / "dense_clip_feature_cache_fp16.npy"
    meta_path = output_root / "dense_clip_feature_cache_meta.json"
    first_arr = np.load(file_infos[0].highdim_path, mmap_mode="r")
    if first_arr.ndim != 3:
        raise ValueError(f"Expected [Gh, Gw, D] in {file_infos[0].highdim_path}, got {first_arr.shape}")
    feature_dim = int(first_arr.shape[-1])
    del first_arr

    total_rows, expected_meta = describe_feature_cache(file_infos, feature_dim)

    if cache_path.exists() and meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            cached_meta = json.load(f)
        if cached_meta == expected_meta:
            return FeatureCacheInfo(
                path=cache_path,
                num_rows=total_rows,
                feature_dim=feature_dim,
                dtype="float16",
            )

    cache = np.lib.format.open_memmap(
        cache_path,
        mode="w+",
        dtype=np.float16,
        shape=(total_rows, feature_dim),
    )
    offset = 0
    for info in tqdm(file_infos, desc="Building fp16 feature cache"):
        arr = np.load(info.highdim_path, mmap_mode="r")
        flat = arr.reshape(-1, feature_dim)
        next_offset = offset + flat.shape[0]
        cache[offset:next_offset] = flat.astype(np.float16)
        offset = next_offset
    cache.flush()
    del cache

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(expected_meta, f, indent=2)

    return FeatureCacheInfo(
        path=cache_path,
        num_rows=total_rows,
        feature_dim=feature_dim,
        dtype="float16",
    )


def iterate_feature_cache_batches(
    cache_info: FeatureCacheInfo,
    batch_size: int,
    block_rows: int,
    shuffle: bool,
    seed: int,
    device: torch.device,
    drop_single_sample: bool,
) -> Iterator[torch.Tensor]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    effective_block_rows = max(batch_size, block_rows)
    block_starts = np.arange(0, cache_info.num_rows, effective_block_rows, dtype=np.int64)
    rng = np.random.default_rng(seed)
    if shuffle and block_starts.size > 1:
        rng.shuffle(block_starts)

    cache = np.load(cache_info.path, mmap_mode="r")
    pin_memory = device.type == "cuda"
    for block_start in block_starts.tolist():
        block_end = min(block_start + effective_block_rows, cache_info.num_rows)
        block_np = np.array(cache[block_start:block_end], copy=True)
        if shuffle and block_np.shape[0] > 1:
            order = rng.permutation(block_np.shape[0])
            block_np = np.ascontiguousarray(block_np[order])

        block = torch.from_numpy(block_np)
        if pin_memory:
            block = block.pin_memory()

        for start in range(0, int(block.shape[0]), batch_size):
            batch = block[start:start + batch_size]
            if drop_single_sample and batch.shape[0] == 1:
                continue
            yield batch.to(device=device, dtype=torch.float32, non_blocking=pin_memory)


def maybe_load_feature_cache_to_device(
    cache_info: FeatureCacheInfo,
    device: torch.device,
) -> torch.Tensor | None:
    if device.type != "cuda":
        return None

    cache_bytes = cache_info.num_rows * cache_info.feature_dim * 4
    free_bytes, _ = torch.cuda.mem_get_info(device)
    if cache_bytes > int(free_bytes * 0.7):
        return None

    cache_np = np.load(cache_info.path, mmap_mode="r+")
    cache_cpu = torch.from_numpy(cache_np)
    return cache_cpu.to(device=device, dtype=torch.float32)


def maybe_allocate_device_feature_cache(
    num_rows: int,
    feature_dim: int,
    device: torch.device,
) -> torch.Tensor | None:
    if device.type != "cuda":
        return None

    cache_bytes = num_rows * feature_dim * 4
    free_bytes, _ = torch.cuda.mem_get_info(device)
    if cache_bytes > int(free_bytes * 0.7):
        return None

    return torch.empty((num_rows, feature_dim), device=device, dtype=torch.float32)


def iterate_device_feature_cache_batches(
    cache_tensor: torch.Tensor,
    batch_size: int,
    shuffle: bool,
    seed: int,
    drop_single_sample: bool,
) -> Iterator[torch.Tensor]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    num_rows = int(cache_tensor.shape[0])
    if shuffle:
        generator = torch.Generator(device=cache_tensor.device)
        generator.manual_seed(seed)
        order = torch.randperm(num_rows, device=cache_tensor.device, generator=generator)
        for start in range(0, num_rows, batch_size):
            batch_indices = order[start:start + batch_size]
            if drop_single_sample and batch_indices.shape[0] == 1:
                continue
            yield cache_tensor.index_select(0, batch_indices)
        return

    for start in range(0, num_rows, batch_size):
        batch = cache_tensor[start:start + batch_size]
        if drop_single_sample and batch.shape[0] == 1:
            continue
        yield batch


def resolve_decoder_dims(decoder_dims: Sequence[int], output_dim: int) -> list[int]:
    resolved = list(decoder_dims)
    if not resolved or resolved[-1] != output_dim:
        resolved.append(output_dim)
    return resolved


def cuda_autocast_context(device: torch.device):
    if device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def train_autoencoder(
    feature_cache: FeatureCacheInfo,
    output_root: Path,
    encoder_dims: Sequence[int],
    decoder_dims: Sequence[int],
    lr: float,
    num_epochs: int,
    batch_size: int,
    eval_batch_size: int,
    cache_block_rows: int,
    eval_every: int,
    seed: int,
    device: torch.device,
    device_feature_cache: torch.Tensor | None = None,
) -> tuple[Autoencoder, list[dict], TrainSummary]:
    ckpt_dir = output_root / "clip_autoencoder_ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if eval_every <= 0:
        raise ValueError(f"eval_every must be positive, got {eval_every}")

    resolved_decoder_dims = resolve_decoder_dims(decoder_dims, feature_cache.feature_dim)
    model = Autoencoder(
        encoder_dims=encoder_dims,
        decoder_dims=resolved_decoder_dims,
        input_dim=feature_cache.feature_dim,
    ).to(device)
    if device_feature_cache is None:
        device_feature_cache = maybe_load_feature_cache_to_device(feature_cache, device=device)
    if device_feature_cache is not None:
        print(
            f"Loaded full feature cache on {device}: "
            f"({device_feature_cache.shape[0]}, {device_feature_cache.shape[1]})"
        )
    runtime_model: nn.Module = model
    if device.type == "cuda":
        runtime_model = torch.compile(model, mode="reduce-overhead")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, fused=device.type == "cuda")

    best_eval = float("inf")
    best_epoch = -1
    best_state_dict: dict[str, torch.Tensor] | None = None
    train_log: list[dict] = []

    for epoch in range(num_epochs):
        runtime_model.train()
        epoch_mse = 0.0
        epoch_seen = 0

        train_batches = (
            iterate_device_feature_cache_batches(
                device_feature_cache,
                batch_size=batch_size,
                shuffle=True,
                seed=seed + epoch,
                drop_single_sample=True,
            )
            if device_feature_cache is not None
            else iterate_feature_cache_batches(
                feature_cache,
                batch_size=batch_size,
                block_rows=cache_block_rows,
                shuffle=True,
                seed=seed + epoch,
                device=device,
                drop_single_sample=True,
            )
        )
        for batch in train_batches:
            with cuda_autocast_context(device):
                recon = runtime_model(batch)
                mse = mse_loss(recon, batch)
            optimizer.zero_grad(set_to_none=True)
            mse.backward()
            optimizer.step()

            bs = int(batch.shape[0])
            epoch_seen += bs
            epoch_mse += float(mse.item()) * bs

        if epoch_seen == 0:
            raise RuntimeError("No training batches were processed.")

        train_epoch_mse = epoch_mse / epoch_seen
        should_eval = ((epoch + 1) % eval_every == 0) or (epoch == num_epochs - 1)
        if should_eval:
            runtime_model.eval()
            eval_mse_sum = 0.0
            eval_seen = 0
            with torch.no_grad():
                eval_batches = (
                    iterate_device_feature_cache_batches(
                        device_feature_cache,
                        batch_size=eval_batch_size,
                        shuffle=False,
                        seed=seed,
                        drop_single_sample=False,
                    )
                    if device_feature_cache is not None
                    else iterate_feature_cache_batches(
                        feature_cache,
                        batch_size=eval_batch_size,
                        block_rows=cache_block_rows,
                        shuffle=False,
                        seed=seed,
                        device=device,
                        drop_single_sample=False,
                    )
                )
                for batch in eval_batches:
                    with cuda_autocast_context(device):
                        recon = runtime_model(batch)
                        mse = mse_loss(recon, batch)
                    bs = int(batch.shape[0])
                    eval_seen += bs
                    eval_mse_sum += float(mse.item()) * bs
            eval_epoch_mse = eval_mse_sum / eval_seen
        else:
            eval_epoch_mse = float("nan")

        log_row = {
            "epoch": epoch,
            "train_mse": train_epoch_mse,
            "eval_mse": eval_epoch_mse,
        }
        train_log.append(log_row)
        print(
            f"epoch={epoch:03d} train_mse={train_epoch_mse:.6f} eval_mse={eval_epoch_mse:.6f}"
        )

        if should_eval and eval_epoch_mse < best_eval:
            best_eval = eval_epoch_mse
            best_epoch = epoch
            best_state_dict = {
                name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()
            }

        if epoch % 10 == 0 or epoch == num_epochs - 1:
            torch.save(model.state_dict(), ckpt_dir / f"epoch_{epoch:03d}.pth")

    with open(ckpt_dir / "train_log.json", "w", encoding="utf-8") as f:
        json.dump(train_log, f, indent=2)

    with open(ckpt_dir / "best.txt", "w", encoding="utf-8") as f:
        f.write(json.dumps({"best_epoch": best_epoch, "best_eval": best_eval}, indent=2))

    if best_state_dict is None:
        raise RuntimeError("Failed to capture a best checkpoint during training.")
    torch.save(best_state_dict, ckpt_dir / "best_ckpt.pth")
    model.load_state_dict(best_state_dict)
    model.eval()
    return model, train_log, TrainSummary(best_epoch=best_epoch, best_eval_mse=best_eval)


@torch.inference_mode()
def export_lowdim_features(
    model: Autoencoder,
    file_infos: Sequence[FileInfo],
    batch_size: int,
    device: torch.device,
    device_feature_cache: torch.Tensor | None = None,
) -> None:
    for info in tqdm(file_infos, desc="Exporting 3-D latent maps"):
        latents: list[np.ndarray] = []
        if device_feature_cache is not None:
            flat = device_feature_cache[info.cache_offset:info.cache_offset + info.count]
            for start in range(0, int(flat.shape[0]), batch_size):
                chunk = flat[start:start + batch_size]
                with cuda_autocast_context(device):
                    z = model.encode(chunk)
                z = z.float().cpu().numpy()
                latents.append(z)
        else:
            arr = np.load(info.highdim_path).astype(np.float32)
            flat = arr.reshape(-1, arr.shape[-1])
            for start in range(0, flat.shape[0], batch_size):
                chunk = torch.from_numpy(flat[start:start + batch_size]).to(device)
                with cuda_autocast_context(device):
                    z = model.encode(chunk)
                z = z.float().cpu().numpy()
                latents.append(z)
        out = np.concatenate(latents, axis=0).reshape(info.grid_h, info.grid_w, -1)
        np.save(info.lowdim_path, out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal dense-CLIP LangSplat-style autoencoder trainer")
    parser.add_argument("--images-dir", type=Path, required=True, help="Directory containing scene images")
    parser.add_argument("--output-root", type=Path, default=None, help="Where outputs are written")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--clip-model-type", type=str, default="ViT-L-14-336-quickgelu")
    parser.add_argument("--clip-pretrained", type=str, default="openai")
    parser.add_argument(
        "--clip-load-size",
        type=int,
        default=1024,
        help="Resize shortest side like clip_main. Use 0 to disable; negative values scale the shortest side.",
    )
    parser.add_argument("--clip-center-crop", action="store_true", help="Center crop after resize like clip_main")
    parser.add_argument("--clip-padding-mode", type=str, default="constant", help="Padding mode for patch-grid alignment")
    parser.add_argument("--extract-batch-size", type=int, default=32, help="Max batch size for images with matching preprocessed shapes")
    parser.add_argument("--reextract", action="store_true", help="Recompute dense CLIP features even if cached")

    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3, help="README uses 7e-4; released train.py default is 1e-4")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--eval-every", type=int, default=2, help="Run full evaluation every N epochs, always including the final epoch.")
    parser.add_argument(
        "--cache-block-rows",
        type=int,
        default=65536,
        help="Rows to stream from the fp16 training cache at once before local shuffle/batching.",
    )
    parser.add_argument("--encoder-dims", nargs="+", type=int, default=[256, 128, 64, 32, 3])
    parser.add_argument(
        "--decoder-dims",
        nargs="+",
        type=int,
        default=[16, 32, 64, 128, 256, 256],
        help="Decoder hidden dims. The script appends the CLIP feature dimension automatically when needed.",
    )
    parser.add_argument("--viz-num-images", type=int, default=2, help="Number of random images to visualize in the final PCA comparison gallery")
    parser.add_argument("--viz-scale", type=int, default=4, help="Nearest-neighbor upsample factor for the saved PCA comparison gallery")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    images_dir = args.images_dir.resolve()
    output_root = resolve_output_root(images_dir, args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    configure_cuda_math_mode(device)
    clip_load_size = None if args.clip_load_size == 0 else args.clip_load_size
    total_start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    config = {
        "images_dir": str(images_dir),
        "output_root": str(output_root),
        "device": str(device),
        "seed": args.seed,
        "clip_model_type": args.clip_model_type,
        "clip_pretrained": args.clip_pretrained,
        "clip_load_size": clip_load_size,
        "clip_center_crop": args.clip_center_crop,
        "clip_padding_mode": args.clip_padding_mode,
        "extract_batch_size": args.extract_batch_size,
        "num_epochs": args.num_epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "eval_every": args.eval_every,
        "cache_block_rows": args.cache_block_rows,
        "encoder_dims": args.encoder_dims,
        "decoder_dims": args.decoder_dims,
        "viz_num_images": args.viz_num_images,
        "viz_scale": args.viz_scale,
    }
    ckpt_dir = output_root / "clip_autoencoder_ckpt"
    viz_dir = output_root / "clip_autoencoder_viz"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with open(ckpt_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    extract_start = time.perf_counter()
    extract_result = extract_dense_features(
        images_dir=images_dir,
        output_root=output_root,
        model_name=args.clip_model_type,
        pretrained=args.clip_pretrained,
        load_size=clip_load_size,
        center_crop=args.clip_center_crop,
        padding_mode=args.clip_padding_mode,
        batch_size=args.extract_batch_size,
        reextract=args.reextract,
        device=device,
    )
    extract_seconds = time.perf_counter() - extract_start
    file_infos = extract_result.file_infos
    device_feature_cache = extract_result.device_feature_cache

    cache_start = time.perf_counter()
    feature_cache = extract_result.feature_cache
    if feature_cache is None:
        feature_cache = build_feature_cache(file_infos, output_root=output_root)
    cache_seconds = time.perf_counter() - cache_start
    print(f"Prepared fp16 feature cache: ({feature_cache.num_rows}, {feature_cache.feature_dim})")

    sample_infos = choose_visualization_samples(file_infos, args.viz_num_images)
    save_selection_manifest(sample_infos, viz_dir / "sample_selection.json")

    config["feature_dim"] = feature_cache.feature_dim
    config["num_feature_vectors"] = feature_cache.num_rows
    config["feature_cache_path"] = str(feature_cache.path)
    config["feature_cache_dtype"] = feature_cache.dtype
    config["decoder_dims_resolved"] = resolve_decoder_dims(args.decoder_dims, feature_cache.feature_dim)
    config["viz_samples"] = [info.stem for info in sample_infos]
    with open(ckpt_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    train_start = time.perf_counter()
    model, train_log, train_summary = train_autoencoder(
        feature_cache=feature_cache,
        output_root=output_root,
        encoder_dims=args.encoder_dims,
        decoder_dims=args.decoder_dims,
        lr=args.lr,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        cache_block_rows=args.cache_block_rows,
        eval_every=args.eval_every,
        seed=args.seed,
        device=device,
        device_feature_cache=device_feature_cache,
    )
    train_seconds = time.perf_counter() - train_start

    export_start = time.perf_counter()
    export_lowdim_features(
        model=model,
        file_infos=file_infos,
        batch_size=args.eval_batch_size,
        device=device,
        device_feature_cache=device_feature_cache,
    )
    export_seconds = time.perf_counter() - export_start

    viz_start = time.perf_counter()
    save_final_pca_comparison(
        model=model,
        sample_infos=sample_infos,
        out_path=viz_dir / "pca_comparison.png",
        batch_size=args.eval_batch_size,
        device=device,
        scale=max(1, args.viz_scale),
    )
    viz_seconds = time.perf_counter() - viz_start

    total_seconds = time.perf_counter() - total_start
    peak_vram_mb = torch.cuda.max_memory_allocated(device) / 1024.0 / 1024.0 if device.type == "cuda" else 0.0
    final_train_mse = float(train_log[-1]["train_mse"]) if train_log else 0.0
    final_eval_mse = float(train_log[-1]["eval_mse"]) if train_log else 0.0
    summary = {
        "num_images": len(file_infos),
        "num_feature_vectors": feature_cache.num_rows,
        "feature_dim": feature_cache.feature_dim,
        "num_epochs": args.num_epochs,
        "best_epoch": train_summary.best_epoch,
        "best_eval_mse": train_summary.best_eval_mse,
        "final_train_mse": final_train_mse,
        "final_eval_mse": final_eval_mse,
        "extract_seconds": extract_seconds,
        "cache_seconds": cache_seconds,
        "train_seconds": train_seconds,
        "export_seconds": export_seconds,
        "viz_seconds": viz_seconds,
        "total_seconds": total_seconds,
        "peak_vram_mb": peak_vram_mb,
        "extract_vectors_per_second": safe_rate(feature_cache.num_rows, extract_seconds),
        "cache_vectors_per_second": safe_rate(feature_cache.num_rows, cache_seconds),
        "train_vectors_per_second": safe_rate(feature_cache.num_rows * args.num_epochs, train_seconds),
        "export_vectors_per_second": safe_rate(feature_cache.num_rows, export_seconds),
    }
    with open(ckpt_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("---")
    print(f"num_images:                 {summary['num_images']}")
    print(f"num_feature_vectors:        {summary['num_feature_vectors']}")
    print(f"feature_dim:                {summary['feature_dim']}")
    print(f"best_eval_mse:              {summary['best_eval_mse']:.6f}")
    print(f"extract_seconds:            {summary['extract_seconds']:.3f}")
    print(f"cache_seconds:              {summary['cache_seconds']:.3f}")
    print(f"train_seconds:              {summary['train_seconds']:.3f}")
    print(f"export_seconds:             {summary['export_seconds']:.3f}")
    print(f"viz_seconds:                {summary['viz_seconds']:.3f}")
    print(f"total_seconds:              {summary['total_seconds']:.3f}")
    print(f"peak_vram_mb:               {summary['peak_vram_mb']:.1f}")

    print(f"Done. Outputs written under: {output_root}")


if __name__ == "__main__":
    main()
