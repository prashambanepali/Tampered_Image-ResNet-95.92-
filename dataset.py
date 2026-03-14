"""
dataset.py — 9-Channel version
Reads images from features/ using index files written by split_dataset.py.

Channel layout (9 total):
  0–2   RGB             original image
  3–5   ELA             multi-scale Error Level Analysis (RGB)
  6–8   Noise (3ch)     R=gaussian_residual | G=srm_linear_residual | B=srm_edge_residual

features/
  <class>/
    rgb/    original images
    ela/    ELA maps (.png, RGB 3ch)
    noise/  noise maps (.png, RGB 3ch)

dataset/
  train/<class>/index.txt
  val/<class>/index.txt
  test/<class>/index.txt
"""

import os
import random
import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import io

import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFilter, ImageEnhance
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import numpy as np

IMG_EXTS = (".jpg", ".jpeg", ".png")

FEATURES_ROOT = "Features1"
DATASET_ROOT  = "dataset"

CLASS_NAMES = [
    "authentic",
    "copy_move",
    "enhancement",
    "removal_inpainting",
    "splicing",
]

CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}


# ─────────────────────────────────────────────
# PairedAugConfig
# ─────────────────────────────────────────────
@dataclass
class PairedAugConfig:
    enable: bool = True

    # Mild crop — preserves forensic boundary signal
    crop_scale:  Tuple[float, float] = (0.90, 1.00)
    crop_ratio:  Tuple[float, float] = (0.95, 1.05)

    hflip_p:     float = 0.5
    vflip_p:     float = 0.20
    rotate_deg:  float = 15.0
    rotate_90_p: float = 0.40

    # Color jitter — RGB only
    color_jitter_p:    float = 0.70
    jitter_brightness: float = 0.20
    jitter_contrast:   float = 0.20
    jitter_saturation: float = 0.15
    jitter_hue:        float = 0.04

    # JPEG domain randomization — KEY anti-bias technique
    jpeg_p:    float = 0.60
    jpeg_qmin: int   = 55
    jpeg_qmax: int   = 98

    # Double JPEG compression simulation
    double_jpeg_p: float = 0.30

    # Gaussian noise injection
    gaussian_noise_p:   float = 0.40
    gaussian_noise_std: float = 0.02

    # Gaussian blur
    blur_p:      float = 0.25
    blur_radius: Tuple[float, float] = (0.3, 1.5)

    # Sharpening
    sharpen_p: float = 0.20

    # Random erasing
    erasing_p: float = 0.25

    # Grid shuffle
    grid_shuffle_p: float = 0.20


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def _random_resized_crop_params(img, scale, ratio):
    w, h = img.size
    area = h * w
    for _ in range(10):
        ta  = area * random.uniform(scale[0], scale[1])
        lr  = (math.log(ratio[0]), math.log(ratio[1]))
        asp = math.exp(random.uniform(lr[0], lr[1]))
        nw  = int(round((ta * asp) ** 0.5))
        nh  = int(round((ta / asp) ** 0.5))
        if 0 < nw <= w and 0 < nh <= h:
            return random.randint(0, h - nh), random.randint(0, w - nw), nh, nw
    in_r = w / h
    if in_r < ratio[0]:
        nw, nh = w, int(round(w / ratio[0]))
    elif in_r > ratio[1]:
        nh, nw = h, int(round(h * ratio[1]))
    else:
        nw, nh = w, h
    return (h - nh) // 2, (w - nw) // 2, nh, nw


def _jpeg_recompress(img: Image.Image, q: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _double_jpeg(img: Image.Image,
                 q1_range=(65, 95), q2_range=(55, 90)) -> Image.Image:
    """Simulate double JPEG compression — common on images shared online."""
    return _jpeg_recompress(_jpeg_recompress(img, random.randint(*q1_range)),
                            random.randint(*q2_range))


def _random_erase_tensor(t: torch.Tensor, p: float) -> torch.Tensor:
    if random.random() >= p:
        return t
    _, H, W = t.shape
    for _ in range(10):
        ea = H * W * random.uniform(0.02, 0.12)
        ar = random.uniform(0.3, 3.3)
        eh = int(round(math.sqrt(ea / ar)))
        ew = int(round(math.sqrt(ea * ar)))
        if 0 < eh < H and 0 < ew < W:
            i = random.randint(0, H - eh)
            j = random.randint(0, W - ew)
            t[:, i:i+eh, j:j+ew] = torch.randn_like(t[:, i:i+eh, j:j+ew]) * 0.1
            break
    return t


def _add_gaussian_noise(t: torch.Tensor, std: float) -> torch.Tensor:
    return torch.clamp(t + torch.randn_like(t) * std, 0.0, 1.0)


def _grid_shuffle(img: Image.Image, grid: int = 4) -> Image.Image:
    w, h   = img.size
    cw, ch = w // grid, h // grid
    cells  = [img.crop((c*cw, r*ch, (c+1)*cw, (r+1)*ch))
              for r in range(grid) for c in range(grid)]
    random.shuffle(cells)
    out = img.copy()
    for idx, (r, c) in enumerate((r, c) for r in range(grid) for c in range(grid)):
        out.paste(cells[idx], (c*cw, r*ch))
    return out


# ─────────────────────────────────────────────
# PairedAugment  — applies same spatial ops to RGB, ELA, Noise
# ─────────────────────────────────────────────
class PairedAugment:
    """
    Applies identical spatial transforms to all three modalities
    (RGB, ELA, Noise) so their pixel alignment is preserved.

    Noise is now 3-channel RGB (gaussian|srm_linear|srm_edge),
    so it is treated exactly like ELA — resize/crop/flip via BILINEAR.
    Color/JPEG/blur transforms are applied only to RGB.
    """
    def __init__(self, cfg: PairedAugConfig):
        self.cfg = cfg

    def __call__(self, rgb: Image.Image, ela: Image.Image, noise: Image.Image,
                 out_size=(224, 224)):
        if not self.cfg.enable:
            rgb   = TF.resize(rgb,   out_size, interpolation=InterpolationMode.BILINEAR)
            ela   = TF.resize(ela,   out_size, interpolation=InterpolationMode.BILINEAR)
            noise = TF.resize(noise, out_size, interpolation=InterpolationMode.BILINEAR)
            return rgb, ela, noise

        # ── Mild RandomResizedCrop (shared) ──────────────────────────
        i, j, h, w = _random_resized_crop_params(rgb, self.cfg.crop_scale, self.cfg.crop_ratio)
        rgb   = TF.resized_crop(rgb,   i, j, h, w, out_size, InterpolationMode.BILINEAR)
        ela   = TF.resized_crop(ela,   i, j, h, w, out_size, InterpolationMode.BILINEAR)
        noise = TF.resized_crop(noise, i, j, h, w, out_size, InterpolationMode.BILINEAR)

        # ── Flips (shared) ───────────────────────────────────────────
        if random.random() < self.cfg.hflip_p:
            rgb = TF.hflip(rgb); ela = TF.hflip(ela); noise = TF.hflip(noise)

        if random.random() < self.cfg.vflip_p:
            rgb = TF.vflip(rgb); ela = TF.vflip(ela); noise = TF.vflip(noise)

        # ── 90° rotation (shared) ────────────────────────────────────
        if random.random() < self.cfg.rotate_90_p:
            k = random.choice([1, 2, 3])
            rgb   = rgb.rotate(k * 90)
            ela   = ela.rotate(k * 90)
            noise = noise.rotate(k * 90)

        # ── Small-angle rotation (shared) ────────────────────────────
        angle = random.uniform(-self.cfg.rotate_deg, self.cfg.rotate_deg)
        if abs(angle) > 0.5:
            rgb   = TF.rotate(rgb,   angle, InterpolationMode.BILINEAR, fill=0)
            ela   = TF.rotate(ela,   angle, InterpolationMode.BILINEAR, fill=0)
            noise = TF.rotate(noise, angle, InterpolationMode.BILINEAR, fill=0)

        # ── Grid shuffle (shared — spatial consistency) ───────────────
        if random.random() < self.cfg.grid_shuffle_p:
            rgb   = _grid_shuffle(rgb)
            ela   = _grid_shuffle(ela)
            noise = _grid_shuffle(noise)

        # ── RGB-only: double JPEG compression ────────────────────────
        if random.random() < self.cfg.double_jpeg_p:
            rgb = _double_jpeg(rgb)
        elif random.random() < self.cfg.jpeg_p:
            rgb = _jpeg_recompress(rgb, random.randint(self.cfg.jpeg_qmin, self.cfg.jpeg_qmax))

        # ── RGB-only: color jitter ────────────────────────────────────
        if random.random() < self.cfg.color_jitter_p:
            b  = random.uniform(1 - self.cfg.jitter_brightness, 1 + self.cfg.jitter_brightness)
            c  = random.uniform(1 - self.cfg.jitter_contrast,   1 + self.cfg.jitter_contrast)
            s  = random.uniform(1 - self.cfg.jitter_saturation, 1 + self.cfg.jitter_saturation)
            h_ = random.uniform(-self.cfg.jitter_hue,           self.cfg.jitter_hue)
            rgb = TF.adjust_brightness(rgb, b)
            rgb = TF.adjust_contrast(rgb,   c)
            rgb = TF.adjust_saturation(rgb, s)
            rgb = TF.adjust_hue(rgb,        h_)

        # ── RGB-only: Gaussian blur ───────────────────────────────────
        if random.random() < self.cfg.blur_p:
            rgb = rgb.filter(ImageFilter.GaussianBlur(
                radius=random.uniform(*self.cfg.blur_radius)))

        # ── RGB-only: sharpening ──────────────────────────────────────
        if random.random() < self.cfg.sharpen_p:
            rgb = ImageEnhance.Sharpness(rgb).enhance(random.uniform(1.2, 2.5))

        return rgb, ela, noise


# ─────────────────────────────────────────────
# MixUp / CutMix  (called from train loop)
# ─────────────────────────────────────────────
def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.4):
    lam   = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    index = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[index], y, y[index], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def cutmix_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0):
    lam            = np.random.beta(alpha, alpha)
    B, _, H, W     = x.shape
    index          = torch.randperm(B, device=x.device)
    cut_rat        = math.sqrt(1.0 - lam)
    cut_w, cut_h   = int(W * cut_rat), int(H * cut_rat)
    cx, cy         = random.randint(0, W), random.randint(0, H)
    x1, y1         = max(cx - cut_w // 2, 0), max(cy - cut_h // 2, 0)
    x2, y2         = min(cx + cut_w // 2, W), min(cy + cut_h // 2, H)
    mixed          = x.clone()
    mixed[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    lam            = 1 - (x2 - x1) * (y2 - y1) / (W * H)
    return mixed, y, y[index], lam


# ─────────────────────────────────────────────
# TamperedDataset
# ─────────────────────────────────────────────
class TamperedDataset(Dataset):
    """
    Loads (rgb, ela, noise) triplets and returns:
      x : (9, H, W)  — RGB(3) + ELA(3) + Noise(3)
      y : int  label 0–4

    Noise is now a 3-channel RGB PNG storing:
      R = gaussian_residual
      G = srm_linear_residual
      B = srm_edge_residual
    It is loaded with .convert("RGB") and contributes 3 channels.
    """

    def __init__(
        self,
        split: str,
        image_size: tuple = (224, 224),
        normalize_transform=None,
        paired_augment: Optional[PairedAugment] = None,
        dataset_root: str = DATASET_ROOT,
        features_root: str = FEATURES_ROOT,
    ):
        self.image_size          = image_size
        self.normalize_transform = normalize_transform
        self.paired_augment      = paired_augment
        self.features_root       = features_root

        self.items: List[Tuple[str, str, str, int]] = []

        for class_name in CLASS_NAMES:
            label      = CLASS_TO_IDX[class_name]
            index_path = os.path.join(dataset_root, split, class_name, "index.txt")

            if not os.path.exists(index_path):
                raise FileNotFoundError(
                    f"Index file not found: {index_path}\n"
                    f"Run split_dataset.py first."
                )

            with open(index_path) as f:
                filenames = [l.strip() for l in f.read().splitlines() if l.strip()]

            rgb_dir   = os.path.join(features_root, class_name, "rgb")
            ela_dir   = os.path.join(features_root, class_name, "ela")
            noise_dir = os.path.join(features_root, class_name, "noise")

            for fname in filenames:
                base       = os.path.splitext(fname)[0]
                rgb_path   = os.path.join(rgb_dir,   fname)
                ela_path   = os.path.join(ela_dir,   base + ".png")
                noise_path = os.path.join(noise_dir, base + ".png")
                self.items.append((rgb_path, ela_path, noise_path, label))

        if not self.items:
            raise RuntimeError(f"No items loaded for split='{split}'")

        from collections import Counter
        counts = Counter(lbl for (_, _, _, lbl) in self.items)
        print(f"[Dataset:{split}]  total={len(self.items)}  channels=9 (RGB+ELA+Noise3ch)")
        for i, name in enumerate(CLASS_NAMES):
            print(f"  {name:20s}: {counts.get(i, 0)}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        rgb_path, ela_path, noise_path, label = self.items[idx]

        rgb = Image.open(rgb_path).convert("RGB")

        # ELA: RGB 3-channel PNG
        ela = (Image.open(ela_path).convert("RGB")
               if os.path.exists(ela_path)
               else Image.new("RGB", rgb.size, 0))

        # Noise: 3-channel RGB PNG (R=gaussian, G=srm_linear, B=srm_edge)
        # Load as RGB so all 3 channels are preserved
        noise = (Image.open(noise_path).convert("RGB")
                 if os.path.exists(noise_path)
                 else Image.new("RGB", rgb.size, 0))

        # ── Spatial augmentation (identical for all 3 modalities) ────
        if self.paired_augment is not None:
            rgb, ela, noise = self.paired_augment(rgb, ela, noise,
                                                   out_size=self.image_size)
        else:
            rgb   = TF.resize(rgb,   self.image_size, InterpolationMode.BILINEAR)
            ela   = TF.resize(ela,   self.image_size, InterpolationMode.BILINEAR)
            noise = TF.resize(noise, self.image_size, InterpolationMode.BILINEAR)

        # ── To tensor ────────────────────────────────────────────────
        rgb_t   = TF.to_tensor(rgb)    # (3, H, W)
        ela_t   = TF.to_tensor(ela)    # (3, H, W)
        noise_t = TF.to_tensor(noise)  # (3, H, W)  ← was (1,H,W), now 3ch

        # ── RGB-only tensor augmentations ────────────────────────────
        if self.paired_augment is not None:
            cfg = self.paired_augment.cfg
            if cfg.gaussian_noise_p > 0 and random.random() < cfg.gaussian_noise_p:
                rgb_t = _add_gaussian_noise(rgb_t, cfg.gaussian_noise_std)
            if cfg.erasing_p > 0:
                rgb_t = _random_erase_tensor(rgb_t, cfg.erasing_p)

        # ── Concatenate: RGB(3) + ELA(3) + Noise(3) = 9 channels ─────
        x = torch.cat([rgb_t, ela_t, noise_t], dim=0)  # (9, H, W)

        if self.normalize_transform is not None:
            x = self.normalize_transform(x)

        return x, label
