"""
precompute_maps.py — 3-Channel Noise version
Reads from raw/ and generates ELA + 3-channel Noise maps into features/

Final tensor channels: RGB(3) + ELA(3) + Noise(3) = 9 channels total

Noise 3 channels (saved as a single RGB PNG):
  Ch-0  gaussian_residual   — Gaussian high-pass (sigma=1.5)
                              Detects smooth inpainted regions vs. textured originals
  Ch-1  srm_linear_residual — SRM 5x5 linear prediction residual
                              Captures pixel-dependency violations from copy-move/splicing
  Ch-2  srm_edge_residual   — SRM 5x5 edge-sensitive residual
                              Highlights splice boundaries and sharpness inconsistencies

Output structure:
  features/
    <class>/
      rgb/    *.jpg   — original image copy
      ela/    *.png   — multi-scale ELA (RGB 3-ch PNG)
      noise/  *.png   — 3-channel noise map (RGB PNG: R=gaussian, G=srm_linear, B=srm_edge)
"""

import os
import io
import cv2
import shutil
import numpy as np
from PIL import Image, ImageChops
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# ── Config ────────────────────────────────────────────────────────────
SRC_ROOT    = "raw"
DST_ROOT    = "Features1"
OVERWRITE   = False
NUM_WORKERS = 10
IMG_EXTS    = (".jpg", ".jpeg", ".png", ".tif", ".bmp")

# Multi-scale ELA quality levels
ELA_QUALITIES = (60, 75, 85, 95)

CLASSES = [
    "authentic",
    "copy_move",
    "enhancement",
    "removal_inpainting",
    "splicing",
]
# ─────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────
# SRM Filter Kernels  (Fridrich & Kodovsky, 2012)
# Zero-sum 5x5 high-pass filters used in image steganalysis/forensics.
# ─────────────────────────────────────────────

# Ch-1: SRM Linear residual
# Models pixel as linear combination of neighbors; residual = deviation.
# Catches noise-pattern mismatch introduced by copy-move or splicing.
SRM_LINEAR_KERNEL = np.array([
    [ 0,  0, -1,  0,  0],
    [ 0, -1,  2, -1,  0],
    [-1,  2,  0,  2, -1],   # center=0 → zero-sum by construction
    [ 0, -1,  2, -1,  0],
    [ 0,  0, -1,  0,  0],
], dtype=np.float32) / 8.0

# Ch-2: SRM Edge residual
# Ring-shaped kernel — strongly activates at region boundaries.
# Exposes splice edges and inconsistent sharpness from enhancement/removal.
SRM_EDGE_KERNEL = np.array([
    [-1, -1, -1, -1, -1],
    [-1,  2,  2,  2, -1],
    [-1,  2,  8,  2, -1],
    [-1,  2,  2,  2, -1],
    [-1, -1, -1, -1, -1],
], dtype=np.float32)
# Force zero-sum: adjust center so filter sum = 0
_edge_sum_no_center = SRM_EDGE_KERNEL.sum() - SRM_EDGE_KERNEL[2, 2]
SRM_EDGE_KERNEL[2, 2] = -_edge_sum_no_center
SRM_EDGE_KERNEL /= 8.0


def _percentile_norm(arr: np.ndarray) -> np.ndarray:
    """Clip to p2–p98 and scale to [0, 255] uint8. Stable across domains."""
    p2, p98 = np.percentile(arr, [2, 98])
    denom   = max(float(p98 - p2), 1e-6)
    return np.clip((arr - p2) / denom * 255.0, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────
# ELA — multi-scale, percentile normalized
# ─────────────────────────────────────────────
def _ela_single(img_rgb: Image.Image, quality: int) -> np.ndarray:
    buf = io.BytesIO()
    img_rgb.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    recomp = Image.open(buf).convert("RGB")
    diff   = ImageChops.difference(img_rgb, recomp)
    arr    = np.array(diff, dtype=np.float32)
    p2, p98 = np.percentile(arr, [2, 98])
    denom   = max(float(p98 - p2), 1e-6)
    return np.clip((arr - p2) / denom * 255.0, 0, 255)


def compute_ela_multiscale(img: Image.Image) -> Image.Image:
    """
    Average ELA across quality levels (60, 75, 85, 95).
    Returns PIL RGB image — 3 channels.
    """
    img_rgb = img.convert("RGB")
    maps    = [_ela_single(img_rgb, q) for q in ELA_QUALITIES]
    avg     = np.clip(np.mean(maps, axis=0), 0, 255).astype(np.uint8)
    return Image.fromarray(avg, mode="RGB")


# ─────────────────────────────────────────────
# 3-Channel Noise Map
# ─────────────────────────────────────────────
def compute_noise_3ch(img: Image.Image) -> Image.Image:
    """
    Computes 3 forensic residual channels and packs them into one RGB PNG.

      R = gaussian_residual
          img_gray - GaussianBlur(sigma=1.5)
          General noise inconsistency detector. Smooth inpainted patches
          have near-zero residual; textured originals have high residual.

      G = srm_linear_residual
          SRM 5x5 linear prediction filter applied to grayscale.
          Detects copy-move/splicing by exposing pixel statistical anomalies
          when the noise source of a pasted region differs from the host.

      B = srm_edge_residual
          SRM 5x5 ring/edge filter applied to grayscale.
          Strongly activates at region boundaries — exposes splice seams,
          enhancement halos, and removal boundary artifacts.

    Each channel is independently percentile-normalized to [0, 255].
    Stored as a single 3-channel RGB PNG (R=Ch0, G=Ch1, B=Ch2).
    In dataset.py: loaded with .convert("RGB") → TF.to_tensor() → (3,H,W).
    """
    gray = np.array(img.convert("L"), dtype=np.float32) / 255.0  # (H,W) in [0,1]

    # ── Ch-0: Gaussian residual ──────────────────────────────────────
    blurred          = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.5)
    gaussian_residual = gray - blurred                             # (H,W)

    # ── Ch-1: SRM linear residual ────────────────────────────────────
    srm_linear_residual = cv2.filter2D(
        gray, ddepth=-1,
        kernel=SRM_LINEAR_KERNEL,
        borderType=cv2.BORDER_REFLECT
    )                                                              # (H,W)

    # ── Ch-2: SRM edge residual ──────────────────────────────────────
    srm_edge_residual = cv2.filter2D(
        gray, ddepth=-1,
        kernel=SRM_EDGE_KERNEL,
        borderType=cv2.BORDER_REFLECT
    )                                                              # (H,W)

    # ── Normalize each channel independently ────────────────────────
    ch_r = _percentile_norm(gaussian_residual)     # R
    ch_g = _percentile_norm(srm_linear_residual)   # G
    ch_b = _percentile_norm(srm_edge_residual)     # B

    # ── Pack into RGB PNG ────────────────────────────────────────────
    noise_rgb = np.stack([ch_r, ch_g, ch_b], axis=2)             # (H,W,3)
    return Image.fromarray(noise_rgb, mode="RGB")


# ─────────────────────────────────────────────
# Worker
# ─────────────────────────────────────────────
def worker(job: tuple) -> int:
    """
    job = (src_path, rgb_out, ela_out, noise_out)
    Writes: rgb copy, ela RGB PNG, noise 3-ch RGB PNG.
    """
    src_path, rgb_out, ela_out, noise_out = job

    img = Image.open(src_path).convert("RGB")

    if not os.path.exists(rgb_out):
        shutil.copy2(src_path, rgb_out)

    if not os.path.exists(ela_out):
        compute_ela_multiscale(img).save(ela_out)

    if not os.path.exists(noise_out):
        compute_noise_3ch(img).save(noise_out)

    return 1


# ─────────────────────────────────────────────
# Job collection
# ─────────────────────────────────────────────
def collect_jobs(class_name: str) -> list:
    src_folder   = os.path.join(SRC_ROOT,  class_name)
    rgb_folder   = os.path.join(DST_ROOT,  class_name, "rgb")
    ela_folder   = os.path.join(DST_ROOT,  class_name, "ela")
    noise_folder = os.path.join(DST_ROOT,  class_name, "noise")

    os.makedirs(rgb_folder,   exist_ok=True)
    os.makedirs(ela_folder,   exist_ok=True)
    os.makedirs(noise_folder, exist_ok=True)

    jobs = []
    for fname in sorted(os.listdir(src_folder)):
        if not fname.lower().endswith(IMG_EXTS):
            continue

        base      = os.path.splitext(fname)[0]
        src_path  = os.path.join(src_folder,   fname)
        rgb_out   = os.path.join(rgb_folder,   fname)
        ela_out   = os.path.join(ela_folder,   base + ".png")
        noise_out = os.path.join(noise_folder, base + ".png")

        if (not OVERWRITE) and all(os.path.exists(p) for p in [rgb_out, ela_out, noise_out]):
            continue

        jobs.append((src_path, rgb_out, ela_out, noise_out))

    return jobs


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    print(f"Source  : {os.path.abspath(SRC_ROOT)}")
    print(f"Output  : {os.path.abspath(DST_ROOT)}")
    print(f"Workers : {NUM_WORKERS}")
    print(f"ELA     : multi-scale q={ELA_QUALITIES}  (percentile norm, RGB 3ch)")
    print(f"Noise   : 3-channel RGB PNG")
    print(f"          R = gaussian_residual   (sigma=1.5 Gaussian high-pass)")
    print(f"          G = srm_linear_residual (5x5 SRM linear kernel)")
    print(f"          B = srm_edge_residual   (5x5 SRM ring/edge kernel)")
    print(f"Input channels total: RGB(3) + ELA(3) + Noise(3) = 9")

    for class_name in CLASSES:
        src_folder = os.path.join(SRC_ROOT, class_name)
        if not os.path.isdir(src_folder):
            print(f"\n[SKIP] {src_folder} not found")
            continue

        jobs = collect_jobs(class_name)
        if not jobs:
            print(f"\n[SKIP] {class_name} — all files already generated")
            continue

        print(f"\nProcessing: {class_name}  ({len(jobs)} images)")

        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as ex:
            futures = [ex.submit(worker, j) for j in jobs]
            for _ in tqdm(as_completed(futures), total=len(futures),
                          unit="img", dynamic_ncols=True):
                pass

    print("\n─── Features Summary ───")
    for class_name in CLASSES:
        rgb_dir = os.path.join(DST_ROOT, class_name, "rgb")
        ela_dir = os.path.join(DST_ROOT, class_name, "ela")
        noi_dir = os.path.join(DST_ROOT, class_name, "noise")
        if os.path.isdir(rgb_dir):
            n_rgb = len(os.listdir(rgb_dir))
            n_ela = len(os.listdir(ela_dir))
            n_noi = len(os.listdir(noi_dir))
            status = "✓" if n_rgb == n_ela == n_noi else "⚠ MISMATCH"
            print(f"  {class_name:20s}  rgb={n_rgb}  ela={n_ela}  noise={n_noi}  {status}")

    print(f"\nDone. Features saved to: {os.path.abspath(DST_ROOT)}/")
    print("Noise maps: 3-channel RGB PNG  (R=gaussian | G=srm_linear | B=srm_edge)")
    print("\nNext step: run  python split_dataset.py")


if __name__ == "__main__":
    main()
