"""
predict.py — Multi-Scale Aspect-Ratio Preserving Prediction

The core problem with high-res images:
  - Model trained on 256×256 full scene images
  - Resizing 4000px → 224px destroys forensic signal (ELA, SRM, boundaries)
  - Patch cropping breaks scene context → enhancement bias

Correct solution:
  1. Aspect-ratio preserving resize to training resolution (256px)
     → preserves relative scale of forensic artifacts
  2. Center crop to 224×224 for model input
     → minimal information loss (256→224 = 12% trim, not 98%)
  3. Multi-scale: also predict at 0.75× and 1.25× of native res
     → catches artifacts that appear at different scales
  4. Weighted vote across scales

This is fundamentally different from patch cropping:
  PATCH: crops a fragment → loses scene context → enhancement bias
  THIS:  downscales whole image → preserves scene context → correct class
"""

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image, ImageOps
import numpy as np
from train import build_resnet50_9ch
from precompute_maps import compute_ela_multiscale, compute_noise_3ch

CLASS_NAMES = ["authentic", "copy_move", "enhancement",
               "removal_inpainting", "splicing"]

# ── Config ──────────────────────────────────────────────────────────────
IMAGE_PATH  = "test.jpg"
MODEL_PATH  = "outputs0/best_model.pth"
TEMPERATURE = 0.5

# Multi-scale prediction
# Scales relative to training resolution (256px)
# Each scale resizes the image so its shorter side = 256 * scale
SCALES      = [0.75, 1.0, 1.5, 2.0]   # predict at 4 different downscale levels
SCALE_WEIGHTS = [0.15, 0.40, 0.30, 0.15]  # weight each scale's prediction
                                            # 1.0 (= 256px) gets highest weight
                                            # as it matches training distribution

# TTA flips per scale
TTA_FLIPS   = True
# ────────────────────────────────────────────────────────────────────────


def load_model(model_path, device):
    model = build_resnet50_9ch(5, pretrained=False).to(device)
    state = torch.load(model_path, map_location=device)
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v
                 for k, v in state.items()
                 if k != "n_averaged"}
    model.load_state_dict(state)
    model.eval()
    return model


def aspect_preserving_resize(pil_img, target_short_side):
    """
    Resize image so shorter side = target_short_side,
    preserving aspect ratio.
    This is the CORRECT way to resize forensic images —
    it maintains the relative pixel density of artifacts.
    """
    W, H        = pil_img.size
    short_side  = min(W, H)
    scale       = target_short_side / short_side
    new_W       = int(round(W * scale))
    new_H       = int(round(H * scale))
    return pil_img.resize((new_W, new_H), Image.LANCZOS)


def center_crop_224(pil_img):
    """Center crop to 224×224 for model input."""
    W, H = pil_img.size
    left  = (W - 224) // 2
    top   = (H - 224) // 2
    return pil_img.crop((left, top, left + 224, top + 224))


def pad_to_224(pil_img):
    """
    If image is smaller than 224, pad with edge pixels (not zeros).
    Zero padding looks like a black border — very unusual in natural images
    and confuses the model. Edge padding is more natural.
    """
    W, H    = pil_img.size
    if W >= 224 and H >= 224:
        return pil_img
    new_W   = max(W, 224)
    new_H   = max(H, 224)
    padded  = ImageOps.expand(pil_img,
                               border=((new_W-W)//2, (new_H-H)//2,
                                       (new_W-W+1)//2, (new_H-H+1)//2),
                               fill=0)
    return padded.resize((224, 224), Image.LANCZOS)


def prepare_input(pil_img, normalize_9, device, target_short_side=256):
    """
    Full preprocessing pipeline for one scale:
    1. Aspect-preserving resize to target_short_side
    2. Compute ELA + noise at THAT resolution (key: before center crop)
    3. Center crop / pad to 224×224
    4. Stack to 9-channel tensor
    """
    # Step 1: resize whole image to target scale
    resized = aspect_preserving_resize(pil_img, target_short_side)

    # Step 2: compute forensic maps at resized resolution
    #         (NOT at 224 — forensic maps degrade with resizing)
    ela     = compute_ela_multiscale(resized)
    noise   = compute_noise_3ch(resized)

    # Step 3: center crop to 224 (or pad if smaller)
    W, H    = resized.size
    if W >= 224 and H >= 224:
        rgb_crop   = center_crop_224(resized)
        ela_crop   = center_crop_224(ela)
        noise_crop = center_crop_224(noise)
    else:
        rgb_crop   = pad_to_224(resized)
        ela_crop   = pad_to_224(ela)
        noise_crop = pad_to_224(noise)

    # Step 4: to tensor
    rgb_t   = TF.to_tensor(rgb_crop)    # (3,224,224)
    ela_t   = TF.to_tensor(ela_crop)    # (3,224,224)
    noise_t = TF.to_tensor(noise_crop)  # (3,224,224)

    x = normalize_9(torch.cat([rgb_t, ela_t, noise_t], dim=0))
    return x.unsqueeze(0).to(device)


def predict_single(model, x, temperature, use_tta=True):
    """Predict on one input tensor, with optional TTA."""
    if use_tta:
        flips = [
            x,
            torch.flip(x, dims=[3]),
            torch.flip(x, dims=[2]),
            torch.flip(x, dims=[2, 3]),
        ]
    else:
        flips = [x]

    with torch.no_grad():
        logits_list = [model(f) for f in flips]
        avg_logits  = torch.stack(logits_list).mean(0)
        probs       = torch.softmax(avg_logits / temperature, dim=1)[0]
    return probs.cpu().numpy()


def predict(image_path, model_path,
            temperature=0.5,
            scales=None,
            scale_weights=None,
            tta_flips=True):

    if scales is None:
        scales = SCALES
    if scale_weights is None:
        scale_weights = SCALE_WEIGHTS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device      : {device}")
    print(f"Model       : {model_path}")
    print(f"Temperature : {temperature}")

    model       = load_model(model_path, device)
    normalize_9 = transforms.Normalize(
        mean=[0.485, 0.456, 0.406, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        std =[0.229, 0.224, 0.225, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    )

    pil_orig = Image.open(image_path).convert("RGB")
    W, H     = pil_orig.size
    print(f"Image size  : {W}×{H}  (shorter side = {min(W,H)}px)")
    print(f"Scales      : {scales}  (×256px shorter side)\n")

    # ── Multi-scale prediction ────────────────────────────────────────
    all_probs   = []
    all_weights = []

    for scale, weight in zip(scales, scale_weights):
        target_px = int(256 * scale)
        x         = prepare_input(pil_orig, normalize_9, device,
                                   target_short_side=target_px)

        probs     = predict_single(model, x, temperature, use_tta=tta_flips)
        pred_c    = CLASS_NAMES[probs.argmax()]
        conf_c    = probs.max()

        print(f"  Scale {scale:.2f}× ({target_px:4d}px short side) → "
              f"{pred_c:22s}  {conf_c:.2%}  (weight={weight})")

        all_probs.append(probs)
        all_weights.append(weight)

    # ── Weighted fusion across scales ─────────────────────────────────
    weights_arr = np.array(all_weights) / sum(all_weights)  # normalize
    probs_arr   = np.stack(all_probs, axis=0)               # (n_scales, 5)
    final_probs = (probs_arr * weights_arr[:, None]).sum(axis=0)  # (5,)

    pred_name = CLASS_NAMES[final_probs.argmax()]
    conf      = final_probs.max()

    # ── Display ───────────────────────────────────────────────────────
    print(f"\nImage  : {image_path}")
    print("=" * 54)
    for name, p in zip(CLASS_NAMES, final_probs):
        bar    = "█" * int(p * 40)
        marker = " ◄" if name == pred_name else ""
        print(f"  {name:22s}: {p:.4f}  {bar}{marker}")
    print("=" * 54)
    print(f"  Prediction : {pred_name}")
    print(f"  Confidence : {conf:.2%}")
    print(f"  Mode       : multi-scale ({len(scales)} scales, weighted vote)\n")

    return pred_name, conf, final_probs


if __name__ == "__main__":
    predict(
        image_path    = IMAGE_PATH,
        model_path    = MODEL_PATH,
        temperature   = TEMPERATURE,
        scales        = SCALES,
        scale_weights = SCALE_WEIGHTS,
        tta_flips     = TTA_FLIPS,
    )
