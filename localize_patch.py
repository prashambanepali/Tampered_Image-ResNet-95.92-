"""
localize_advanced.py
Advanced multi-layer GradCAM++ + multi-layer Score-CAM forensic localization
for a 9-channel tampering classifier.

Improvements over the previous version:
- Uses multiple convolution layers: layer2, layer3, layer4
- Fuses semantic + fine-detail CAMs
- Uses overlapping tiled inference over the full image
- Multi-scale classification and localization
- Uses ELA + SRM prior gating to suppress false positives
- Adaptive mask generation for sharper suspected-region localization
"""

import os
import io
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from train import build_resnet50_9ch
from precompute_maps import compute_ela_multiscale, compute_noise_3ch


# =========================================================
# CONFIG
# =========================================================
CLASS_NAMES = [
    "authentic",
    "copy_move",
    "enhancement",
    "removal_inpainting",
    "splicing",
]

IMAGE_PATH = r"C:\Tampered_9channel\Unseen_coco_images\COCO_DF_S000B00000_00024776.jpg"
MODEL_PATH = r"outputs0\swa_model.pth"
OUTPUT_DIR = "localization_output"

TEMPERATURE = 0.65

# multi-scale analysis
SCALES = [1.0, 1.25, 1.50]
SCALE_WEIGHTS = [0.45, 0.30, 0.25]

# full-image tiled localization
BASE_SHORT_SIDE = 256
TILE_SIZE = 224
TILE_STRIDE = 112

# score-cam
SCORECAM_TOP_K = 16

# priors
ELA_BLOCK_SIZE = 8
SRM_PATCH = 32
SRM_STRIDE = 8

# mask / decision tuning
AUTHENTIC_CONF_STRONG = 0.72
MIN_TOP1_TOP2_MARGIN = 0.10
MIN_SUSPICIOUS_AREA_RATIO = 0.0008


# =========================================================
# MODEL
# =========================================================
def load_model(model_path, device):
    model = build_resnet50_9ch(5, pretrained=False).to(device)

    state = torch.load(model_path, map_location=device)

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    if any(k.startswith("module.") for k in state.keys()):
        state = {
            k.replace("module.", "", 1): v
            for k, v in state.items()
            if k != "n_averaged"
        }

    model.load_state_dict(state, strict=True)
    model.eval()
    return model


# =========================================================
# IMAGE HELPERS
# =========================================================
def aspect_preserving_resize(pil_img, target_short_side):
    w, h = pil_img.size
    scale = target_short_side / min(w, h)
    new_w = max(int(round(w * scale)), 1)
    new_h = max(int(round(h * scale)), 1)
    return pil_img.resize((new_w, new_h), Image.LANCZOS)


def normalize_9ch_tensor(x):
    mean = torch.tensor(
        [0.485, 0.456, 0.406, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        dtype=x.dtype
    ).view(9, 1, 1)
    std = torch.tensor(
        [0.229, 0.224, 0.225, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        dtype=x.dtype
    ).view(9, 1, 1)
    return (x - mean) / std


def build_9ch_from_pil(rgb_pil, ela_pil, noise_pil):
    rgb_t = TF.to_tensor(rgb_pil)
    ela_t = TF.to_tensor(ela_pil)
    noi_t = TF.to_tensor(noise_pil)
    x = torch.cat([rgb_t, ela_t, noi_t], dim=0)
    x = normalize_9ch_tensor(x)
    return x


# =========================================================
# TILING
# =========================================================
def get_tile_positions(width, height, tile=TILE_SIZE, stride=TILE_STRIDE):
    xs = list(range(0, max(width - tile + 1, 1), stride))
    ys = list(range(0, max(height - tile + 1, 1), stride))

    if len(xs) == 0:
        xs = [0]
    if len(ys) == 0:
        ys = [0]

    if xs[-1] != max(width - tile, 0):
        xs.append(max(width - tile, 0))
    if ys[-1] != max(height - tile, 0):
        ys.append(max(height - tile, 0))

    positions = []
    for y in ys:
        for x in xs:
            positions.append((x, y))
    return positions


def prepare_full_maps_and_tiles(resized_rgb):
    """
    Returns:
      tiles: list of dicts:
             {
               "x": int, "y": int,
               "tensor": 9ch tensor [9,224,224]
             }
      ela_full_pil
      noise_full_pil
    """
    ela_full = compute_ela_multiscale(resized_rgb)
    noise_full = compute_noise_3ch(resized_rgb)

    w, h = resized_rgb.size
    positions = get_tile_positions(w, h, TILE_SIZE, TILE_STRIDE)

    tiles = []
    for x, y in positions:
        rgb_tile = resized_rgb.crop((x, y, x + TILE_SIZE, y + TILE_SIZE))
        ela_tile = ela_full.crop((x, y, x + TILE_SIZE, y + TILE_SIZE))
        noi_tile = noise_full.crop((x, y, x + TILE_SIZE, y + TILE_SIZE))

        x9 = build_9ch_from_pil(rgb_tile, ela_tile, noi_tile)

        tiles.append({
            "x": x,
            "y": y,
            "tensor": x9,
        })

    return tiles, ela_full, noise_full


# =========================================================
# TTA
# =========================================================
@torch.no_grad()
def tta_logits(model, x):
    variants = [
        x,
        torch.flip(x, dims=[3]),
        torch.flip(x, dims=[2]),
        torch.flip(x, dims=[2, 3]),
    ]
    logits = torch.stack([model(v) for v in variants], dim=0).mean(dim=0)
    return logits


# =========================================================
# MULTI-LAYER GRADCAM++
# =========================================================
class MultiLayerGradCAMPP:
    def __init__(self, model):
        self.model = model
        self.activations = {}
        self.gradients = {}

        self.target_layers = {
            "layer2": self.model.layer2,
            "layer3": self.model.layer3[0],
            "layer4": self.model.layer4[0],
        }

        for name, layer in self.target_layers.items():
            layer.register_forward_hook(self._make_fwd_hook(name))
            layer.register_full_backward_hook(self._make_bwd_hook(name))

    def _make_fwd_hook(self, name):
        def hook(module, inp, out):
            self.activations[name] = out
        return hook

    def _make_bwd_hook(self, name):
        def hook(module, grad_in, grad_out):
            self.gradients[name] = grad_out[0]
        return hook

    def _gradcampp_single(self, acts, grads):
        grads_sq = grads ** 2
        grads_cu = grads ** 3

        alpha_denom = 2.0 * grads_sq + (grads_cu * acts).sum(dim=(2, 3), keepdim=True)
        alpha_denom = alpha_denom + 1e-8
        alpha = grads_sq / alpha_denom

        weights = (alpha * F.relu(grads)).sum(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1)
        cam = F.relu(cam)

        cam = cam[0].detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def generate(self, x, class_idx, out_size=(224, 224)):
        self.model.zero_grad(set_to_none=True)

        logits = self.model(x)
        logits[:, class_idx].sum().backward(retain_graph=True)

        cams = {}
        for name in self.target_layers.keys():
            acts = self.activations[name]
            grads = self.gradients[name]
            cam = self._gradcampp_single(acts, grads)
            cam = cv2.resize(cam, out_size, interpolation=cv2.INTER_CUBIC)
            cams[name] = cam

        return cams


# =========================================================
# MULTI-LAYER SCORE-CAM
# =========================================================
class MultiLayerScoreCAM:
    def __init__(self, model):
        self.model = model
        self.activations = {}

        self.target_layers = {
            "layer2": self.model.layer2,
            "layer3": self.model.layer3[0],
            "layer4": self.model.layer4[0],
        }

        for name, layer in self.target_layers.items():
            layer.register_forward_hook(self._make_hook(name))

    def _make_hook(self, name):
        def hook(module, inp, out):
            self.activations[name] = out.detach()
        return hook

    @torch.no_grad()
    def _scorecam_single(self, x, class_idx, acts, top_k=16):
        acts = acts[0]  # C,H,W
        c, h, w = acts.shape
        H, W = x.shape[2], x.shape[3]

        baseline = torch.zeros_like(x)
        base_score = torch.softmax(self.model(baseline), dim=1)[0, class_idx].item()

        scores = acts.mean(dim=(1, 2))
        top_idx = scores.topk(min(top_k, c)).indices

        cam = torch.zeros((H, W), device=x.device)

        for idx in top_idx:
            act = acts[idx][None, None, ...]
            act = F.interpolate(act, size=(H, W), mode="bilinear", align_corners=False)[0, 0]
            act = (act - act.min()) / (act.max() - act.min() + 1e-8)

            masked_x = x * act.unsqueeze(0).unsqueeze(0)
            score = torch.softmax(self.model(masked_x), dim=1)[0, class_idx].item()
            weight = max(score - base_score, 0.0)
            cam += weight * act

        cam = cam.detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    @torch.no_grad()
    def generate(self, x, class_idx, top_k=16):
        _ = self.model(x)

        cams = {}
        for name, acts in self.activations.items():
            cams[name] = self._scorecam_single(x, class_idx, acts, top_k=top_k)
        return cams


# =========================================================
# CAM FUSION / PRIORS
# =========================================================
def norm01(x):
    x = x.astype(np.float32)
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def robust_normalize(x):
    x = x.astype(np.float32)
    lo = np.percentile(x, 1.0)
    hi = np.percentile(x, 99.0)
    x = np.clip((x - lo) / (hi - lo + 1e-8), 0, 1)
    return x


def fuse_multilayer_cams(gcams, scams, ela=None, srm=None):
    g2 = robust_normalize(gcams["layer2"])
    g3 = robust_normalize(gcams["layer3"])
    g4 = robust_normalize(gcams["layer4"])

    s2 = robust_normalize(scams["layer2"])
    s3 = robust_normalize(scams["layer3"])
    s4 = robust_normalize(scams["layer4"])

    cam_fused = (
        0.12 * g2 + 0.18 * g3 + 0.20 * g4 +
        0.12 * s2 + 0.18 * s3 + 0.20 * s4
    )

    if ela is not None:
        cam_fused += 0.12 * robust_normalize(ela)

    if srm is not None:
        cam_fused += 0.08 * robust_normalize(srm)

    return robust_normalize(cam_fused)


def build_suspected_area_prior(ela, srm):
    prior = 0.55 * robust_normalize(ela) + 0.45 * robust_normalize(srm)
    prior = cv2.GaussianBlur(prior, (0, 0), sigmaX=2.0)
    return robust_normalize(prior)


def gate_cam_with_prior(cam, prior, power_cam=1.2, power_prior=1.6):
    cam = np.power(np.clip(cam, 0, 1), power_cam)
    prior = np.power(np.clip(prior, 0, 1), power_prior)
    gated = cam * prior
    return norm01(gated)


# =========================================================
# ELA / SRM MAPS
# =========================================================
def ela_block_map(img_pil, block_size=8, qualities=(70, 85, 95)):
    img_rgb = img_pil.convert("RGB")
    arr = np.array(img_rgb, dtype=np.float32)
    h, w = arr.shape[:2]
    score = np.zeros((h, w), dtype=np.float32)

    for q in qualities:
        buf = io.BytesIO()
        img_rgb.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        recomp = np.array(Image.open(buf).convert("RGB"), dtype=np.float32)
        diff = np.abs(arr - recomp).mean(axis=2)

        for r in range(0, h - block_size + 1, block_size):
            for c in range(0, w - block_size + 1, block_size):
                score[r:r + block_size, c:c + block_size] = diff[r:r + block_size, c:c + block_size].mean()

    return norm01(score)


SRM_LIN = np.array(
    [[0, 0, -1, 0, 0],
     [0, -1, 2, -1, 0],
     [-1, 2, 0, 2, -1],
     [0, -1, 2, -1, 0],
     [0, 0, -1, 0, 0]], dtype=np.float32
) / 8.0

SRM_EDGE = np.array(
    [[-1, -1, -1, -1, -1],
     [-1,  2,  2,  2, -1],
     [-1,  2,  8,  2, -1],
     [-1,  2,  2,  2, -1],
     [-1, -1, -1, -1, -1]], dtype=np.float32
)

_s = SRM_EDGE.sum() - SRM_EDGE[2, 2]
SRM_EDGE[2, 2] = -_s
SRM_EDGE /= 8.0


def srm_inconsistency_map(img_pil, patch=32, stride=8):
    gray = np.array(img_pil.convert("L"), dtype=np.float32) / 255.0
    h, w = gray.shape

    rl = cv2.filter2D(gray, -1, SRM_LIN, borderType=cv2.BORDER_REFLECT)
    re = cv2.filter2D(gray, -1, SRM_EDGE, borderType=cv2.BORDER_REFLECT)
    rg = gray - cv2.GaussianBlur(gray, (0, 0), sigmaX=1.5)

    ml, sl = rl.mean(), rl.std() + 1e-8
    me, se = re.mean(), re.std() + 1e-8
    mg, sg = rg.mean(), rg.std() + 1e-8

    sm = np.zeros((h, w), dtype=np.float32)
    cm = np.zeros((h, w), dtype=np.float32)

    for r in range(0, h - patch + 1, stride):
        for c in range(0, w - patch + 1, stride):
            pl = rl[r:r + patch, c:c + patch]
            pe = re[r:r + patch, c:c + patch]
            pg = rg[r:r + patch, c:c + patch]

            ps = (
                0.30 * abs(pl.mean() - ml) / sl +
                0.30 * abs(pe.mean() - me) / se +
                0.20 * abs(pg.mean() - mg) / sg +
                0.10 * abs(pl.std() - sl) / sl +
                0.10 * abs(pe.std() - se) / se
            )
            sm[r:r + patch, c:c + patch] += ps
            cm[r:r + patch, c:c + patch] += 1.0

    sm /= np.maximum(cm, 1.0)
    return norm01(sm)


# =========================================================
# CLASSIFICATION
# =========================================================
def classify_multiscale_tiled(model, pil_orig, device):
    per_scale_probs = []
    scale_debug = []

    for scale, scale_w in zip(SCALES, SCALE_WEIGHTS):
        target_short = int(BASE_SHORT_SIDE * scale)
        resized_rgb = aspect_preserving_resize(pil_orig, target_short)
        tiles, _, _ = prepare_full_maps_and_tiles(resized_rgb)

        tile_probs = []
        tile_weights = []

        for tile in tiles:
            x = tile["tensor"].unsqueeze(0).to(device)
            logits = tta_logits(model, x)
            probs = torch.softmax(logits / TEMPERATURE, dim=1)[0].cpu().numpy()

            tamper_score = 1.0 - probs[0]
            weight = 0.6 + 0.4 * tamper_score

            tile_probs.append(probs)
            tile_weights.append(weight)

        tile_weights = np.array(tile_weights, dtype=np.float32)
        tile_weights = tile_weights / (tile_weights.sum() + 1e-8)

        fused_scale_prob = np.sum(np.stack(tile_probs, axis=0) * tile_weights[:, None], axis=0)
        per_scale_probs.append(fused_scale_prob * scale_w)

        pred_idx = int(fused_scale_prob.argmax())
        scale_debug.append((scale, CLASS_NAMES[pred_idx], float(fused_scale_prob[pred_idx])))

    final_probs = np.sum(np.stack(per_scale_probs, axis=0), axis=0)
    final_probs = final_probs / (final_probs.sum() + 1e-8)

    sort_idx = np.argsort(-final_probs)
    top1 = int(sort_idx[0])
    top2 = int(sort_idx[1])
    margin = float(final_probs[top1] - final_probs[top2])

    return final_probs, top1, top2, margin, scale_debug


# =========================================================
# MAP ACCUMULATION
# =========================================================
def accumulate_map(acc_map, acc_weight, tile_map, x, y, tile=TILE_SIZE):
    acc_map[y:y + tile, x:x + tile] += tile_map
    acc_weight[y:y + tile, x:x + tile] += 1.0


def compute_multiscale_advanced_maps(model, gcam, scam, pil_orig, pred_idx, device):
    orig_w, orig_h = pil_orig.size

    gcams_total = {
        "layer2": np.zeros((orig_h, orig_w), dtype=np.float32),
        "layer3": np.zeros((orig_h, orig_w), dtype=np.float32),
        "layer4": np.zeros((orig_h, orig_w), dtype=np.float32),
    }
    scams_total = {
        "layer2": np.zeros((orig_h, orig_w), dtype=np.float32),
        "layer3": np.zeros((orig_h, orig_w), dtype=np.float32),
        "layer4": np.zeros((orig_h, orig_w), dtype=np.float32),
    }

    for scale, scale_w in zip(SCALES, SCALE_WEIGHTS):
        target_short = int(BASE_SHORT_SIDE * scale)
        resized_rgb = aspect_preserving_resize(pil_orig, target_short)
        rw, rh = resized_rgb.size

        tiles, _, _ = prepare_full_maps_and_tiles(resized_rgb)

        gc_scale = {
            "layer2": np.zeros((rh, rw), dtype=np.float32),
            "layer3": np.zeros((rh, rw), dtype=np.float32),
            "layer4": np.zeros((rh, rw), dtype=np.float32),
        }
        sc_scale = {
            "layer2": np.zeros((rh, rw), dtype=np.float32),
            "layer3": np.zeros((rh, rw), dtype=np.float32),
            "layer4": np.zeros((rh, rw), dtype=np.float32),
        }
        gc_wsum = {
            "layer2": np.zeros((rh, rw), dtype=np.float32),
            "layer3": np.zeros((rh, rw), dtype=np.float32),
            "layer4": np.zeros((rh, rw), dtype=np.float32),
        }
        sc_wsum = {
            "layer2": np.zeros((rh, rw), dtype=np.float32),
            "layer3": np.zeros((rh, rw), dtype=np.float32),
            "layer4": np.zeros((rh, rw), dtype=np.float32),
        }

        for tile in tiles:
            x0 = tile["x"]
            y0 = tile["y"]
            inp = tile["tensor"].unsqueeze(0).to(device)

            with torch.no_grad():
                logits = tta_logits(model, inp)
                probs = torch.softmax(logits / TEMPERATURE, dim=1)[0].cpu().numpy()

            class_conf = float(probs[pred_idx])
            evidence_w = 0.5 + 0.5 * class_conf

            gc_dict = gcam.generate(inp, pred_idx, out_size=(TILE_SIZE, TILE_SIZE))
            sc_dict = scam.generate(inp, pred_idx, top_k=SCORECAM_TOP_K)

            for name in ["layer2", "layer3", "layer4"]:
                gc = gc_dict[name]
                sc = sc_dict[name]

                if gc.shape != (TILE_SIZE, TILE_SIZE):
                    gc = cv2.resize(gc, (TILE_SIZE, TILE_SIZE), interpolation=cv2.INTER_CUBIC)
                if sc.shape != (TILE_SIZE, TILE_SIZE):
                    sc = cv2.resize(sc, (TILE_SIZE, TILE_SIZE), interpolation=cv2.INTER_CUBIC)

                accumulate_map(gc_scale[name], gc_wsum[name], gc * evidence_w, x0, y0, TILE_SIZE)
                accumulate_map(sc_scale[name], sc_wsum[name], sc * evidence_w, x0, y0, TILE_SIZE)

        for name in ["layer2", "layer3", "layer4"]:
            gc_scale[name] = gc_scale[name] / np.maximum(gc_wsum[name], 1e-8)
            sc_scale[name] = sc_scale[name] / np.maximum(sc_wsum[name], 1e-8)

            gc_scale[name] = cv2.resize(gc_scale[name], (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
            sc_scale[name] = cv2.resize(sc_scale[name], (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

            gcams_total[name] += scale_w * gc_scale[name]
            scams_total[name] += scale_w * sc_scale[name]

    for name in ["layer2", "layer3", "layer4"]:
        gcams_total[name] = robust_normalize(gcams_total[name])
        scams_total[name] = robust_normalize(scams_total[name])

    return gcams_total, scams_total


# =========================================================
# MASK / OVERLAY
# =========================================================
def build_precise_mask_advanced(score_map, img_np):
    score = np.clip(score_map, 0, 1).astype(np.float32)

    score = cv2.bilateralFilter((score * 255).astype(np.uint8), 7, 50, 50).astype(np.float32) / 255.0
    score = cv2.GaussianBlur(score, (0, 0), sigmaX=1.2)

    p85 = np.percentile(score, 85)
    p92 = np.percentile(score, 92)
    thr = max(0.38, 0.5 * p85 + 0.5 * p92)

    mask = (score >= thr).astype(np.uint8) * 255

    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k2)

    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    if dist.max() > 0:
        sure = (dist > 0.22 * dist.max()).astype(np.uint8) * 255
        mask = cv2.bitwise_or(mask, sure)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k3)

    min_area = img_np.shape[0] * img_np.shape[1] * MIN_SUSPICIOUS_AREA_RATIO
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cleaned = np.zeros_like(mask)
    for cnt in cnts:
        if cv2.contourArea(cnt) >= min_area:
            cv2.drawContours(cleaned, [cnt], -1, 255, -1)

    return cleaned, score


def draw_precise_overlay(img_np, mask, score_map):
    result = img_np.copy()
    overlay = img_np.copy()

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    min_area = img_np.shape[0] * img_np.shape[1] * MIN_SUSPICIOUS_AREA_RATIO
    valid = [c for c in cnts if cv2.contourArea(c) >= min_area]

    if valid:
        cv2.drawContours(overlay, valid, -1, (220, 40, 40), -1)
        result = cv2.addWeighted(overlay, 0.30, result, 0.70, 0)

        for cnt in valid:
            a = cv2.contourArea(cnt)
            pct = a / (img_np.shape[0] * img_np.shape[1]) * 100.0
            cv2.drawContours(result, [cnt], -1, (255, 40, 40), 2)

            hull = cv2.convexHull(cnt)
            cv2.drawContours(result, [hull], -1, (255, 160, 0), 1)

            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(result, (x, y), (x + w, y + h), (255, 210, 0), 1)

            local_score = float(score_map[y:y + h, x:x + w].mean())
            label = f"{pct:.1f}%  s={local_score:.2f}"

            ty = max(y - 6, 14)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(result, (x, ty - th - 2), (x + tw + 4, ty + 2), (0, 0, 0), -1)
            cv2.putText(result, label, (x + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 210, 0), 1)

    return result, valid


# =========================================================
# SAVE HELPERS
# =========================================================
def blend_heatmap(img_np, hm, cmap, alpha=0.55):
    hm = np.clip(hm, 0, 1)
    heat = cv2.applyColorMap((hm * 255).astype(np.uint8), cmap)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    out = np.clip((1 - alpha) * img_np + alpha * heat, 0, 255).astype(np.uint8)
    return out


def save_img(arr, path):
    Image.fromarray(arr.astype(np.uint8)).save(path)


# =========================================================
# MAIN LOCALIZE
# =========================================================
def localize(image_path, model_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice : {device}")
    print(f"Image  : {image_path}")
    print(f"Model  : {model_path}\n")

    model = load_model(model_path, device)
    gcam = MultiLayerGradCAMPP(model)
    scam = MultiLayerScoreCAM(model)

    pil_orig = Image.open(image_path).convert("RGB")
    img_np = np.array(pil_orig)
    orig_w, orig_h = pil_orig.size

    print(f"Image size : {orig_w} x {orig_h}")

    # ---------------- Classification ----------------
    final_probs, pred_idx, top2_idx, margin, scale_debug = classify_multiscale_tiled(
        model, pil_orig, device
    )

    for scale, cls_name, conf in scale_debug:
        print(f"  Scale {scale:.2f}x -> {cls_name:22s} {conf:.2%}")

    pred_name = CLASS_NAMES[pred_idx]
    pred_conf = float(final_probs[pred_idx])

    print(f"\nPrediction : {pred_name} ({pred_conf:.2%})")
    print(f"Top-2      : {CLASS_NAMES[top2_idx]} ({final_probs[top2_idx]:.2%})")
    print(f"Margin     : {margin:.4f}\n")

    # ---------------- CAM fusion ----------------
    print("Computing multi-layer multi-scale CAMs...")
    gcams_total, scams_total = compute_multiscale_advanced_maps(
        model, gcam, scam, pil_orig, pred_idx, device
    )

    # ---------------- forensic priors ----------------
    print("Computing forensic priors...")
    resized_1x = aspect_preserving_resize(pil_orig, BASE_SHORT_SIDE)

    ela_1x = ela_block_map(resized_1x, block_size=ELA_BLOCK_SIZE)
    srm_1x = srm_inconsistency_map(resized_1x, patch=SRM_PATCH, stride=SRM_STRIDE)

    ela_full = cv2.resize(ela_1x, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
    srm_full = cv2.resize(srm_1x, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

    prior = build_suspected_area_prior(ela_full, srm_full)

    # ---------------- layer fusion ----------------
    print("Fusing CAM layers and applying prior gating...")
    fused_cam = fuse_multilayer_cams(gcams_total, scams_total, ela_full, srm_full)
    gated_cam = gate_cam_with_prior(fused_cam, prior)

    # ---------------- mask ----------------
    print("Building precise mask...")
    mask, score_smooth = build_precise_mask_advanced(gated_cam, img_np)

    # strong authentic suppression
    if pred_idx == 0 and pred_conf >= AUTHENTIC_CONF_STRONG and margin >= MIN_TOP1_TOP2_MARGIN:
        mask[:] = 0

    annotated, contours = draw_precise_overlay(img_np, mask, score_smooth)

    # ---------------- save outputs ----------------
    save_img(annotated, os.path.join(output_dir, "annotated.png"))
    save_img(mask, os.path.join(output_dir, "mask.png"))
    save_img(blend_heatmap(img_np, score_smooth, cv2.COLORMAP_INFERNO, 0.60), os.path.join(output_dir, "combined_overlay.png"))
    save_img(blend_heatmap(img_np, gated_cam, cv2.COLORMAP_MAGMA, 0.60), os.path.join(output_dir, "gated_cam_overlay.png"))
    save_img(blend_heatmap(img_np, prior, cv2.COLORMAP_BONE, 0.50), os.path.join(output_dir, "prior_overlay.png"))
    save_img(blend_heatmap(img_np, ela_full, cv2.COLORMAP_HOT, 0.55), os.path.join(output_dir, "ela_overlay.png"))
    save_img(blend_heatmap(img_np, srm_full, cv2.COLORMAP_COOL, 0.55), os.path.join(output_dir, "srm_overlay.png"))

    for name in ["layer2", "layer3", "layer4"]:
        save_img(
            blend_heatmap(img_np, gcams_total[name], cv2.COLORMAP_JET, 0.55),
            os.path.join(output_dir, f"gradcampp_{name}.png")
        )
        save_img(
            blend_heatmap(img_np, scams_total[name], cv2.COLORMAP_PLASMA, 0.55),
            os.path.join(output_dir, f"scorecam_{name}.png")
        )

    # ---------------- forensic report ----------------
    fig = plt.figure(figsize=(24, 16), facecolor="#0a0a0a")
    fig.suptitle(
        f"Forensic Report | {os.path.basename(image_path)} | "
        f"{pred_name.upper()} ({pred_conf:.1%}) | {orig_w}x{orig_h}",
        color="white", fontsize=11, fontweight="bold", y=0.99
    )

    gs = gridspec.GridSpec(
        3, 4, figure=fig,
        hspace=0.25, wspace=0.06,
        left=0.02, right=0.98, top=0.94, bottom=0.03
    )

    panels = [
        (0, 0, img_np, "Original", None, False),
        (0, 1, annotated, "Detected Regions", None, False),
        (0, 2, mask, "Binary Mask", "hot", True),
        (0, 3, score_smooth, "Final Score Map", "inferno", True),

        (1, 0, blend_heatmap(img_np, gcams_total["layer2"], cv2.COLORMAP_JET, 0.65), "GradCAM++ Layer2", None, False),
        (1, 1, blend_heatmap(img_np, gcams_total["layer3"], cv2.COLORMAP_JET, 0.65), "GradCAM++ Layer3", None, False),
        (1, 2, blend_heatmap(img_np, gcams_total["layer4"], cv2.COLORMAP_JET, 0.65), "GradCAM++ Layer4", None, False),
        (1, 3, blend_heatmap(img_np, gated_cam, cv2.COLORMAP_MAGMA, 0.65), "Gated Fused CAM", None, False),

        (2, 0, blend_heatmap(img_np, scams_total["layer2"], cv2.COLORMAP_PLASMA, 0.65), "Score-CAM Layer2", None, False),
        (2, 1, blend_heatmap(img_np, scams_total["layer3"], cv2.COLORMAP_PLASMA, 0.65), "Score-CAM Layer3", None, False),
        (2, 2, blend_heatmap(img_np, scams_total["layer4"], cv2.COLORMAP_PLASMA, 0.65), "Score-CAM Layer4", None, False),
        (2, 3, blend_heatmap(img_np, prior, cv2.COLORMAP_BONE, 0.65), "ELA+SRM Prior", None, False),
    ]

    for row, col, data, title, cmap, use_cmap in panels:
        ax = fig.add_subplot(gs[row, col])
        if use_cmap:
            vmax = 255 if data.max() > 1.5 else 1.0
            ax.imshow(data, cmap=cmap, vmin=0, vmax=vmax)
        else:
            ax.imshow(data)
        ax.set_title(title, color="#dddddd", fontsize=8, pad=3)
        ax.axis("off")

    report_path = os.path.join(output_dir, "forensic_report.png")
    fig.savefig(report_path, dpi=150, bbox_inches="tight", facecolor="#0a0a0a")
    plt.close(fig)

    # ---------------- summary ----------------
    print("\n" + "=" * 58)
    for name, p in zip(CLASS_NAMES, final_probs):
        bar = "█" * int(float(p) * 40)
        marker = " ◄" if name == pred_name else ""
        print(f"  {name:22s}: {float(p):.4f}  {bar}{marker}")
    print("=" * 58)
    print(f"  Prediction  : {pred_name} ({pred_conf:.2%})")
    print(f"  Top2 margin : {margin:.4f}")

    if contours:
        suspicious_pct = (mask.sum() / 255.0) / (orig_h * orig_w) * 100.0
        print(f"  Suspicious area : {suspicious_pct:.2f}%")
        print(f"  Regions found   : {len(contours)}")
    else:
        print("  No reliable suspicious region found.")

    print(f"  Output dir      : {os.path.abspath(output_dir)}")
    print("=" * 58 + "\n")


if __name__ == "__main__":
    localize(IMAGE_PATH, MODEL_PATH, OUTPUT_DIR)