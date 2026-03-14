"""
localize.py — Forensic Tamper Localization
Generates:
  1. GradCAM heatmap        — where the model looks to make its decision
  2. ELA visualization      — JPEG compression inconsistencies
  3. Noise residual map     — SRM-based pixel manipulation traces
  4. Binary suspicion mask  — thresholded combination of all signals
  5. Annotated overlay      — original image with suspected regions highlighted
  6. Full forensic report   — all maps saved as a single PNG grid

Usage:
  Set IMAGE_PATH below and run:
    python localize.py
"""

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from train import build_resnet50_9ch
from precompute_maps import compute_ela_multiscale, compute_noise_3ch

CLASS_NAMES = ["authentic", "copy_move", "enhancement",
               "removal_inpainting", "splicing"]

# ── Config ────────────────────────────────────────────────────────────
IMAGE_PATH   = r"C:\Tampered_9channel\Im20_cmfr1.jpg"                  # ← your image path here
MODEL_PATH   = "outputs0/swa_model.pth"   # ← model weights
OUTPUT_DIR   = "localization_output"       # ← results saved here
TEMPERATURE  = 0.5                         # ← confidence sharpening
MASK_THRESH  = 0.45                        # ← suspicion threshold (0–1), lower = more sensitive
# ─────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────
# Model loader
# ─────────────────────────────────────────────
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


# ─────────────────────────────────────────────
# GradCAM
# ─────────────────────────────────────────────
class GradCAM:
    """
    Hooks into the last conv layer (layer4) to compute class-discriminative
    activation maps. Shows WHICH spatial regions the model used to classify.
    """
    def __init__(self, model):
        self.model       = model
        self.gradients   = None
        self.activations = None
        self._hook_layer()

    def _hook_layer(self):
        # Hook into layer4[0] — the main ResNet conv block before CBAM
        target = self.model.layer4[0]
        target.register_forward_hook(self._save_activation)
        target.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, x, class_idx=None):
        """
        Returns GradCAM heatmap (H, W) normalized to [0, 1].
        If class_idx is None, uses the predicted class.
        """
        self.model.zero_grad()
        logits = self.model(x)

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        score = logits[0, class_idx]
        score.backward()

        # Global average pool gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam     = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam     = F.relu(cam)
        cam     = cam.squeeze().cpu().numpy()

        # Normalize
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx


# ─────────────────────────────────────────────
# EigenCAM (fallback — no gradients needed)
# ─────────────────────────────────────────────
class EigenCAM:
    """
    Uses PCA of feature maps instead of gradients.
    More stable than GradCAM for uniform/simple images.
    """
    def __init__(self, model):
        self.model       = model
        self.activations = None
        self.model.layer4[0].register_forward_hook(self._save)

    def _save(self, module, input, output):
        self.activations = output.detach()

    @torch.no_grad()
    def generate(self, x):
        self.model(x)
        act = self.activations.squeeze(0)           # (C, H, W)
        C, H, W = act.shape
        flat = act.view(C, -1).T                    # (H*W, C)
        flat = flat - flat.mean(0)
        _, _, Vt = torch.linalg.svd(flat, full_matrices=False)
        first_pc = Vt[0]                            # (C,)
        cam = (act * first_pc[:, None, None]).sum(0)  # (H, W)
        cam = cam.cpu().numpy()
        cam = np.maximum(cam, 0)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


# ─────────────────────────────────────────────
# Map processing helpers
# ─────────────────────────────────────────────
def resize_to(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    """Resize 2D float array to (h, w) using INTER_LINEAR."""
    return cv2.resize(arr, (w, h), interpolation=cv2.INTER_LINEAR)


def apply_colormap(arr: np.ndarray, cmap=cv2.COLORMAP_JET) -> np.ndarray:
    """Apply OpenCV colormap to normalized [0,1] float array → BGR uint8."""
    return cv2.applyColorMap((arr * 255).astype(np.uint8), cmap)


def blend_heatmap(img_rgb: np.ndarray, heatmap_bgr: np.ndarray,
                  alpha=0.55) -> np.ndarray:
    """Blend RGB image with BGR heatmap. Returns RGB uint8."""
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    return np.clip(
        (1 - alpha) * img_rgb.astype(np.float32) +
        alpha       * heatmap_rgb.astype(np.float32), 0, 255
    ).astype(np.uint8)


def build_suspicion_mask(gradcam: np.ndarray,
                         ela_gray: np.ndarray,
                         noise_gray: np.ndarray,
                         threshold: float = 0.45) -> np.ndarray:
    """
    Combines GradCAM + ELA + noise into a single suspicion score,
    then thresholds to a binary mask.

    Weights:
      GradCAM  0.50 — model attention (most discriminative)
      ELA      0.30 — compression artifact inconsistency
      Noise    0.20 — pixel-level manipulation traces
    """
    g = (gradcam   - gradcam.min())   / (gradcam.max()   - gradcam.min()   + 1e-8)
    e = (ela_gray  - ela_gray.min())  / (ela_gray.max()  - ela_gray.min()  + 1e-8)
    n = (noise_gray- noise_gray.min())/ (noise_gray.max()- noise_gray.min()+ 1e-8)

    combined = 0.50 * g + 0.30 * e + 0.20 * n

    # Morphological cleanup — remove noise, fill holes
    binary = (combined > threshold).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    return binary, combined


def draw_contours_on_image(img_rgb: np.ndarray,
                            mask: np.ndarray) -> np.ndarray:
    """
    Draws contours of suspicious regions on the image.
    Fills region with semi-transparent red, outlines with solid red.
    """
    result  = img_rgb.copy()
    overlay = img_rgb.copy()

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Fill suspicious regions with red overlay
        cv2.drawContours(overlay, contours, -1, (255, 50, 50), -1)
        result = cv2.addWeighted(overlay, 0.35, result, 0.65, 0)

        # Draw contour border
        cv2.drawContours(result, contours, -1, (255, 30, 30), 2)

        # Draw bounding boxes for each region
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500:   # skip tiny noise regions
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(result, (x, y), (x+w, y+h), (255, 200, 0), 2)

            # Label with area percentage
            img_area  = img_rgb.shape[0] * img_rgb.shape[1]
            area_pct  = area / img_area * 100
            label     = f"{area_pct:.1f}%"
            cv2.putText(result, label, (x, max(y - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 2)

    return result, contours


# ─────────────────────────────────────────────
# Main localization pipeline
# ─────────────────────────────────────────────
def localize(image_path, model_path, output_dir,
             temperature=0.5, mask_thresh=0.45):

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device     : {device}")
    print(f"Image      : {image_path}")
    print(f"Model      : {model_path}")

    # ── Load model ────────────────────────────────────────────────────
    model    = load_model(model_path, device)
    gradcam  = GradCAM(model)
    eigcam   = EigenCAM(model)

    # ── Prepare image ─────────────────────────────────────────────────
    normalize_9 = transforms.Normalize(
        mean=[0.485, 0.456, 0.406, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        std =[0.229, 0.224, 0.225, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    )
    size     = (224, 224)
    pil_orig = Image.open(image_path).convert("RGB")
    W_orig, H_orig = pil_orig.size

    ela   = compute_ela_multiscale(pil_orig)
    noise = compute_noise_3ch(pil_orig)

    rgb_t   = TF.to_tensor(TF.resize(pil_orig, size, InterpolationMode.BILINEAR))
    ela_t   = TF.to_tensor(TF.resize(ela,      size, InterpolationMode.BILINEAR))
    noise_t = TF.to_tensor(TF.resize(noise,    size, InterpolationMode.BILINEAR))

    x = normalize_9(torch.cat([rgb_t, ela_t, noise_t], dim=0)).unsqueeze(0).to(device)
    x.requires_grad_(True)

    # ── Classification ────────────────────────────────────────────────
    with torch.no_grad():
        logits = model(x)
        probs  = torch.softmax(logits / temperature, dim=1)[0].cpu().tolist()

    pred_idx  = probs.index(max(probs))
    pred_name = CLASS_NAMES[pred_idx]
    conf      = max(probs)
    print(f"\nPrediction : {pred_name}  ({conf:.2%})")

    # ── GradCAM ───────────────────────────────────────────────────────
    x_grad = x.clone().detach().requires_grad_(True)
    cam_map, _ = gradcam.generate(x_grad, class_idx=pred_idx)

    # ── EigenCAM (supplementary) ──────────────────────────────────────
    with torch.no_grad():
        eig_map = eigcam.generate(x)

    # ── Resize all maps to original image size ────────────────────────
    img_np     = np.array(pil_orig)                                 # (H,W,3) uint8
    cam_full   = resize_to(cam_map,  H_orig, W_orig)
    eig_full   = resize_to(eig_map,  H_orig, W_orig)

    # ELA grayscale (mean of 3 channels)
    ela_np     = np.array(ela.convert("L"), dtype=np.float32) / 255.0

    # Noise grayscale (mean of 3 channels — gaussian+srm_linear+srm_edge)
    noise_np   = np.array(noise, dtype=np.float32).mean(axis=2) / 255.0

    # ── Suspicion mask ────────────────────────────────────────────────
    mask_bin, combined_map = build_suspicion_mask(
        cam_full, ela_np, noise_np, threshold=mask_thresh)

    # ── Annotated overlay ─────────────────────────────────────────────
    annotated, contours = draw_contours_on_image(img_np, mask_bin)

    # ── Individual channel noise maps ─────────────────────────────────
    noise_arr  = np.array(noise)  # (H,W,3)
    gauss_map  = noise_arr[:, :, 0]   # R = gaussian_residual
    srm_lin    = noise_arr[:, :, 1]   # G = srm_linear_residual
    srm_edge   = noise_arr[:, :, 2]   # B = srm_edge_residual

    # ── Build full forensic report figure ────────────────────────────
    fig = plt.figure(figsize=(22, 14), facecolor="#0d0d0d")
    fig.suptitle(
        f"Forensic Localization Report  |  {os.path.basename(image_path)}  "
        f"|  Prediction: {pred_name.upper()}  ({conf:.1%})",
        color="white", fontsize=14, fontweight="bold", y=0.98
    )

    gs = gridspec.GridSpec(3, 4, figure=fig,
                           hspace=0.35, wspace=0.15,
                           left=0.03, right=0.97,
                           top=0.93, bottom=0.03)

    panels = [
        # Row 0
        (0, 0, img_np,      "Original Image",          "gray",       False),
        (0, 1, annotated,   "Suspected Regions",        "gray",       False),
        (0, 2, mask_bin,    "Suspicion Mask",           "hot",        True),
        (0, 3, combined_map,"Combined Score Map",       "inferno",    True),
        # Row 1
        (1, 0, cam_full,    "GradCAM (model attention)","jet",        True),
        (1, 1, eig_full,    "EigenCAM (feature PCA)",   "plasma",     True),
        (1, 2, np.array(ela),"ELA Map (RGB)",           "gray",       False),
        (1, 3, ela_np,      "ELA Intensity",            "hot",        True),
        # Row 2
        (2, 0, gauss_map,   "Noise Ch0: Gaussian",      "coolwarm",   True),
        (2, 1, srm_lin,     "Noise Ch1: SRM Linear",    "coolwarm",   True),
        (2, 2, srm_edge,    "Noise Ch2: SRM Edge",      "coolwarm",   True),
        (2, 3, noise_np,    "Noise Combined (mean)",    "hot",        True),
    ]

    for row, col, data, title, cmap, use_cmap in panels:
        ax = fig.add_subplot(gs[row, col])
        if use_cmap:
            ax.imshow(data, cmap=cmap, vmin=0,
                      vmax=255 if data.max() > 1 else 1)
        else:
            ax.imshow(data)
        ax.set_title(title, color="#cccccc", fontsize=8, pad=4)
        ax.axis("off")

    # ── Save full report ───────────────────────────────────────────────
    report_path = os.path.join(output_dir, "forensic_report.png")
    fig.savefig(report_path, dpi=150, bbox_inches="tight",
                facecolor="#0d0d0d")
    plt.close(fig)
    print(f"Report     : {report_path}")

    # ── Save individual outputs ────────────────────────────────────────
    # Annotated image (suspected regions highlighted)
    ann_path = os.path.join(output_dir, "annotated.png")
    Image.fromarray(annotated).save(ann_path)
    print(f"Annotated  : {ann_path}")

    # Binary mask
    mask_path = os.path.join(output_dir, "mask.png")
    Image.fromarray(mask_bin).save(mask_path)
    print(f"Mask       : {mask_path}")

    # GradCAM heatmap blended
    cam_color   = apply_colormap(cam_full, cv2.COLORMAP_JET)
    cam_blended = blend_heatmap(img_np, cam_color, alpha=0.55)
    cam_path    = os.path.join(output_dir, "gradcam.png")
    Image.fromarray(cam_blended).save(cam_path)
    print(f"GradCAM    : {cam_path}")

    # ELA visualization
    ela_color   = apply_colormap(ela_np, cv2.COLORMAP_HOT)
    ela_blended = blend_heatmap(img_np, ela_color, alpha=0.6)
    ela_path    = os.path.join(output_dir, "ela_overlay.png")
    Image.fromarray(ela_blended).save(ela_path)
    print(f"ELA overlay: {ela_path}")

    # Combined score heatmap blended
    comb_color   = apply_colormap(combined_map, cv2.COLORMAP_INFERNO)
    comb_blended = blend_heatmap(img_np, comb_color, alpha=0.55)
    comb_path    = os.path.join(output_dir, "combined_overlay.png")
    Image.fromarray(comb_blended).save(comb_path)
    print(f"Combined   : {comb_path}")

    # ── Console summary ────────────────────────────────────────────────
    print("\n" + "="*52)
    print("  CLASSIFICATION SCORES")
    print("="*52)
    for name, p in zip(CLASS_NAMES, probs):
        bar    = "█" * int(p * 40)
        marker = " ◄" if name == pred_name else ""
        print(f"  {name:22s}: {p:.4f}  {bar}{marker}")
    print("="*52)
    print(f"  Prediction : {pred_name}")
    print(f"  Confidence : {conf:.2%}")

    if contours:
        total_mask_area = mask_bin.sum() / 255
        total_img_area  = H_orig * W_orig
        pct = total_mask_area / total_img_area * 100
        print(f"\n  Suspicious area : {pct:.1f}% of image")
        print(f"  Regions found   : {len([c for c in contours if cv2.contourArea(c) > 500])}")
    else:
        print(f"\n  No suspicious regions detected above threshold={mask_thresh}")

    print(f"\n  Output folder: {os.path.abspath(output_dir)}/")
    print("="*52 + "\n")


if __name__ == "__main__":
    localize(
        image_path  = IMAGE_PATH,
        model_path  = MODEL_PATH,
        output_dir  = OUTPUT_DIR,
        temperature = TEMPERATURE,
        mask_thresh = MASK_THRESH,
    )
