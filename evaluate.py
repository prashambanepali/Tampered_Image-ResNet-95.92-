"""
evaluate.py — Test Set Evaluation
==================================
Runs the trained model on the test split and produces:
  - Overall accuracy (with and without TTA)
  - Per-class precision, recall, F1-score
  - Confusion matrix (saved as PNG)
  - Per-image prediction CSV log
  - Top-K misclassified images report

Reads test images from:
  dataset/test/<class>/index.txt  →  paths to rgb/ela/noise in Features1/

Usage:
  python evaluate.py
"""

import os
import io
import csv
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageChops
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ── Config ──────────────────────────────────────────────────────────────
MODEL_PATH     = "outputs0/swa_model.pth"   # or swa_model.pth
OUTPUT_DIR     = "eval_output(swa)"
DATASET_ROOT   = "dataset"
FEATURES_ROOT  = "Features1"
BATCH_SIZE     = 32
NUM_WORKERS    = 4
TEMPERATURE    = 1.0    # no temperature scaling for evaluation (fair comparison)
USE_TTA        = True   # test-time augmentation (4 flips)
TOP_K_ERRORS   = 20     # save this many worst misclassified images to CSV

CLASS_NAMES    = ["authentic", "copy_move", "enhancement",
                  "removal_inpainting", "splicing"]
# ────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────
# Model — copied inline so no dependency on train.py
# ─────────────────────────────────────────────
class ChannelAttention(nn.Module):
    def __init__(self, in_ch, reduction=16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.mx  = nn.AdaptiveMaxPool2d(1)
        self.fc  = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_ch // reduction, in_ch, 1, bias=False),
        )
    def forward(self, x):
        return torch.sigmoid(self.fc(self.avg(x)) + self.fc(self.mx(x))) * x


class SpatialAttention(nn.Module):
    def __init__(self, k=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, k, padding=k // 2, bias=False)
    def forward(self, x):
        avg = x.mean(1, keepdim=True)
        mx, _ = x.max(1, keepdim=True)
        return torch.sigmoid(self.conv(torch.cat([avg, mx], 1))) * x


class CBAM(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.ca = ChannelAttention(in_ch)
        self.sa = SpatialAttention()
    def forward(self, x):
        return self.sa(self.ca(x))


class DropBlock2D(nn.Module):
    def __init__(self, block_size=7, drop_prob=0.10):
        super().__init__()
        self.block_size = block_size
        self.drop_prob  = drop_prob
    def forward(self, x):
        if not self.training or self.drop_prob == 0:
            return x
        B, C, H, W = x.shape
        mask_h     = H - self.block_size + 1
        mask_w     = W - self.block_size + 1
        seed_rate  = (self.drop_prob / self.block_size ** 2
                      * H * W / (mask_h * mask_w))
        mask = torch.bernoulli(
            torch.ones(B, C, mask_h, mask_w, device=x.device) * seed_rate)
        mask = F.max_pool2d(mask, (self.block_size, self.block_size),
                            stride=1, padding=self.block_size // 2)
        if mask.shape[-2:] != x.shape[-2:]:
            mask = F.interpolate(mask, size=x.shape[-2:])
        mask = 1 - mask
        return x * mask * (mask.numel() / (mask.sum() + 1e-6))


def build_resnet50_9ch(num_classes=5, pretrained=False, drop_prob=0.10):
    weights  = ResNet50_Weights.DEFAULT if pretrained else None
    model    = models.resnet50(weights=weights)
    old_conv = model.conv1
    model.conv1 = nn.Conv2d(
        9, old_conv.out_channels,
        old_conv.kernel_size, old_conv.stride,
        old_conv.padding, bias=False
    )
    model.layer3 = nn.Sequential(
        model.layer3, CBAM(1024), DropBlock2D(block_size=7, drop_prob=drop_prob))
    model.layer4 = nn.Sequential(
        model.layer4, CBAM(2048), DropBlock2D(block_size=5, drop_prob=drop_prob))
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(2048, num_classes),
    )
    return model


def load_model(model_path, device):
    model = build_resnet50_9ch(len(CLASS_NAMES), pretrained=False).to(device)
    state = torch.load(model_path, map_location=device)
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v
                 for k, v in state.items()
                 if k != "n_averaged"}
    model.load_state_dict(state)
    model.eval()
    return model


# ─────────────────────────────────────────────
# Dataset — reads from dataset/test/<class>/index.txt
# ─────────────────────────────────────────────
class TestDataset(Dataset):
    """
    Reads test split index files and loads precomputed
    RGB + ELA + noise (3ch) maps from Features1/.
    """
    def __init__(self, dataset_root, features_root,
                 class_names, image_size=(224, 224)):
        self.image_size    = image_size
        self.normalize_9   = transforms.Normalize(
            mean=[0.485, 0.456, 0.406, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            std =[0.229, 0.224, 0.225, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        )
        self.items = []   # list of (rgb_path, ela_path, noise_path, label, rel_path)

        for label, cls in enumerate(class_names):
            index_file = os.path.join(dataset_root, "test", cls, "index.txt")
            if not os.path.exists(index_file):
                print(f"  [WARN] Missing index: {index_file}")
                continue

            with open(index_file) as f:
                lines = [l.strip() for l in f if l.strip()]

            for line in lines:
                # index.txt stores relative paths like:
                # Features1/authentic/rgb/img001.jpg
                # We derive ela and noise paths from it
                rgb_path = line
                if not os.path.isabs(rgb_path):
                    rgb_path = os.path.join(os.path.dirname(index_file),
                                            "..", "..", "..", rgb_path)
                rgb_path = os.path.normpath(rgb_path)

                # Derive ela/noise paths by replacing /rgb/ with /ela/ and /noise/
                base      = os.path.splitext(rgb_path)[0]
                ela_path  = base.replace(
                    os.sep + "rgb" + os.sep,
                    os.sep + "ela"   + os.sep) + ".png"
                noi_path  = base.replace(
                    os.sep + "rgb" + os.sep,
                    os.sep + "noise" + os.sep) + ".png"

                if not os.path.exists(rgb_path):
                    # Try features_root-based path
                    fname    = os.path.basename(rgb_path)
                    base_n   = os.path.splitext(fname)[0]
                    rgb_path = os.path.join(features_root, cls, "rgb",   fname)
                    ela_path = os.path.join(features_root, cls, "ela",   base_n + ".png")
                    noi_path = os.path.join(features_root, cls, "noise", base_n + ".png")

                self.items.append((rgb_path, ela_path, noi_path, label,
                                   os.path.basename(rgb_path)))

        print(f"  Test samples loaded: {len(self.items)}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        rgb_path, ela_path, noi_path, label, fname = self.items[idx]

        rgb   = Image.open(rgb_path).convert("RGB")
        ela   = Image.open(ela_path).convert("RGB")
        noise = Image.open(noi_path).convert("RGB")

        size  = self.image_size
        rgb_t   = TF.to_tensor(TF.resize(rgb,   size, TF.InterpolationMode.BILINEAR))
        ela_t   = TF.to_tensor(TF.resize(ela,   size, TF.InterpolationMode.BILINEAR))
        noi_t   = TF.to_tensor(TF.resize(noise, size, TF.InterpolationMode.BILINEAR))

        x = self.normalize_9(torch.cat([rgb_t, ela_t, noi_t], dim=0))
        return x, label, fname


# ─────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────
def predict_batch(model, x, device, temperature=1.0, use_tta=True):
    """Returns softmax probs (B, C) for a batch."""
    x = x.to(device)
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
        avg_logits = torch.stack([model(f) for f in flips]).mean(0)
        probs      = torch.softmax(avg_logits / temperature, dim=1)
    return probs.cpu()


def compute_metrics(y_true, y_pred, class_names):
    """
    Computes per-class and overall metrics without sklearn.
    Returns dict of metrics.
    """
    n = len(class_names)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1

    per_class = {}
    for i, cls in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        f1        = 2 * precision * recall / (precision + recall + 1e-8)
        support   = cm[i, :].sum()
        per_class[cls] = dict(precision=precision, recall=recall,
                               f1=f1, support=int(support), tp=int(tp))

    overall_acc    = np.diag(cm).sum() / cm.sum()
    macro_precision= np.mean([v["precision"] for v in per_class.values()])
    macro_recall   = np.mean([v["recall"]    for v in per_class.values()])
    macro_f1       = np.mean([v["f1"]        for v in per_class.values()])

    return dict(
        confusion_matrix = cm,
        per_class        = per_class,
        overall_acc      = overall_acc,
        macro_precision  = macro_precision,
        macro_recall     = macro_recall,
        macro_f1         = macro_f1,
    )


def save_confusion_matrix(cm, class_names, output_path, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(9, 7))

    # Normalize for color (keep raw numbers as labels)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, linecolor="gray",
                cbar_kws={"label": "Normalized (row)"}, ax=ax)

    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True",      fontsize=12)
    ax.set_title(title,        fontsize=13, fontweight="bold")
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.yticks(rotation=0,  fontsize=9)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Confusion matrix → {output_path}")


def print_report(metrics, class_names, elapsed_sec, n_samples, use_tta):
    w = 54
    print("\n" + "="*w)
    print("  TEST SET EVALUATION REPORT")
    print("="*w)
    print(f"  Samples   : {n_samples}")
    print(f"  TTA       : {'Yes (4 flips)' if use_tta else 'No'}")
    print(f"  Time      : {elapsed_sec:.1f}s  ({elapsed_sec/n_samples*1000:.1f}ms/img)")
    print("-"*w)
    print(f"  {'Class':<22}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'Support':>8}")
    print("-"*w)
    for cls in class_names:
        m = metrics["per_class"][cls]
        print(f"  {cls:<22}  {m['precision']:6.4f}  {m['recall']:6.4f}  "
              f"{m['f1']:6.4f}  {m['support']:8d}")
    print("-"*w)
    print(f"  {'Macro avg':<22}  {metrics['macro_precision']:6.4f}  "
          f"{metrics['macro_recall']:6.4f}  {metrics['macro_f1']:6.4f}")
    print("="*w)
    print(f"  Overall Accuracy : {metrics['overall_acc']:.4f}  "
          f"({metrics['overall_acc']*100:.2f}%)")
    print("="*w + "\n")


# ─────────────────────────────────────────────
# Main evaluation loop
# ─────────────────────────────────────────────
def evaluate(model_path, dataset_root, features_root, output_dir,
             batch_size=32, num_workers=4,
             temperature=1.0, use_tta=True, top_k_errors=20):

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice     : {device}")
    print(f"Model      : {model_path}")
    print(f"TTA        : {use_tta}")
    print(f"Temperature: {temperature}\n")

    # ── Load model ────────────────────────────────────────────────────
    model = load_model(model_path, device)
    print(f"Model loaded. Parameters: "
          f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M\n")

    # ── Load test dataset ─────────────────────────────────────────────
    print("Loading test dataset...")
    dataset = TestDataset(dataset_root, features_root,
                          CLASS_NAMES, image_size=(224, 224))

    if len(dataset) == 0:
        print("\n[ERROR] No test samples found.")
        print("Check that dataset/test/<class>/index.txt files exist.")
        return

    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=num_workers,
                        pin_memory=True)

    # ── Run inference ─────────────────────────────────────────────────
    all_probs   = []
    all_preds   = []
    all_labels  = []
    all_fnames  = []
    all_confs   = []

    print(f"\nRunning inference on {len(dataset)} test images...")
    t0 = time.time()

    for x, labels, fnames in tqdm(loader, unit="batch", dynamic_ncols=True):
        probs  = predict_batch(model, x, device, temperature, use_tta)
        preds  = probs.argmax(dim=1)
        confs  = probs.max(dim=1).values

        all_probs.extend(probs.numpy())
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())
        all_fnames.extend(fnames)
        all_confs.extend(confs.numpy())

    elapsed = time.time() - t0

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_confs  = np.array(all_confs)

    # ── Compute metrics ───────────────────────────────────────────────
    metrics = compute_metrics(all_labels, all_preds, CLASS_NAMES)
    print_report(metrics, CLASS_NAMES, elapsed, len(dataset), use_tta)

    # ── Save confusion matrix ─────────────────────────────────────────
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    cm_title   = (f"Confusion Matrix — {model_name}\n"
                  f"Acc={metrics['overall_acc']:.4f}  "
                  f"F1={metrics['macro_f1']:.4f}  "
                  f"{'TTA' if use_tta else 'No TTA'}")
    save_confusion_matrix(
        metrics["confusion_matrix"], CLASS_NAMES,
        os.path.join(output_dir, "confusion_matrix.png"),
        title=cm_title
    )

    # ── Save per-image prediction CSV ─────────────────────────────────
    csv_path = os.path.join(output_dir, "predictions.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "true_label", "true_class",
                         "pred_label", "pred_class", "confidence", "correct"]
                        + [f"prob_{c}" for c in CLASS_NAMES])
        for i in range(len(all_fnames)):
            row = [
                all_fnames[i],
                int(all_labels[i]),
                CLASS_NAMES[all_labels[i]],
                int(all_preds[i]),
                CLASS_NAMES[all_preds[i]],
                f"{all_confs[i]:.4f}",
                int(all_labels[i] == all_preds[i]),
            ] + [f"{p:.4f}" for p in all_probs[i]]
            writer.writerow(row)
    print(f"  Per-image CSV  → {csv_path}")

    # ── Save top-K misclassified images ───────────────────────────────
    errors_path = os.path.join(output_dir, "top_errors.csv")
    wrong_mask  = all_preds != all_labels
    wrong_idx   = np.where(wrong_mask)[0]

    # Sort by confidence of wrong prediction (most confidently wrong = worst)
    wrong_conf  = all_confs[wrong_idx]
    sort_order  = np.argsort(-wrong_conf)
    wrong_idx   = wrong_idx[sort_order]

    with open(errors_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "filename", "true_class",
                         "pred_class", "confidence"])
        for rank, idx in enumerate(wrong_idx[:top_k_errors], 1):
            writer.writerow([
                rank,
                all_fnames[idx],
                CLASS_NAMES[all_labels[idx]],
                CLASS_NAMES[all_preds[idx]],
                f"{all_confs[idx]:.4f}",
            ])
    print(f"  Top-{top_k_errors} errors → {errors_path}")

    # ── ROC / AUC curves ──────────────────────────────────────────────
    all_probs_arr  = np.array(all_probs)   # (N, C)
    all_labels_arr = np.array(all_labels)  # (N,)
    n_classes      = len(CLASS_NAMES)

    # One-vs-Rest ROC for each class
    fpr_dict  = {}
    tpr_dict  = {}
    auc_dict  = {}

    for i, cls in enumerate(CLASS_NAMES):
        # Binary labels: 1 if true class == i, else 0
        y_bin   = (all_labels_arr == i).astype(int)
        y_score = all_probs_arr[:, i]

        # Sort by descending score to trace the ROC curve
        sort_idx = np.argsort(-y_score)
        y_bin_s  = y_bin[sort_idx]
        y_sc_s   = y_score[sort_idx]

        P = y_bin.sum()          # total positives
        N = len(y_bin) - P       # total negatives

        tprs = [0.0]; fprs = [0.0]
        tp = 0; fp = 0

        # Unique thresholds (descending)
        thresholds = np.unique(y_sc_s)[::-1]
        for thr in thresholds:
            mask  = y_sc_s >= thr
            tp    = (y_bin_s[mask]).sum()
            fp    = (~y_bin_s[mask].astype(bool)).sum()
            tprs.append(tp / (P + 1e-8))
            fprs.append(fp / (N + 1e-8))

        tprs.append(1.0); fprs.append(1.0)
        fprs = np.array(fprs); tprs = np.array(tprs)

        # AUC via trapezoidal rule
        auc = float(np.trapz(tprs, fprs))
        # AUC should be in [0,1]; flip if < 0.5 due to sort direction
        auc = max(auc, 1 - auc) if auc < 0.4 else auc

        fpr_dict[cls] = fprs
        tpr_dict[cls] = tprs
        auc_dict[cls] = auc

    # Micro-average ROC (flatten all classes)
    y_bin_all   = np.eye(n_classes)[all_labels_arr]          # (N, C) one-hot
    y_score_all = all_probs_arr                               # (N, C)

    sort_idx_m  = np.argsort(-y_score_all.ravel())
    y_bin_flat  = y_bin_all.ravel()[sort_idx_m]
    y_sc_flat   = y_score_all.ravel()[sort_idx_m]

    P_m = y_bin_flat.sum(); N_m = len(y_bin_flat) - P_m
    tprs_m = [0.0]; fprs_m = [0.0]
    for thr in np.unique(y_sc_flat)[::-1]:
        mask  = y_sc_flat >= thr
        tp_m  = y_bin_flat[mask].sum()
        fp_m  = (~y_bin_flat[mask].astype(bool)).sum()
        tprs_m.append(tp_m / (P_m + 1e-8))
        fprs_m.append(fp_m / (N_m + 1e-8))
    tprs_m.append(1.0); fprs_m.append(1.0)
    fprs_m = np.array(fprs_m); tprs_m = np.array(tprs_m)
    auc_micro = float(np.trapz(tprs_m, fprs_m))
    auc_micro = max(auc_micro, 1 - auc_micro) if auc_micro < 0.4 else auc_micro

    # ── Plot ROC curves ───────────────────────────────────────────────
    COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        f"ROC Curves — {model_name}  |  "
        f"Acc={metrics['overall_acc']:.4f}  "
        f"Macro-AUC={np.mean(list(auc_dict.values())):.4f}",
        fontsize=13, fontweight="bold"
    )

    # Left: all classes on one plot
    ax = axes[0]
    ax.set_facecolor("#f8f9fa")
    ax.plot([0,1],[0,1], "k--", lw=1, alpha=0.5, label="Random (AUC=0.50)")
    for i, cls in enumerate(CLASS_NAMES):
        ax.plot(fpr_dict[cls], tpr_dict[cls],
                color=COLORS[i], lw=2,
                label=f"{cls}  (AUC={auc_dict[cls]:.4f})")
    # Micro-average
    ax.plot(fprs_m, tprs_m, color="black", lw=2.5, linestyle="-.",
            label=f"Micro-avg  (AUC={auc_micro:.4f})")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate",  fontsize=11)
    ax.set_title("All Classes (One-vs-Rest)", fontsize=11)
    ax.legend(loc="lower right", fontsize=8.5)
    ax.set_xlim([-0.01, 1.01]); ax.set_ylim([-0.01, 1.05])
    ax.grid(True, alpha=0.3)

    # Right: one subplot per class arranged in a 5-panel grid
    fig2, axes2 = plt.subplots(2, 3, figsize=(16, 10))
    fig2.suptitle(
        f"Per-Class ROC Curves — {model_name}",
        fontsize=13, fontweight="bold"
    )
    axes2_flat = axes2.flatten()

    for i, cls in enumerate(CLASS_NAMES):
        ax2 = axes2_flat[i]
        ax2.set_facecolor("#f8f9fa")
        ax2.plot([0,1],[0,1], "k--", lw=1, alpha=0.5)
        ax2.fill_between(fpr_dict[cls], tpr_dict[cls],
                          alpha=0.15, color=COLORS[i])
        ax2.plot(fpr_dict[cls], tpr_dict[cls],
                 color=COLORS[i], lw=2.5,
                 label=f"AUC = {auc_dict[cls]:.4f}")
        ax2.set_title(cls, fontsize=11, fontweight="bold", color=COLORS[i])
        ax2.set_xlabel("FPR", fontsize=9)
        ax2.set_ylabel("TPR", fontsize=9)
        ax2.legend(loc="lower right", fontsize=10)
        ax2.set_xlim([-0.01, 1.01]); ax2.set_ylim([-0.01, 1.05])
        ax2.grid(True, alpha=0.3)

        # Annotate optimal threshold point (closest to top-left corner)
        dist    = np.sqrt(fpr_dict[cls]**2 + (1 - tpr_dict[cls])**2)
        opt_idx = np.argmin(dist)
        ax2.plot(fpr_dict[cls][opt_idx], tpr_dict[cls][opt_idx],
                 "ko", markersize=7,
                 label=f"Optimal point\n"
                       f"FPR={fpr_dict[cls][opt_idx]:.3f} "
                       f"TPR={tpr_dict[cls][opt_idx]:.3f}")
        ax2.legend(loc="lower right", fontsize=8)

    # Hide unused 6th subplot
    axes2_flat[5].set_visible(False)

    plt.tight_layout()
    roc_combined = os.path.join(output_dir, "roc_combined.png")
    roc_perclass = os.path.join(output_dir, "roc_per_class.png")
    fig.savefig(roc_combined, dpi=150, bbox_inches="tight")
    fig2.savefig(roc_perclass, dpi=150, bbox_inches="tight")
    plt.close(fig); plt.close(fig2)
    print(f"  ROC combined   → {roc_combined}")
    print(f"  ROC per-class  → {roc_perclass}")

    # ── Save AUC summary to CSV ───────────────────────────────────────
    auc_csv = os.path.join(output_dir, "auc_scores.csv")
    with open(auc_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "auc", "f1", "precision", "recall", "support"])
        for cls in CLASS_NAMES:
            m = metrics["per_class"][cls]
            writer.writerow([cls,
                              f"{auc_dict[cls]:.4f}",
                              f"{m['f1']:.4f}",
                              f"{m['precision']:.4f}",
                              f"{m['recall']:.4f}",
                              m["support"]])
        writer.writerow(["micro_avg", f"{auc_micro:.4f}",
                          f"{metrics['macro_f1']:.4f}",
                          f"{metrics['macro_precision']:.4f}",
                          f"{metrics['macro_recall']:.4f}",
                          len(dataset)])
    print(f"  AUC scores CSV → {auc_csv}")

    # ── Save per-class accuracy bar chart ─────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Test Set Results — {model_name}  "
                 f"(Acc={metrics['overall_acc']:.4f})",
                 fontsize=13, fontweight="bold")

    # Per-class F1
    f1_vals = [metrics["per_class"][c]["f1"] for c in CLASS_NAMES]
    colors  = ["#2ecc71" if v >= 0.90 else
               "#f39c12" if v >= 0.75 else "#e74c3c" for v in f1_vals]
    axes[0].barh(CLASS_NAMES, f1_vals, color=colors, edgecolor="white")
    axes[0].set_xlim(0, 1.05)
    axes[0].set_xlabel("F1 Score")
    axes[0].set_title("Per-Class F1 Score")
    for i, v in enumerate(f1_vals):
        axes[0].text(v + 0.01, i, f"{v:.4f}", va="center", fontsize=9)

    # Per-class accuracy (TP / support)
    acc_vals = [metrics["per_class"][c]["tp"] /
                max(metrics["per_class"][c]["support"], 1)
                for c in CLASS_NAMES]
    colors2  = ["#3498db" if v >= 0.90 else
                "#e67e22" if v >= 0.75 else "#c0392b" for v in acc_vals]
    axes[1].barh(CLASS_NAMES, acc_vals, color=colors2, edgecolor="white")
    axes[1].set_xlim(0, 1.05)
    axes[1].set_xlabel("Accuracy")
    axes[1].set_title("Per-Class Accuracy")
    for i, v in enumerate(acc_vals):
        axes[1].text(v + 0.01, i, f"{v:.4f}", va="center", fontsize=9)

    plt.tight_layout()
    chart_path = os.path.join(output_dir, "class_metrics.png")
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Class metrics  → {chart_path}")

    # ── Final summary ─────────────────────────────────────────────────
    print(f"\n{'='*54}")
    print(f"  SUMMARY")
    print(f"{'='*54}")
    print(f"  Overall Accuracy : {metrics['overall_acc']*100:.2f}%")
    print(f"  Macro F1         : {metrics['macro_f1']:.4f}")
    print(f"  Macro Precision  : {metrics['macro_precision']:.4f}")
    print(f"  Macro Recall     : {metrics['macro_recall']:.4f}")
    print(f"  Micro AUC        : {auc_micro:.4f}")
    print(f"  Macro AUC        : {np.mean(list(auc_dict.values())):.4f}")
    print(f"  Wrong preds      : {wrong_mask.sum()} / {len(dataset)}")
    print(f"\n  Output folder: {os.path.abspath(output_dir)}/")
    print(f"    confusion_matrix.png  ← heatmap")
    print(f"    class_metrics.png     ← F1 + accuracy bars")
    print(f"    roc_combined.png      ← all classes on one ROC plot")
    print(f"    roc_per_class.png     ← individual ROC per class")
    print(f"    auc_scores.csv        ← AUC + F1 + P/R per class")
    print(f"    predictions.csv       ← per-image results")
    print(f"    top_errors.csv        ← worst {top_k_errors} mistakes")
    print(f"{'='*54}\n")

    return metrics


if __name__ == "__main__":
    evaluate(
        model_path    = MODEL_PATH,
        dataset_root  = DATASET_ROOT,
        features_root = FEATURES_ROOT,
        output_dir    = OUTPUT_DIR,
        batch_size    = BATCH_SIZE,
        num_workers   = NUM_WORKERS,
        temperature   = TEMPERATURE,
        use_tta       = USE_TTA,
        top_k_errors  = TOP_K_ERRORS,
    )
