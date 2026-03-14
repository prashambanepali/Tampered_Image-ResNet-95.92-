"""
train.py — 9-Channel Anti-bias version
  Input: RGB(3) + ELA(3) + Noise(3) = 9 channels

Anti-bias features:
  ✓ Training accuracy displayed live every epoch
  ✓ MixUp + CutMix
  ✓ DropBlock2D on layer3/layer4
  ✓ Test-Time Augmentation (TTA) for validation
  ✓ Strong augmentation without aggressive crops
  ✓ Label smoothing (0.10) in Focal Loss
  ✓ SWA (Stochastic Weight Averaging) in final epochs
  ✓ Layer-wise learning rate decay
  ✓ Per-epoch CSV log (epoch, train_loss, train_acc, val_loss, val_acc)
"""

import os
import copy
import random
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from dataset import (
    TamperedDataset, PairedAugment, PairedAugConfig,
    mixup_data, mixup_criterion, cutmix_data,
)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────
# CBAM
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


# ─────────────────────────────────────────────
# DropBlock2D
# ─────────────────────────────────────────────
class DropBlock2D(nn.Module):
    def __init__(self, block_size=7, drop_prob=0.10):
        super().__init__()
        self.block_size = block_size
        self.drop_prob  = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0:
            return x
        B, C, H, W   = x.shape
        mask_h        = H - self.block_size + 1
        mask_w        = W - self.block_size + 1
        seed_rate     = (self.drop_prob / self.block_size ** 2
                         * H * W / (mask_h * mask_w))
        mask = torch.bernoulli(
            torch.ones(B, C, mask_h, mask_w, device=x.device) * seed_rate)
        mask = F.max_pool2d(mask, (self.block_size, self.block_size),
                            stride=1, padding=self.block_size // 2)
        if mask.shape[-2:] != x.shape[-2:]:
            mask = F.interpolate(mask, size=x.shape[-2:])
        mask = 1 - mask
        return x * mask * (mask.numel() / (mask.sum() + 1e-6))


# ─────────────────────────────────────────────
# Model — 9-channel input
# ─────────────────────────────────────────────
def build_resnet50_9ch(num_classes=5, pretrained=True, drop_prob=0.10):
    """
    ResNet-50 with 9-channel first conv.

    Weight initialization for extra channels:
      ch 0-2  (RGB)   ← pretrained ImageNet weights
      ch 3-5  (ELA)   ← mean of pretrained RGB weights  (smooth start)
      ch 6-8  (Noise) ← mean of pretrained RGB weights  (smooth start)

    All extra channels initialized identically so the model can gradually
    learn to use forensic features without disrupting RGB features early on.
    """
    weights   = ResNet50_Weights.DEFAULT if pretrained else None
    model     = models.resnet50(weights=weights)
    old_conv  = model.conv1                       # (64, 3, 7, 7)

    model.conv1 = nn.Conv2d(
        9, old_conv.out_channels,
        old_conv.kernel_size, old_conv.stride,
        old_conv.padding, bias=False
    )
    with torch.no_grad():
        model.conv1.weight[:, :3]  = old_conv.weight            # RGB ← pretrained
        mean_w = old_conv.weight.mean(dim=1, keepdim=True)      # (64,1,7,7)
        model.conv1.weight[:, 3:6] = mean_w.repeat(1, 3, 1, 1) # ELA
        model.conv1.weight[:, 6:9] = mean_w.repeat(1, 3, 1, 1) # Noise

    model.layer3 = nn.Sequential(
        model.layer3, CBAM(1024), DropBlock2D(block_size=7, drop_prob=drop_prob))
    model.layer4 = nn.Sequential(
        model.layer4, CBAM(2048), DropBlock2D(block_size=5, drop_prob=drop_prob))

    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(2048, num_classes),
    )
    return model


def freeze_backbone(model):
    for p in model.parameters():  p.requires_grad = False
    for p in model.conv1.parameters(): p.requires_grad = True
    for p in model.fc.parameters():    p.requires_grad = True


def unfreeze_all(model):
    for p in model.parameters(): p.requires_grad = True


def get_layer_wise_params(model, base_lr, decay=0.65):
    """Layer-wise LR decay — earlier layers get smaller LR for better generalization."""
    return [
        {"params": list(model.conv1.parameters()),  "lr": base_lr * (decay ** 4)},
        {"params": list(model.bn1.parameters()),    "lr": base_lr * (decay ** 4)},
        {"params": list(model.layer1.parameters()), "lr": base_lr * (decay ** 3)},
        {"params": list(model.layer2.parameters()), "lr": base_lr * (decay ** 2)},
        {"params": list(model.layer3.parameters()), "lr": base_lr * (decay ** 1)},
        {"params": list(model.layer4.parameters()), "lr": base_lr * (decay ** 0)},
        {"params": list(model.fc.parameters()),     "lr": base_lr * 2.0},
    ]


# ─────────────────────────────────────────────
# Focal Loss
# ─────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, smoothing=0.10):
        super().__init__()
        self.gamma     = gamma
        self.smoothing = smoothing
        if alpha is None:
            alpha = torch.tensor([1.0, 2.5, 2.0, 1.0, 1.2], dtype=torch.float32)
        elif not isinstance(alpha, torch.Tensor):
            alpha = torch.tensor(alpha, dtype=torch.float32)
        self.register_buffer("alpha", alpha.float())

    def forward(self, logits, targets):
        n        = logits.size(1)
        log_prob = F.log_softmax(logits, dim=1)
        with torch.no_grad():
            smooth = torch.full_like(log_prob, self.smoothing / (n - 1))
            smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        ce   = -(smooth * log_prob).sum(dim=1)
        pt   = torch.exp(-ce)
        loss = self.alpha[targets] * (1 - pt) ** self.gamma * ce
        return loss.mean()


# ─────────────────────────────────────────────
# Early Stopping
# ─────────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience; self.min_delta = min_delta
        self.best = -float("inf"); self.bad = 0

    def step(self, v):
        if v > self.best + self.min_delta:
            self.best = v; self.bad = 0; return False
        self.bad += 1
        return self.bad >= self.patience


# ─────────────────────────────────────────────
# Logger
# ─────────────────────────────────────────────
class TrainingLogger:
    def __init__(self, path="outputs0/training_log.csv"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        with open(path, "w") as f:
            f.write("epoch,train_loss,train_acc,val_loss,val_acc\n")

    def log(self, epoch, train_loss, train_acc, val_loss, val_acc):
        with open(self.path, "a") as f:
            f.write(f"{epoch},{train_loss:.6f},{train_acc:.6f},"
                    f"{val_loss:.6f},{val_acc:.6f}\n")
        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f} | "
            f"Val Loss:   {val_loss:.4f}  Val Acc:   {val_acc:.4f}"
        )


# ─────────────────────────────────────────────
# Evaluate
# ─────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total = correct = 0; loss_sum = 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with autocast('cuda'):
            logits = model(x)
            loss   = criterion(logits, y)
        loss_sum += loss.item()
        correct  += (logits.argmax(1) == y).sum().item()
        total    += y.size(0)
    return correct / max(1, total), loss_sum / max(1, len(loader))


@torch.no_grad()
def evaluate_with_tta(model, loader, device, criterion, n_tta=4):
    """TTA over original + hflip + vflip + both-flips."""
    model.eval()
    total = correct = 0; loss_sum = 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        all_probs = []
        for t in range(n_tta):
            xa = x.clone()
            if t == 1: xa = torch.flip(xa, dims=[3])
            if t == 2: xa = torch.flip(xa, dims=[2])
            if t == 3: xa = torch.flip(xa, dims=[2, 3])
            with autocast('cuda'):
                all_probs.append(F.softmax(model(xa), dim=1))
        avg = torch.stack(all_probs).mean(0)
        with autocast('cuda'):
            loss = criterion(model(x), y)
        loss_sum += loss.item()
        correct  += (avg.argmax(1) == y).sum().item()
        total    += y.size(0)
    return correct / max(1, total), loss_sum / max(1, len(loader))


# ─────────────────────────────────────────────
# Train one epoch
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, scheduler, scaler, criterion,
                    device, epoch, num_epochs,
                    mixup_alpha=0.4, cutmix_alpha=1.0, mix_prob=0.5):
    model.train()
    running_loss = total_correct = total_samples = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)
    for x, y in pbar:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if random.random() < mix_prob:
            if random.random() < 0.5:
                x_m, ya, yb, lam = mixup_data(x, y, mixup_alpha)
            else:
                x_m, ya, yb, lam = cutmix_data(x, y, cutmix_alpha)
            with autocast('cuda'):
                logits = model(x_m)
                loss   = mixup_criterion(criterion, logits, ya, yb, lam)
            pred = logits.argmax(1)
            batch_correct = (lam * (pred == ya).float()
                             + (1 - lam) * (pred == yb).float()).sum()
        else:
            with autocast('cuda'):
                logits = model(x)
                loss   = criterion(logits, y)
            pred = logits.argmax(1)
            batch_correct = (pred == y).sum().float()

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer); scaler.update(); scheduler.step()

        running_loss  += loss.item()
        total_correct += batch_correct.item()
        total_samples += y.size(0)

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{total_correct / max(1, total_samples):.4f}",
            lr=f"{scheduler.get_last_lr()[0]:.2e}",
        )

    return running_loss / max(1, len(loader)), total_correct / max(1, total_samples)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    set_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.backends.cudnn.benchmark        = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {torch.cuda.get_device_name(0)}  VRAM: {vram:.1f} GB")

    # ── Config ────────────────────────────────────────────────────────
    num_classes   = 5
    image_size    = (224, 224)
    batch_size    = 32
    num_workers   = 4
    num_epochs    = 40
    freeze_epochs = 3
    swa_start     = 30
    lr_frozen     = 1e-4
    lr_unfrozen   = 5e-5
    mixup_alpha   = 0.4
    cutmix_alpha  = 1.0
    mix_prob      = 0.50
    tta_n         = 4

    os.makedirs("outputs0", exist_ok=True)
    best_path  = "outputs0/best_model.pth"
    swa_path   = "outputs0/swa_model.pth"
    logger     = TrainingLogger("outputs0/training_log.csv")
    early_stop = EarlyStopping(patience=10)

    # ── Normalization — 9 channels ────────────────────────────────────
    # RGB  : ImageNet mean/std
    # ELA  : neutral 0.5/0.5  (no prior statistics available)
    # Noise: neutral 0.5/0.5 per channel (gaussian|srm_linear|srm_edge)
    normalize_9 = transforms.Normalize(
        mean=[0.485, 0.456, 0.406,   # RGB
              0.5,   0.5,   0.5,     # ELA
              0.5,   0.5,   0.5],    # Noise (gaussian | srm_linear | srm_edge)
        std =[0.229, 0.224, 0.225,   # RGB
              0.5,   0.5,   0.5,     # ELA
              0.5,   0.5,   0.5],    # Noise
    )

    # ── Augmentation config ───────────────────────────────────────────
    aug_cfg = PairedAugConfig(
        enable=True,
        crop_scale=(0.90, 1.00),
        crop_ratio=(0.95, 1.05),
        hflip_p=0.5,
        vflip_p=0.20,
        rotate_deg=15.0,
        rotate_90_p=0.40,
        color_jitter_p=0.70,
        jitter_brightness=0.20,
        jitter_contrast=0.20,
        jitter_saturation=0.15,
        jitter_hue=0.04,
        jpeg_p=0.60,
        jpeg_qmin=55,
        jpeg_qmax=98,
        double_jpeg_p=0.30,
        gaussian_noise_p=0.40,
        gaussian_noise_std=0.02,
        blur_p=0.25,
        blur_radius=(0.3, 1.5),
        erasing_p=0.25,
        grid_shuffle_p=0.20,
        sharpen_p=0.20,
    )
    paired_aug = PairedAugment(aug_cfg)

    # ── Datasets ──────────────────────────────────────────────────────
    train_dataset = TamperedDataset("train", image_size=image_size,
                                     normalize_transform=normalize_9,
                                     paired_augment=paired_aug)
    val_dataset   = TamperedDataset("val",   image_size=image_size,
                                     normalize_transform=normalize_9,
                                     paired_augment=None)

    # ── Balanced sampler ──────────────────────────────────────────────
    train_labels  = [lbl for (_, _, _, lbl) in train_dataset.items]
    class_counts  = torch.bincount(torch.tensor(train_labels),
                                    minlength=num_classes).float()
    class_weights = 1.0 / (class_counts + 1e-8)
    sample_weights= torch.tensor([class_weights[l] for l in train_labels])
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights),
                                     replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                               sampler=sampler, num_workers=num_workers,
                               pin_memory=True,
                               persistent_workers=(num_workers > 0))
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                               shuffle=False, num_workers=num_workers,
                               pin_memory=True,
                               persistent_workers=(num_workers > 0))

    # ── Model ─────────────────────────────────────────────────────────
    model = build_resnet50_9ch(num_classes, pretrained=True,
                                drop_prob=0.10).to(device)
    total_p = sum(p.numel() for p in model.parameters()) / 1e6
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Params total: {total_p:.1f}M  Trainable: {train_p:.1f}M")
    freeze_backbone(model)

    swa_model = AveragedModel(model)

    criterion = FocalLoss(
        alpha=[1.0, 2.5, 2.0, 1.0, 1.2],
        gamma=2.0,
        smoothing=0.10,
    ).to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr_frozen, weight_decay=2e-4)
    scheduler = OneCycleLR(optimizer, max_lr=lr_frozen,
                            steps_per_epoch=len(train_loader),
                            epochs=freeze_epochs, pct_start=0.2)
    scaler = GradScaler('cuda')

    best_val_acc = -float("inf")
    best_state: Optional[Dict] = None
    best_epoch  = -1

    print("\n" + "="*80)
    print("9-channel model: RGB(3) + ELA(3) + Noise(3)")
    print("Noise channels : gaussian_residual | srm_linear_residual | srm_edge_residual")
    print("Anti-bias      : MixUp + CutMix + DropBlock + JPEG-domain-rand + SWA + TTA")
    print("="*80 + "\n")

    for epoch in range(num_epochs):

        # ── Unfreeze ─────────────────────────────────────────────────
        if epoch == freeze_epochs:
            unfreeze_all(model)
            param_groups = get_layer_wise_params(model, lr_unfrozen, decay=0.65)
            optimizer    = torch.optim.AdamW(param_groups, weight_decay=2e-4)
            scheduler    = OneCycleLR(optimizer, max_lr=lr_unfrozen,
                                       steps_per_epoch=len(train_loader),
                                       epochs=(swa_start - epoch), pct_start=0.1)
            print(f"[Epoch {epoch+1}] Unfroze all layers (layer-wise LR decay).")

        # ── SWA phase ────────────────────────────────────────────────
        if epoch == swa_start:
            scheduler     = CosineAnnealingLR(optimizer,
                                               T_max=(num_epochs - swa_start),
                                               eta_min=1e-7)
            swa_scheduler = SWALR(optimizer, swa_lr=5e-6, anneal_epochs=5)
            print(f"[Epoch {epoch+1}] SWA phase started.")

        # ── Train ─────────────────────────────────────────────────────
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, criterion,
            device, epoch, num_epochs,
            mixup_alpha=mixup_alpha, cutmix_alpha=cutmix_alpha,
            mix_prob=mix_prob,
        )

        if epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()

        # ── Validate ──────────────────────────────────────────────────
        use_tta = (tta_n > 1) and (epoch >= freeze_epochs)
        if use_tta:
            val_acc, val_loss = evaluate_with_tta(
                model, val_loader, device, criterion, n_tta=tta_n)
        else:
            val_acc, val_loss = evaluate(model, val_loader, device, criterion)

        logger.log(epoch + 1, train_loss, train_acc, val_loss, val_acc)

        # ── Save best ─────────────────────────────────────────────────
        if val_acc > best_val_acc + 1e-8:
            best_val_acc = val_acc; best_epoch = epoch + 1
            best_state   = copy.deepcopy(model.state_dict())
            torch.save(best_state, best_path)
            print(f"  ✓ Saved best model  (val_acc={val_acc:.4f})")

        if early_stop.step(val_acc):
            print(f"Early stopping at epoch {epoch+1}. Best: {best_val_acc:.4f}")
            break

    # ── SWA: update BatchNorm and save ────────────────────────────────
    print("\nUpdating SWA BatchNorm statistics on training data …")
    update_bn(train_loader, swa_model, device=device)
    torch.save(swa_model.state_dict(), swa_path)
    print(f"SWA model → {swa_path}")

    swa_acc, _ = evaluate_with_tta(swa_model, val_loader, device,
                                    criterion, n_tta=tta_n)
    print(f"\nSWA  Val Acc (TTA x{tta_n}): {swa_acc:.4f}")
    print(f"Best Val Acc             : {best_val_acc:.4f}  (epoch {best_epoch})")
    print(f"Log saved                : outputs0/training_log.csv")


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
