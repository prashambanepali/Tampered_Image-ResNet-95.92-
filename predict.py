# predict.py
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
from train import build_resnet50_9ch
from precompute_maps import compute_ela_multiscale, compute_noise_3ch

CLASS_NAMES = ["authentic", "copy_move", "enhancement", "removal_inpainting", "splicing"]

# ── Config ────────────────────────────────────────────────────────────
IMAGE_PATH  = r"C:\Tampered_9channel\Unseen_coco_images\COCO_DF_E000B05114_00750983.jpg"
MODEL_PATH  = "outputs0/swa_model.pth"

# Temperature scaling — controls confidence sharpness
#   T = 1.0  → raw model output (soft, low confidence)
#   T = 0.5  → sharper predictions (recommended start)
#   T = 0.3  → very sharp, high confidence
# Lower T if predictions are correct but confidence is too low.
# Raise T if model is overconfident on wrong predictions.
TEMPERATURE = 0.5
# ─────────────────────────────────────────────────────────────────────


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


def predict(image_path, model_path, temperature=1.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device      : {device}")
    print(f"Model       : {model_path}")
    print(f"Temperature : {temperature}")

    model = load_model(model_path, device)

    normalize_9 = transforms.Normalize(
        mean=[0.485, 0.456, 0.406, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        std =[0.229, 0.224, 0.225, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    )
    size = (224, 224)

    img   = Image.open(image_path).convert("RGB")
    ela   = compute_ela_multiscale(img)
    noise = compute_noise_3ch(img)

    rgb_t   = TF.to_tensor(TF.resize(img,   size, InterpolationMode.BILINEAR))
    ela_t   = TF.to_tensor(TF.resize(ela,   size, InterpolationMode.BILINEAR))
    noise_t = TF.to_tensor(TF.resize(noise, size, InterpolationMode.BILINEAR))

    x = normalize_9(torch.cat([rgb_t, ela_t, noise_t], dim=0)).unsqueeze(0).to(device)

    flips = [
        x,
        torch.flip(x, dims=[3]),
        torch.flip(x, dims=[2]),
        torch.flip(x, dims=[2, 3]),
    ]

    with torch.no_grad():
        # Collect raw logits (before softmax) for all TTA passes
        logits_list = [model(f) for f in flips]
        # Average logits first, THEN apply temperature + softmax
        # (more correct than averaging softmax outputs)
        avg_logits = torch.stack(logits_list).mean(0)
        probs = torch.softmax(avg_logits / temperature, dim=1)[0].cpu().tolist()

    pred  = CLASS_NAMES[probs.index(max(probs))]
    conf  = max(probs)

    print(f"\nImage  : {image_path}")
    print("=" * 52)
    for name, p in zip(CLASS_NAMES, probs):
        bar    = "█" * int(p * 40)
        marker = " ◄" if name == pred else ""
        print(f"  {name:22s}: {p:.4f}  {bar}{marker}")
    print("=" * 52)
    print(f"  Prediction : {pred}")
    print(f"  Confidence : {conf:.2%}")
    print(f"  (Temperature={temperature} — lower = sharper)\n")


if __name__ == "__main__":
    predict(IMAGE_PATH, MODEL_PATH, temperature=TEMPERATURE)