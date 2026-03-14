"""
split_dataset.py
Splits features/ → dataset/train, dataset/val, dataset/test

Reads filenames from features/<class>/rgb/, shuffles, then writes
a plain-text index.txt per split — NO file copying, instant re-split.

To change ratio later:
  1. Edit SPLITS below
  2. Delete dataset/ folder
  3. Re-run: python split_dataset.py
"""

import os
import random

# ── Config ────────────────────────────────────────────────────────────
FEATURES_ROOT = "Features1"
DATASET_ROOT  = "dataset"

SPLITS = {"train": 0.80, "val": 0.10, "test": 0.10}
SEED   = 42

CLASSES = [
    "authentic",
    "copy_move",
    "enhancement",
    "removal_inpainting",
    "splicing",
]
IMG_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".bmp")
# ─────────────────────────────────────────────────────────────────────


def split_class(class_name: str):
    rgb_dir = os.path.join(FEATURES_ROOT, class_name, "rgb")
    if not os.path.isdir(rgb_dir):
        print(f"  [SKIP] {rgb_dir} not found — run precompute_maps.py first")
        return

    files = sorted([f for f in os.listdir(rgb_dir) if f.lower().endswith(IMG_EXTS)])
    if not files:
        print(f"  [SKIP] No images in {rgb_dir}")
        return

    random.shuffle(files)
    n     = len(files)
    n_tr  = int(n * SPLITS["train"])
    n_val = int(n * SPLITS["val"])

    split_files = {
        "train": files[:n_tr],
        "val":   files[n_tr: n_tr + n_val],
        "test":  files[n_tr + n_val:],
    }

    print(f"\n  {class_name:20s}  total={n}")
    for split_name, flist in split_files.items():
        index_dir  = os.path.join(DATASET_ROOT, split_name, class_name)
        os.makedirs(index_dir, exist_ok=True)
        index_path = os.path.join(index_dir, "index.txt")
        with open(index_path, "w") as f:
            f.write("\n".join(flist))
        print(f"    {split_name:5s}: {len(flist):5d}  →  {index_path}")


def print_summary():
    print("\n─── Split Summary ───")
    print(f"  {'Class':20s}  {'train':>7}  {'val':>7}  {'test':>7}  {'total':>7}")
    print("  " + "-" * 56)
    grand = {"train": 0, "val": 0, "test": 0}
    for cls in CLASSES:
        counts = {}
        for split_name in ["train", "val", "test"]:
            idx = os.path.join(DATASET_ROOT, split_name, cls, "index.txt")
            counts[split_name] = 0
            if os.path.exists(idx):
                with open(idx) as f:
                    counts[split_name] = len([l for l in f.read().splitlines() if l])
            grand[split_name] += counts[split_name]
        total = sum(counts.values())
        print(f"  {cls:20s}  {counts['train']:7d}  {counts['val']:7d}  "
              f"{counts['test']:7d}  {total:7d}")
    print("  " + "-" * 56)
    g_total = sum(grand.values())
    print(f"  {'TOTAL':20s}  {grand['train']:7d}  {grand['val']:7d}  "
          f"{grand['test']:7d}  {g_total:7d}")


def main():
    random.seed(SEED)
    print(f"Features : {os.path.abspath(FEATURES_ROOT)}")
    print(f"Output   : {os.path.abspath(DATASET_ROOT)}")
    print(f"Ratio    : train={SPLITS['train']}  val={SPLITS['val']}  test={SPLITS['test']}  (seed={SEED})")

    for cls in CLASSES:
        split_class(cls)

    print_summary()
    print(f"\nDone. Re-split anytime: edit SPLITS, delete dataset/, re-run.")
    print("Next: python train.py")


if __name__ == "__main__":
    main()
