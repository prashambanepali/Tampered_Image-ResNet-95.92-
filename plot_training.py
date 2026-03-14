import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("./outputs0/training_log.csv")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Training History", fontsize=15, fontweight="bold")

# --- Accuracy Plot ---
ax1 = axes[0]
ax1.plot(df["epoch"], df["train_acc"], label="Train Accuracy", color="#2196F3", linewidth=2, marker="o", markersize=3)
ax1.plot(df["epoch"], df["val_acc"],   label="Val Accuracy",   color="#FF5722", linewidth=2, marker="o", markersize=3)
ax1.set_title("Epoch vs Accuracy")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Accuracy")
ax1.legend()
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.set_xlim(df["epoch"].min(), df["epoch"].max())

# --- Loss Plot ---
ax2 = axes[1]
ax2.plot(df["epoch"], df["train_loss"], label="Train Loss", color="#2196F3", linewidth=2, marker="o", markersize=3)
ax2.plot(df["epoch"], df["val_loss"],   label="Val Loss",   color="#FF5722", linewidth=2, marker="o", markersize=3)
ax2.set_title("Epoch vs Loss")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")
ax2.legend()
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.set_xlim(df["epoch"].min(), df["epoch"].max())

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150, bbox_inches="tight")
print("Saved: training_curves.png")
plt.show()
