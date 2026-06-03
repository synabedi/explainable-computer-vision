"""
evaluate_xai.py
──────────────────────────────────────────────────────────────────────────────
Phase 3 – Quantitative XAI Evaluation

Metrics computed for BOTH Grad-CAM and Guided Backpropagation:
  1. Entropy          – sharpness / focus of saliency map (lower = more focused)
  2. Deletion AOPC    – faithfulness: does removing top pixels kill confidence?
  3. Insertion AOPC   – faithfulness: does inserting top pixels restore confidence?

Outputs
  outputs/deletion_insertion_curves.png  – side-by-side Deletion & Insertion plots
  outputs/metrics_comparison_table.png   – summary table
"""

import os, sys
import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import models

from dataset import get_transforms
from explain import GradCAM, GuidedBackprop, load_model, CLASS_NAMES

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
DATA_DIR   = os.path.join(BASE_DIR, "..", "COVID_19_dataset", "train")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SAMPLE_PATH = os.path.join(DATA_DIR, "Viral Pneumonia", "Viral Pneumonia-17.png")


# ── Metrics ───────────────────────────────────────────────────────────────────
def entropy(mask):
    f = mask.flatten(); s = f.sum()
    if s == 0: return 0.0
    p = f / s; p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def _sorted_indices(mask):
    return np.argsort(mask.flatten())[::-1]


def deletion_test(model, x, mask, device, steps=10):
    model.eval()
    idx   = _sorted_indices(mask)
    step  = len(idx) // steps
    mod   = x.clone()
    scores = []
    with torch.no_grad():
        out = torch.softmax(model(mod), 1)
        cls = int(torch.argmax(out, 1))
        scores.append(float(out[0, cls]))
        for i in range(steps):
            for k in idx[i*step:(i+1)*step]:
                h, w = divmod(int(k), 224)
                mod[0, :, h, w] = 0.0
            scores.append(float(torch.softmax(model(mod), 1)[0, cls]))
    return scores, cls


def insertion_test(model, x, mask, device, steps=10):
    """Start from blurred image; progressively reveal most salient pixels."""
    model.eval()
    img_np  = x[0].cpu().numpy().transpose(1, 2, 0)
    blur_np = cv2.GaussianBlur(img_np, (51, 51), 0)
    blur_t  = torch.tensor(blur_np.transpose(2, 0, 1),
                            dtype=torch.float32).unsqueeze(0).to(device)

    idx   = _sorted_indices(mask)
    step  = len(idx) // steps
    mod   = blur_t.clone()
    scores = []

    with torch.no_grad():
        # class determined by original image
        cls = int(torch.argmax(torch.softmax(model(x), 1), 1))
        scores.append(float(torch.softmax(model(mod), 1)[0, cls]))
        for i in range(steps):
            for k in idx[i*step:(i+1)*step]:
                h, w = divmod(int(k), 224)
                mod[0, :, h, w] = x[0, :, h, w]
            scores.append(float(torch.softmax(model(mod), 1)[0, cls]))
    return scores, cls


def aopc(scores):
    """Deletion AOPC: initial confidence minus mean over removal steps (higher = better saliency)."""
    return float(scores[0] - np.mean(scores[1:]))

def insertion_aopc(scores):
    """Insertion AOPC: mean confidence over insertion steps minus baseline (higher = better saliency)."""
    return float(np.mean(scores[1:]) - scores[0])


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_curves(del_gc, ins_gc, del_gbp, ins_gbp, save_path):
    x = np.linspace(0, 100, len(del_gc))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Faithfulness Evaluation: Deletion & Insertion Curves",
                 fontsize=13, fontweight='bold')

    axes[0].plot(x, del_gc,  'o-', color='red',    lw=2, label='Grad-CAM')
    axes[0].plot(x, del_gbp, 's-', color='purple', lw=2, label='Guided Backprop')
    axes[0].set_title("Deletion  (lower AUC = better)"); axes[0].set_xlabel("% Pixels Removed")
    axes[0].set_ylabel("Confidence"); axes[0].legend(); axes[0].grid(True)

    axes[1].plot(x, ins_gc,  'o-', color='green', lw=2, label='Grad-CAM')
    axes[1].plot(x, ins_gbp, 's-', color='blue',  lw=2, label='Guided Backprop')
    axes[1].set_title("Insertion  (higher AUC = better)"); axes[1].set_xlabel("% Pixels Inserted")
    axes[1].set_ylabel("Confidence"); axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=120)
    plt.close()
    print(f"  Saved → {save_path}")


def plot_table(results, save_path):
    methods = list(results.keys())
    rows    = ['Entropy\n(↓ sharper)', 'Deletion AOPC\n(↑ better)', 'Insertion AOPC\n(↑ better)']
    data    = [
        [f"{results[m]['entropy']:.4f}"  for m in methods],
        [f"{results[m]['del_aopc']:.4f}" for m in methods],
        [f"{results[m]['ins_aopc']:.4f}" for m in methods],
    ]
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.axis('off')
    tbl = ax.table(cellText=data, rowLabels=rows, colLabels=methods,
                   cellLoc='center', loc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1.4, 2.2)
    ax.set_title("XAI Method Comparison – Grad-CAM vs Guided Backpropagation",
                 fontsize=12, fontweight='bold', pad=18)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=120)
    plt.close()
    print(f"  Saved → {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model(DEVICE)
    _, vt  = get_transforms()

    if not os.path.exists(SAMPLE_PATH):
        print(f"⚠  Sample image not found: {SAMPLE_PATH}"); sys.exit(1)

    x = vt(Image.open(SAMPLE_PATH).convert("RGB")).unsqueeze(0).to(DEVICE)

    print("\n🔬 Phase 3: Generating saliency masks …")
    gc          = GradCAM(model, model.features.norm5)
    gc_mask, _  = gc.generate(x)

    gbp          = GuidedBackprop(model)
    gbp_mask, _  = gbp.generate(x)

    print("  Computing entropy …")
    ent_gc  = entropy(gc_mask)
    ent_gbp = entropy(gbp_mask)
    print(f"    Grad-CAM entropy        : {ent_gc:.4f}")
    print(f"    Guided Backprop entropy : {ent_gbp:.4f}")

    print("  Running Deletion tests …")
    del_gc,  _ = deletion_test(model, x, gc_mask,  DEVICE)
    del_gbp, _ = deletion_test(model, x, gbp_mask, DEVICE)

    print("  Running Insertion tests …")
    ins_gc,  _ = insertion_test(model, x, gc_mask,  DEVICE)
    ins_gbp, _ = insertion_test(model, x, gbp_mask, DEVICE)

    a_del_gc  = aopc(del_gc);           a_del_gbp = aopc(del_gbp)
    a_ins_gc  = insertion_aopc(ins_gc); a_ins_gbp = insertion_aopc(ins_gbp)

    print(f"\n  Grad-CAM        Del AOPC={a_del_gc:.4f}  Ins AOPC={a_ins_gc:.4f}")
    print(f"  Guided Backprop Del AOPC={a_del_gbp:.4f}  Ins AOPC={a_ins_gbp:.4f}")

    print("\n  Saving plots …")
    plot_curves(del_gc, ins_gc, del_gbp, ins_gbp,
                os.path.join(OUTPUT_DIR, "deletion_insertion_curves.png"))
    plot_table({
        "Grad-CAM":        {"entropy": ent_gc,  "del_aopc": a_del_gc,  "ins_aopc": a_ins_gc},
        "Guided Backprop": {"entropy": ent_gbp, "del_aopc": a_del_gbp, "ins_aopc": a_ins_gbp},
    }, os.path.join(OUTPUT_DIR, "metrics_comparison_table.png"))

    print("\n✅ Evaluation complete.\n")


if __name__ == "__main__":
    main()
