"""
eval_only.py
──────────────────────────────────────────────────────────────────────────────
Uses the existing best_densenet.pth checkpoint to:
  - Evaluate on the test set (accuracy, precision, recall, F1, classification report)
  - Plot confusion matrix
  - Save test_metrics.txt for the report

Skips re-training since the checkpoint was trained for 10 epochs already.
"""

import os, sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                              classification_report, confusion_matrix)
from tqdm import tqdm

from dataset import ChestXRayDataset, get_transforms

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
TEST_DIR   = os.path.join(BASE_DIR, "..", "COVID_19_dataset", "test")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_NAMES = ["COVID", "Normal", "Viral Pneumonia"]


def load_model(device):
    m = models.densenet121(weights=None)
    nf = m.classifier.in_features
    m.classifier = nn.Sequential(
        nn.Linear(nf, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 3))
    w = os.path.join(OUTPUT_DIR, "best_densenet.pth")
    m.load_state_dict(torch.load(w, map_location=device))
    m.to(device)
    return m


def plot_confusion_matrix(cm, class_names, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks); ax.set_xticklabels(class_names, rotation=30, ha='right')
    ax.set_yticks(ticks); ax.set_yticklabels(class_names)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black', fontsize=13)
    ax.set_ylabel('True Label'); ax.set_xlabel('Predicted Label')
    ax.set_title('Confusion Matrix – Test Set', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=120)
    plt.close()
    print(f"  Saved → {save_path}")


def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥  Device: {DEVICE}")

    model = load_model(DEVICE)
    model.eval()

    _, val_tf = get_transforms()
    test_ds   = ChestXRayDataset(TEST_DIR, transform=val_tf)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=1)
    print(f"  Test set: {len(test_ds)} images\n")

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="  Evaluating"):
            images = images.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    acc  = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0)

    print(f"\n  ── Test Results ──────────────────────────────")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"\n  Classification Report:")
    report = classification_report(all_labels, all_preds, target_names=CLASS_NAMES)
    print(report)

    # Save metrics
    with open(os.path.join(OUTPUT_DIR, "test_metrics.txt"), "w") as f:
        f.write(f"Test Accuracy  : {acc:.4f}\n")
        f.write(f"Test Precision : {prec:.4f}\n")
        f.write(f"Test Recall    : {rec:.4f}\n")
        f.write(f"Test F1-Score  : {f1:.4f}\n\n")
        f.write(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))
    print(f"  Saved → {OUTPUT_DIR}/test_metrics.txt")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, CLASS_NAMES,
                          os.path.join(OUTPUT_DIR, "confusion_matrix.png"))

    # Per-class bar chart
    report_dict = {}
    from sklearn.metrics import classification_report as cr
    import json
    rd = cr(all_labels, all_preds, target_names=CLASS_NAMES, output_dict=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(CLASS_NAMES))
    w = 0.25
    ax.bar(x - w,   [rd[c]['precision'] for c in CLASS_NAMES], w, label='Precision', color='steelblue')
    ax.bar(x,       [rd[c]['recall']    for c in CLASS_NAMES], w, label='Recall',    color='seagreen')
    ax.bar(x + w,   [rd[c]['f1-score']  for c in CLASS_NAMES], w, label='F1-Score',  color='coral')
    ax.set_xticks(x); ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylim(0, 1.15); ax.set_ylabel("Score"); ax.legend()
    ax.set_title(f"Per-Class Metrics – Test Set  (Macro F1={f1:.4f})",
                 fontsize=12, fontweight='bold')
    ax.grid(axis='y', ls='--', alpha=0.5)
    for i, c in enumerate(CLASS_NAMES):
        for j, (val, offset) in enumerate([(rd[c]['precision'], -w),
                                            (rd[c]['recall'],    0),
                                            (rd[c]['f1-score'],  w)]):
            ax.text(i + offset, val + 0.02, f"{val:.2f}", ha='center',
                    va='bottom', fontsize=8, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "per_class_metrics.png"),
                bbox_inches='tight', dpi=120)
    plt.close()
    print(f"  Saved → {OUTPUT_DIR}/per_class_metrics.png")
    print("\n✅ Test evaluation complete.\n")


if __name__ == "__main__":
    main()
