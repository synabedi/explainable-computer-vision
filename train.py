import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                              classification_report, confusion_matrix)
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from dataset import ChestXRayDataset, get_transforms

# ── paths (absolute so the script runs from any cwd) ─────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR  = os.path.join(BASE_DIR, "..", "COVID_19_dataset", "train")
VAL_DIR    = os.path.join(BASE_DIR, "..", "COVID_19_dataset", "val")
TEST_DIR   = os.path.join(BASE_DIR, "..", "COVID_19_dataset", "test")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_NAMES = ["COVID", "Normal", "Viral Pneumonia"]


def build_model(num_classes=3):
    model = models.densenet121(weights=None)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )
    return model


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, all_preds, all_labels = 0.0, [], []
    for images, labels in tqdm(dataloader, desc="  Train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return running_loss / len(dataloader.dataset), accuracy_score(all_labels, all_preds)


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss, all_preds, all_labels = 0.0, [], []
    for images, labels in tqdm(dataloader, desc="  Eval ", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        running_loss += criterion(outputs, labels).item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0)
    return running_loss / len(dataloader.dataset), acc, prec, rec, f1, all_preds, all_labels


def plot_training_curves(history, save_path):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Training History – DenseNet-121", fontsize=13, fontweight='bold')

    axes[0].plot(epochs, history['train_loss'], 'b-o', label='Train Loss')
    axes[0].plot(epochs, history['val_loss'],   'r-o', label='Val Loss')
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch")
    axes[0].legend(); axes[0].grid(True)

    axes[1].plot(epochs, history['train_acc'], 'b-o', label='Train Acc')
    axes[1].plot(epochs, history['val_acc'],   'r-o', label='Val Acc')
    axes[1].plot(epochs, history['val_f1'],    'g-s', label='Val F1')
    axes[1].set_title("Accuracy & F1"); axes[1].set_xlabel("Epoch")
    axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=120)
    plt.close()
    print(f"  Saved → {save_path}")


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
                    color='white' if cm[i, j] > thresh else 'black', fontsize=12)
    ax.set_ylabel('True Label'); ax.set_xlabel('Predicted Label')
    ax.set_title('Confusion Matrix – Test Set', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=120)
    plt.close()
    print(f"  Saved → {save_path}")


def main():
    BATCH_SIZE = 32
    EPOCHS     = 10
    LR         = 1e-4
    DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥  Device: {DEVICE}")

    train_tf, val_tf = get_transforms()
    train_ds = ChestXRayDataset(TRAIN_DIR, transform=train_tf)
    val_ds   = ChestXRayDataset(VAL_DIR,   transform=val_tf)
    test_ds  = ChestXRayDataset(TEST_DIR,  transform=val_tf)
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model     = build_model(num_classes=3).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    history = dict(train_loss=[], train_acc=[], val_loss=[], val_acc=[], val_f1=[])
    best_f1 = 0.0

    print("\n🚀 Training …")
    for epoch in range(EPOCHS):
        print(f"\n  Epoch {epoch+1}/{EPOCHS}")
        tl, ta = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        vl, va, vp, vr, vf, _, _ = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step()

        history['train_loss'].append(tl); history['train_acc'].append(ta)
        history['val_loss'].append(vl);   history['val_acc'].append(va)
        history['val_f1'].append(vf)

        print(f"  Train  Loss={tl:.4f}  Acc={ta:.4f}")
        print(f"  Val    Loss={vl:.4f}  Acc={va:.4f}  Prec={vp:.4f}  Rec={vr:.4f}  F1={vf:.4f}")

        if vf > best_f1:
            best_f1 = vf
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_densenet.pth"))
            print("  ✅ Saved new best checkpoint")

    plot_training_curves(history, os.path.join(OUTPUT_DIR, "training_curves.png"))

    # ── Test evaluation ───────────────────────────────────────────────────────
    print("\n🔬 Evaluating on test set …")
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_densenet.pth"),
                                     map_location=DEVICE))
    _, test_acc, test_prec, test_rec, test_f1, test_preds, test_labels = \
        evaluate(model, test_loader, criterion, DEVICE)

    print(f"\n  Test Acc={test_acc:.4f}  Prec={test_prec:.4f}  Rec={test_rec:.4f}  F1={test_f1:.4f}")
    print("\n  Classification Report:")
    print(classification_report(test_labels, test_preds, target_names=CLASS_NAMES))

    cm = confusion_matrix(test_labels, test_preds)
    plot_confusion_matrix(cm, CLASS_NAMES, os.path.join(OUTPUT_DIR, "confusion_matrix.png"))

    # Save metrics to txt for easy copy-paste into report
    with open(os.path.join(OUTPUT_DIR, "test_metrics.txt"), "w") as f:
        f.write(f"Test Accuracy  : {test_acc:.4f}\n")
        f.write(f"Test Precision : {test_prec:.4f}\n")
        f.write(f"Test Recall    : {test_rec:.4f}\n")
        f.write(f"Test F1        : {test_f1:.4f}\n\n")
        f.write(classification_report(test_labels, test_preds, target_names=CLASS_NAMES))
    print(f"  Saved → {OUTPUT_DIR}/test_metrics.txt")
    print("\n✅ Training complete.\n")


if __name__ == "__main__":
    main()
