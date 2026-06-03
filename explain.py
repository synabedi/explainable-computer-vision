"""
explain.py
──────────────────────────────────────────────────────────────────────────────
Phase 2 – Explainability
  • Grad-CAM      : coarse class-discriminative heatmap via feature gradients
  • Guided Backprop : pixel-sharp saliency via guided ReLU gradient propagation

Runs on one representative image per class and saves side-by-side figures.
"""

import os
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

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
DATA_DIR   = os.path.join(BASE_DIR, "..", "COVID_19_dataset", "train")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_NAMES = ["COVID", "Normal", "Viral Pneumonia"]


# ── Grad-CAM ─────────────────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._fwd)
        target_layer.register_full_backward_hook(self._bwd)

    def _fwd(self, m, i, o):  self.activations = o
    def _bwd(self, m, gi, go): self.gradients  = go[0]

    def generate(self, x, class_idx=None):
        self.model.eval()
        out = self.model(x)
        if class_idx is None:
            class_idx = int(torch.argmax(out, 1))
        self.model.zero_grad()
        out[0, class_idx].backward()

        g  = self.gradients.detach().cpu().numpy()[0]        # (C,H,W)
        a  = self.activations.detach().cpu().numpy()[0]
        w  = g.mean(axis=(1, 2))
        cam = (w[:, None, None] * a).sum(0)
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam -= cam.min(); cam /= (cam.max() + 1e-8)
        return cam, class_idx


# ── Guided Backpropagation ────────────────────────────────────────────────────
class GuidedBackprop:
    """
    Registers modified backward hooks on every ReLU so only positive
    gradients that arrived at a positively-activated unit pass through.
    This gives pixel-level spatial precision — unlike Grad-CAM which
    operates at the 7×7 feature-map resolution.
    """
    def __init__(self, model):
        self.model = model
        self._hooks = []
        for m in model.modules():
            if isinstance(m, nn.ReLU):
                self._hooks.append(
                    m.register_backward_hook(self._guided_relu))

    @staticmethod
    def _guided_relu(module, grad_in, grad_out):
        return (torch.clamp(grad_in[0], min=0.0),)

    def generate(self, x, class_idx=None):
        self.model.eval()
        x = x.clone().requires_grad_(True)
        out = self.model(x)
        if class_idx is None:
            class_idx = int(torch.argmax(out, 1))
        self.model.zero_grad()
        out[0, class_idx].backward()
        sal = x.grad.cpu().numpy()[0]               # (3,H,W)
        sal = np.abs(sal).mean(0)                   # (H,W)
        sal -= sal.min(); sal /= (sal.max() + 1e-8)
        self._remove()
        return sal, class_idx

    def _remove(self):
        for h in self._hooks: h.remove()
        self._hooks = []


# ── Overlay helper ────────────────────────────────────────────────────────────
def overlay(img_bgr, mask, colormap=cv2.COLORMAP_JET):
    hm = cv2.applyColorMap(np.uint8(255 * mask), colormap).astype(np.float32) / 255
    blend = hm + img_bgr.astype(np.float32) / 255
    blend /= blend.max()
    return np.uint8(255 * blend)


# ── Save 3-panel comparison ───────────────────────────────────────────────────
def save_figure(img_path, gc_mask, gbp_mask, save_path, title):
    bgr = cv2.resize(cv2.imread(img_path), (224, 224))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gc_rgb  = cv2.cvtColor(overlay(bgr, gc_mask,  cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    gbp_rgb = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * gbp_mask),
                                              cv2.COLORMAP_HOT), cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle(title, fontsize=12, fontweight='bold')
    for a, im, t in zip(ax,
                        [rgb, gc_rgb, gbp_rgb],
                        ["Original X-Ray", "Grad-CAM", "Guided Backpropagation"]):
        a.imshow(im); a.set_title(t, fontsize=11); a.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=120)
    plt.close()
    print(f"  Saved → {save_path}")


# ── Build model util (shared across scripts) ──────────────────────────────────
def load_model(device):
    m = models.densenet121(weights=None)
    nf = m.classifier.in_features
    m.classifier = nn.Sequential(
        nn.Linear(nf, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 3))
    w = os.path.join(OUTPUT_DIR, "best_densenet.pth")
    m.load_state_dict(torch.load(w, map_location=device))
    m.to(device)
    return m


def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model(DEVICE)
    _, vt  = get_transforms()

    # One image per class
    samples = {
        "COVID":           os.path.join(DATA_DIR, "COVID",          "COVID-1.png"),
        "Normal":          os.path.join(DATA_DIR, "Normal",         "Normal-10004.png"),
        "Viral Pneumonia": os.path.join(DATA_DIR, "Viral Pneumonia","Viral Pneumonia-17.png"),
    }

    print("\n🔍 Generating Grad-CAM + Guided Backpropagation …\n")
    for label, path in samples.items():
        if not os.path.exists(path):
            print(f"  ⚠ Not found: {path}"); continue
        print(f"  {label}")
        x = vt(Image.open(path).convert("RGB")).unsqueeze(0).to(DEVICE)

        gc  = GradCAM(model, model.features.norm5)
        gc_mask, pred_idx = gc.generate(x)

        gbp = GuidedBackprop(model)
        gbp_mask, _ = gbp.generate(x, class_idx=pred_idx)

        tag = label.replace(" ", "_")
        save_figure(path, gc_mask, gbp_mask,
                    save_path=os.path.join(OUTPUT_DIR, f"explain_{tag}.png"),
                    title=f"True: {label}  |  Predicted: {CLASS_NAMES[pred_idx]}")

    print("\n✅ Explanation maps saved.\n")


if __name__ == "__main__":
    main()
