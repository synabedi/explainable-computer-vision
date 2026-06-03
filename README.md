#  Explainable Computer Vision — Chest X-Ray Classification

**AIMS DTU Research Intern 2026** | *Syna Bedi (25/A06/020)*

A DenseNet-121 classifier trained on a three-class chest X-ray dataset (COVID-19 / Normal / Viral Pneumonia), with Grad-CAM and Guided Backpropagation explainability, quantitative XAI evaluation, and a novel multi-baseline deletion study as the bonus task.

---

## Results at a Glance

| Metric | COVID | Normal | Viral Pneumonia | Macro Avg |
|--------|-------|--------|-----------------|-----------|
| Precision | 0.98 | 0.97 | 0.98 | **0.977** |
| Recall | 0.98 | 0.96 | 0.99 | **0.977** |
| F1-Score | 0.98 | 0.97 | 0.99 | **0.977** |

Test set: 435 images (145 per class) · Overall accuracy: **97.7%**

---

##  Repository Structure

```
├── dataset.py          # Dataset loader + augmentation transforms
├── train.py            # Full training loop (DenseNet-121, AdamW, cosine LR)
├── eval_only.py        # Evaluate saved checkpoint → metrics + confusion matrix
├── explain.py          # Grad-CAM + Guided Backpropagation visualisation
├── evaluate_xai.py     # Quantitative XAI: entropy, deletion/insertion AOPC
├── bonus_task.py       # Multi-baseline deletion OOD study
├── run_all.py          # Master runner (phases 1–4 in order)
```

---

##  Quickstart

### 1. Clone & install dependencies

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install torch torchvision scikit-learn matplotlib opencv-python tqdm pillow
```

### 2. Download the dataset

Download the [COVID-19 Chest X-Ray dataset](https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset) and place it at:

```
../COVID_19_dataset/
    train/  COVID/  Normal/  Viral Pneumonia/
    val/    COVID/  Normal/  Viral Pneumonia/
    test/   COVID/  Normal/  Viral Pneumonia/
```

### 3. Download model weights

Download `best_densenet.pth` from the [Google Drive link](https://drive.google.com/file/d/1EciMDuzTHZ4uNrYFED1OlkSF_pVTmp6J/view?usp=sharing) and place it in `outputs/`.

### 4. Run everything

```bash
# Run all phases (eval → explain → XAI eval → bonus task)
python run_all.py

# Or run phases individually:
python eval_only.py       # Phase 1: test metrics
python explain.py         # Phase 2: saliency maps
python evaluate_xai.py    # Phase 3: deletion/insertion/entropy
python bonus_task.py      # Phase 4: multi-baseline deletion

# To retrain from scratch (~20 min GPU / 4–6 hrs CPU):
python train.py
```

---

##  Model Architecture

**DenseNet-121** backbone with a custom classification head:

```
Input (224×224 RGB)
  → Conv + BN + ReLU + Pool (Stem)
  → Dense Block 1 + Transition
  → Dense Block 2 + Transition
  → Dense Block 3 + Transition
  → Dense Block 4 + norm5  ← Grad-CAM hook
  → Global Average Pooling
  → Linear(1024 → 256) → ReLU → Dropout(0.3)
  → Linear(256 → 3)
```

**Training config:** 10 epochs · batch 32 · AdamW (lr=1e-4, wd=1e-2) · Cosine Annealing LR · Cross-Entropy loss · Best checkpoint saved by validation macro F1.

---

##  Explainability Methods

### Grad-CAM
Computes gradients of the predicted class score w.r.t. the final feature map (`model.features.norm5`), globally averages them to get channel weights, and produces a weighted heatmap. Spatially coarse (7×7 before upsampling) but strongly class-discriminative.

### Guided Backpropagation
Modifies ReLU backward hooks to pass only gradients that are both positive and entering a positively-activated unit. Produces pixel-resolution saliency maps with fine structural detail, though less class-discriminative than Grad-CAM.

---

##  Quantitative XAI Evaluation

| Metric | Grad-CAM | Guided Backprop |
|--------|----------|-----------------|
| Entropy ↓ (sharper = better) | 15.31 | **14.71** |
| Deletion AOPC ↑ | 0.877 | **0.987** |
| Insertion AOPC ↑ | **0.825** | 0.594 |

**Interpretation:** Guided Backprop pinpoints the most individually diagnostic pixels (higher deletion AOPC). Grad-CAM covers broader clinically relevant regions more effectively (higher insertion AOPC). Both are capturing genuine information at different spatial granularities.

---

## Bonus Task — Multi-Baseline Deletion (OOD Mitigation)

**Problem:** Standard deletion replaces pixels with black zeros, creating an out-of-distribution (OOD) artefact. The model was never trained on images with black holes, so confidence drops may partly reflect distribution shift rather than removal of truly informative content.

**Proposed solution:** Report deletion under three baselines and treat their spread as a diagnostic signal.

| Strategy | AOPC | Monotonicity Violations |
|----------|------|-------------------------|
| Blackout | 0.877 | 2/10 |
| Mean-Pixel | 0.877 | 2/10 |
| Smooth Blur (21×21 Gaussian) | **0.198** | 3/10 |

**Key finding:** The wide gap between Blackout (0.877) and Smooth Blur (0.198) confirms OOD inflation in the standard metric. Non-zero monotonicity violations even under smooth replacement show that no single strategy is perfectly OOD-neutral — motivating future work with learned in-painting baselines.

---

##  Outputs & Assets

All generated outputs are available on Google Drive: **[Link to Drive](https://drive.google.com/drive/folders/1YgEM8j8K8Ga3Q-1xn3tODYYAZMiNjSpj?usp=sharing)**

| File | Description |
|------|-------------|
| `best_densenet.pth` | Trained model weights |
| `confusion_matrix.png` | Test set confusion matrix |
| `per_class_metrics.png` | Per-class precision / recall / F1 bar chart |
| `explain_COVID.png` | Grad-CAM + Guided Backprop for COVID sample |
| `explain_Normal.png` | Grad-CAM + Guided Backprop for Normal sample |
| `explain_Viral_Pneumonia.png` | Grad-CAM + Guided Backprop for Viral Pneumonia sample |
| `deletion_insertion_curves.png` | Faithfulness evaluation curves |
| `metrics_comparison_table.png` | XAI method comparison table |
| `bonus_comparison_curve.png` | Multi-baseline deletion study |
| `test_metrics.txt` | Full classification report |
| `bonus_findings.txt` | Bonus task numerical results |

---

## Requirements

```
torch >= 2.0
torchvision >= 0.15
scikit-learn
matplotlib
opencv-python
tqdm
pillow
numpy
```

---

##  Report

The full project report (methodology, results, explainability analysis, limitations, and future work) is included as `EXPLAINABLE_COMPUTER_VISION.pdf`.

---

##  Future Work

- **Guided Grad-CAM** — element-wise product of Grad-CAM and Guided Backprop for pixel-sharp, class-discriminative maps
- **Vision Transformer** backbone with native attention rollout
- **SHAP pixel attribution** for Shapley value guarantees
- **Learned in-painting** as a replacement baseline to fully eliminate OOD artefacts
- **Multi-image AOPC evaluation** (mean ± std over 50–100 images) for statistically robust conclusions

---


