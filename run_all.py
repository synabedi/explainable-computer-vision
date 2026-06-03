"""
run_all.py  –  Master runner
─────────────────────────────
Runs every phase in order:
  1. eval_only.py     → test metrics, confusion matrix, per-class bar chart
  2. explain.py       → Grad-CAM + Guided Backprop for all 3 classes
  3. evaluate_xai.py  → Deletion, Insertion, Entropy, AOPC table
  4. bonus_task.py    → Multi-baseline OOD deletion study

NOTE: Training was already done; best_densenet.pth is included.
      To retrain from scratch, run:  python train.py
      (requires ~4-6 hours on CPU or ~20 min on GPU)

Usage:
  python run_all.py
"""
import subprocess, sys, os

scripts = [
    ("Phase 1 – Test Evaluation",    "eval_only.py"),
    ("Phase 2 – Explainability",     "explain.py"),
    ("Phase 3 – XAI Evaluation",     "evaluate_xai.py"),
    ("Phase 4 – Bonus Task",         "bonus_task.py"),
]

base = os.path.dirname(os.path.abspath(__file__))

for title, script in scripts:
    print(f"\n{'='*60}\n  {title}\n{'='*60}\n")
    r = subprocess.run([sys.executable, os.path.join(base, script)], cwd=base)
    if r.returncode != 0:
        print(f"\n❌  {script} failed (exit {r.returncode})")
        sys.exit(r.returncode)

print("\n" + "="*60)
print("  ALL PHASES COMPLETE – outputs are in outputs/")
print("="*60 + "\n")
