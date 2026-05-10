"""
Evaluation module for HybSwinEff.

Provides full test-set evaluation for both the MLP head (gradient-trained)
and the ELM head (analytically fitted), including:
    - Per-class precision / recall / F1 / support table
    - Macro and weighted average metrics
    - ROC-AUC curves (one-vs-rest)
    - Confusion matrices
    - Per-image inference timing

Entry point:
    python -m Model.src.test

Author : Rashel Mahmud Rabbi
Date   : November 2025
"""

import sys
import time
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

from .HybSwinEff import HybSwinEffELM
from .preprocessing import BloodSmearDataset, create_train_val_test_split, get_transforms
from .utils import (
    Config,
    set_seed,
    style_dataframe,
    plot_multiclass_confusion,
    plot_multiclass_roc,
)

if 'IPython' in sys.modules:
    from IPython.display import display as _display
else:
    _display = lambda df: print(df.to_string(index=False))


# ---------------------------------------------------------------------------
# Classification report table
# ---------------------------------------------------------------------------

def create_classification_report_table(
    y_true,
    y_pred,
    class_names: List[str]
) -> object:
    """
    Build a styled per-class precision / recall / F1 / support table
    including macro-average and weighted-average rows.

    Args:
        y_true       : Ground-truth integer labels.
        y_pred       : Predicted integer labels.
        class_names  : Ordered list of class name strings.

    Returns:
        Pandas Styler object ready for display.
    """
    n_classes = len(class_names)
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred,
        average=None,
        labels=range(n_classes),
        zero_division=0
    )

    rows = [
        {
            'Class':     class_names[i],
            'Precision': f"{p[i]:.4f}",
            'Recall':    f"{r[i]:.4f}",
            'F1-Score':  f"{f[i]:.4f}",
            'Support':   int(s[i])
        }
        for i in range(n_classes)
    ]

    # Macro average
    pm, rm, fm, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    rows.append({
        'Class': 'macro avg',
        'Precision': f"{pm:.4f}", 'Recall': f"{rm:.4f}",
        'F1-Score': f"{fm:.4f}", 'Support': int(sum(s))
    })

    # Weighted average
    pw, rw, fw, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    rows.append({
        'Class': 'weighted avg',
        'Precision': f"{pw:.4f}", 'Recall': f"{rw:.4f}",
        'F1-Score': f"{fw:.4f}", 'Support': int(sum(s))
    })

    return style_dataframe(pd.DataFrame(rows))


# ---------------------------------------------------------------------------
# Full evaluation
# ---------------------------------------------------------------------------

def evaluate_multiclass_model(
    model: HybSwinEffELM,
    test_loader: DataLoader,
    class_names: List[str],
    device: str = 'cuda'
) -> Dict:
    """
    Run complete test-set evaluation for both MLP and ELM heads.

    Metrics computed
    ----------------
    - Per-class and averaged precision / recall / F1
    - Overall accuracy
    - Macro-average ROC-AUC
    - Average per-image inference time (ms)
    - Total testing wall-clock time (s)

    Args:
        model        : Trained HybSwinEffELM with fitted ELM (model.elm.beta set).
        test_loader  : DataLoader for the held-out test split.
        class_names  : Ordered list of class name strings.
        device       : Compute device string.

    Returns:
        results dict with keys:
            'mlp' → {y_true, y_pred, y_score, f1, acc}
            'elm' → {y_true, y_pred, y_score, f1, acc}
            'test_loss', 'testing_time', 'avg_inference_time'
    """
    assert model.elm.beta is not None, (
        "ELM has not been fitted yet. "
        "Run fit_elm_on_features() before evaluate_multiclass_model()."
    )

    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    y_true, y_pred_mlp, y_pred_elm = [], [], []
    y_score_mlp, y_score_elm = [], []
    test_loss = n_test = 0
    inference_times = []

    t_start = time.time()

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs   = imgs.to(device)
            labels = labels.to(device)

            t_batch = time.time()
            out = model.forward(imgs, return_intermediate=True)
            inference_times.append(
                (time.time() - t_batch) / imgs.size(0)
            )

            # MLP head
            mlp_logits = out['logits_mlp']
            mlp_probs  = F.softmax(mlp_logits, dim=1)

            # ELM head
            elm_logits = model.elm.predict_logits(out['F_head'])
            elm_probs  = F.softmax(elm_logits, dim=1)

            test_loss += criterion(mlp_logits, labels).item() * imgs.size(0)
            n_test    += imgs.size(0)

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred_mlp.extend(mlp_logits.argmax(1).cpu().numpy().tolist())
            y_pred_elm.extend(elm_logits.argmax(1).cpu().numpy().tolist())
            y_score_mlp.extend(mlp_probs.cpu().numpy().tolist())
            y_score_elm.extend(elm_probs.cpu().numpy().tolist())

    testing_time = time.time() - t_start
    test_loss    = test_loss / n_test
    avg_inf_time = np.mean(inference_times)

    y_true      = np.array(y_true)
    y_score_mlp = np.array(y_score_mlp)
    y_score_elm = np.array(y_score_elm)

    # ── MLP report ────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("TEST SET EVALUATION — MLP HEAD")
    print("=" * 80)
    _display(create_classification_report_table(y_true, y_pred_mlp, class_names))
    mlp_f1  = f1_score(y_true, y_pred_mlp, average='macro')
    mlp_acc = accuracy_score(y_true, y_pred_mlp)
    print(f"  MLP Macro F1-Score : {mlp_f1:.4f}")
    print(f"  MLP Accuracy       : {mlp_acc:.4f}")

    # ── ELM report ────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("TEST SET EVALUATION — ELM HEAD")
    print("=" * 80)
    _display(create_classification_report_table(y_true, y_pred_elm, class_names))
    elm_f1  = f1_score(y_true, y_pred_elm, average='macro')
    elm_acc = accuracy_score(y_true, y_pred_elm)
    print(f"  ELM Macro F1-Score : {elm_f1:.4f}")
    print(f"  ELM Accuracy       : {elm_acc:.4f}")

    # ── Timing summary ────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("TIMING")
    print("=" * 80)
    print(f"  Total testing time          : {testing_time:.2f} s")
    print(f"  Avg inference time / image  : {avg_inf_time * 1000:.2f} ms")
    print(f"  Test loss (MLP)             : {test_loss:.4f}")

    return {
        'mlp': {
            'y_true': y_true, 'y_pred': y_pred_mlp,
            'y_score': y_score_mlp, 'f1': mlp_f1, 'acc': mlp_acc
        },
        'elm': {
            'y_true': y_true, 'y_pred': y_pred_elm,
            'y_score': y_score_elm, 'f1': elm_f1, 'acc': elm_acc
        },
        'test_loss': test_loss,
        'testing_time': testing_time,
        'avg_inference_time': avg_inf_time
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    from .train import build_feature_bank, fit_elm_on_features

    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[Test] Using device: {device}")

    _, val_tf = get_transforms()
    test_tf = val_tf

    # ── Load dataset ──────────────────────────────────────────────────
    base_dataset = BloodSmearDataset(
        root_dir=Config.ROOT_DIR, transform=None, verbose=True
    )
    train_df, val_df, test_df = create_train_val_test_split(base_dataset)
    test_dataset = BloodSmearDataset(
        df=test_df, transform=test_tf, verbose=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=Config.BATCH_SIZE,
        shuffle=False, num_workers=Config.NUM_WORKERS
    )

    NUM_CLASSES = len(test_dataset.label2idx)
    class_names = [test_dataset.idx2label[i] for i in range(NUM_CLASSES)]

    # ── Load model ────────────────────────────────────────────────────
    checkpoint = torch.load('hybswineff_model.pth', map_location=device)
    model = HybSwinEffELM(
        num_classes=NUM_CLASSES, D_f=256,
        elm_hidden=Config.ELM_HIDDEN, device=device
    )
    model.load_state_dict(checkpoint['model_state'])
    model.to(device)

    # Refit ELM from training features (needed if loaded from .pth)
    from .preprocessing import get_transforms as _gt
    train_tf, _ = _gt()
    train_dataset = BloodSmearDataset(
        df=train_df, transform=train_tf, verbose=False
    )
    feat_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE,
        shuffle=False, num_workers=Config.NUM_WORKERS
    )
    import numpy as np
    X_train, Y_train = build_feature_bank(model, feat_loader, device=device)
    fit_elm_on_features(model.elm, X_train, Y_train)

    # ── Evaluate ──────────────────────────────────────────────────────
    results = evaluate_multiclass_model(
        model, test_loader, class_names, device=device
    )

    # ── Plots ─────────────────────────────────────────────────────────
    for head in ('mlp', 'elm'):
        plot_multiclass_roc(
            results[head]['y_true'], results[head]['y_score'],
            class_names,
            title=f"{head.upper()} Head — Multiclass ROC Curves",
            savepath=f'roc_{head}.png'
        )
        plot_multiclass_confusion(
            results[head]['y_true'], results[head]['y_pred'],
            class_names,
            title=f"{head.upper()} Head — Confusion Matrix",
            savepath=f'confusion_{head}.png'
        )

    print("\n[Test] Evaluation complete.")
