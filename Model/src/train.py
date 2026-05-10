"""
Training module for HybSwinEff.

Two-stage training pipeline
----------------------------
Stage 1 — Gradient-based optimisation of the full model (backbone + heads)
    using AdamW with differential learning rates, ReduceLROnPlateau
    scheduling, label smoothing, and class-balanced cross-entropy loss.

Stage 2 — Analytic ELM fitting on frozen Stage-1 features extracted from
    the Residual Fusion Block (F_head) of the trained model.

Entry point:
    python -m Model.src.train

Author : Rashel Mahmud Rabbi
Date   : November 2025
"""

import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from .HybSwinEff import ELMClassifier, HybSwinEffELM
from .preprocessing import BloodSmearDataset, create_train_val_test_split, get_transforms
from .utils import Config, set_seed, plot_training_history


# ---------------------------------------------------------------------------
# Stage 1: Gradient-based training
# ---------------------------------------------------------------------------

def fit_stage1(
    model: HybSwinEffELM,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = Config.NUM_EPOCHS,
    lr_backbone: float = Config.LR_BACKBONE,
    lr_heads: float = Config.LR_HEADS,
    device: str = 'cuda',
    weight_decay: float = Config.WEIGHT_DECAY,
    class_weights: Optional[torch.Tensor] = None,
    patience: int = Config.PATIENCE
) -> Dict[str, List[float]]:
    """
    Train HybSwinEffELM end-to-end with differential learning rates.

    Backbone parameters (EfficientNetV2-S, Swin-Tiny) are fine-tuned at a
    lower rate (lr_backbone) while projection, fusion, and head parameters
    use a higher rate (lr_heads).  Training uses:
        - AdamW optimiser with L2 weight decay
        - ReduceLROnPlateau scheduler (factor=0.5, patience=10)
        - Cross-entropy loss with label smoothing and class-balanced weights
        - Early stopping on validation loss

    Args:
        model         : HybSwinEffELM instance.
        train_loader  : DataLoader for training set.
        val_loader    : DataLoader for validation set.
        test_loader   : DataLoader for test set (monitored but not used for ES).
        epochs        : Maximum training epochs.
        lr_backbone   : Learning rate for backbone parameters.
        lr_heads      : Learning rate for non-backbone parameters.
        device        : 'cuda' or 'cpu'.
        weight_decay  : AdamW L2 coefficient.
        class_weights : Optional tensor of per-class loss weights (on device).
        patience      : Early-stopping patience in epochs.

    Returns:
        history dict with keys:
            'train_loss', 'train_acc', 'val_loss', 'val_acc',
            'test_loss', 'test_acc', 'test_f1'
    """
    # ── Optimiser ─────────────────────────────────────────────────────────
    backbone_params = (
        list(model.cnn.parameters()) + list(model.swin.parameters())
    )
    backbone_ids = {id(p) for p in backbone_params}
    other_params = [
        p for p in model.parameters() if id(p) not in backbone_ids
    ]

    optimizer = AdamW(
        [
            {'params': backbone_params, 'lr': lr_backbone},
            {'params': other_params,    'lr': lr_heads}
        ],
        weight_decay=weight_decay
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=Config.LABEL_SMOOTHING
    )

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss':   [], 'val_acc':   [],
        'test_loss':  [], 'test_acc':  [], 'test_f1': []
    }
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        # ── Training ──────────────────────────────────────────────────
        model.train()
        running_loss = correct = n_train = 0

        for imgs, labels in tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{epochs} [Train]",
            leave=False
        ):
            imgs   = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            preds     = logits.argmax(dim=1)
            correct  += (preds == labels).sum().item()
            running_loss += loss.item() * imgs.size(0)
            n_train      += imgs.size(0)

        train_loss = running_loss / n_train
        train_acc  = correct / n_train

        # ── Validation ────────────────────────────────────────────────
        model.eval()
        val_loss, val_acc = _evaluate_loss_acc(model, val_loader, criterion, device)

        # ── Test (monitor only) ───────────────────────────────────────
        test_loss, test_acc, test_f1 = _evaluate_loss_acc_f1(
            model, test_loader, criterion, device
        )

        scheduler.step(val_loss)

        # ── Record ────────────────────────────────────────────────────
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['test_f1'].append(test_f1)

        print(
            f"Epoch {epoch + 1:>3}/{epochs} | "
            f"train_acc={train_acc:.4f}  train_loss={train_loss:.4f} | "
            f"val_acc={val_acc:.4f}  val_loss={val_loss:.4f} | "
            f"test_acc={test_acc:.4f}  test_f1={test_f1:.4f}"
        )

        # ── Early stopping ─────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {'model_state': model.state_dict()},
                'best_model_stage1.pth'
            )
            print(f"  → New best model saved (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    f"Early stopping triggered at epoch {epoch + 1} "
                    f"(patience={patience})."
                )
                break

    return history


# ---------------------------------------------------------------------------
# Internal evaluation helpers
# ---------------------------------------------------------------------------

def _evaluate_loss_acc(model, loader, criterion, device):
    """Return (loss, accuracy) on a given DataLoader."""
    total_loss = correct = n = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            total_loss += criterion(logits, labels).item() * imgs.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            n += imgs.size(0)
    return total_loss / n, correct / n


def _evaluate_loss_acc_f1(model, loader, criterion, device):
    """Return (loss, accuracy, macro-F1) on a given DataLoader."""
    total_loss = n = 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            total_loss += criterion(logits, labels).item() * imgs.size(0)
            n += imgs.size(0)
            y_pred.extend(logits.argmax(1).cpu().numpy().tolist())
            y_true.extend(labels.cpu().numpy().tolist())
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average='macro')
    return total_loss / n, acc, f1


# ---------------------------------------------------------------------------
# Stage 2: ELM feature bank and fitting
# ---------------------------------------------------------------------------

def build_feature_bank(
    model: HybSwinEffELM,
    loader: DataLoader,
    device: str = 'cuda'
):
    """
    Extract F_head features from the frozen Stage-1 model for ELM fitting.

    Args:
        model  : Trained HybSwinEffELM.
        loader : DataLoader (typically the training set, shuffle=False).
        device : Compute device.

    Returns:
        X        : Feature matrix  (N, 256) — numpy float32 array.
        Y_onehot : One-hot targets (N, num_classes) — numpy float32 array.
    """
    model.eval()
    X_list, Y_list = [], []

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Building ELM feature bank"):
            imgs = imgs.to(device)
            out = model.forward(imgs, return_intermediate=True)
            X_list.append(out['F_head'].cpu().numpy())
            Y_list.extend(labels.numpy().tolist())

    X = np.concatenate(X_list, axis=0)
    K = model.num_classes
    Y_onehot = np.zeros((len(Y_list), K), dtype=np.float32)
    for i, y in enumerate(Y_list):
        Y_onehot[i, int(y)] = 1.0

    print(f"[ELM] Feature bank: X={X.shape}, Y={Y_onehot.shape}")
    return X, Y_onehot


def fit_elm_on_features(
    elm: ELMClassifier,
    X: np.ndarray,
    Y_onehot: np.ndarray
) -> None:
    """
    Analytically fit ELM output weights beta using the feature bank.

    Args:
        elm      : ELMClassifier instance (from model.elm).
        X        : Feature matrix  (N, in_dim).
        Y_onehot : One-hot targets (N, num_classes).
    """
    print("[ELM] Fitting beta via Ridge regression …")
    elm.fit(X, Y_onehot)
    print("[ELM] Fitting complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[Main] Using device: {device}")

    # ── Transforms ────────────────────────────────────────────────────
    train_tf, val_tf = get_transforms()
    test_tf = val_tf

    # ── Datasets ──────────────────────────────────────────────────────
    print("\n[Main] Scanning dataset …")
    base_dataset = BloodSmearDataset(
        root_dir=Config.ROOT_DIR, transform=None, verbose=True
    )
    train_df, val_df, test_df = create_train_val_test_split(base_dataset)

    train_dataset = BloodSmearDataset(df=train_df, transform=train_tf, verbose=False)
    val_dataset   = BloodSmearDataset(df=val_df,   transform=val_tf,   verbose=False)
    test_dataset  = BloodSmearDataset(df=test_df,  transform=test_tf,  verbose=False)

    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE,
        shuffle=True, num_workers=Config.NUM_WORKERS
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE,
        shuffle=False, num_workers=Config.NUM_WORKERS
    )
    test_loader = DataLoader(
        test_dataset, batch_size=Config.BATCH_SIZE,
        shuffle=False, num_workers=Config.NUM_WORKERS
    )

    # ── Class weights ─────────────────────────────────────────────────
    cw_np = compute_class_weight(
        'balanced',
        classes=np.unique(train_dataset.df['label_idx']),
        y=train_dataset.df['label_idx']
    )
    class_weights = torch.FloatTensor(cw_np).to(device)
    print(f"[Main] Class weights: {cw_np}")

    # ── Build model ───────────────────────────────────────────────────
    NUM_CLASSES = len(train_dataset.label2idx)
    print(f"\n[Main] Building model with {NUM_CLASSES} classes …")
    model = HybSwinEffELM(
        num_classes=NUM_CLASSES,
        D_f=256,
        elm_hidden=Config.ELM_HIDDEN,
        device=device
    )

    # ── Stage 1: Train ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STAGE 1 — GRADIENT-BASED TRAINING")
    print("=" * 70)
    t0 = time.time()
    history = fit_stage1(
        model, train_loader, val_loader, test_loader,
        device=device, class_weights=class_weights
    )
    stage1_time = time.time() - t0
    plot_training_history(history, savepath='training_history.png')

    # ── Stage 2: ELM ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STAGE 2 — ELM FITTING")
    print("=" * 70)
    t0 = time.time()
    feat_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE,
        shuffle=False, num_workers=Config.NUM_WORKERS
    )
    X_train, Y_train = build_feature_bank(model, feat_loader, device=device)
    fit_elm_on_features(model.elm, X_train, Y_train)
    stage2_time = time.time() - t0

    # ── Save final model ─────────────────────────────────────────────
    class_names = [train_dataset.idx2label[i] for i in range(NUM_CLASSES)]
    torch.save(
        {
            'model_state': model.state_dict(),
            'label2idx': train_dataset.label2idx,
            'idx2label': train_dataset.idx2label,
            'class_names': class_names
        },
        'hybswineff_model.pth'
    )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Stage 1 training time : {stage1_time:.1f} s")
    print(f"  Stage 2 ELM fit time  : {stage2_time:.1f} s")
    print(f"  Total training time   : {stage1_time + stage2_time:.1f} s")
    print(f"  Model saved           : hybswineff_model.pth")
    print(f"  Classes               : {class_names}")