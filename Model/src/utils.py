"""
Utility module for HybSwinEff.

Contains:
    - Config                  : Central hyperparameter dataclass
    - set_seed                : Reproducibility helper
    - maybe_load_pretrained_backbone : timm backbone factory
    - style_dataframe         : Pandas styler for summary tables
    - print_model_summary     : Keras-like layer summary
    - detailed_model_analysis : Per-component parameter counts
    - count_parameters        : Total parameter count
    - count_trainable_parameters : Trainable parameter count
    - plot_training_history   : Loss / accuracy curve plots
    - plot_multiclass_confusion : Confusion matrix visualisation
    - plot_multiclass_roc     : One-vs-rest ROC curve plot

Author : Rashel Mahmud Rabbi
Date   : November 2025
"""

import os
import random
import sys
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.preprocessing import label_binarize

try:
    import timm
except ImportError:
    timm = None

if 'IPython' in sys.modules:
    from IPython.display import display as _display
else:
    _display = lambda df: print(df.to_string(index=False))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class Config:
    """
    Central configuration for dataset paths, model dimensions, and
    training hyperparameters.

    Attributes
    ----------
    ROOT_DIR     : Path to the root dataset directory.
    IMAGE_SIZE   : Model input spatial resolution (square).
    RESIZE_SIZE  : Resize target before random crop augmentation.
    BATCH_SIZE   : Mini-batch size for all data loaders.
    NUM_WORKERS  : DataLoader worker processes.
    NUM_EPOCHS   : Maximum training epochs (Stage 1).
    LR_BACKBONE  : Learning rate for pretrained backbone parameters.
    LR_HEADS     : Learning rate for projection / head parameters.
    WEIGHT_DECAY : AdamW L2 regularisation coefficient.
    PATIENCE     : Early-stopping patience in epochs.
    ELM_HIDDEN   : Number of ELM random hidden units.
    RIDGE_LAMBDA : ELM Ridge regression regularisation coefficient.
    LABEL_SMOOTHING : Cross-entropy label smoothing factor.
    """

    ROOT_DIR = '/kaggle/input/datasets/rashelmahmud/blood-cell-cancer-all'
    IMAGE_SIZE = 224
    RESIZE_SIZE = 512
    BATCH_SIZE = 16
    NUM_WORKERS = 2
    NUM_EPOCHS = 5
    LR_BACKBONE = 1e-5
    LR_HEADS = 1e-4
    WEIGHT_DECAY = 1e-3
    PATIENCE = 10
    ELM_HIDDEN = 512
    RIDGE_LAMBDA = 5e-2
    LABEL_SMOOTHING = 0.1


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """
    Fix all random-number generators for reproducible training.

    Args:
        seed : Integer seed (default 42, used in all experiments).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Backbone factory
# ---------------------------------------------------------------------------

def maybe_load_pretrained_backbone(
    name: str,
    pretrained: bool = True,
    device: str = 'cpu'
) -> nn.Module:
    """
    Load a timm backbone with no classification head.

    Args:
        name      : timm model name (e.g. 'efficientnetv2_s').
        pretrained: Whether to load ImageNet-1K pretrained weights.
        device    : Target device string.

    Returns:
        nn.Module with global_pool disabled and num_classes=0.

    Raises:
        ImportError : If timm is not installed.
    """
    if timm is None:
        raise ImportError(
            "timm is required for pretrained backbones.\n"
            "Install with: pip install timm"
        )
    model = timm.create_model(
        name, pretrained=pretrained, num_classes=0, global_pool=''
    )
    return model.to(device)


# ---------------------------------------------------------------------------
# DataFrame styling
# ---------------------------------------------------------------------------

def style_dataframe(df: pd.DataFrame):
    """
    Apply a consistent HTML style to a summary DataFrame.

    Args:
        df : Pandas DataFrame to style.
    Returns:
        Pandas Styler object.
    """
    return (
        df.style
        .set_table_styles([
            {'selector': 'th', 'props': [
                ('background-color', 'lightblue'),
                ('font-weight', 'bold'),
                ('text-align', 'left')
            ]},
            {'selector': 'tr:nth-child(even)', 'props': [
                ('background-color', '#f9f9f9')
            ]},
            {'selector': 'tr:hover', 'props': [
                ('background-color', '#e6f3ff')
            ]},
        ])
        .set_properties(**{'text-align': 'left', 'border': '1px solid #ddd'})
        .hide(axis='index')
    )


# ---------------------------------------------------------------------------
# Model summary
# ---------------------------------------------------------------------------

def print_model_summary(
    model: nn.Module,
    input_size=(1, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE)
) -> None:
    """
    Print a Keras-style layer summary table including output shapes and
    parameter counts, then call detailed_model_analysis.

    Args:
        model      : HybSwinEffELM instance.
        input_size : Tuple for the dummy input tensor.
    """

    def hook_fn(module, input, output):
        if isinstance(output, torch.Tensor):
            shape = list(output.shape)
            shape[0] = 'None'
            module.output_shape = tuple(shape)
        else:
            module.output_shape = 'N/A'

    handles = []
    for name in ['cnn', 'swin']:
        module = getattr(model, name)
        handles.append(module.register_forward_hook(hook_fn))

    for name, module in model.named_children():
        if name not in ['cnn', 'swin'] and not list(module.children()):
            handles.append(module.register_forward_hook(hook_fn))

    model.eval()
    device = next(model.parameters()).device
    dummy_input = torch.randn(input_size).to(device)

    with torch.no_grad():
        try:
            _ = model(dummy_input)
        except Exception as e:
            print(f"[Warning] Forward pass error during summary: {e}")
            for h in handles:
                h.remove()
            return

    total_params, trainable_params = detailed_model_analysis(model)

    data = [
        {
            'Layer (type)': 'input_layer (InputLayer)',
            'Output Shape': f'(None, 3, {Config.IMAGE_SIZE}, {Config.IMAGE_SIZE})',
            'Param #': 0,
            'Connected to': ''
        }
    ]

    prev = 'input_layer'
    for name in ['cnn', 'swin']:
        module = getattr(model, name)
        params = sum(p.numel() for p in module.parameters())
        data.append({
            'Layer (type)': f"{name} ({type(module).__name__})",
            'Output Shape': str(getattr(module, 'output_shape', 'N/A')),
            'Param #': f"{params:,}",
            'Connected to': f"{prev}[0][0]"
        })
        prev = name

    for name, module in model.named_children():
        if name in ['cnn', 'swin']:
            continue
        params = sum(p.numel() for p in module.parameters())
        data.append({
            'Layer (type)': f"{name} ({type(module).__name__})",
            'Output Shape': str(getattr(module, 'output_shape', 'N/A')),
            'Param #': f"{params:,}",
            'Connected to': f"{prev}[0][0]"
        })
        prev = name

    for h in handles:
        h.remove()

    print('\nModel Summary:')
    print('Model: "HybSwinEffELM"')
    _display(style_dataframe(pd.DataFrame(data)))
    print(f"\nTotal params        : {total_params:,}")
    print(f"Trainable params    : {trainable_params:,}")
    print(f"Non-trainable params: {total_params - trainable_params:,}")
    print("-" * 85)


def detailed_model_analysis(model: nn.Module):
    """
    Print a breakdown of parameter counts per major component.

    Args:
        model : HybSwinEffELM instance.

    Returns:
        (total_params, trainable_params) tuple of ints.
    """
    print("\n" + "=" * 80)
    print("HYBSWINEFF-ELM MODEL DETAILED ANALYSIS")
    print("=" * 80)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nOVERVIEW:")
    print(f"  Total parameters       : {total_params:,}")
    print(f"  Trainable parameters   : {trainable_params:,}")
    print(f"  Non-trainable params   : {total_params - trainable_params:,}")
    print(f"  Percentage trainable   : {100 * trainable_params / total_params:.2f}%")

    components = {
        'CNN Backbone (EfficientNetV2-S)': model.cnn,
        'Swin Transformer Backbone (Tiny)': model.swin,
        'Projection Layers (proj_c + proj_v)': [model.proj_c, model.proj_v],
        'HAFB (Hierarchical Attention Fusion)': model.hafb,
        'Residual Fusion Block (RFB)': model.rfb,
        'MLP Classification Head': model.mlp_head,
    }

    print(f"\nCOMPONENT BREAKDOWN:")
    for component_name, component in components.items():
        if isinstance(component, list):
            params = sum(
                sum(p.numel() for p in sub.parameters()) for sub in component
            )
            trainable = sum(
                sum(p.numel() for p in sub.parameters() if p.requires_grad)
                for sub in component
            )
        else:
            params = sum(p.numel() for p in component.parameters())
            trainable = sum(
                p.numel() for p in component.parameters() if p.requires_grad
            )
        print(
            f"  {component_name:<42}: {params:>8,} params "
            f"({trainable:>8,} trainable)"
        )

    return total_params, trainable_params


def count_parameters(model: nn.Module) -> int:
    """Return total number of parameters in the model."""
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model: nn.Module) -> int:
    """Return number of trainable (requires_grad) parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Training history plots
# ---------------------------------------------------------------------------

def plot_training_history(
    history: Dict[str, List[float]],
    savepath: Optional[str] = None
) -> None:
    """
    Plot and optionally save the Stage 1 loss and accuracy curves.

    Args:
        history  : Dict with keys 'train_loss', 'val_loss', 'test_loss',
                   'train_acc', 'val_acc', 'test_acc'.
        savepath : File path to save the figure (PNG, 600 dpi).
    """
    epochs = list(range(1, len(history['train_loss']) + 1))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'],   label='Val Loss')
    plt.plot(epochs, history['test_loss'],  label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], label='Train Acc')
    plt.plot(epochs, history['val_acc'],   label='Val Acc')
    plt.plot(epochs, history['test_acc'],  label='Test Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curves')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=600, bbox_inches='tight')
        print(f"[Plot] Saved training history → {savepath}")
    plt.show()


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def plot_multiclass_confusion(
    y_true,
    y_pred,
    class_names: List[str],
    title: str,
    savepath: Optional[str] = None
) -> None:
    """
    Plot a colour-coded confusion matrix with per-cell counts.

    Args:
        y_true       : Ground-truth label array.
        y_pred       : Predicted label array.
        class_names  : Ordered list of class name strings.
        title        : Figure title.
        savepath     : Optional save path (600 dpi PNG).
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        plt.text(
            j, i, format(cm[i, j], 'd'),
            ha='center', va='center',
            color='white' if cm[i, j] > thresh else 'black'
        )

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=600, bbox_inches='tight')
        print(f"[Plot] Saved confusion matrix → {savepath}")
    plt.show()


# ---------------------------------------------------------------------------
# ROC curves
# ---------------------------------------------------------------------------

def plot_multiclass_roc(
    y_true,
    y_score,
    class_names: List[str],
    title: str,
    savepath: Optional[str] = None
) -> None:
    """
    Plot one-vs-rest ROC curves for all classes plus the macro-average AUC.

    Args:
        y_true      : Ground-truth label array (integer encoded).
        y_score     : Predicted probability matrix (N, n_classes).
        class_names : Ordered list of class name strings.
        title       : Figure title.
        savepath    : Optional save path (600 dpi PNG).
    """
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))

    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = roc_auc_score(y_true_bin[:, i], y_score[:, i])

    colors = [
        'blue', 'red', 'green', 'orange', 'purple',
        'brown', 'pink', 'gray', 'olive', 'cyan'
    ]

    plt.figure(figsize=(10, 8))
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i], tpr[i], color=color, lw=2,
            label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})'
        )

    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True)

    if savepath:
        plt.savefig(savepath, dpi=600, bbox_inches='tight')
        print(f"[Plot] Saved ROC curve → {savepath}")
    plt.show()