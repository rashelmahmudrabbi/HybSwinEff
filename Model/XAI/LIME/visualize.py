"""
Visualisation utilities for HybSwinEff LIME outputs.

Saves side-by-side figures showing the original blood smear image
alongside the LIME superpixel explanation overlay.

Author : Rashel Mahmud Rabbi
Date   : November 2025
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from config import OUTPUT_DIR


def save_lime_figure(
    original: np.ndarray,
    lime_vis: np.ndarray,
    class_name: str,
    true_class: str,
    confidence: float,
    savepath: str = None
) -> None:
    """
    Save a side-by-side figure of the original image and LIME overlay.

    Left panel  : Original (denormalised) blood smear image.
    Right panel : LIME explanation with yellow superpixel boundaries.

    Args:
        original    : Original image as HWC float32 numpy array in [0, 1].
        lime_vis    : LIME overlay as HWC float32 numpy array in [0, 1].
        class_name  : Predicted class name (used in title and filename).
        true_class  : Ground-truth class name (used in title).
        confidence  : Model confidence for the predicted class.
        savepath    : Full output file path.  If None, saves to
                      ``OUTPUT_DIR/lime_<class_name>.png``.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if savepath is None:
        safe_name = class_name.replace(' ', '_').replace('/', '-')
        savepath  = os.path.join(OUTPUT_DIR, f"lime_{safe_name}.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].imshow(np.clip(original, 0, 1))
    axes[0].set_title(
        f"Original\nTrue: {true_class} | Pred: {class_name} ({confidence:.3f})",
        fontsize=11
    )
    axes[0].axis('off')

    axes[1].imshow(np.clip(lime_vis, 0, 1))
    axes[1].set_title(
        f"LIME Explanation\nHighlighted regions → '{class_name}'",
        fontsize=11
    )
    axes[1].axis('off')

    plt.suptitle(
        f"HybSwinEff — LIME Interpretability: {class_name}",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(savepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[LIME] Saved → {savepath}")