"""
Core LIME explanation generator for HybSwinEff.

Uses the lime-image library to produce superpixel-level explanations
that highlight which regions of a blood smear drive each prediction.
Yellow boundary overlays (mark_boundaries) indicate positive-influence
regions for the top predicted class.

Author : Rashel Mahmud Rabbi
Date   : November 2025
"""

import numpy as np
import torch
from PIL import Image
from lime import lime_image
from skimage.morphology import binary_dilation, disk
from skimage.segmentation import mark_boundaries

from data_utils import test_transform
from config import NUM_SAMPLES, NUM_FEATURES


def _predict_fn(images_np, model, device):
    """
    Prediction function compatible with LIME's perturbation interface.

    Args:
        images_np : numpy array (N, H, W, 3) uint8 — LIME's perturbed samples.
        model     : HybSwinEffELM in eval mode.
        device    : Compute device string.

    Returns:
        numpy array (N, num_classes) of softmax probabilities.
    """
    import torch.nn.functional as F

    batch = []
    for img in images_np:
        pil_img = Image.fromarray(img)
        batch.append(test_transform(pil_img))

    batch_tensor = torch.stack(batch).to(device)
    with torch.no_grad():
        probs = F.softmax(model(batch_tensor), dim=1)
    return probs.cpu().numpy()


def generate_lime_explanation(
    image_uint8: np.ndarray,
    model,
    device: str = 'cpu'
) -> np.ndarray:
    """
    Generate a LIME visual explanation for a single blood smear image.

    The explanation highlights the top-``NUM_FEATURES`` superpixel segments
    that most positively influence the model's top predicted class.
    Segment boundaries are thickened with a 3-pixel dilation for visibility
    and coloured yellow (RGB 1, 1, 0).

    Args:
        image_uint8 : Input image as numpy uint8 array (H, W, 3).
        model       : Loaded HybSwinEffELM instance in eval mode.
        device      : Compute device string.

    Returns:
        lime_overlay : numpy float32 array (H, W, 3) in [0, 1] with
                       yellow superpixel boundaries overlaid on the
                       masked image.
    """
    explainer   = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        image_uint8,
        lambda imgs: _predict_fn(imgs, model, device),
        top_labels=1,
        hide_color=0,
        num_samples=NUM_SAMPLES,
        num_features=NUM_FEATURES
    )

    top_label = explanation.top_labels[0]
    temp, mask = explanation.get_image_and_mask(
        top_label,
        positive_only=True,
        num_features=NUM_FEATURES,
        hide_rest=True
    )

    # Thicken mask boundary for visual clarity
    thick_mask = binary_dilation(mask, footprint=disk(3))
    lime_overlay = mark_boundaries(
        temp / 255.0, thick_mask, color=(1, 1, 0)
    )
    return lime_overlay