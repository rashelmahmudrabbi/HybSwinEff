"""
Model loader for HybSwinEff LIME explainability.

Loads a saved HybSwinEffELM checkpoint and exposes a unified
``predict(image_tensor)`` interface suitable for the LIME wrapper.

Author : Rashel Mahmud Rabbi
Date   : November 2025
"""

import sys
import os

import torch
import torch.nn.functional as F

# Allow imports from the project root when running lime scripts directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from Model.src.HybSwinEff import HybSwinEffELM
from Model.src.utils import Config
from config import MODEL_PATH, CATEGORIES


def load_model(device: str = 'cpu') -> HybSwinEffELM:
    """
    Load a trained HybSwinEffELM from a checkpoint file.

    The checkpoint must have been saved by ``train.py`` and must contain
    the keys ``model_state``, ``label2idx``, and ``class_names``.

    Args:
        device : Target device string ('cuda' or 'cpu').

    Returns:
        HybSwinEffELM instance in eval mode on the specified device.

    Raises:
        FileNotFoundError : If MODEL_PATH does not exist.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}\n"
            f"Run Model/src/train.py first to generate the checkpoint."
        )

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    num_classes = len(CATEGORIES)

    model = HybSwinEffELM(
        num_classes=num_classes,
        D_f=256,
        elm_hidden=Config.ELM_HIDDEN,
        device=device
    )
    model.load_state_dict(checkpoint['model_state'])
    model.to(device)
    model.eval()

    print(f"[LIME] Model loaded from '{MODEL_PATH}' ({num_classes} classes)")
    return model


def predict_proba(images_np, model, device: str = 'cpu'):
    """
    Batch-predict class probabilities for LIME's perturbation sampler.

    LIME passes numpy uint8 arrays (N, H, W, 3); this function applies
    the test transform and returns softmax probabilities.

    Args:
        images_np : numpy array of shape (N, H, W, 3), dtype uint8.
        model     : Loaded HybSwinEffELM instance.
        device    : Compute device string.

    Returns:
        numpy array of shape (N, num_classes) with probability values.
    """
    from PIL import Image
    from data_utils import test_transform

    batch = []
    for img in images_np:
        pil_img = Image.fromarray(img)
        batch.append(test_transform(pil_img))

    batch_tensor = torch.stack(batch).to(device)

    with torch.no_grad():
        logits = model(batch_tensor)
        probs  = F.softmax(logits, dim=1)

    return probs.cpu().numpy()