"""
Data utilities for HybSwinEff LIME explainability.

Provides:
    - test_transform : Deterministic val/test torchvision pipeline.
    - denormalize    : Reverse ImageNet normalisation for display.

Author : Rashel Mahmud Rabbi
Date   : November 2025
"""

import numpy as np
from torchvision import transforms
from config import IMAGE_SIZE, RESIZE_SIZE

# ── Validation / test transform (matches preprocessing.py val_transforms) ──
test_transform = transforms.Compose([
    transforms.Resize(RESIZE_SIZE),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


def denormalize(tensor):
    """
    Reverse ImageNet normalisation and return a numpy HWC float32 array
    clipped to [0, 1].

    Args:
        tensor : Normalised CHW torch.Tensor on any device.

    Returns:
        numpy ndarray of shape (H, W, 3) with values in [0, 1].
    """
    img  = tensor.cpu().numpy().transpose(1, 2, 0)   # CHW → HWC
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = img * std + mean
    return np.clip(img, 0.0, 1.0)