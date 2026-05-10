"""
Preprocessing module for HybSwinEff.

Contains:
    - BloodSmearDataset        : PyTorch Dataset for blood smear images.
    - create_train_val_test_split : Stratified split helper.
    - get_transforms           : Returns train and val/test transform pipelines.

The preprocessing pipeline follows Algorithm 1 from the thesis:

Training augmentation (Eq. 1):
    X' = N(μ,σ) ∘ T ∘ J(b,c,s,h) ∘ RRC_s ∘ RS_r ∘ RVF ∘ RHF ∘ RR_θ (X)

Validation / test (centre-crop only):
    X' = N(μ,σ) ∘ T ∘ CC_s ∘ RS_r (X)

ImageNet statistics:
    μ = [0.485, 0.456, 0.406]
    σ = [0.229, 0.224, 0.225]

Author : Rashel Mahmud Rabbi
Date   : November 2025
"""

import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

from .utils import Config


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def get_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Build and return the training and validation/test transform pipelines.

    Training pipeline applies random geometric and colour augmentations to
    simulate blood-smear orientation and staining variations (Eq. 1).
    Validation/test pipeline uses deterministic centre-crop only.

    Returns:
        (train_transforms, val_transforms) — both are torchvision Compose objects.
    """
    # ImageNet normalisation statistics (matching pretrained backbone input)
    _mean = [0.485, 0.456, 0.406]
    _std  = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([
        transforms.RandomRotation(10),                           # RR_θ, θ~U(−10°,10°)
        transforms.RandomHorizontalFlip(),                       # RHF, p=0.5
        transforms.RandomVerticalFlip(),                         # RVF, p=0.5
        transforms.Resize(Config.RESIZE_SIZE),                  # RS_r, r=512
        transforms.RandomResizedCrop(Config.IMAGE_SIZE),        # RRC_s, s=224
        transforms.ColorJitter(                                  # J(b,c,s,h)
            brightness=0.1,
            contrast=0.1,
            saturation=0.1,
            hue=0.02
        ),
        transforms.ToTensor(),                                   # T → [0,1]
        transforms.Normalize(mean=_mean, std=_std),             # N(μ,σ)
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(Config.RESIZE_SIZE),                  # RS_r
        transforms.CenterCrop(Config.IMAGE_SIZE),               # CC_s
        transforms.ToTensor(),
        transforms.Normalize(mean=_mean, std=_std),
    ])

    return train_transforms, val_transforms


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BloodSmearDataset(Dataset):
    """
    PyTorch Dataset for the Blood Cell Cancer (ALL) and Acute Lymphoblastic
    Leukemia datasets stored as class-named sub-directories.

    Expected directory layout
    -------------------------
    root_dir/
        Benign/
            image_001.jpg
            image_002.jpg
            ...
        [Early Pre-B/]
            ...
        [Pre-B/]
            ...
        [Pro-B/]
            ...

    Each class folder name becomes the label string.  Labels are integer-
    encoded in alphabetical class order for reproducibility.

    Args:
        root_dir  : Path to the dataset root directory.
                    Required when ``df`` is None.
        df        : Pre-split Pandas DataFrame with columns
                    ``['image_path', 'label']``.  When provided, ``root_dir``
                    is ignored for scanning but stored for reference.
        transform : torchvision Compose transform to apply per sample.
        verbose   : Print a summary of the class distribution.
    """

    SUPPORTED_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

    def __init__(
        self,
        root_dir: Optional[str] = None,
        df: Optional[pd.DataFrame] = None,
        transform=None,
        verbose: bool = True
    ):
        self.root_dir = root_dir
        self.transform = transform

        if df is None:
            # ── Scan directory tree ────────────────────────────────────
            if root_dir is None:
                raise ValueError("Either root_dir or df must be provided.")
            if not os.path.isdir(root_dir):
                raise FileNotFoundError(
                    f"Dataset root directory not found: {root_dir}"
                )

            records = []
            for class_name in sorted(os.listdir(root_dir)):
                class_dir = os.path.join(root_dir, class_name)
                if not os.path.isdir(class_dir):
                    continue
                images = sorted([
                    f for f in os.listdir(class_dir)
                    if f.lower().endswith(self.SUPPORTED_EXTS)
                ])
                for img_file in images:
                    records.append({
                        'image_path': os.path.join(class_dir, img_file),
                        'label': class_name
                    })
                if verbose:
                    print(f"  Found {len(images):>5} images in class '{class_name}'")

            if not records:
                raise ValueError(
                    f"No supported images found under: {root_dir}"
                )
            self.df = pd.DataFrame(records)
        else:
            # ── Use pre-split DataFrame ────────────────────────────────
            self.df = df.reset_index(drop=True)

        # ── Build label ↔ index mappings ───────────────────────────────
        unique_labels = sorted(self.df['label'].unique())
        self.label2idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx2label = {idx: label for label, idx in self.label2idx.items()}
        self.df['label_idx'] = self.df['label'].map(self.label2idx)

        if verbose:
            self._print_summary()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def _print_summary(self) -> None:
        """Print class distribution to stdout."""
        total = len(self.df)
        print(f"\nDataset summary:")
        print(f"  Total images : {total}")
        print(f"  Classes      : {len(self.label2idx)}")
        print(f"  Distribution:")
        for label, idx in self.label2idx.items():
            count = int((self.df['label_idx'] == idx).sum())
            pct = 100.0 * count / total
            print(f"    [{idx}] {label:<20}: {count:>5} images ({pct:.1f}%)")

    # ------------------------------------------------------------------
    # PyTorch Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        """
        Returns:
            (image_tensor, label_int) tuple.

        A plain white image is returned on read failure so that training
        does not crash on a single corrupt file.
        """
        row = self.df.iloc[idx]
        img_path = row['image_path']

        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"[Warning] Could not read image {img_path}: {e}")
            img = Image.new('RGB', (Config.IMAGE_SIZE, Config.IMAGE_SIZE), 'white')

        if self.transform:
            img = self.transform(img)

        return img, int(row['label_idx'])


# ---------------------------------------------------------------------------
# Train / Validation / Test split
# ---------------------------------------------------------------------------

def create_train_val_test_split(
    dataset: BloodSmearDataset,
    test_size: float = 0.10,
    val_size: float = 0.10,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified split of the dataset into train, validation, and test sets.

    The split is performed on the raw DataFrame of the base dataset so that
    class proportions are maintained in every split (matching Table 2 of the
    thesis: 80 % / 10 % / 10 %).

    Args:
        dataset      : BloodSmearDataset to split.
        test_size    : Fraction reserved for testing.
        val_size     : Fraction reserved for validation.
        random_state : Random seed for reproducibility (default 42).

    Returns:
        (train_df, val_df, test_df) — three Pandas DataFrames.
    """
    df = dataset.df

    # First split off (val + test)
    train_df, temp_df = train_test_split(
        df,
        test_size=test_size + val_size,
        stratify=df['label_idx'],
        random_state=random_state
    )

    # Split the remainder evenly into val and test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df['label_idx'],
        random_state=random_state
    )

    total = len(df)
    print(f"\nStratified dataset split (seed={random_state}):")
    print(f"  Training   : {len(train_df):>5} samples ({100*len(train_df)/total:.1f}%)")
    print(f"  Validation : {len(val_df):>5} samples ({100*len(val_df)/total:.1f}%)")
    print(f"  Test       : {len(test_df):>5} samples ({100*len(test_df)/total:.1f}%)")

    return train_df, val_df, test_df