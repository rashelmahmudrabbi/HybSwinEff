# HybSwinEff: Hybrid CNN-Transformer Fusion for Binary and Multi-Stage Blood Cell Cancer Classification

## Overview

HybSwinEff is a lightweight hybrid CNN-Transformer architecture for automated,
interpretable, and multi-stage Acute Lymphoblastic Leukemia (ALL) classification
from peripheral blood smear images.

### Key Features

- **Hierarchical Pipeline**: First distinguishes benign from malignant, then subtypes
  malignant cases into Early Pre-B, Pre-B, and Pro-B stages
- **Hybrid Architecture**: EfficientNetV2-RW-T (local texture) + Swin Transformer Tiny
  (global context) fused via a Residual Fusion Block (RFB)
- **Dual Heads**: Simultaneous binary + stage classification with ELM and MLP heads
- **XAI Integration**: LIME visualizations for clinical interpretability
- **High Performance**: 99.69% binary accuracy, 100% staging accuracy on Blood Cell Cancer

### Technical Specifications

- **Framework**: PyTorch 2.x
- **Language**: Python 3.10
- **License**: MIT
- **Datasets**: Blood Cell Cancer (ALL), Acute Lymphoblastic Leukemia (ALL)
- **Parameters**: ~40.54M
- **Inference Speed**: ~3.40 ms/image (GPU)

## Project Structure
HybSwinEff/
│
├── Model/
│   ├── src/
│   │   ├── init.py
│   │   ├── HybSwinEff.py
│   │   ├── train.py
│   │   ├── test.py
│   │   ├── preprocessing.py
│   │   └── utils.py
│   ├── XAI/
│   │   └── LIME/
│   ├── environment.yml
│   ├── requirements.txt
│   └── README.md
│
├── .gitignore
├── LICENSE
└── README.md"# HybSwinEff" 
