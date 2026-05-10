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
```
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
└── README.md
```
## Installation

### Prerequisites
- Python 3.10+
- CUDA-compatible GPU (recommended)
- pip or conda

### Setup

1. **Clone the repository**
```bash
   git clone https://github.com/rashelmahmudrabbi/HybSwinEff.git
   cd HybSwinEff
```

2. **Create environment**

   Using conda:
```bash
   conda env create -f Model/environment.yml
   conda activate hybswineff
```

   Using pip:
```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r Model/requirements.txt
```

3. **Download Datasets**
   - [Blood Cell Cancer (ALL)](https://www.kaggle.com/datasets/mohammadamireshraghi/blood-cell-cancer-all-4class)
   - [Acute Lymphoblastic Leukemia](https://www.kaggle.com/datasets/mehradaria/leukemia)

## Usage

### Training

```bash
python Model/src/train.py
```

### Evaluation

```bash
python Model/src/test.py
```

### LIME Explainability

```bash
python Model/XAI/LIME/run_lime.py
```

## Performance Results

| Dataset | Task | Accuracy | Precision | Recall | F1 Score | AUC |
|---|---|---|---|---|---|---|
| Blood Cell Cancer | Binary | 99.69% | 99.04% | 99.82% | 99.42% | 1.000 |
| Blood Cell Cancer | Stage | 100% | 100% | 100% | 100% | 1.000 |
| Acute Lymphoblastic Leukemia | Binary | 100% | 100% | 100% | 100% | 1.000 |
| Acute Lymphoblastic Leukemia | Stage | 100% | 100% | 100% | 100% | 1.000 |

## Architecture

HybSwinEff processes blood smear images through two parallel backbones:

1. **EfficientNetV2-RW-T** extracts fine-grained local texture features
2. **Swin Transformer Tiny** captures hierarchical global context
3. Features are mean-pooled, projected to 128-dim, concatenated to 256-dim
4. A **Residual Fusion Block (RFB)** refines the joint representation
5. **Dual MLP heads** produce binary + stage logits
6. An **ELM classifier** is fitted on top of frozen head features (Stage 2)

## Citation

If you use this work, please cite:

@thesis{rabbi2025hybswineff,
title={HybSwinEff: Hybrid CNN-Transformer Fusion for Binary and Multi-Stage
Blood Cell Cancer Classification},
author={Rashel Mahmud Rabbi},
school={North Bengal International University},
year={2025}
}

## License

MIT License — see [LICENSE](LICENSE)

## Acknowledgements

Supervised by Md Shafiuzzaman, Lecturer, CSE, NBIU.
External supervision by Saifur Rahman, Lecturer, CSE, NBIU.
