# HybSwinEff Model

This directory contains the core model implementation.

## Structure
- `src/HybSwinEff.py` — Full model architecture
- `src/train.py` — Training pipeline (Stage 1 + Stage 2 ELM)
- `src/test.py` — Evaluation metrics and reporting
- `src/preprocessing.py` — Dataset loader and augmentation
- `src/utils.py` — Utilities, visualization, summaries
- `XAI/LIME/` — LIME explainability module

## Quick Start
```bash
pip install -r requirements.txt
python src/train.py
```