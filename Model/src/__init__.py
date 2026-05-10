from .HybSwinEff import HybSwinEffELM, ELMClassifier
from .preprocessing import BloodSmearDataset, create_train_val_test_split
from .utils import set_seed, plot_training_history

"""
HybSwinEff — src package.

Public API
----------
    HybSwinEffELM               : Full hybrid CNN-Transformer model.
    ELMClassifier               : Stage-2 analytic ELM classifier.
    BloodSmearDataset           : PyTorch dataset for blood smear images.
    create_train_val_test_split : Stratified 80/10/10 split helper.
    get_transforms              : Returns (train_tf, val_tf) transform pipelines.
    fit_stage1                  : Stage-1 gradient-based training loop.
    build_feature_bank          : Extract F_head features for ELM.
    fit_elm_on_features         : Analytically fit ELM beta weights.
    evaluate_multiclass_model   : Full test-set evaluation (MLP + ELM).
    set_seed                    : Fix all RNG seeds for reproducibility.
    plot_training_history       : Loss / accuracy curve plots.
    plot_multiclass_confusion   : Confusion matrix visualisation.
    plot_multiclass_roc         : One-vs-rest ROC curve plot.
    Config                      : Central hyperparameter configuration.

Author  : Rashel Mahmud Rabbi
Version : 1.0.0
"""

from .HybSwinEff import (
    ELMClassifier,
    GatedFusion,
    HAFB,
    HybSwinEffELM,
    MLPHead,
    MultiHeadCrossAttention,
    ProjToken,
    ResidualFusionBlock,
)
from .preprocessing import (
    BloodSmearDataset,
    create_train_val_test_split,
    get_transforms,
)
from .train import (
    build_feature_bank,
    fit_elm_on_features,
    fit_stage1,
)
from .test import (
    create_classification_report_table,
    evaluate_multiclass_model,
)
from .utils import (
    Config,
    count_parameters,
    count_trainable_parameters,
    detailed_model_analysis,
    maybe_load_pretrained_backbone,
    plot_multiclass_confusion,
    plot_multiclass_roc,
    plot_training_history,
    print_model_summary,
    set_seed,
    style_dataframe,
)

__version__ = "1.0.0"
__author__  = "Rashel Mahmud Rabbi"
__email__   = "rashelmahmud@example.com"

__all__ = [
    # Model components
    "HybSwinEffELM",
    "ELMClassifier",
    "ProjToken",
    "MultiHeadCrossAttention",
    "GatedFusion",
    "HAFB",
    "ResidualFusionBlock",
    "MLPHead",
    # Data
    "BloodSmearDataset",
    "create_train_val_test_split",
    "get_transforms",
    # Training
    "fit_stage1",
    "build_feature_bank",
    "fit_elm_on_features",
    # Evaluation
    "evaluate_multiclass_model",
    "create_classification_report_table",
    # Utilities
    "Config",
    "set_seed",
    "maybe_load_pretrained_backbone",
    "style_dataframe",
    "print_model_summary",
    "detailed_model_analysis",
    "count_parameters",
    "count_trainable_parameters",
    "plot_training_history",
    "plot_multiclass_confusion",
    "plot_multiclass_roc",
]