# config.py — LIME explainability configuration for HybSwinEff

# ── Class labels (must match training label2idx alphabetical order) ────────
CATEGORIES = [
    "Benign",
    "Early Pre-B",
    "Pre-B",
    "Pro-B",
]

# ── Paths ──────────────────────────────────────────────────────────────────
# Root directory of the Blood Cell Cancer (ALL) dataset
DATASET_PATH = "/kaggle/input/datasets/rashelmahmud/blood-cell-cancer-all"

# Path to a saved model checkpoint (output of train.py)
MODEL_PATH = "hybswineff_model.pth"

# Directory where LIME output images will be saved
OUTPUT_DIR = "lime_outputs"

# ── Image settings ─────────────────────────────────────────────────────────
IMAGE_SIZE   = 224    # Model input resolution (square)
RESIZE_SIZE  = 512    # Intermediate resize before centre-crop

# ── LIME hyperparameters ───────────────────────────────────────────────────
NUM_SAMPLES  = 2000   # Number of perturbed samples per explanation
NUM_FEATURES = 10     # Number of superpixel segments to highlight