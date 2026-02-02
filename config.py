from pathlib import Path

# =====================================================
# PROJECT ROOT
# =====================================================

# Racine du projet (à adapter UNE SEULE FOIS)
PROJECT_ROOT = Path(__file__).resolve().parent

# =====================================================
# DATA PATHS
# =====================================================

DATA_DIR = PROJECT_ROOT / "Data"
COEF_DIR = DATA_DIR / "coef"

INPUT_DATA_FILE = DATA_DIR / "INPUT_SP1_ALL_exploration_subset(in).csv"
OPTIMIZED_OUTPUT_FILE = DATA_DIR / "optimized_output.csv"

# =====================================================
# METADATA
# =====================================================

METADATA_DIR = PROJECT_ROOT 
INDICATOR_METADATA_FILE = METADATA_DIR / "SP1_metadata(SP1).csv"

# =====================================================
# DEFAULT PARAMETERS
# =====================================================

DEFAULT_START_YEAR = 2022
DEFAULT_END_YEAR = 2030

# Optimization options
MAX_ITER = 1000
LOWER_BOUND_EXP = 1e-6

# =====================================================
# STREAMLIT
# =====================================================

APP_TITLE = "SP Model Calibration & Optimization"
PAGE_LAYOUT = "wide"

# =====================================================
# R CONFIG
# =====================================================

RSCRIPT_EXECUTABLE = "Rscript"  # ou chemin complet si nécessaire
CALIBRATION_SCRIPT = PROJECT_ROOT / "calibrate.R"
