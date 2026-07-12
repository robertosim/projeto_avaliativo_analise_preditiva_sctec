# config.py - Caminhos e parâmetros do projeto

from pathlib import Path

# Raiz do projeto
ROOT = Path(__file__).resolve().parents[1]

# Dados
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_FINAL = ROOT / "data" / "final"

# Modelos
MODELS_DIR = ROOT / "models"

# Outputs
OUTPUTS_DIR = ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"

# Dataset
DATASET_FILE = DATA_RAW / "kc_house_data.csv"

# Parâmetros
RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET = "price"
