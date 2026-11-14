"""Single place for paths, constants and hyper-parameters (change here, not in the code)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "raw"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


def data_path() -> Path:
    """data/raw/Dataset.csv (preferred) or Dataset.csv in the project root (legacy)."""
    for p in (DATA_DIR / "Dataset.csv", ROOT / "Dataset.csv"):
        if p.exists():
            return p
    raise FileNotFoundError("Dataset.csv not found in data/raw/ or project root")


SEED = 42
N_FOLDS = 5
HORIZONS = (3, 6, 12)      # hours; 6 is the main result
MAIN_HORIZON = 6
MIN_ONSET = 6              # patients with onset in the first 6 ICU hours are dropped
TARGET_RECALL = 0.80       # operating point, chosen on a patient-held-out validation split
N_BOOTSTRAP = 100

# Leaky columns: non-sepsis stays are truncated at ~59h, so stay length reveals the label.
LEAKY_COLS = ("ICULOS", "HospAdmTime")

MEASURED = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2", "BaseExcess", "HCO3",
            "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN", "Alkalinephos", "Calcium", "Chloride",
            "Creatinine", "Bilirubin_direct", "Glucose", "Lactate", "Magnesium", "Phosphate",
            "Potassium", "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
            "Fibrinogen", "Platelets"]

ROLLING = ["HR", "O2Sat", "Temp", "SBP", "MAP", "Resp"]   # vitals that get 6h trend features
ROLL_WINDOW = 6

LGBM_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_child_samples=200,
                   feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0,
                   verbose=-1, seed=SEED, n_jobs=-1)
EARLY_STOPPING = 100
MAX_ROUNDS = 1500
