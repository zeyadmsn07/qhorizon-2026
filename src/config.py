from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Keep True while developing and for judge reruns (< 3 mins); set False for the full 10-seed sweep
FAST_MODE = True

MASTER_SPLIT_SEED = 2026
ALL_SEEDS = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
SEEDS = ALL_SEEDS[:2] if FAST_MODE else ALL_SEEDS

N_TUNING_CONFIGS = 3 if FAST_MODE else 20
N_TEST = 3000
N_VAL = 1500
SAMPLE_SWEEP = [100, 200, 500] if FAST_MODE else [100, 200, 500, 1000]
