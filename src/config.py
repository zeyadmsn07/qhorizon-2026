import os
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
DATA_CACHE_DIR = ROOT_DIR / "data_cache"
RESULTS_DIR = ROOT_DIR / "results"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MASTER_SPLIT_SEED = 2026
ALL_SEEDS = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
TUNE_SEEDS = [200, 201, 202]
N_TEST = 3000
N_VAL = 1500
COLLAPSE_PR_AUC = 0.70


@dataclass(frozen=True)
class ModeConfig:
    name: str
    eval_seeds: tuple[int, ...]
    tune_seeds: tuple[int, ...]
    n_tuning_configs: int
    sample_sweep: tuple[int, ...]
    vqc_max_epochs: int
    mlp_max_epochs: int


# Separate mode configs prevent quick checks from ever overwriting full 10-seed results
MODES = {
    "smoke": ModeConfig("smoke", (100, 101), (200,), 2, (100, 200), 6, 25),
    "fast": ModeConfig("fast", (100, 101, 102), (200,), 4, (100, 200, 500), 20, 60),
    "full": ModeConfig("full", tuple(ALL_SEEDS), tuple(TUNE_SEEDS), 20, (100, 200, 500, 1000), 55, 120),
}


def get_mode_config(mode: str | None = None) -> ModeConfig:
    m = (mode or os.environ.get("QHORIZON_MODE", "fast")).strip().lower()
    if m not in MODES:
        raise ValueError(f"Unknown mode '{m}'. Valid modes: {list(MODES)}")
    return MODES[m]


def get_results_dir(mode: str | None = None) -> Path:
    cfg = get_mode_config(mode)
    out = RESULTS_DIR / cfg.name
    out.mkdir(parents=True, exist_ok=True)
    return out
