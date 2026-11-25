import numpy as np
from sklearn.model_selection import ParameterSampler

from src.config import N_TUNING_CONFIGS, SEEDS
from src.data import sample_and_preprocess
from src.metrics import evaluate_predictions
from src.models_classical import fit_predict_classical
from src.models_quantum import train_predict_quantum_kernel_svm, train_predict_vqc

SEARCH_SPACES = {
    "Dummy": [{}],
    "LogReg": {
        "C": np.logspace(-3, 3, 50),
        "class_weight": [None, "balanced"],
    },
    "RBF-SVM": {
        "C": np.logspace(-2, 3, 50),
        "gamma": np.logspace(-3, 1, 50),
        "class_weight": [None, "balanced", {0: 1, 1: 5}, {0: 1, 1: 10}],
    },
    "RandomForest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 8, None],
        "min_samples_leaf": [1, 3, 8],
        "class_weight": [None, "balanced", "balanced_subsample"],
    },
    "Tiny-MLP": {
        # Fixed at 41 params for an equal parameter-budget comparison against VQC
        "hidden_dim": [4],
        "lr": [1e-3, 3e-3, 1e-2, 3e-2],
        "weight_decay": [0.0, 1e-4, 1e-2],
        "pos_weight_mult": [0.5, 1.0, 1.5],
        "batch_size": [32, 64],
    },
    "Q1-VQC": {
        # Input angle bandwidth alpha is the primary quantum performance lever
        "alpha": [0.25, 0.5, 1.0, 1.5, 2.5],
        "lr": [0.005, 0.01, 0.02, 0.04],
        "weight_decay": [0.0, 1e-4, 1e-3],
        "pos_weight_mult": [0.5, 1.0, 1.5],
        "batch_size": [128],
    },
    "Q2-QuantumKernel": {
        "alpha": np.logspace(-1.0, 0.5, 30),
        "depth": [1, 2],
        "C": np.logspace(-2, 3, 40),
        "class_weight": [None, "balanced", {0: 1, 1: 5}, {0: 1, 1: 10}],
    },
}


def _canonical_key(family: str, n_train: int, fixed_overrides: dict) -> tuple:
    norm_overrides = dict(fixed_overrides)
    if family == "Q1-VQC":
        # Normalize defaults so functionally identical overrides share cache entries
        norm_overrides.setdefault("encoding", "angle")
        norm_overrides.setdefault("entanglement", "ring")
        norm_overrides.setdefault("n_layers", 2)
        norm_overrides.setdefault("reupload", False)
    items = tuple(sorted((k, tuple(v) if isinstance(v, (list, tuple)) else v) for k, v in norm_overrides.items()))
    return (family, n_train, items)


def _run_once(family: str, params: dict, data: dict, seed: int) -> dict:
    init_grad_norm = float("nan")
    if family in {"Dummy", "LogReg", "RBF-SVM", "RandomForest", "Tiny-MLP"}:
        vs, ts, npa, sec = fit_predict_classical(family, params, data, seed)
    elif family == "Q1-VQC":
        vs, ts, npa, sec, init_grad_norm, _ = train_predict_vqc(params, data, seed, return_model=True)
    elif family == "Q2-QuantumKernel":
        vs, ts, npa, sec = train_predict_quantum_kernel_svm(params, data, seed)
    else:
        raise ValueError(f"Unknown family: {family}")

    metrics = evaluate_predictions(data["y_val"], vs, data["y_test"], ts, npa, sec)
    metrics["init_grad_norm"] = init_grad_norm
    return metrics


def tune_and_evaluate_10_seeds(
    splits: dict,
    n_train: int,
    family: str,
    fixed_overrides: dict | None = None,
    label: str | None = None,
    cache: dict | None = None,
) -> list[dict]:
    # Shared validation random search enforces strict equal-budget fairness across all model families
    fixed_overrides = fixed_overrides or {}
    label = label or family

    key = _canonical_key(family, n_train, fixed_overrides)
    if cache is not None and key in cache:
        cached_entry = cache[key]
        records = []
        for rec in cached_entry["records"]:
            r = dict(rec)
            r["model"] = label
            records.append(r)
        return records

    space = SEARCH_SPACES[family]
    if isinstance(space, list):
        candidates = [{**c, **fixed_overrides} for c in space]
    else:
        sampler = ParameterSampler(space, n_iter=N_TUNING_CONFIGS, random_state=42)
        candidates = [{**c, **fixed_overrides} for c in sampler]

    ref_data = sample_and_preprocess(splits, n_train=n_train, seed=SEEDS[0])
    best_cfg = candidates[0]
    best_val_pr = -1.0

    for cfg in candidates:
        res = _run_once(family, cfg, ref_data, seed=SEEDS[0])
        if res["val_pr_auc"] > best_val_pr:
            best_val_pr = res["val_pr_auc"]
            best_cfg = cfg

    records = []
    for s in SEEDS:
        d = sample_and_preprocess(splits, n_train=n_train, seed=s)
        metrics = _run_once(family, best_cfg, d, seed=s)
        metrics.update({"model": label, "family": family, "n_train": n_train, "seed": s})
        records.append(metrics)

    if cache is not None:
        cache[key] = {"records": records, "best_cfg": best_cfg}

    return records


def get_best_config(cache: dict, family: str, n_train: int, fixed_overrides: dict | None = None) -> dict:
    key = _canonical_key(family, n_train, fixed_overrides or {})
    if key in cache:
        return dict(cache[key]["best_cfg"])
    raise KeyError(f"No tuned config found in cache for key: {key}")
