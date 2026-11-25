import numpy as np
from sklearn.model_selection import ParameterSampler
from src.config import ModeConfig, get_mode_config
from src.data import sample_and_preprocess
from src.metrics import evaluate_predictions
from src.models_classical import fit_predict_classical
from src.models_quantum import train_predict_quantum_kernel_svm, train_predict_vqc

SEARCH_SPACES = {
    "Dummy": [{}],
    "LogReg": {"C": np.logspace(-3, 3, 50), "class_weight": [None, "balanced"]},
    "RBF-SVM": {"C": np.logspace(-2, 3, 50), "gamma": np.logspace(-3, 1, 50), "class_weight": [None, "balanced", {0: 1, 1: 5}, {0: 1, 1: 10}]},
    "RandomForest": {"n_estimators": [50, 100, 200], "max_depth": [3, 5, 8, None], "min_samples_leaf": [1, 3, 8], "class_weight": [None, "balanced", "balanced_subsample"]},
    "Tiny-MLP": {"hidden_dim": [4], "lr": [1e-3, 3e-3, 1e-2, 3e-2], "weight_decay": [0.0, 1e-4, 1e-2], "pos_weight_mult": [0.5, 1.0, 1.5], "batch_size": [32, 64]},
    "Q1-VQC": {"alpha": [0.25, 0.5, 1.0, 1.5, 2.5], "lr": [0.005, 0.01, 0.02, 0.04], "weight_decay": [0.0, 1e-4, 1e-3], "pos_weight_mult": [0.5, 1.0, 1.5], "batch_size": [128]},
    "Q2-QuantumKernel": {"alpha": np.logspace(-1.0, 0.5, 30), "depth": [1, 2], "C": np.logspace(-2, 3, 40), "class_weight": [None, "balanced", {0: 1, 1: 5}, {0: 1, 1: 10}]},
}

def _canonical_key(family: str, n_train: int, fixed_overrides: dict) -> tuple:
    no = dict(fixed_overrides)
    if family == "Q1-VQC":
        no.setdefault("encoding", "angle"); no.setdefault("entanglement", "ring"); no.setdefault("n_layers", 2); no.setdefault("reupload", False)
    return (family, n_train, tuple(sorted((k, tuple(v) if isinstance(v, (list, tuple)) else v) for k, v in no.items())))

def _run_once(family: str, params: dict, data: dict, seed: int, mode_cfg: ModeConfig) -> dict:
    p = dict(params)
    if family == "Q1-VQC": p.setdefault("max_epochs", mode_cfg.vqc_max_epochs)
    if family == "Tiny-MLP": p.setdefault("max_epochs", mode_cfg.mlp_max_epochs)
    grad = float("nan")
    if family in {"Dummy", "LogReg", "RBF-SVM", "RandomForest", "Tiny-MLP"}:
        vs, ts, npa, sec = fit_predict_classical(family, p, data, seed)
    elif family == "Q1-VQC":
        vs, ts, npa, sec, grad, _ = train_predict_vqc(p, data, seed, return_model=True)
    else:
        vs, ts, npa, sec = train_predict_quantum_kernel_svm(p, data, seed)
    m = evaluate_predictions(data["y_val"], vs, data["y_test"], ts, npa, sec)
    m["init_grad_norm"] = grad
    return m

def tune_and_evaluate_10_seeds(splits: dict, n_train: int, family: str, fixed_overrides: dict | None = None, label: str | None = None, cache: dict | None = None, mode_cfg: ModeConfig | None = None) -> list[dict]:
    mode_cfg = mode_cfg or get_mode_config()
    fixed_overrides = fixed_overrides or {}
    label = label or family
    key = _canonical_key(family, n_train, fixed_overrides)
    if cache is not None and key in cache:
        return [{**r, "model": label} for r in cache[key]["records"]]
    space = SEARCH_SPACES[family]
    cands = [{**c, **fixed_overrides} for c in (space if isinstance(space, list) else ParameterSampler(space, n_iter=mode_cfg.n_tuning_configs, random_state=42))]
    ref = sample_and_preprocess(splits, n_train=n_train, seed=mode_cfg.eval_seeds[0])
    best_cfg = max(cands, key=lambda c: _run_once(family, c, ref, mode_cfg.eval_seeds[0], mode_cfg)["val_pr_auc"])
    recs = []
    for s in mode_cfg.eval_seeds:
        d = sample_and_preprocess(splits, n_train=n_train, seed=s)
        m = _run_once(family, best_cfg, d, s, mode_cfg)
        m.update({"model": label, "family": family, "n_train": n_train, "seed": s})
        recs.append(m)
    if cache is not None: cache[key] = {"records": recs, "best_cfg": best_cfg}
    return recs

def get_best_config(cache: dict, family: str, n_train: int, fixed_overrides: dict | None = None) -> dict:
    return dict(cache[_canonical_key(family, n_train, fixed_overrides or {})]["best_cfg"])
