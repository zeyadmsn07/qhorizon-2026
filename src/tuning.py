import json, re
from pathlib import Path
import numpy as np
from sklearn.model_selection import ParameterSampler

from src.config import COLLAPSE_PR_AUC, ModeConfig, TUNE_SEEDS, get_mode_config, get_results_dir
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


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def to_jsonable(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]
    return obj


def save_pred_npz(out_dir: Path, family: str, label: str, n_train: int, seed: int, y_val: np.ndarray, val_scores: np.ndarray, y_test: np.ndarray, test_scores: np.ndarray, threshold: float) -> str:
    pdir = out_dir / "predictions"
    pdir.mkdir(parents=True, exist_ok=True)
    rel = f"predictions/{_safe(family)}__{_safe(label)}__N{n_train}__seed{seed}.npz"
    np.savez_compressed(
        out_dir / rel,
        y_val=y_val.astype(np.int8),
        val_scores=val_scores.astype(np.float32),
        y_test=y_test.astype(np.int8),
        test_scores=test_scores.astype(np.float32),
        threshold=np.float32(threshold),
    )
    return rel


def _canonical_key(family: str, n_train: int, fixed_overrides: dict) -> tuple:
    no = dict(fixed_overrides)
    if family == "Q1-VQC":
        no.setdefault("encoding", "angle"); no.setdefault("entanglement", "ring"); no.setdefault("n_layers", 2); no.setdefault("reupload", False); no.setdefault("init_mode", "small_normal"); no.setdefault("sign_offset", False); no.setdefault("readout", "mean_z"); no.setdefault("n_restarts", 1)
    if family == "Tiny-MLP":
        no.setdefault("hidden_dim", 4)
    return (family, n_train, tuple(sorted((k, tuple(v) if isinstance(v, (list, tuple)) else v) for k, v in no.items())))


def _run_once(family: str, params: dict, data: dict, seed: int, mode_cfg: ModeConfig, diag_path: Path | None = None):
    p = dict(params)
    if family == "Q1-VQC": p.setdefault("max_epochs", mode_cfg.vqc_max_epochs)
    if family == "Tiny-MLP": p.setdefault("max_epochs", mode_cfg.mlp_max_epochs)
    grad = float("nan")
    if family in {"Dummy", "LogReg", "RBF-SVM", "RandomForest", "Tiny-MLP"}:
        vs, ts, npa, sec = fit_predict_classical(family, p, data, seed)
    elif family == "Q1-VQC":
        vs, ts, npa, sec, grad, _ = train_predict_vqc(p, data, seed, return_model=True, diag_path=diag_path)
    else:
        vs, ts, npa, sec = train_predict_quantum_kernel_svm(p, data, seed)
    m = evaluate_predictions(data["y_val"], vs, data["y_test"], ts, npa, sec)
    m["init_grad_norm"] = grad
    m["collapsed"] = int(m["test_pr_auc"] < COLLAPSE_PR_AUC)
    return m, vs, ts


def tune_and_evaluate_10_seeds(
    splits: dict,
    n_train: int,
    family: str,
    fixed_overrides: dict | None = None,
    label: str | None = None,
    cache: dict | None = None,
    mode_cfg: ModeConfig | None = None,
    out_dir: Path | None = None,
    n_tune_seeds: int | None = None,
    tuning_log: list | None = None,
) -> list[dict]:
    mode_cfg = mode_cfg or get_mode_config()
    out_dir = out_dir or get_results_dir(mode_cfg.name)
    fixed_overrides = fixed_overrides or {}
    label = label or family

    tune_seeds = tuple(TUNE_SEEDS[:n_tune_seeds]) if n_tune_seeds is not None else mode_cfg.tune_seeds
    # Disjoint tune vs evaluation seeds eliminate optimistic selection leakage
    assert set(tune_seeds).isdisjoint(set(mode_cfg.eval_seeds)), "Tune and eval seeds must be disjoint!"

    key = _canonical_key(family, n_train, fixed_overrides)
    if cache is not None and key in cache:
        out_recs = []
        for r, pack in zip(cache[key]["records"], cache[key]["preds"]):
            rel = save_pred_npz(out_dir, family, label, n_train, r["seed"], *pack)
            out_recs.append({**r, "model": label, "pred_file": rel})
        return out_recs

    space = SEARCH_SPACES[family]
    cands = [{**c, **fixed_overrides} for c in (space if isinstance(space, list) else ParameterSampler(space, n_iter=mode_cfg.n_tuning_configs, random_state=42))]
    tune_datasets = [(ts, sample_and_preprocess(splits, n_train=n_train, seed=ts)) for ts in tune_seeds]

    best_cfg, best_rank = cands[0], (-1.0, -float("inf"))
    cand_logs = []
    for cfg in cands:
        scores = [_run_once(family, cfg, td, ts, mode_cfg)[0]["val_pr_auc"] for ts, td in tune_datasets]
        m_val = float(np.mean(scores))
        s_val = float(np.std(scores))
        cand_logs.append({"config": to_jsonable(cfg), "per_seed_val_pr_auc": scores, "mean_val_pr_auc": m_val, "std_val_pr_auc": s_val})
        if (m_val, -s_val) > best_rank:
            best_rank = (m_val, -s_val)
            best_cfg = cfg

    if tuning_log is not None:
        tuning_log.append({
            "family": family, "label": label, "n_train": n_train, "tune_seeds": list(tune_seeds),
            "n_candidates": len(cands), "chosen_config": to_jsonable(best_cfg),
            "chosen_mean_val_pr_auc": best_rank[0], "chosen_std_val_pr_auc": -best_rank[1],
            "candidates": cand_logs,
        })
        (out_dir / "tuning_log.json").write_text(json.dumps(tuning_log, indent=2), encoding="utf-8")

    recs, preds = [], []
    for s in mode_cfg.eval_seeds:
        d = sample_and_preprocess(splits, n_train=n_train, seed=s)
        dp = (out_dir / "diagnostics" / f"{_safe(label)}__N{n_train}__seed{s}.json") if family == "Q1-VQC" else None
        m, vs, ts = _run_once(family, best_cfg, d, s, mode_cfg, diag_path=dp)
        pack = (d["y_val"], vs, d["y_test"], ts, m["threshold"])
        rel = save_pred_npz(out_dir, family, label, n_train, s, *pack)
        m.update({"model": label, "family": family, "n_train": n_train, "seed": s, "pred_file": rel})
        recs.append(m); preds.append(pack)

    if cache is not None:
        cache[key] = {"records": recs, "best_cfg": best_cfg, "preds": preds}
    return recs


def get_best_config(cache: dict, family: str, n_train: int, fixed_overrides: dict | None = None) -> dict:
    return dict(cache[_canonical_key(family, n_train, fixed_overrides or {})]["best_cfg"])
