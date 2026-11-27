import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, PowerTransformer
from sklearn.svm import SVC

from src.config import get_mode_config, get_results_dir
from src.data import get_locked_splits, sample_and_preprocess
from src.stats import stratified_bootstrap_pr_auc, two_level_bootstrap_pr_auc
from src.tuning import SEARCH_SPACES, tune_and_evaluate_10_seeds


def _tune_poly2_10_seeds(splits: dict, cfg) -> list[np.ndarray]:
    c_grid = [0.01, 0.05, 0.2, 1.0, 5.0]
    cw_grid = [None, "balanced"]
    cands = [(c, cw) for c in c_grid for cw in cw_grid][: cfg.n_tuning_configs]
    tune_sets = [sample_and_preprocess(splits, n_train=500, seed=ts) for ts in cfg.tune_seeds]

    best_cfg, best_val = cands[0], -1.0
    for c, cw in cands:
        vals = []
        for td in tune_sets:
            clf = make_pipeline(
                PolynomialFeatures(2, include_bias=False),
                LogisticRegression(C=c, class_weight=cw, max_iter=1000, random_state=42),
            ).fit(td["X_train"], td["y_train"])
            vals.append(average_precision_score(td["y_val"], clf.decision_function(td["X_val"])))
        mv = float(np.mean(vals))
        if mv > best_val:
            best_val, best_cfg = mv, (c, cw)

    c_opt, cw_opt = best_cfg
    out = []
    for s in cfg.eval_seeds:
        d = sample_and_preprocess(splits, n_train=500, seed=s)
        clf = make_pipeline(
            PolynomialFeatures(2, include_bias=False),
            LogisticRegression(C=c_opt, class_weight=cw_opt, max_iter=1000, random_state=s),
        ).fit(d["X_train"], d["y_train"])
        out.append(clf.decision_function(d["X_test"]))
    return out


def run_linearity_check(cfg_name: str | None = None) -> pd.DataFrame:
    cfg = get_mode_config(cfg_name)
    res_dir = get_results_dir(cfg.name)
    diag_dir = res_dir / "linearity_preds"
    diag_dir.mkdir(parents=True, exist_ok=True)
    splits = get_locked_splits()
    y_test = splits["y_test"]

    # 1. N=500 across eval_seeds using the shared validation tuning protocol
    cache = {}
    lr_recs = tune_and_evaluate_10_seeds(splits, 500, "LogReg", {}, "LogReg (9p)", cache=cache, mode_cfg=cfg, out_dir=diag_dir)
    svm_recs = tune_and_evaluate_10_seeds(splits, 500, "RBF-SVM", {}, "RBF-SVM", cache=cache, mode_cfg=cfg, out_dir=diag_dir)
    mlp_recs = tune_and_evaluate_10_seeds(splits, 500, "Tiny-MLP", {}, "Tiny-MLP (41p)", cache=cache, mode_cfg=cfg, out_dir=diag_dir)

    n500_scores = {
        "LogReg (9p)": [np.load(diag_dir / r["pred_file"])["test_scores"] for r in sorted(lr_recs, key=lambda x: x["seed"])],
        "Poly2-LogReg (45p)": _tune_poly2_10_seeds(splits, cfg),
        "RBF-SVM": [np.load(diag_dir / r["pred_file"])["test_scores"] for r in sorted(svm_recs, key=lambda x: x["seed"])],
        "Tiny-MLP (41p)": [np.load(diag_dir / r["pred_file"])["test_scores"] for r in sorted(mlp_recs, key=lambda x: x["seed"])],
    }

    rows = []
    for name, sc_list in n500_scores.items():
        pt, lo, hi = two_level_bootstrap_pr_auc(y_test, sc_list, n_boot=600)
        rows.append({"regime": "N=500", "n_train": 500, "model": name, "test_pr_auc": pt, "ci_lo": lo, "ci_hi": hi})

    # 2. Full training pool (~13.4k) classical ceiling with validation-selected hyperparameters
    pt_tr = PowerTransformer(method="yeo-johnson", standardize=True).fit(splits["X_pool"])
    full_d = {
        "X_train": pt_tr.transform(splits["X_pool"]).astype(np.float32), "y_train": splits["y_pool"],
        "X_val": pt_tr.transform(splits["X_val"]).astype(np.float32), "y_val": splits["y_val"],
        "X_test": pt_tr.transform(splits["X_test"]).astype(np.float32), "y_test": y_test,
    }
    n_pool = int(len(full_d["X_train"]))

    best_lr = max(
        [(c, cw) for c in [0.05, 0.2, 1.0, 5.0, 20.0] for cw in [None, "balanced"]],
        key=lambda p: average_precision_score(
            full_d["y_val"],
            LogisticRegression(C=p[0], class_weight=p[1], max_iter=1000, random_state=2026)
            .fit(full_d["X_train"], full_d["y_train"]).decision_function(full_d["X_val"]),
        ),
    )
    lr_f = LogisticRegression(C=best_lr[0], class_weight=best_lr[1], max_iter=1000, random_state=2026).fit(full_d["X_train"], full_d["y_train"])

    best_poly = max(
        [(c, cw) for c in [0.02, 0.1, 0.5, 2.0] for cw in [None, "balanced"]],
        key=lambda p: average_precision_score(
            full_d["y_val"],
            make_pipeline(PolynomialFeatures(2, include_bias=False), LogisticRegression(C=p[0], class_weight=p[1], max_iter=1000, random_state=2026))
            .fit(full_d["X_train"], full_d["y_train"]).decision_function(full_d["X_val"]),
        ),
    )
    poly_f = make_pipeline(
        PolynomialFeatures(2, include_bias=False),
        LogisticRegression(C=best_poly[0], class_weight=best_poly[1], max_iter=1000, random_state=2026),
    ).fit(full_d["X_train"], full_d["y_train"])

    svm_grid = [(c, g, cw) for c in [1.0, 5.0, 20.0] for g in [0.002, 0.005, 0.02, 0.08] for cw in [None, "balanced"]]
    best_svm = max(
        svm_grid,
        key=lambda p: average_precision_score(
            full_d["y_val"],
            SVC(kernel="rbf", C=p[0], gamma=p[1], class_weight=p[2], random_state=2026)
            .fit(full_d["X_train"], full_d["y_train"]).decision_function(full_d["X_val"]),
        ),
    )
    svm_f = SVC(kernel="rbf", C=best_svm[0], gamma=best_svm[1], class_weight=best_svm[2], random_state=2026).fit(full_d["X_train"], full_d["y_train"])

    from src.models_classical import fit_predict_classical
    mlp_cands = list(ParameterSampler(SEARCH_SPACES["Tiny-MLP"], n_iter=min(6, cfg.n_tuning_configs), random_state=42))
    best_mlp_sc, best_mlp_ts = -1.0, None
    for mp in mlp_cands:
        mp_run = {**mp, "max_epochs": cfg.mlp_max_epochs}
        vs_m, ts_m, _, _ = fit_predict_classical("Tiny-MLP", mp_run, full_d, seed=2026)
        vp = float(average_precision_score(full_d["y_val"], vs_m))
        if vp > best_mlp_sc:
            best_mlp_sc, best_mlp_ts = vp, ts_m

    for name, sc in [
        ("LogReg (9p)", lr_f.decision_function(full_d["X_test"])),
        ("Poly2-LogReg (45p)", poly_f.decision_function(full_d["X_test"])),
        ("RBF-SVM", svm_f.decision_function(full_d["X_test"])),
        ("Tiny-MLP (41p)", best_mlp_ts),
    ]:
        pt, lo, hi = stratified_bootstrap_pr_auc(y_test, sc, n_boot=600)
        rows.append({"regime": f"Full Pool (N={n_pool})", "n_train": n_pool, "model": name, "test_pr_auc": pt, "ci_lo": lo, "ci_hi": hi})

    df = pd.DataFrame(rows)
    df.to_csv(res_dir / "linearity_check.csv", index=False)

    models = ["LogReg (9p)", "Poly2-LogReg (45p)", "RBF-SVM", "Tiny-MLP (41p)"]
    x, w = np.arange(len(models)), 0.35
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    for i, (reg, col) in enumerate([("N=500", "#2B7BBA"), (f"Full Pool (N={n_pool})", "#1B9E77")]):
        sub = df[df["regime"] == reg].set_index("model").loc[models]
        m = sub["test_pr_auc"].values
        yerr = [np.maximum(0.0, m - sub["ci_lo"].values), np.maximum(0.0, sub["ci_hi"].values - m)]
        bars = ax.bar(x + (i - 0.5) * w, m, width=w, yerr=yerr, capsize=4, color=col, label=reg)
        for b, v in zip(bars, m):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(models); ax.set_ylim(0.84, 0.98)
    ax.set_ylabel("Test PR-AUC (95% Bootstrap CI)")
    ax.set_title("HTRU2 Linearity & Data-Regime Ceiling Check (Validation-Tuned)")
    ax.legend(loc="lower right", frameon=True); fig.tight_layout()
    fig.savefig(res_dir / "linearity_check.png", dpi=300); plt.close(fig)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "fast", "full"], default=None)
    print(run_linearity_check(ap.parse_args().mode).to_string(index=False))