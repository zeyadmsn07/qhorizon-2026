import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, PowerTransformer
from sklearn.svm import SVC

from src.config import RESULTS_DIR, get_mode_config, get_results_dir
from src.data import get_locked_splits, sample_and_preprocess
from src.models_classical import fit_predict_classical
from src.stats import stratified_bootstrap_pr_auc, two_level_bootstrap_pr_auc


def run_linearity_check(cfg_name: str | None = None) -> pd.DataFrame:
    cfg = get_mode_config(cfg_name)
    res_dir = get_results_dir(cfg.name)
    splits = get_locked_splits()
    y_test = splits["y_test"]

    # 1. N=500 across eval_seeds with two-level bootstrap 95% CIs
    n500_scores = {"LogReg (9p)": [], "Poly2-LogReg (45p)": [], "RBF-SVM": [], "Tiny-MLP (41p)": []}
    mlp_p = {"hidden_dim": 4, "lr": 0.01, "weight_decay": 1e-4, "pos_weight_mult": 1.0, "batch_size": 64, "max_epochs": cfg.mlp_max_epochs}
    for s in cfg.eval_seeds:
        d = sample_and_preprocess(splits, n_train=500, seed=s)
        lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=s).fit(d["X_train"], d["y_train"])
        poly = make_pipeline(PolynomialFeatures(2, include_bias=False), LogisticRegression(C=0.2, class_weight="balanced", max_iter=1000, random_state=s)).fit(d["X_train"], d["y_train"])
        svm = SVC(kernel="rbf", C=2.0, gamma="scale", class_weight="balanced", random_state=s).fit(d["X_train"], d["y_train"])
        _, ts_mlp, _, _ = fit_predict_classical("Tiny-MLP", mlp_p, d, seed=s)
        n500_scores["LogReg (9p)"].append(lr.decision_function(d["X_test"]))
        n500_scores["Poly2-LogReg (45p)"].append(poly.decision_function(d["X_test"]))
        n500_scores["RBF-SVM"].append(svm.decision_function(d["X_test"]))
        n500_scores["Tiny-MLP (41p)"].append(ts_mlp)

    rows = []
    for name, sc_list in n500_scores.items():
        pt, lo, hi = two_level_bootstrap_pr_auc(y_test, sc_list, n_boot=600)
        rows.append({"regime": "N=500", "n_train": 500, "model": name, "test_pr_auc": pt, "ci_lo": lo, "ci_hi": hi})

    # 2. Full training pool (~13.4k) classical ceiling (fit Yeo-Johnson strictly on X_pool)
    pt_tr = PowerTransformer(method="yeo-johnson", standardize=True).fit(splits["X_pool"])
    full_d = {
        "X_train": pt_tr.transform(splits["X_pool"]).astype(np.float32), "y_train": splits["y_pool"],
        "X_val": pt_tr.transform(splits["X_val"]).astype(np.float32), "y_val": splits["y_val"],
        "X_test": pt_tr.transform(splits["X_test"]).astype(np.float32), "y_test": y_test,
    }
    n_pool = int(len(full_d["X_train"]))
    lr_f = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=2026).fit(full_d["X_train"], full_d["y_train"])
    poly_f = make_pipeline(PolynomialFeatures(2, include_bias=False), LogisticRegression(C=0.2, class_weight="balanced", max_iter=1000, random_state=2026)).fit(full_d["X_train"], full_d["y_train"])
    svm_f = SVC(kernel="rbf", C=2.0, gamma="scale", class_weight="balanced", random_state=2026).fit(full_d["X_train"], full_d["y_train"])
    _, ts_mlp_f, _, _ = fit_predict_classical("Tiny-MLP", mlp_p, full_d, seed=2026)

    for name, sc in [("LogReg (9p)", lr_f.decision_function(full_d["X_test"])), ("Poly2-LogReg (45p)", poly_f.decision_function(full_d["X_test"])), ("RBF-SVM", svm_f.decision_function(full_d["X_test"])), ("Tiny-MLP (41p)", ts_mlp_f)]:
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
    ax.set_title("HTRU2 Linearity & Data-Regime Ceiling Check (Linear vs. Nonlinear)")
    ax.legend(loc="lower right", frameon=True); fig.tight_layout()
    fig.savefig(res_dir / "linearity_check.png", dpi=300); plt.close(fig)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "fast", "full"], default=None)
    print(run_linearity_check(ap.parse_args().mode).to_string(index=False))