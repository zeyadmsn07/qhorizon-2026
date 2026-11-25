import argparse, json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from src.config import MASTER_SPLIT_SEED, get_mode_config, get_results_dir
from src.data import sample_and_preprocess
from src.stats import two_level_bootstrap_pr_auc
from src.tuning import tune_and_evaluate_10_seeds


def _build_concentric_shell_splits() -> dict:
    # Controlled 8D nonlinear task: 12% positive class inside a 4D radial shell + 4 nuisance features
    rng = np.random.default_rng(MASTER_SPLIT_SEED)
    X = rng.normal(0.0, 1.0, size=(7500, 8))
    r2 = np.sum(X[:, :4] ** 2, axis=1) + 0.35 * np.sin(2.5 * X[:, 0] * X[:, 1])
    y = (r2 <= np.quantile(r2, 0.12)).astype(np.int64)
    X_rem, X_te, y_rem, y_te = train_test_split(X, y, test_size=2000, stratify=y, random_state=MASTER_SPLIT_SEED)
    X_pool, X_va, y_pool, y_va = train_test_split(X_rem, y_rem, test_size=1000, stratify=y_rem, random_state=MASTER_SPLIT_SEED)
    return {"X_pool": X_pool, "y_pool": y_pool, "X_val": X_va, "y_val": y_va, "X_test": X_te, "y_test": y_te}


def run_nonlinear_benchmark(cfg_name: str | None = None, n_tune_seeds: int | None = None) -> pd.DataFrame:
    cfg = get_mode_config(cfg_name)
    res_dir = get_results_dir(cfg.name)
    nl_dir = res_dir / "nonlinear_task"
    nl_dir.mkdir(parents=True, exist_ok=True)
    splits = _build_concentric_shell_splits()

    d0 = sample_and_preprocess(splits, n_train=500, seed=100)
    lr_pre = float(average_precision_score(d0["y_test"], LogisticRegression(class_weight="balanced", max_iter=1000).fit(d0["X_train"], d0["y_train"]).decision_function(d0["X_test"])))
    svm_pre = float(average_precision_score(d0["y_test"], SVC(kernel="rbf", C=5.0, class_weight="balanced").fit(d0["X_train"], d0["y_train"]).decision_function(d0["X_test"])))
    sel_gap = svm_pre - lr_pre
    assert sel_gap >= 0.05, f"Dataset failed nonlinearity selection rule: gap={sel_gap:.4f} < 0.05"
    print(f"Pre-check passed on ConcentricShell-8D (N=500): RBF-SVM={svm_pre:.4f} vs LogReg={lr_pre:.4f} (gap=+{sel_gap:.4f} >= 0.05)", flush=True)

    sel_p = res_dir / "headline_vqc_selection.json"
    hl_ov = json.loads(sel_p.read_text(encoding="utf-8"))["chosen_overrides"] if sel_p.exists() else {"encoding": "angle", "entanglement": "bipartite", "n_layers": 2, "sign_offset": True, "readout": "linear_z"}

    models = [("LogReg", {}, "LogReg"), ("RBF-SVM", {}, "RBF-SVM"), ("Tiny-MLP", {}, "Tiny-MLP (41p)"), ("Q2-QuantumKernel", {}, "Q2-QuantumKernel (8q)"), ("Q1-VQC", hl_ov, "Q1-VQC (Headline)")]
    cache, tlog, raw_rows = {}, [], []
    n_list = [100, 200] if cfg.name == "smoke" else [100, 200, 500]
    for N in n_list:
        for fam, ov, lbl in models:
            raw_rows.extend(tune_and_evaluate_10_seeds(splits, N, fam, ov, lbl, cache=cache, mode_cfg=cfg, out_dir=nl_dir, n_tune_seeds=n_tune_seeds, tuning_log=tlog))

    raw_df = pd.DataFrame(raw_rows)
    raw_df.to_csv(res_dir / "nonlinear_dataset_runs.csv", index=False)

    ci_rows = []
    for N in n_list:
        for _, _, lbl in models:
            sub = raw_df[(raw_df["n_train"] == N) & (raw_df["model"] == lbl)].sort_values("seed")
            yt = np.load(nl_dir / str(sub["pred_file"].iloc[0]))["y_test"]
            sc = [np.load(nl_dir / str(p))["test_scores"] for p in sub["pred_file"]]
            pt, lo, hi = two_level_bootstrap_pr_auc(yt, sc, n_boot=500)
            ci_rows.append({"n_train": N, "model": lbl, "mean_pr_auc": pt, "ci_lo": lo, "ci_hi": hi, "precheck_rbf_minus_logreg": sel_gap})
    ci_df = pd.DataFrame(ci_rows)
    ci_df.to_csv(res_dir / "nonlinear_dataset_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for lbl, col in [("RBF-SVM", "#374151"), ("Tiny-MLP (41p)", "#1B9E77"), ("Q2-QuantumKernel (8q)", "#2B7BBA"), ("Q1-VQC (Headline)", "#D95F02"), ("LogReg", "#DC2626")]:
        s = ci_df[ci_df["model"] == lbl].sort_values("n_train")
        ax.plot(s["n_train"], s["mean_pr_auc"], marker="o", linewidth=2, color=col, label=lbl)
        ax.fill_between(s["n_train"], s["ci_lo"], s["ci_hi"], color=col, alpha=0.14)
    ax.set_xticks(n_list); ax.set_xlabel("Training Set Size (N_train)"); ax.set_ylabel("Test PR-AUC (95% Bootstrap CI)")
    ax.set_title("Generalization to Strongly Nonlinear 8D Task (ConcentricShell-8D)")
    ax.legend(loc="best", frameon=True); fig.tight_layout()
    fig.savefig(res_dir / "nonlinear_dataset.png", dpi=300); plt.close(fig)
    return ci_df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "fast", "full"], default=None)
    ap.add_argument("--tune-seeds", type=int, choices=[1, 3], default=None)
    args = ap.parse_args()
    print(run_nonlinear_benchmark(args.mode, args.tune_seeds).to_string(index=False))