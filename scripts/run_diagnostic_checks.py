import glob
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import get_mode_config, get_results_dir
from src.data import get_locked_splits, sample_and_preprocess
from src.kernel_analysis import _centered_kta, _eff_rank, _zz_gram
from src.models_classical import TinyMLP, count_torch_params
from src.nonlinear_benchmark import _build_concentric_shell_splits
from src.stats import paired_delta_pr_auc, two_level_bootstrap_pr_auc
from src.tuning import tune_and_evaluate_10_seeds


def main():
    cfg = get_mode_config("full")
    res_dir = get_results_dir("full")
    out_dir = res_dir / "diagnostics_extra"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check A: ConcentricShell-8D MLP Width Sweep (hidden_dim = 4, 8, 16, 32)
    print("=== CHECK A: ConcentricShell-8D Tiny-MLP Width Sweep (20 configs, 3 tune seeds, 10 eval seeds) ===")
    splits_nl = _build_concentric_shell_splits()
    cache = {}
    width_rows = []
    for N in [100, 200, 500]:
        for h in [4, 8, 16, 32]:
            n_p = count_torch_params(TinyMLP(8, h))
            lbl = f"MLP_h{h}_{n_p}p"
            recs = tune_and_evaluate_10_seeds(
                splits_nl, N, "Tiny-MLP", {"hidden_dim": h}, lbl,
                cache=cache, mode_cfg=cfg, out_dir=out_dir
            )
            recs_ord = sorted(recs, key=lambda r: r["seed"])
            yt = np.load(out_dir / recs_ord[0]["pred_file"])["y_test"]
            sc = [np.load(out_dir / r["pred_file"])["test_scores"] for r in recs_ord]
            pt, lo, hi = two_level_bootstrap_pr_auc(yt, sc, n_boot=500)
            width_rows.append({"n_train": N, "hidden_dim": h, "n_params": n_p, "mean_pr_auc": pt, "ci_lo": lo, "ci_hi": hi})
    df_w = pd.DataFrame(width_rows)
    df_w.to_csv(out_dir / "concentric_mlp_width_sweep.csv", index=False)
    print(df_w.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Check B: Sub-0.1 Hybrid VQC + Tiny-MLP Probability Blend Grid across N=100..1000
    print("\n=== CHECK B: Hybrid VQC + Tiny-MLP Probability Blend Sweep (w_vqc in 0.00 .. 1.00) ===")
    pred_dir = res_dir / "predictions"
    weights = [0.0, 0.01, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70, 1.0]

    def _prob(arr):
        s = np.asarray(arr, dtype=np.float64).ravel()
        if s.min() < 0.0 or s.max() > 1.0:
            return 1.0 / (1.0 + np.exp(-np.clip(s, -30.0, 30.0)))
        return s

    blend_rows = []
    for N in [100, 200, 500, 1000]:
        vqc_probs_val, mlp_probs_val, y_vals = [], [], []
        vqc_probs_te, mlp_probs_te, y_tes = [], [], []
        for s in cfg.eval_seeds:
            vf = sorted(glob.glob(str(pred_dir / f"Q1-VQC__*Bipartite*__N{N}__seed{s}.npz")))[0]
            mf = [p for p in sorted(glob.glob(str(pred_dir / f"Tiny-MLP__*41p*__N{N}__seed{s}.npz"))) if "h_2" not in p and "h_8" not in p][0]
            vd, md = np.load(vf), np.load(mf)
            vqc_probs_val.append(_prob(vd["val_scores"])); mlp_probs_val.append(_prob(md["val_scores"])); y_vals.append(vd["y_val"])
            vqc_probs_te.append(_prob(vd["test_scores"])); mlp_probs_te.append(_prob(md["test_scores"])); y_tes.append(vd["y_test"])

        for w in weights:
            val_prs = [average_precision_score(yv, w * pv + (1.0 - w) * pm) for yv, pv, pm in zip(y_vals, vqc_probs_val, mlp_probs_val)]
            blend_te = [w * pv + (1.0 - w) * pm for pv, pm in zip(vqc_probs_te, mlp_probs_te)]
            d_stat = paired_delta_pr_auc(y_tes[0], blend_te, mlp_probs_te, n_boot=500)
            blend_rows.append({
                "N": N, "w_vqc": w,
                "mean_val_pr_auc": float(np.mean(val_prs)),
                "mean_test_pr_auc": float(np.mean([average_precision_score(y_tes[0], b) for b in blend_te])),
                "delta_vs_mlp": d_stat["mean_delta"],
                "ci_lo": d_stat["ci_lo"],
                "ci_hi": d_stat["ci_hi"],
            })
    df_b = pd.DataFrame(blend_rows)
    df_b.to_csv(out_dir / "blend_fine_grid.csv", index=False)
    print(df_b[df_b["N"] == 100].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Check C: Sub-0.1 ZZ Quantum Kernel Bandwidth Sweep (alpha = 0.02, 0.05, 0.10, 0.20)
    print("\n=== CHECK C: Sub-0.1 ZZ Quantum Kernel Bandwidth Check (N=500, seed=100) ===")
    d = sample_and_preprocess(get_locked_splits(), n_train=500, seed=100)
    Xtr, ytr, Xva, yva, Xte, yte = d["X_train"], d["y_train"], d["X_val"], d["y_val"], d["X_test"], d["y_test"]
    off = ~np.eye(len(Xtr), dtype=bool)
    k_rows = []
    for a in [0.02, 0.05, 0.10, 0.20]:
        Ktr, Kva, Kte = _zz_gram(Xtr, Xtr, a, 1), _zz_gram(Xva, Xtr, a, 1), _zz_gram(Xte, Xtr, a, 1)
        best_c = max([0.5, 2.0, 10.0, 30.0, 100.0], key=lambda c: average_precision_score(yva, SVC(kernel="precomputed", C=c, class_weight="balanced").fit(Ktr, ytr).decision_function(Kva)))
        q_pr = float(average_precision_score(yte, SVC(kernel="precomputed", C=best_c, class_weight="balanced").fit(Ktr, ytr).decision_function(Kte)))
        oq = Ktr[off]
        k_rows.append({"alpha": a, "best_C": best_c, "off_diag_mean": float(oq.mean()), "kta": _centered_kta(Ktr, ytr), "eff_rank": _eff_rank(Ktr), "q_test_pr_auc": q_pr})
    df_k = pd.DataFrame(k_rows)
    df_k.to_csv(out_dir / "kernel_sub01_check.csv", index=False)
    print(df_k.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()