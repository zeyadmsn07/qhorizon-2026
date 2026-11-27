import glob
import os
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

PRED_DIR = os.path.join("results", "full", "predictions")
OUT_DIR = os.path.join("results", "full", "bonus")
os.makedirs(OUT_DIR, exist_ok=True)

def load_npz_preds(path):
    d = np.load(path, allow_pickle=True)
    y = np.asarray(d["y_test"]).ravel().astype(int)
    s = np.asarray(d["test_scores"]).ravel().astype(float)
    thr = float(np.asarray(d["threshold"]).ravel()[0]) if "threshold" in d.keys() else None
    # Convert raw logits to [0, 1] probabilities if scores fall outside [0, 1]
    if np.min(s) < 0.0 or np.max(s) > 1.0:
        p = 1.0 / (1.0 + np.exp(-np.clip(s, -30.0, 30.0)))
    else:
        p = s.copy()
    return y, s, p, thr

def rank_pct(x):
    order = np.argsort(np.argsort(x))
    return (order + 1.0) / len(x)

rows = []
for n_samples in [100, 200, 500, 1000]:
    for seed in range(100, 110):
        vqc_candidates = sorted(glob.glob(os.path.join(PRED_DIR, f"Q1-VQC__*Bipartite*__N{n_samples}__seed{seed}.npz")))
        mlp_candidates = [
            p for p in sorted(glob.glob(os.path.join(PRED_DIR, f"Tiny-MLP__*41p*__N{n_samples}__seed{seed}.npz")))
            if "h_2" not in p and "h_8" not in p
        ]
        if not vqc_candidates or not mlp_candidates:
            continue

        y_vqc, s_vqc, p_vqc, thr_vqc = load_npz_preds(vqc_candidates[0])
        y_mlp, s_mlp, p_mlp, thr_mlp = load_npz_preds(mlp_candidates[0])
        assert np.array_equal(y_vqc, y_mlp), f"Label mismatch at N={n_samples}, seed={seed}"
        y = y_vqc

        p_blend = 0.5 * p_vqc + 0.5 * p_mlp
        r_blend = 0.5 * rank_pct(s_vqc) + 0.5 * rank_pct(s_mlp)

        pos = y == 1
        k = int(pos.sum())

        # 1. Top-K comparison (K = true positive count in test set)
        top_vqc = np.zeros_like(y, dtype=bool)
        top_mlp = np.zeros_like(y, dtype=bool)
        top_vqc[np.argsort(s_vqc)[-k:]] = True
        top_mlp[np.argsort(s_mlp)[-k:]] = True

        # 2. Operating-point comparison using saved validation threshold
        pred_vqc = s_vqc >= thr_vqc if thr_vqc is not None else top_vqc
        pred_mlp = s_mlp >= thr_mlp if thr_mlp is not None else top_mlp

        pos_corr = float(np.corrcoef(p_vqc[pos], p_mlp[pos])[0, 1]) if pos.sum() > 2 else np.nan

        rows.append({
            "N": n_samples,
            "seed": seed,
            "n_test": len(y),
            "n_pos": k,
            "vqc_pr_auc": average_precision_score(y, s_vqc),
            "mlp_pr_auc": average_precision_score(y, s_mlp),
            "prob_blend_pr_auc": average_precision_score(y, p_blend),
            "rank_blend_pr_auc": average_precision_score(y, r_blend),
            "vqc_roc_auc": roc_auc_score(y, s_vqc),
            "mlp_roc_auc": roc_auc_score(y, s_mlp),
            "prob_blend_roc_auc": roc_auc_score(y, p_blend),
            "topk_both_tp": int(np.sum(pos & top_vqc & top_mlp)),
            "topk_vqc_only_tp": int(np.sum(pos & top_vqc & ~top_mlp)),
            "topk_mlp_only_tp": int(np.sum(pos & ~top_vqc & top_mlp)),
            "thr_both_tp": int(np.sum(pos & pred_vqc & pred_mlp)),
            "thr_vqc_only_tp": int(np.sum(pos & pred_vqc & ~pred_mlp)),
            "thr_mlp_only_tp": int(np.sum(pos & ~pred_vqc & pred_mlp)),
            "pos_prob_corr": pos_corr,
        })

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT_DIR, "exp3_disagreement_runs.csv"), index=False)

summary = df.groupby("N").agg(
    seeds=("seed", "count"),
    vqc_pr=("vqc_pr_auc", "mean"),
    mlp_pr=("mlp_pr_auc", "mean"),
    prob_blend_pr=("prob_blend_pr_auc", "mean"),
    rank_blend_pr=("rank_blend_pr_auc", "mean"),
    topk_both=("topk_both_tp", "mean"),
    topk_vqc_only=("topk_vqc_only_tp", "mean"),
    topk_mlp_only=("topk_mlp_only_tp", "mean"),
    thr_both=("thr_both_tp", "mean"),
    thr_vqc_only=("thr_vqc_only_tp", "mean"),
    thr_mlp_only=("thr_mlp_only_tp", "mean"),
    pos_corr=("pos_prob_corr", "mean"),
).reset_index()
summary.to_csv(os.path.join(OUT_DIR, "exp3_disagreement_summary.csv"), index=False)

print("\n=== EXPERIMENT 3: HYBRID ENSEMBLE & PULSAR DISAGREEMENT (10 SEEDS) ===")
print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
