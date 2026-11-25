import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pennylane as qml
import torch
from sklearn.metrics import average_precision_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.svm import SVC

from src.config import get_mode_config, get_results_dir
from src.data import get_locked_splits, sample_and_preprocess


def _zz_gram(Xa: np.ndarray, Xb: np.ndarray, alpha: float, depth: int) -> np.ndarray:
    dev = qml.device("default.qubit", wires=8)

    @qml.qnode(dev, interface="torch", diff_method=None)
    def st(xb):
        ang = 2.0 * torch.atan(float(alpha) * xb)
        for _ in range(int(depth)):
            for w in range(8):
                qml.Hadamard(wires=w); qml.RZ(ang[:, w], wires=w)
            for w in range(8):
                wn = (w + 1) % 8
                qml.CNOT(wires=[w, wn]); qml.RZ(ang[:, w] * ang[:, wn], wires=wn); qml.CNOT(wires=[w, wn])
        return qml.state()

    with torch.no_grad():
        pa = st(torch.tensor(Xa, dtype=torch.float32))
        pb = st(torch.tensor(Xb, dtype=torch.float32))
        return torch.abs(pa @ pb.conj().T).pow(2).cpu().numpy()


def _centered_kta(K: np.ndarray, y: np.ndarray) -> float:
    yp = np.where(y == 1, 1.0, -1.0)[:, None]
    H = np.eye(len(y)) - np.ones((len(y), len(y))) / len(y)
    Kc, Yc = H @ K @ H, H @ (yp @ yp.T) @ H
    return float(np.sum(Kc * Yc) / max(np.linalg.norm(Kc, "fro") * np.linalg.norm(Yc, "fro"), 1e-12))


def _eff_rank(K: np.ndarray) -> float:
    ev = np.clip(np.linalg.eigvalsh(K), 0.0, None)
    p = ev / max(ev.sum(), 1e-12)
    p = p[p > 1e-15]
    return float(np.exp(-np.sum(p * np.log(p))))


def run_kernel_analysis(cfg_name: str | None = None) -> pd.DataFrame:
    cfg = get_mode_config(cfg_name)
    res_dir = get_results_dir(cfg.name)
    d = sample_and_preprocess(get_locked_splits(), n_train=500, seed=100)
    Xtr, ytr, Xva, yva, Xte, yte = d["X_train"], d["y_train"], d["X_val"], d["y_val"], d["X_test"], d["y_test"]
    off = ~np.eye(len(Xtr), dtype=bool)

    # Tune reference RBF-SVM (gamma, C) strictly on validation
    best_rbf = (-1.0, 0.1, 1.0)
    for g in np.geomspace(1e-3, 2.0, 12):
        R_tr, R_va = rbf_kernel(Xtr, Xtr, gamma=g), rbf_kernel(Xva, Xtr, gamma=g)
        for C in [0.5, 2.0, 10.0]:
            vp = average_precision_score(yva, SVC(kernel="precomputed", C=C, class_weight="balanced").fit(R_tr, ytr).decision_function(R_va))
            if vp > best_rbf[0]:
                best_rbf = (vp, float(g), float(C))
    _, g_star, c_star = best_rbf
    R_star_tr = rbf_kernel(Xtr, Xtr, gamma=g_star)
    R_star_te = rbf_kernel(Xte, Xtr, gamma=g_star)
    rbf_star_pr = float(average_precision_score(yte, SVC(kernel="precomputed", C=c_star, class_weight="balanced").fit(R_star_tr, ytr).decision_function(R_star_te)))
    rbf_off = R_star_tr[off]

    alphas = [0.1, 0.2, 0.4, 0.8, 1.4, 2.5]
    rows = []
    for depth in [1, 2]:
        for a in alphas:
            Ktr, Kva, Kte = _zz_gram(Xtr, Xtr, a, depth), _zz_gram(Xva, Xtr, a, depth), _zz_gram(Xte, Xtr, a, depth)
            best_c = max([0.5, 2.0, 10.0, 30.0], key=lambda c: average_precision_score(yva, SVC(kernel="precomputed", C=c, class_weight="balanced").fit(Ktr, ytr).decision_function(Kva)))
            q_pr = float(average_precision_score(yte, SVC(kernel="precomputed", C=best_c, class_weight="balanced").fit(Ktr, ytr).decision_function(Kte)))
            Rm_tr, Rm_te = rbf_kernel(Xtr, Xtr, gamma=float(a ** 2)), rbf_kernel(Xte, Xtr, gamma=float(a ** 2))
            rm_pr = float(average_precision_score(yte, SVC(kernel="precomputed", C=best_c, class_weight="balanced").fit(Rm_tr, ytr).decision_function(Rm_te)))
            oq = Ktr[off]
            corr = float(np.corrcoef(oq, rbf_off)[0, 1]) if np.std(oq) > 1e-12 else 0.0
            rows.append({"depth": depth, "alpha": a, "off_diag_mean": float(oq.mean()), "off_diag_std": float(oq.std()), "kta": _centered_kta(Ktr, ytr), "eff_rank": _eff_rank(Ktr), "q_test_pr_auc": q_pr, "rbf_matched_pr_auc": rm_pr, "rbf_tuned_pr_auc": rbf_star_pr, "tuned_rbf_gamma": g_star, "corr_with_tuned_rbf": corr})

    df = pd.DataFrame(rows)
    df.to_csv(res_dir / "kernel_analysis.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.0))
    for depth, col in [(1, "#2B7BBA"), (2, "#D95F02")]:
        s = df[df["depth"] == depth]
        axes[0].plot(s["alpha"], s["off_diag_mean"], marker="o", color=col, label=f"Mean K_ij (d={depth})")
        axes[0].fill_between(s["alpha"], np.clip(s["off_diag_mean"] - s["off_diag_std"], 0, 1), np.clip(s["off_diag_mean"] + s["off_diag_std"], 0, 1), color=col, alpha=0.14)
        axes[1].plot(s["alpha"], s["kta"], marker="s", color=col, label=f"Centered KTA (d={depth})")
        axes[2].plot(s["alpha"], s["q_test_pr_auc"], marker="^", color=col, label=f"Q-Kernel (d={depth})")
    s1 = df[df["depth"] == 1]
    axes[2].plot(s1["alpha"], s1["rbf_matched_pr_auc"], linestyle="--", marker="d", color="#4B5563", label="RBF (gamma=alpha^2)")
    axes[2].axhline(rbf_star_pr, linestyle=":", color="#111827", label=f"Tuned RBF ({rbf_star_pr:.3f})")
    for ax, t in zip(axes, ["Kernel Concentration vs. alpha", "Kernel-Target Alignment vs. alpha", "Test PR-AUC vs. alpha"]):
        ax.set_xscale("log"); ax.set_xlabel("Bandwidth alpha"); ax.set_title(t, fontsize=10.5, fontweight="bold"); ax.legend(fontsize=7.5)
    fig.tight_layout(); fig.savefig(res_dir / "kernel_analysis.png", dpi=300); plt.close(fig)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "fast", "full"], default=None)
    print(run_kernel_analysis(ap.parse_args().mode).to_string(index=False))