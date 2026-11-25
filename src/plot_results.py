import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.config import COLLAPSE_PR_AUC, RESULTS_DIR
from src.stats import two_level_bootstrap_pr_auc

plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "font.size": 9.5, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--",
})


def _ci(res_dir: Path, sub: pd.DataFrame, col: str = "test_pr_auc") -> tuple[float, float, float]:
    vals = sub[col].to_numpy(dtype=np.float64)
    mv = float(np.mean(vals))
    if col == "test_pr_auc" and "pred_file" in sub.columns and sub["pred_file"].notna().all():
        p0 = res_dir / str(sub["pred_file"].iloc[0])
        if p0.exists():
            ord_s = sub.sort_values("seed")
            yt = np.load(res_dir / str(ord_s["pred_file"].iloc[0]))["y_test"]
            sc = [np.load(res_dir / str(p))["test_scores"] for p in ord_s["pred_file"]]
            return two_level_bootstrap_pr_auc(yt, sc, n_boot=400)
    sd = 0.0 if len(vals) < 2 else float(np.std(vals, ddof=1))
    return mv, max(0.0, mv - sd), min(1.0, mv + sd)


def plot_main_benchmark(res_dir: Path) -> None:
    df = pd.read_csv(res_dir / "main_benchmark.csv")
    rows = []
    for m in df["model"].unique():
        sub = df[df["model"] == m]
        pm, plo, phi = _ci(res_dir, sub, "test_pr_auc")
        mm, mlo, mhi = _ci(res_dir, sub, "test_mcc")
        rows.append((m, pm, max(0.0, pm - plo), max(0.0, phi - pm), mm, max(0.0, mm - mlo), max(0.0, mhi - mm)))
    rows.sort(key=lambda r: r[1])
    lbls = [r[0] for r in rows]
    y, h = np.arange(len(lbls)), 0.36
    c1 = ["#D9534F" if m == "Dummy" else ("#2B7BBA" if "Q" in m else "#4A5568") for m in lbls]
    c2 = ["#E69F9C" if m == "Dummy" else ("#6BAED6" if "Q" in m else "#A0AEC0") for m in lbls]
    fig, ax = plt.subplots(figsize=(9.8, 5.4))
    ax.barh(y + h / 2, [r[1] for r in rows], height=h, xerr=[[r[2] for r in rows], [r[3] for r in rows]], capsize=3, color=c1, label="Test PR-AUC (95% CI)")
    ax.barh(y - h / 2, [r[4] for r in rows], height=h, xerr=[[r[5] for r in rows], [r[6] for r in rows]], capsize=3, color=c2, label="Test MCC")
    marked = False
    for yi, r in enumerate(rows):
        ax.text(min(r[1] + r[3] + 0.015, 1.03), yi + h / 2, f"{r[1]:.3f}", va="center", fontsize=8.5)
        cv = df[(df["model"] == r[0]) & (df["test_pr_auc"] < COLLAPSE_PR_AUC)]["test_pr_auc"]
        if len(cv) > 0:
            ax.scatter(cv, np.full(len(cv), yi + h / 2), facecolors="none", edgecolors="#B2182B", linewidths=1.5, s=42, zorder=5, label="Collapsed seed (<0.70)" if not marked else "")
            marked = True
    ax.axvline(0.0917, color="#D9534F", linestyle=":", linewidth=1.1, label="Pulsar Floor (0.092)")
    ax.set_yticks(y); ax.set_yticklabels(lbls); ax.set_xlim(0, 1.14)
    ax.set_xlabel("Score (Mean & Two-Level 95% Bootstrap CI, N_train = 500)")
    ax.set_title("Main Benchmark: Classical vs. Quantum Models (Hollow = Collapsed Seed)")
    ax.legend(loc="lower right", frameon=True); fig.tight_layout()
    fig.savefig(res_dir / "main_benchmark.png"); plt.close(fig)


def plot_bar_table(res_dir: Path, csv_name: str, png_name: str, title: str, x_col: str = "model") -> None:
    p = res_dir / csv_name
    if not p.exists():
        return
    df = pd.read_csv(p)
    df[x_col] = df[x_col].astype(str)
    keys = list(df[x_col].unique())
    prs = [_ci(res_dir, df[df[x_col] == k], "test_pr_auc") for k in keys]
    mcs = [_ci(res_dir, df[df[x_col] == k], "test_mcc") for k in keys]
    x, w = np.arange(len(keys)), 0.36
    fig, ax = plt.subplots(figsize=(max(8.2, 1.15 * len(keys)), 4.8))
    b1 = ax.bar(x - w / 2, [r[0] for r in prs], width=w, yerr=[[max(0.0, r[0] - r[1]) for r in prs], [max(0.0, r[2] - r[0]) for r in prs]], capsize=3, color="#2B7BBA", label="Test PR-AUC (95% CI)")
    ax.bar(x + w / 2, [r[0] for r in mcs], width=w, yerr=[[max(0.0, r[0] - r[1]) for r in mcs], [max(0.0, r[2] - r[0]) for r in mcs]], capsize=3, color="#74C476", label="Test MCC")
    for xi, (b, r, k) in enumerate(zip(b1, prs, keys)):
        ax.text(b.get_x() + b.get_width() / 2, min(r[2] + 0.015, 1.03), f"{r[0]:.3f}", ha="center", fontsize=8)
        cv = df[(df[x_col] == k) & (df["test_pr_auc"] < COLLAPSE_PR_AUC)]["test_pr_auc"]
        if len(cv) > 0:
            ax.scatter(np.full(len(cv), xi - w / 2), cv, facecolors="none", edgecolors="#B2182B", s=38, zorder=5)
    ax.set_xticks(x); ax.set_xticklabels(keys, rotation=18 if len(keys) > 5 else 0, ha="right" if len(keys) > 5 else "center")
    ax.set_ylim(0.30, 1.08); ax.set_ylabel("Score (95% Bootstrap CI)"); ax.set_title(title)
    ax.legend(loc="lower right", frameon=True); fig.tight_layout()
    fig.savefig(res_dir / png_name); plt.close(fig)


def plot_gold_c_sweep(res_dir: Path) -> None:
    df = pd.read_csv(res_dir / "gold_c_sweep.csv")
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    pal = ["#1B9E77", "#4A5568", "#2B7BBA", "#D95F02", "#7570B3"]
    for i, m in enumerate([k for k in df["model"].unique() if k != "Dummy"]):
        ns = sorted(df["n_train"].unique())
        cis = [_ci(res_dir, df[(df["model"] == m) & (df["n_train"] == n)]) for n in ns]
        col = pal[i % len(pal)]
        ax.plot(ns, [c[0] for c in cis], marker="o", linewidth=2, color=col, label=m)
        ax.fill_between(ns, [c[1] for c in cis], [c[2] for c in cis], color=col, alpha=0.14)
    lin_p = res_dir / "linearity_check.csv"
    if lin_p.exists():
        ldf = pd.read_csv(lin_p)
        fp = ldf[ldf["n_train"] > 5000]
        if not fp.empty:
            c_svm = float(fp[fp["model"] == "RBF-SVM"]["test_pr_auc"].iloc[0])
            c_lr = float(fp[fp["model"] == "LogReg (9p)"]["test_pr_auc"].iloc[0])
            ax.axhline(c_svm, color="#111827", linestyle="--", linewidth=1.2, label=f"Full-Pool RBF Ceiling ({c_svm:.3f})")
            ax.axhline(c_lr, color="#6B7280", linestyle=":", linewidth=1.2, label=f"Full-Pool LogReg ({c_lr:.3f})")
    ax.set_xticks(sorted(df["n_train"].unique())); ax.set_ylim(0.45, 0.99)
    ax.set_xlabel("Training Set Size (N_train)"); ax.set_ylabel("Test PR-AUC (95% Bootstrap CI)")
    ax.set_title("Gold C: Sample Efficiency Sweep with Two-Level 95% CIs")
    ax.legend(loc="lower right", frameon=True); fig.tight_layout()
    fig.savefig(res_dir / "gold_c_sweep.png"); plt.close(fig)


def main(results_dir: Path | None = None) -> None:
    if results_dir is None:
        ap = argparse.ArgumentParser()
        d_dir = RESULTS_DIR / "full" if (RESULTS_DIR / "full" / "main_benchmark.csv").exists() else RESULTS_DIR / "fast"
        ap.add_argument("--results-dir", type=Path, default=d_dir)
        results_dir = ap.parse_args().results_dir
    plot_main_benchmark(results_dir)
    plot_bar_table(results_dir, "gold_a_entanglement.csv", "gold_a_entanglement.png", "Gold A: Entanglement & Depth (95% CIs)")
    plot_bar_table(results_dir, "gold_b_encoding.csv", "gold_b_encoding.png", "Gold B: Encoding & Radial Norm Restoration (95% CIs)")
    plot_gold_c_sweep(results_dir)
    plot_bar_table(results_dir, "table_d_shots.csv", "table_d_shots.png", "Table D: Finite Shot-Noise Degradation (95% CIs)", x_col="shots")
    plot_bar_table(results_dir, "mlp_ablation.csv", "mlp_ablation.png", "Tiny-MLP Width Ablation (21 vs. 41 vs. 81 Params)")
    plot_bar_table(results_dir, "vqc_stability_ablation.csv", "vqc_stability_ablation.png", "Phase 5: VQC Stability & Readout Ablation (95% CIs)")
    print(f"Rendered 300-DPI CI figures in {results_dir}")


if __name__ == "__main__":
    main()