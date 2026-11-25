import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.config import RESULTS_DIR

plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "font.size": 10,
    "axes.titlesize": 11.5, "axes.titleweight": "bold", "axes.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--",
})


def _sstd(s: pd.Series) -> float:
    return 0.0 if len(s) < 2 else float(s.std(ddof=1))


def plot_main_benchmark(res_dir: Path) -> None:
    df = pd.read_csv(res_dir / "main_benchmark.csv")
    agg = df.groupby("model", sort=False)[["test_pr_auc", "test_mcc"]].agg(["mean", _sstd])
    agg.columns = ["pr_m", "pr_s", "mcc_m", "mcc_s"]
    agg = agg.sort_values("pr_m", ascending=True)
    labels = agg.index.tolist()
    y, h = np.arange(len(labels)), 0.36
    c_pr = ["#D9534F" if m == "Dummy" else ("#2B7BBA" if "Q" in m else "#4A5568") for m in labels]
    c_mc = ["#E69F9C" if m == "Dummy" else ("#6BAED6" if "Q" in m else "#A0AEC0") for m in labels]

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.barh(y + h / 2, agg["pr_m"], height=h, xerr=agg["pr_s"], capsize=3, color=c_pr, label="Test PR-AUC (Primary)")
    ax.barh(y - h / 2, agg["mcc_m"], height=h, xerr=agg["mcc_s"], capsize=3, color=c_mc, label="Test MCC")
    for yi, (m, s) in enumerate(zip(agg["pr_m"], agg["pr_s"])):
        ax.text(min(m + s + 0.015, 1.03), yi + h / 2, f"{m:.3f}", va="center", fontsize=8.5)
    ax.axvline(0.0917, color="#D9534F", linestyle=":", linewidth=1.1, label="Pulsar Prevalence Floor (0.092)")
    ax.set_yticks(y); ax.set_yticklabels(labels); ax.set_xlim(0, 1.12)
    ax.set_xlabel("Score (Mean +/- Std across Eval Seeds, N_train = 500)")
    ax.set_title("Main Benchmark: Classical Baselines vs. Quantum Models on UCI HTRU2")
    ax.legend(loc="lower right", frameon=True); fig.tight_layout()
    fig.savefig(res_dir / "main_benchmark.png"); plt.close(fig)


def plot_gold_a_entanglement(res_dir: Path) -> None:
    df = pd.read_csv(res_dir / "gold_a_entanglement.csv")
    topos = [("none", "None (Product)", "#4A5568"), ("ring", "Ring", "#2B7BBA"), ("bipartite", "Bipartite (Profile <-> DM)", "#1B9E77")]
    depths = [1, 2, 4]
    x, w = np.arange(len(depths)), 0.25
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for idx, (t, lbl, col) in enumerate(topos):
        ms = [df[df["model"] == f"VQC_{t}_L{L}"]["test_pr_auc"].mean() for L in depths]
        ss = [_sstd(df[df["model"] == f"VQC_{t}_L{L}"]["test_pr_auc"]) for L in depths]
        bars = ax.bar(x + (idx - 1) * w, ms, width=w, yerr=ss, capsize=3, color=col, label=lbl)
        for b, m in zip(bars, ms):
            ax.text(b.get_x() + b.get_width() / 2, m + 0.012, f"{m:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([f"Depth L = {L}" for L in depths]); ax.set_ylim(0.50, 1.02)
    ax.set_ylabel("Test PR-AUC"); ax.set_title("Gold A: Entanglement Topology & Circuit Depth Ablation")
    ax.legend(loc="lower right", frameon=True); fig.tight_layout()
    fig.savefig(res_dir / "gold_a_entanglement.png"); plt.close(fig)


def plot_gold_b_encoding(res_dir: Path) -> None:
    df = pd.read_csv(res_dir / "gold_b_encoding.csv")
    order = ["3q_Amplitude_L5", "4q_AmplitudePlusNorm_L4", "8q_Angle_L2", "8q_Angle_Reupload_L2"]
    disp = ["3q Amplitude (L=5)\n[Norm Discarded]", "4q Amp + Norm (L=4)\n[Norm on Wire 3]", "8q Angle (L=2)\n[Single]", "8q Angle Re-upload\n[Repeated]"]
    agg = df.groupby("model")[["test_pr_auc", "test_mcc"]].agg(["mean", _sstd]).loc[order]
    x, w = np.arange(len(order)), 0.35
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    b1 = ax.bar(x - w / 2, agg[("test_pr_auc", "mean")], width=w, yerr=agg[("test_pr_auc", "_sstd")], capsize=4, color="#2B7BBA", label="Test PR-AUC")
    ax.bar(x + w / 2, agg[("test_mcc", "mean")], width=w, yerr=agg[("test_mcc", "_sstd")], capsize=4, color="#74C476", label="Test MCC")
    for b, m in zip(b1, agg[("test_pr_auc", "mean")]):
        ax.text(b.get_x() + b.get_width() / 2, m + 0.02, f"{m:.3f}", ha="center", fontsize=8.5, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(disp); ax.set_ylim(0.40, 1.05)
    ax.set_ylabel("Score"); ax.set_title("Gold B: Quantum Encoding Compression & Radial Norm Restoration")
    ax.legend(loc="lower right", frameon=True); fig.tight_layout()
    fig.savefig(res_dir / "gold_b_encoding.png"); plt.close(fig)


def plot_gold_c_sweep(res_dir: Path) -> None:
    df = pd.read_csv(res_dir / "gold_c_sweep.csv")
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    palette = ["#1B9E77", "#4A5568", "#2B7BBA", "#D95F02", "#7570B3"]
    for idx, m_name in enumerate([m for m in df["model"].unique() if m != "Dummy"]):
        sub = df[df["model"] == m_name].groupby("n_train")["test_pr_auc"].agg(["mean", _sstd]).sort_index()
        ns, ms, ss = sub.index.values, sub["mean"].values, sub["_sstd"].values
        col = palette[idx % len(palette)]
        ax.plot(ns, ms, marker="o", linewidth=2, color=col, label=m_name)
        ax.fill_between(ns, ms - ss, ms + ss, color=col, alpha=0.14)
    ax.set_xticks(sorted(df["n_train"].unique())); ax.set_ylim(0.50, 0.98)
    ax.set_xlabel("Training Set Size (N_train)"); ax.set_ylabel("Test PR-AUC")
    ax.set_title("Gold C: Sample Efficiency Across Training Regimes")
    ax.legend(loc="lower right", frameon=True); fig.tight_layout()
    fig.savefig(res_dir / "gold_c_sweep.png"); plt.close(fig)


def plot_table_d_shots(res_dir: Path) -> None:
    df = pd.read_csv(res_dir / "table_d_shots.csv")
    df["shots"] = df["shots"].astype(str)
    order = [s for s in ["analytic", "2000", "500", "100"] if s in set(df["shots"])]
    agg = df.groupby("shots")[["test_pr_auc", "test_mcc"]].agg(["mean", _sstd]).loc[order]
    x, w = np.arange(len(order)), 0.35
    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    b1 = ax.bar(x - w / 2, agg[("test_pr_auc", "mean")], width=w, yerr=agg[("test_pr_auc", "_sstd")], capsize=4, color="#2B7BBA", label="Test PR-AUC")
    ax.bar(x + w / 2, agg[("test_mcc", "mean")], width=w, yerr=agg[("test_mcc", "_sstd")], capsize=4, color="#6BAED6", label="Test MCC")
    for b, m in zip(b1, agg[("test_pr_auc", "mean")]):
        ax.text(b.get_x() + b.get_width() / 2, m + 0.02, f"{m:.3f}", ha="center", fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels(order); ax.set_ylim(0.40, 1.04)
    ax.set_ylabel("Score"); ax.set_title("Table D: Finite Measurement Shot-Noise Degradation")
    ax.legend(loc="lower left", frameon=True); fig.tight_layout()
    fig.savefig(res_dir / "table_d_shots.png"); plt.close(fig)


def plot_mlp_ablation(res_dir: Path) -> None:
    p = res_dir / "mlp_ablation.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    agg = df.groupby("model", sort=False)[["test_pr_auc", "test_mcc"]].agg(["mean", _sstd])
    x, w = np.arange(len(agg)), 0.35
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    b1 = ax.bar(x - w / 2, agg[("test_pr_auc", "mean")], width=w, yerr=agg[("test_pr_auc", "_sstd")], capsize=4, color="#1B9E77", label="Test PR-AUC")
    ax.bar(x + w / 2, agg[("test_mcc", "mean")], width=w, yerr=agg[("test_mcc", "_sstd")], capsize=4, color="#A0AEC0", label="Test MCC")
    for b, m in zip(b1, agg[("test_pr_auc", "mean")]):
        ax.text(b.get_x() + b.get_width() / 2, m + 0.015, f"{m:.3f}", ha="center", fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels(agg.index.tolist()); ax.set_ylim(0.65, 1.02)
    ax.set_ylabel("Score"); ax.set_title("Tiny-MLP Width Ablation (21 vs. 41 vs. 81 Trainable Params)")
    ax.legend(loc="lower right", frameon=True); fig.tight_layout()
    fig.savefig(res_dir / "mlp_ablation.png"); plt.close(fig)


def main(results_dir: Path | None = None) -> None:
    if results_dir is None:
        ap = argparse.ArgumentParser()
        d_dir = RESULTS_DIR / "full" if (RESULTS_DIR / "full" / "main_benchmark.csv").exists() else RESULTS_DIR / "v1_original_10seed"
        ap.add_argument("--results-dir", type=Path, default=d_dir)
        results_dir = ap.parse_args().results_dir
    plot_main_benchmark(results_dir)
    plot_gold_a_entanglement(results_dir)
    plot_gold_b_encoding(results_dir)
    plot_gold_c_sweep(results_dir)
    plot_table_d_shots(results_dir)
    plot_mlp_ablation(results_dir)
    print(f"Generated 300-DPI figures in {results_dir}")


if __name__ == "__main__":
    main()
