from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import RESULTS_DIR

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
})


def plot_main_benchmark() -> None:
    df = pd.read_csv(RESULTS_DIR / "main_benchmark.csv")
    order = [
        "RBF-SVM",
        "LogReg",
        "Tiny-MLP (41p)",
        "Q2-QuantumKernel (8q)",
        "RandomForest",
        "Q1-VQC (8q Ring L=2)",
        "Dummy",
    ]
    agg = df.groupby("model")[["test_pr_auc", "test_mcc"]].agg(["mean", "std"]).loc[order]
    labels = order[::-1]
    pr_m = agg[("test_pr_auc", "mean")].loc[labels].values
    pr_s = agg[("test_pr_auc", "std")].loc[labels].values
    mcc_m = agg[("test_mcc", "mean")].loc[labels].values
    mcc_s = agg[("test_mcc", "std")].loc[labels].values

    y = np.arange(len(labels))
    h = 0.36
    colors_pr = ["#D9534F" if m == "Dummy" else ("#2B7BBA" if "Q" in m else "#4A5568") for m in labels]
    colors_mcc = ["#E69F9C" if m == "Dummy" else ("#6BAED6" if "Q" in m else "#A0AEC0") for m in labels]

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.barh(y + h / 2, pr_m, height=h, xerr=pr_s, capsize=3, color=colors_pr, label="Test PR-AUC (Primary)")
    ax.barh(y - h / 2, mcc_m, height=h, xerr=mcc_s, capsize=3, color=colors_mcc, label="Test MCC")

    for yi, m, s in zip(y, pr_m, pr_s):
        ax.text(min(m + s + 0.015, 1.03), yi + h / 2, f"{m:.3f}", va="center", fontsize=8.5)

    ax.axvline(0.0917, color="#D9534F", linestyle=":", linewidth=1.1, label="Pulsar Prevalence Floor (0.092)")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Score (Mean ± Std across 10 Seeds, N_train = 500)")
    ax.set_title("Main Benchmark: Classical Baselines vs. Quantum Models on UCI HTRU2")
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "main_benchmark.png")
    plt.close(fig)


def plot_gold_a_entanglement() -> None:
    df = pd.read_csv(RESULTS_DIR / "gold_a_entanglement.csv")
    topologies = ["none", "ring", "bipartite"]
    topo_labels = ["None (Product)", "Ring (Nearest-Neighbor)", "Bipartite (Profile ↔ DM-SNR)"]
    depths = [1, 2, 4]
    colors = ["#4A5568", "#2B7BBA", "#1B9E77"]

    x = np.arange(len(depths))
    w = 0.25
    fig, ax = plt.subplots(figsize=(8.8, 4.8))

    for idx, (topo, t_lbl, col) in enumerate(zip(topologies, topo_labels, colors)):
        means, stds = [], []
        for L in depths:
            sub = df[df["model"] == f"VQC_{topo}_L{L}"]["test_pr_auc"]
            means.append(sub.mean())
            stds.append(sub.std())
        offset = (idx - 1) * w
        bars = ax.bar(x + offset, means, width=w, yerr=stds, capsize=3, color=col, label=t_lbl)
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, m + 0.012, f"{m:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Depth L = {L}" for L in depths])
    ax.set_ylim(0.55, 1.00)
    ax.set_ylabel("Test PR-AUC (Mean ± Std across 10 Seeds)")
    ax.set_title("Gold A: Entanglement Topology & Circuit Depth Ablation (8-Qubit Angle VQC)")
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "gold_a_entanglement.png")
    plt.close(fig)


def plot_gold_b_encoding() -> None:
    df = pd.read_csv(RESULTS_DIR / "gold_b_encoding.csv")
    order = [
        "3q_Amplitude_L5",
        "4q_AmplitudePlusNorm_L4",
        "8q_Angle_L2",
        "8q_Angle_Reupload_L2",
    ]
    display = [
        "3q Amplitude (L=5)\n[Norm Discarded]",
        "4q Amplitude + Norm (L=4)\n[Norm Restored on Wire 3]",
        "8q Angle Ring (L=2)\n[Single Encoding]",
        "8q Angle Re-upload (L=2)\n[Repeated Encoding]",
    ]
    agg = df.groupby("model")[["test_pr_auc", "test_mcc"]].agg(["mean", "std"]).loc[order]
    x = np.arange(len(order))
    w = 0.35

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    b1 = ax.bar(x - w / 2, agg[("test_pr_auc", "mean")], width=w, yerr=agg[("test_pr_auc", "std")], capsize=4, color="#2B7BBA", label="Test PR-AUC")
    b2 = ax.bar(x + w / 2, agg[("test_mcc", "mean")], width=w, yerr=agg[("test_mcc", "std")], capsize=4, color="#74C476", label="Test MCC")

    for bar, m in zip(b1, agg[("test_pr_auc", "mean")]):
        ax.text(bar.get_x() + bar.get_width() / 2, m + 0.02, f"{m:.3f}", ha="center", fontsize=8.5, fontweight="bold")

    ax.annotate(
        "+0.225 PR-AUC recovery\nfrom restoring L2 norm",
        xy=(1 - w / 2, 0.843),
        xytext=(0.15, 0.94),
        arrowprops=dict(arrowstyle="->", color="#1B7837", lw=1.3),
        fontsize=9,
        fontweight="bold",
        color="#1B7837",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(display)
    ax.set_ylim(0.45, 1.05)
    ax.set_ylabel("Score (Mean ± Std across 10 Seeds)")
    ax.set_title("Gold B: Quantum Encoding Compression & Radial Norm Restoration")
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "gold_b_encoding.png")
    plt.close(fig)


def plot_gold_c_sweep() -> None:
    df = pd.read_csv(RESULTS_DIR / "gold_c_sweep.csv")
    models = [
        ("Tiny-MLP (41p)", "#1B9E77", "o", "-"),
        ("RBF-SVM", "#4A5568", "s", "--"),
        ("Q2-QuantumKernel (8q)", "#2B7BBA", "^", "-"),
        ("Q1-VQC (8q)", "#D95F02", "D", "-."),
    ]
    fig, ax = plt.subplots(figsize=(8.8, 4.8))

    for m_name, col, marker, ls in models:
        sub = df[df["model"] == m_name].groupby("n_train")["test_pr_auc"].agg(["mean", "std"]).sort_index()
        ns = sub.index.values
        means = sub["mean"].values
        stds = sub["std"].values
        ax.plot(ns, means, marker=marker, linestyle=ls, linewidth=2, color=col, label=m_name)
        ax.fill_between(ns, means - stds, means + stds, color=col, alpha=0.14)

    ax.set_xticks([100, 200, 500, 1000])
    ax.set_ylim(0.55, 0.98)
    ax.set_xlabel("Training Set Size (N_train)")
    ax.set_ylabel("Test PR-AUC (Mean ± 1 Std across 10 Seeds)")
    ax.set_title("Gold C: Sample Efficiency Across Low-to-Moderate Training Regimes")
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "gold_c_sweep.png")
    plt.close(fig)


def plot_table_d_shots() -> None:
    df = pd.read_csv(RESULTS_DIR / "table_d_shots.csv")
    order = ["analytic", "2000", "500", "100"]
    labels = ["Analytic\n(Exact)", "2,000 Shots", "500 Shots", "100 Shots"]
    agg = df.groupby("shots")[["test_pr_auc", "test_mcc"]].agg(["mean", "std"]).loc[order]

    x = np.arange(len(order))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    b1 = ax.bar(x - w / 2, agg[("test_pr_auc", "mean")], width=w, yerr=agg[("test_pr_auc", "std")], capsize=4, color="#2B7BBA", label="Test PR-AUC")
    b2 = ax.bar(x + w / 2, agg[("test_mcc", "mean")], width=w, yerr=agg[("test_mcc", "std")], capsize=4, color="#6BAED6", label="Test MCC")

    for bar, m in zip(b1, agg[("test_pr_auc", "mean")]):
        ax.text(bar.get_x() + bar.get_width() / 2, m + 0.02, f"{m:.3f}", ha="center", fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.50, 1.02)
    ax.set_ylabel("Score (Mean ± Std across 10 Seeds)")
    ax.set_title("Table D: Finite Measurement Shot-Noise Degradation at Inference")
    ax.legend(loc="lower left", frameon=True)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "table_d_shots.png")
    plt.close(fig)


def main() -> None:
    plot_main_benchmark()
    plot_gold_a_entanglement()
    plot_gold_b_encoding()
    plot_gold_c_sweep()
    plot_table_d_shots()
    print("Generated all 5 high-DPI PNG figures in results/.")


if __name__ == "__main__":
    main()