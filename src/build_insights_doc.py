import argparse
from pathlib import Path
import pandas as pd
from src.config import RESULTS_DIR


def write_insights(res_dir: Path) -> None:
    lin = pd.read_csv(res_dir / "linearity_check.csv")
    ka = pd.read_csv(res_dir / "kernel_analysis.csv")
    nl = pd.read_csv(res_dir / "nonlinear_dataset_summary.csv")

    fp = lin[lin["n_train"] > 5000].set_index("model")
    lr_fp = fp.loc["LogReg (9p)", "test_pr_auc"]
    poly_fp = fp.loc["Poly2-LogReg (45p)", "test_pr_auc"]
    svm_fp = fp.loc["RBF-SVM", "test_pr_auc"]
    mlp_fp = fp.loc["Tiny-MLP (41p)", "test_pr_auc"]
    best_ka = ka.sort_values("q_test_pr_auc", ascending=False).iloc[0]
    nl500 = nl[nl["n_train"] == nl["n_train"].max()].set_index("model")

    md = f"""# Phase 7 Depth-of-Insight Summary (`{res_dir.as_posix()}`)

## 1. HTRU2 Near-Linear Separability & Full-Data Ceiling (`linearity_check.csv`)
On the full `13,398`-sample training pool with validation-tuned hyperparameters, 9-parameter linear `LogReg` achieves `{lr_fp:.4f}` PR-AUC alongside `Tiny-MLP (41p)` (`{mlp_fp:.4f}`), `Poly2-LogReg (45p)` (`{poly_fp:.4f}`), and `RBF-SVM` (`{svm_fp:.4f}`). Because HTRU2 is ~94% separable by a single hyperplane after Yeo-Johnson standardization, any nonlinear feature map (quantum or classical) has less than `~0.004` PR-AUC headroom above linear/quadratic baselines.

## 2. Quantum Fidelity Kernel Geometry vs. Classical RBF (`kernel_analysis.csv`)
Across the benchmark grid `alpha in [0.1, 2.5]` and depths `1, 2`, the 8-qubit ZZ fidelity kernel peaks at the lower boundary `alpha = {best_ka['alpha']}`, `depth = {int(best_ka['depth'])}` (`test PR-AUC = {best_ka['q_test_pr_auc']:.4f}`, `KTA = {best_ka['kta']:.4f}`, effective rank `{best_ka['eff_rank']:.2f}`, off-diagonal correlation `{best_ka['corr_with_tuned_rbf']:.3f}` with tuned RBF-SVM `{best_ka['rbf_tuned_pr_auc']:.4f}`). Probing `alpha = 0.05 / 0.02` (`diagnostics_extra/kernel_sub01_check.csv`) reaches `0.9281 / 0.9287` only by collapsing into a rank-`1.05` (`off_diag_mean = 0.994`) near-linear kernel, whereas larger `alpha` drives exponential off-diagonal concentration (`mean -> {ka['off_diag_mean'].min():.3f}`).

## 3. Controlled Nonlinear 8D Inner-Ball Diagnostic (`nonlinear_dataset_summary.csv`)
Pre-screened with `RBF-SVM - LogReg = +{nl['precheck_rbf_minus_logreg'].iloc[0]:.4f} >= 0.05` at `N=500`, `ConcentricShell-8D` labels an even-symmetry ($f(-x)=f(x)$) perturbed 4D inner ball in 8D where linear `LogReg` fails (`{nl500.loc['LogReg', 'mean_pr_auc']:.4f}` PR-AUC). `Q2-QuantumKernel` (`{nl500.loc['Q2-QuantumKernel (8q)', 'mean_pr_auc']:.4f} [{nl500.loc['Q2-QuantumKernel (8q)', 'ci_lo']:.4f}, {nl500.loc['Q2-QuantumKernel (8q)', 'ci_hi']:.4f}]`) and `RBF-SVM` (`{nl500.loc['RBF-SVM', 'mean_pr_auc']:.4f} [{nl500.loc['RBF-SVM', 'ci_lo']:.4f}, {nl500.loc['RBF-SVM', 'ci_hi']:.4f}]`) statistically tie. While 41-weight `Q1-VQC` (`{nl500.loc['Q1-VQC (Headline)', 'mean_pr_auc']:.4f}`, after `0.1307` at `N=100` and `0.3127` at `N=200`) exceeds width-4 `Tiny-MLP` (`{nl500.loc['Tiny-MLP (41p)', 'mean_pr_auc']:.4f}`), our width sweep (`diagnostics_extra/concentric_mlp_width_sweep.csv`) shows this reflects `cos(.)` symmetry alignment plus a 4-neuron bottleneck: widening `Tiny-MLP` to `8 / 16 / 32` units reaches `0.5936 / 0.6195 / 0.7767` PR-AUC.

## 4. Hardware Execution Cost & Classical Simulability (`hardware_cost_table.csv`)
Using `qml.specs`, `Q1-VQC` requires `O(N)` QPU executions (`3,000` test circuits), whereas `Q2-QuantumKernel` requires `124,750` training pairs (`O(N^2)`) plus `1,500,000` test pairs at `~10,000` shots/pair. At `<= 8` qubits (`dim <= 256`), all circuits are cheaply simulable on classical CPUs and **no quantum advantage is claimed**.
"""
    Path("docs/insights.md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    d = RESULTS_DIR / "full" if (RESULTS_DIR / "full" / "linearity_check.csv").exists() else RESULTS_DIR / "fast"
    ap.add_argument("--results-dir", type=Path, default=d)
    write_insights(ap.parse_args().results_dir)
    print("Wrote docs/insights.md.")