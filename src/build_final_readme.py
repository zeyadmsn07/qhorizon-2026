import argparse, json
from pathlib import Path
import pandas as pd
from src.config import RESULTS_DIR


def build_final_readme(res_dir: Path) -> None:
    v = json.loads((res_dir / "verdicts.json").read_text(encoding="utf-8"))
    meta = json.loads((res_dir / "run_metadata.json").read_text(encoding="utf-8"))
    sel = json.loads((res_dir / "headline_vqc_selection.json").read_text(encoding="utf-8"))
    lin = pd.read_csv(res_dir / "linearity_check.csv")
    ka = pd.read_csv(res_dir / "kernel_analysis.csv")
    nl = pd.read_csv(res_dir / "nonlinear_dataset_summary.csv")
    rt = Path("docs/measured_runtimes.md").read_text(encoding="utf-8").strip() if Path("docs/measured_runtimes.md").exists() else ""
    team_lines = [ln for ln in Path("docs/TEAM.md").read_text(encoding="utf-8").splitlines() if not ln.startswith("# ")]
    team_md = "\n".join(team_lines).strip()
    rel = res_dir.as_posix()
    bt = "`" * 3

    fp = lin[lin["n_train"] > 5000].set_index("model")
    lr_fp, svm_fp = fp.loc["LogReg (9p)", "test_pr_auc"], fp.loc["RBF-SVM", "test_pr_auc"]
    b_ka = ka.sort_values("q_test_pr_auc", ascending=False).iloc[0]
    nl_max = nl[nl["n_train"] == nl["n_train"].max()].set_index("model")

    h_rows = "\n".join(
        f"| **{hid}** | `{h['comparison']}` | `{h['mean_delta']:+.4f}` | `[{h['ci_lo']:+.4f}, {h['ci_hi']:+.4f}]` | **{h['verdict']}** |"
        for hid, h in v["hypotheses"].items()
    )
    m_rows = "\n".join(
        f"| **{m}** | `{s['mean']:.4f}` | `[{s['ci_lo']:.4f}, {s['ci_hi']:.4f}]` | {s['collapse']} | `{s['delta_vs_best']:+.4f}` (`[{s['delta_lo']:+.4f}, {s['delta_hi']:+.4f}]`) | {s['status']} |"
        for m, s in v["models"].items()
    )

    banner = "" if meta["mode"] == "full" else f"\n> **Active Artifact Directory:** `{rel}` (`--mode {meta['mode']}`, `eval_seeds={meta['eval_seeds']}`). Running `python -m src.build_final_readme --results-dir results/full` updates all numbers from `results/full/` automatically.\n"
    readme = f"""# Q//Horizon 2026: Fair Quantum vs. Classical Pulsar Classification
{banner}
## 1. Research Question & Why PR-AUC Matters
How do compact (`<= 8` qubit) variational quantum classifiers and quantum fidelity kernels compare against equal-budget classical baselines on imbalanced radio-astronomy tabular data, and when do quantum feature maps actually help?

We benchmark on the **UCI HTRU2 Pulsar Candidate Dataset** (R. J. Lyon, 2017, UCI ID `372`, CC BY 4.0; offline copy in `data_cache/htru2.csv`, see [`DATA_LICENSE.md`](DATA_LICENSE.md)): `17,898` candidates, `8` summary statistics (`4` integrated pulse-profile + `4` DM-SNR curve), and **`9.16%` true pulsars**. Predicting majority noise (`0`) everywhere yields **`90.83%` accuracy** while finding zero pulsars (`0.0917` PR-AUC, `0.0000` MCC), so we rank models by **PR-AUC** alongside **MCC** and **F1**.

## 2. Models & Design Rationale
- **Train-Only Yeo-Johnson + Bounded Angle Encoding:** Fit `PowerTransformer` strictly on `X_train` to tame pulsar skewness/kurtosis tails, then map features via `2 * atan(alpha * x)` into `(-pi, pi)` with tunable bandwidth `alpha` so outliers never wrap modulo `2*pi`.
- **Circuit Topology & Norm Restoration:** We test separable (`none`), `ring`, and domain-coupled `bipartite` entanglement (`0-3 <-> 4-7`), plus 3-qubit amplitude compression (`L2` normalization discards signal magnitude) vs. 4-qubit norm restoration on wire 3.
- **Readout & Symmetry Breaking ([`docs/vqc_stability.md`](docs/vqc_stability.md)):** Standard `RY(theta)|0>` yields `<Z> = cos(theta)`, an *even* function blind to feature sign near `0.05 * randn` init—explaining `2/10` seed collapses in v1 `Ring L=2`. We test `pi/2` sign offsets, 8-wire linear readout (`41` weights, matching `Tiny-MLP`), and restarts, selecting headline `{v['headline_vqc']}` strictly by `{sel['rule']}`.
- **Quantum Fidelity Kernel (`Q2-QuantumKernel`):** 8-qubit `ZZ` feature map with `K_ij = |<psi_i | psi_j>|^2` fed to a dual SVM (exact state-vector product in CPU simulation; `O(N^2)` circuit runs on hardware).

## 3. Fairness Protocol & Statistical Results (`N_train = 500`)
Locked stratified holdouts (`3,000` test, `1,500` val), disjoint tuning seeds (`tune_seeds = {meta['tune_seeds']}`, `eval_seeds = {meta['eval_seeds']}`), equal candidate budgets, validation-only thresholds, and separate complexity columns ([`{rel}/summary_tables.md`]({rel}/summary_tables.md)). Verdicts use a mechanical `margin = 0.01` PR-AUC rule on 95% two-level bootstrap CIs ([`{rel}/verdicts.md`]({rel}/verdicts.md)):

| Model | Test PR-AUC | 95% Two-Level CI | Collapse (<0.70) | Delta vs. `{v['best_classical']}` (95% CI) | Verdict vs. Best |
| :--- | :---: | :---: | :---: | :---: | :---: |
{m_rows}

| ID | Hypothesis Comparison | Delta PR-AUC | 95% Two-Level CI | Mechanical Verdict |
| :--- | :--- | :---: | :---: | :---: |
{h_rows}

![Main Benchmark]({rel}/main_benchmark.png)

## 4. Depth of Insight ([`docs/insights.md`](docs/insights.md))
1. **Why Quantum Doesn't Win on HTRU2 (Near-Linearity):** Even on all `13,398` pool samples, 9-parameter `LogReg` (`{lr_fp:.4f}` PR-AUC) nearly matches `RBF-SVM` (`{svm_fp:.4f}`), leaving `< 0.01` PR-AUC headroom for nonlinear feature maps.
2. **Kernel Concentration & Alignment:** Across `alpha`, `Q2-QuantumKernel` peaks at `alpha={b_ka['alpha']}`, `d={int(b_ka['depth'])}` (`{b_ka['q_test_pr_auc']:.4f}` PR-AUC, `KTA={b_ka['kta']:.4f}`, rank `{b_ka['eff_rank']:.1f}`), while larger `alpha` concentrates off-diagonal overlaps.
3. **Harder Nonlinear 8D Benchmark (`ConcentricShell-8D`):** Pre-screened with `RBF - LogReg = +{nl['precheck_rbf_minus_logreg'].iloc[0]:.4f} >= 0.05`, linear `LogReg` fails (`{nl_max.loc['LogReg', 'mean_pr_auc']:.4f}`), whereas `Q2-QuantumKernel` (`{nl_max.loc['Q2-QuantumKernel (8q)', 'mean_pr_auc']:.4f}`) and `Q1-VQC` (`{nl_max.loc['Q1-VQC (Headline)', 'mean_pr_auc']:.4f}`) learn strong nonlinear boundaries alongside `Tiny-MLP` (`{nl_max.loc['Tiny-MLP (41p)', 'mean_pr_auc']:.4f}`) and `RBF-SVM` (`{nl_max.loc['RBF-SVM', 'mean_pr_auc']:.4f}`).
4. **Hardware Cost ([`{rel}/hardware_cost_table.csv`]({rel}/hardware_cost_table.csv)):** `Q1-VQC` scales as `O(N)` (`3,000` test circuits), while `Q2-QuantumKernel` needs `124,750` train pairs + `1.5M` test pairs. At `<= 8` qubits, all circuits are CPU-simulable; **no quantum advantage is claimed**.

## 5. Limitations
Noiseless `default.qubit` state-vector simulation (shot noise only at inference; no depolarizing gate noise), `O(N^2)` QPU kernel scaling, `N <= 1000` quantum training subsamples, one `1,500`-sample validation split shared across tuning/early-stopping/thresholding, and `~275` test pulsars.

## 6. Reproducibility, Credits, AI Disclosure & Team
{bt}bash
python run_all.py --mode smoke          # ~1.5 min sanity check -> results/smoke/
python run_all.py --mode fast           # fast dev check -> results/fast/
python run_all.py --mode full           # 10-seed benchmark -> results/full/
python -m src.make_tables --results-dir {rel}
python analyze_results.py --results-dir {rel}
python -m src.plot_results --results-dir {rel}
python scripts/check_submission.py      # automated submission audit
{bt}
{rt}

- **Sources & Credits:** PennyLane (`0.45.1`), PyTorch, scikit-learn, SciPy, NumPy, pandas, matplotlib; Dataset: UCI HTRU2 (`id=372`, R. J. Lyon, 2017, CC BY 4.0). Q&A rationale: [`docs/design_rationale.md`](docs/design_rationale.md).
- **AI Tool Disclosure:** Every AI-assisted step and human verification check is logged in [`AI_USAGE_LOG.md`](AI_USAGE_LOG.md) (rules archived in [`docs/ai/AGENTS.md`](docs/ai/AGENTS.md)).

{team_md}
"""
    Path("README.md").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    d = RESULTS_DIR / "full" if (RESULTS_DIR / "full" / "verdicts.json").exists() else RESULTS_DIR / "fast"
    ap.add_argument("--results-dir", type=Path, default=d)
    build_final_readme(ap.parse_args().results_dir)
    print("Built final concise README.md cleanly.")