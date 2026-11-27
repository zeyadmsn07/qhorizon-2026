import argparse
import csv
from pathlib import Path


def csv_to_md(csv_path: Path) -> str:
    if not csv_path.exists():
        return "_Artifact not found._"
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return "_Empty table._"
    header = rows[0]
    body = rows[1:]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join([":---"] * len(header)) + " |",
    ]
    for r in body:
        formatted = []
        for val in r:
            try:
                num = float(val)
                formatted.append(f"{num:.4f}" if "." in val and abs(num) < 100 else val)
            except ValueError:
                formatted.append(val)
        lines.append("| " + " | ".join(formatted) + " |")
    return "\n".join(lines)


def strip_h1(text: str) -> str:
    return "\n".join(ln for ln in text.splitlines() if not ln.startswith("# ")).strip()


def build_readme(results_dir: Path) -> str:
    bt = chr(96) * 3
    summary_md = strip_h1((results_dir / "summary_tables.md").read_text(encoding="utf-8"))
    verdicts_md = strip_h1((results_dir / "verdicts.md").read_text(encoding="utf-8"))
    nonlinear_md = csv_to_md(results_dir / "nonlinear_dataset_summary.csv")
    hw_md = csv_to_md(results_dir / "hardware_cost_table.csv")

    team_md = ""
    if Path("docs/TEAM.md").exists():
        team_md = strip_h1(Path("docs/TEAM.md").read_text(encoding="utf-8"))

    rt_md = ""
    if Path("docs/measured_runtimes.md").exists():
        rt_md = strip_h1(Path("docs/measured_runtimes.md").read_text(encoding="utf-8"))

    res_rel = results_dir.as_posix()

    return f"""# Q//Horizon 2026: Quantum vs. Classical Pulsar Classification

**An end-to-end, statistically rigorous benchmark comparing Variational Quantum Classifiers (VQC) and Quantum Kernels against matched classical baselines on real radio-telescope pulsar data (UCI HTRU2) and a controlled nonlinear 8D diagnostic.**

---

### What We Tested & What We Found

1. **The Problem & Why Accuracy Is a Trap:** The UCI HTRU2 pulsar dataset (R. J. Lyon et al., 2016) contains **17,898** radio-telescope candidate signals, of which only **9.16%** are real pulsars. A trivial model that predicts *"not a pulsar"* every time achieves **90.83% accuracy** while catching zero pulsars. We therefore evaluate all models using **PR-AUC** (Precision-Recall Area Under the Curve) and **MCC** (Matthews Correlation Coefficient) across **10 independent evaluation seeds (`100..109`)** after tuning hyperparameters strictly on **3 disjoint validation seeds (`200..202`)**.
2. **Classical Models Win on HTRU2—Because Pulsar Data Is Mostly Linear:** On `N = 500` training samples, our weight-matched **Tiny-MLP (41 weights)** achieves **`0.9246 ± 0.0079` PR-AUC** (practically tied with **RBF-SVM** at **`0.9227 ± 0.0036`**), ahead of **Q2-QuantumKernel (8 qubits, `0.9116 ± 0.0057`)** and **Q1-VQC (41 weights, `0.9102 ± 0.0025`)**. Against **Random Forest (`0.9016 ± 0.0089`)**, `Q2-QuantumKernel` achieves a statistically supported `+0.0101` lead (**H1c Supported**), whereas `Q1-VQC`'s `+0.0086` edge falls inside our `±0.01` practical-tie band. Full-pool diagnostics (`N = 13,398`) confirm that 9-weight **Logistic Regression (`0.9377` PR-AUC)** nearly matches **Tiny-MLP (`0.9394`)** and **Poly2-LogReg (`0.9411`)**, leaving minimal headroom for nonlinear feature maps.
3. **Disjoint Multi-Seed Tuning & Joint Readout/Wiring Design Stabilized VQC:** In our initial single-seed-tuned baseline (`v1`), `Ring L=2` collapsed below `0.70` PR-AUC on `2/10` seeds (`0.7800 ± 0.1437`). Moving to disjoint 3-seed validation tuning (`TUNE_SEEDS = [200, 201, 202]`) already eliminated all collapses on `Ring L=2` (`0/10` collapses, `0.8860 ± 0.0062`). Single-factor circuit modifications on `Ring L=2`—adding a $\\pi/2$ sign offset alone (`0.8854`) or 8-wire linear readout alone (`0.8869`)—produced negligible change; **only the joint combination** of sparse **bipartite entanglement** (`8` vs. `16` CNOTs; circuit depth `7` vs. `21`) + $\\pi/2$ sign offset + 8-wire linear readout (`Bipartite_SignLinear_41p`) lifted PR-AUC to **`0.9102 ± 0.0025`** (`+0.0242` over `Ring L=2`, `+0.0121` over plain `VQC_bipartite_L2`).
4. **Inductive-Bias Alignment & Width Bottlenecks on `ConcentricShell-8D`:** On a synthetic 8D radial inner-ball benchmark ($r^2 = \\sum_{{i=0}}^3 x_i^2 + 0.35\\sin(2.5 x_0 x_1) \\le q_{{0.12}}$ with 4 nuisance features) where linear `LogReg` fails (`0.1165` PR-AUC), **Q2-QuantumKernel (`0.8832 [0.8512, 0.9104]`)** and **RBF-SVM (`0.8695 [0.8398, 0.8974]`)** are **statistically tied** (overlapping 95% CIs). At `N = 500`, 41-weight **Q1-VQC (`0.6602`)** exceeds the 4-neuron **Tiny-MLP (`0.4075`)** after nearly failing at `N = 100/200` (`0.1307` / `0.3127`), because this task's even coordinate symmetry ($f(-x)=f(x)$) aligns with $\\cos(\\cdot)$ angle encoding while 4 hidden Tanh units bottleneck radial closure: widening `Tiny-MLP` to `8 / 16 / 32` units lifts `N = 500` PR-AUC to `0.5936 / 0.6195 / 0.7767`, surpassing `Q1-VQC`.

<!-- dashboard-guide:start -->
## How to View the Interactive Dashboard

We built an interactive page that shows every result in this study as charts you can hover over, switch between datasets, and explore experiment by experiment. It uses the same numbers as the tables in this README; nothing is re-run.

### Option 1: Open it in your browser (no installation, no internet needed)
1. Download this repository (green **Code** button, then **Download ZIP**) and unzip it. Alternatively, open `dashboard/qhorizon_dashboard.html` on GitHub and click **Download raw file**.
2. Double-click `qhorizon_dashboard.html`. It opens in any modern browser (Chrome, Edge, Firefox or Safari).

### Option 2: Open the online version
The same page is hosted with GitHub Pages: **https://zeyadmsn07.github.io/qhorizon-2026/**

### What You Will See
* **Overview:** the task, the 8 measurements per signal, and why we score models with PR-AUC instead of plain accuracy.
* **Datasets:** a switch between the real pulsar survey (`HTRU2`) and a synthetic radial inner-ball test (`ConcentricShell-8D`).
* **Findings:** a dot-plot ranking every model. Blue marks classical models and purple marks quantum models.
* **Experiments:** six short studies, each with a chart, a "What happened" box and a "What we can conclude" box.
* **Audit tables:** the exact 10-seed numbers and the verdict on each hypothesis.

### Optional: Refresh the Page From the Result Files
Only needed if `results/full/` is regenerated. Requires Python and `pip install -r requirements.txt`.
{bt}bash
python dashboard/serve.py     # rebuilds the data and opens http://localhost:8000
{bt}
<!-- dashboard-guide:end -->

---

## 1. Fair Benchmark Protocol & Main Results (`N_train = 500`)

### How We Kept Every Comparison Fair
* **Exact Weight Parity:** `Q1-VQC (Bipartite_SignLinear_41p)` and `Tiny-MLP (41p)` both have **41 trainable scalar weights** and train with minibatch AdamW + binary cross-entropy early stopping, supplemented by width ablations (`21p` to `321p`) so architectural bottlenecks are never hidden behind parameter matching.
* **Identical Kernel Solver:** `Q2-QuantumKernel (8q)` and `RBF-SVM` both use the exact same dual Support Vector Machine solver (`0` trainable weights); only the kernel matrix changes.
* **Zero Data Leakage:** Yeo-Johnson power scaling and standardization are fit **only on the training split** inside each seed, evaluated against a locked `3,000`-sample test set and `1,500`-sample validation set.

![Main Benchmark]({res_rel}/main_benchmark.png)

> **Reading the Main Benchmark Chart:** Bars and error markers show mean PR-AUC and 95% two-level bootstrap confidence intervals across 10 seeds (`100..109`). Classical `Tiny-MLP (41p)` and `RBF-SVM` sit practically tied at the top (~`0.923–0.925`), followed closely by `LogReg` (`0.9184`). `Q2-QuantumKernel` (`0.9116`) beats `RandomForest` (`0.9016`) beyond our `±0.01` margin (`+0.0101`, **Supported**), while `Q1-VQC` (`0.9102`) sits `+0.0086` above `RandomForest` (within the `±0.01` tie band) with zero training collapses.

{summary_md}

---

## 2. Mechanical Verdict Rules & All 10 Hypothesis Outcomes

Rather than overclaiming *"Confirmed"* from noisy single-seed p-values, every hypothesis is graded automatically by `analyze_results.py` using a **Mechanical Verdict** rule with a **$\\pm 0.01$ PR-AUC practical margin** and a **95% two-level bootstrap confidence interval** (resampling both the 10 seeds and the 3,000 test examples):

* **Supported:** The entire 95% CI lies above `0` and the mean gain is greater than `0.01` PR-AUC.
* **Practically tied:** The entire 95% CI lies strictly inside `[-0.01, +0.01]` (no meaningful practical difference).
* **Contradicted:** The entire 95% CI lies below `0` and the drop exceeds `0.01` PR-AUC (opposite of the hypothesis).
* **Inconclusive:** A directional trend is visible, but the 95% CI either touches `0` or crosses the `±0.01` margin boundary.

{verdicts_md}

### Plain-English Guide to the Verdicts
* **H1a, H1b, H1c (Main Leaderboard):** Classical `Tiny-MLP (41p)` beats both quantum models by `+0.0130` to `+0.0144` PR-AUC (**Supported**), while `Q2-QuantumKernel` beats `RandomForest` by `+0.0101` (**Supported**).
* **H2a & H2b (Entanglement Wiring):** Sparse bipartite wiring beats dense ring wiring by `+0.0120` PR-AUC with half the CNOTs (**H2b Supported**), whereas bipartite vs. unentangled (`+0.0110`) touches `-0.0006` at the lower CI bound (**H2a Inconclusive**) because HTRU2 is largely linear.
* **H3 (Signal Magnitude / Radial Norm):** Standard amplitude encoding normalizes vectors to length 1, erasing radio-signal strength; restoring the radial norm on qubit 3 boosts PR-AUC by **`+0.2450`** (**Supported**).
* **H4a (Low-Data Regime & Hybrid Blend Check):** At $N = 100$ training samples, standalone `RBF-SVM` beats `Q2-QuantumKernel` by `0.0460` PR-AUC (**Contradicted**). Blending `Q1-VQC` + `Tiny-MLP` probabilities at $N = 100$ yields a smooth interior validation optimum around $w_{{\\text{{VQC}}}} = 0.20\\text{{--}}0.30$ (`test PR-AUC = 0.9132`, `+0.0091 [+0.0009, +0.0200]` over `Tiny-MLP` alone; `+0.0077` at $w = 0.10$), a modest low-data synergy that remains within the `±0.01` practical margin and disappears by $N \\ge 500$ (`+0.0009`).
* **H4b (Measurement Shot Noise):** Running `2,000` or `500` shots per circuit stays within `0.0011` PR-AUC of exact statevector simulation (**Practically tied**), while `100` shots begins showing mild sampling variance (`-0.0068`, **Inconclusive**).

---

## 3. Focused Ablations & VQC Stability Diagnostics

### A. Disentangling Multi-Seed Tuning, Readout & Entanglement Topology (Gold A & Phase 5)
* **What Actually Stopped VQC Collapse:** In `v1`, tuning hyperparameters on a single seed (`seed 100`) selected an overly narrow input scaling (`alpha = 0.25`) where standard `RY` + $\\langle Z \\rangle = \\cos(\\theta)$ readout starts nearly even ($f(-x)=f(x)$), causing `2/10` evaluation seeds to stall below `0.70` (`0.7800 ± 0.1437`). Switching to disjoint 3-seed validation tuning (`seeds 200..202`) already eliminated collapses on `Ring_L2_Baseline` (`0/10` collapses, `0.8860 ± 0.0062`).
* **Single Factors vs. Synergistic Combination (`docs/vqc_stability.md`):** On `Ring L=2`, adding a $\\pi/2$ sign offset alone (`0.8854`) or an 8-wire linear readout alone (`0.8869`) yielded no standalone gain, while switching `mean_z` from ring to bipartite wiring gave `+0.0120` (`VQC_bipartite_L2 = 0.8981`, **H2b Supported**). Combining **bipartite wiring + $\\pi/2$ sign offset + 8-wire linear readout** (`Bipartite_SignLinear_41p`, `depth 7`, `8` CNOTs) produced an additional `+0.0121` jump to **`0.9102 ± 0.0025`** (`2.5×` lower seed std).

| VQC Training Stability & Loss Curves | Entanglement Topology & Depth (Gold A) |
| :---: | :---: |
| ![VQC Loss Diagnostics]({res_rel}/vqc_loss_diagnostics.png) | ![Gold A Entanglement]({res_rel}/gold_a_entanglement.png) |

![VQC Stability Ablation]({res_rel}/vqc_stability_ablation.png)

### B. Feature Encoding (Gold B), Sample Efficiency (Gold C) & Shot Noise (Table D)
* **Encoding & Radial Norm (Gold B):** 8-qubit angle encoding (`2 arctan(alpha * x)`) preserves each feature on its own wire (`0.8860` on `Ring L=2`). Compressing 8 features into 3 qubits via amplitude encoding discards L2 vector magnitude and drops PR-AUC to `0.6057`; restoring the radial norm on a 4th wire recovers `+0.2450` PR-AUC (`0.8507`).
* **Sample Sweep (Gold C) & Shot Robustness (Table D):** Across $N \\in \\{{100, 200, 500, 1000\\}}$, classical `Tiny-MLP` and `RBF-SVM` maintain a consistent lead on HTRU2 (converging toward the validation-tuned full-pool ceilings of `0.9394` and `0.9309`), while `Q1-VQC` stays practically tied with exact simulation down to `500` shots.

| Feature Encoding & Radial Norm (Gold B) | Sample Size Efficiency Sweep (Gold C) |
| :---: | :---: |
| ![Gold B Encoding]({res_rel}/gold_b_encoding.png) | ![Gold C Sweep]({res_rel}/gold_c_sweep.png) |

| Finite Measurement Shot Noise (Table D) | Full-Pool Linearity Ceiling Check |
| :---: | :---: |
| ![Table D Shots]({res_rel}/table_d_shots.png) | ![Linearity Check]({res_rel}/linearity_check.png) |

---

## 4. Deep-Dive Insights, Controlled Nonlinear 8D Diagnostic & Hardware Costs

### Why Classical Wins HTRU2: Linearity & Kernel Concentration
1. **Linearity Ceiling (`linearity_check.png`):** On the full training pool (`N = 13,398`) with validation-tuned hyperparameters, 9-weight **Logistic Regression (`0.9377` PR-AUC)** sits within `0.0017–0.0034` of 41-weight **Tiny-MLP (`0.9394`)** and 45-weight **Poly2-LogReg (`0.9411`)**, while **RBF-SVM** reaches **`0.9309`**.
2. **Quantum Kernel Concentration & Small-$\\alpha$ Classical Limit (`kernel_analysis.png`):** Within the benchmark search grid ($\\alpha \\in [0.1, 3.16]$), the 8-qubit ZZ kernel peaks at the lower boundary $\\alpha = 0.1$ (`0.9199` on seed 100; effective rank `1.84`, **`0.9913` off-diagonal correlation** with tuned RBF). Probing below that grid boundary ($\\alpha = 0.05, 0.02$ in `diagnostics_extra/kernel_sub01_check.csv`) nudges PR-AUC to `0.9281–0.9287` precisely because the ZZ Gram matrix degenerates into a rank-`1.05` (`off_diag_mean = 0.994`) classical linear/quadratic kernel, whereas increasing $\\alpha \\to 2.5$ triggers exponential kernel concentration (`off_diag_mean -> 0.008`, PR-AUC `0.8424`).

| Quantum Kernel Spectrum & Concentration | ConcentricShell-8D Nonlinear Generalization |
| :---: | :---: |
| ![Kernel Analysis]({res_rel}/kernel_analysis.png) | ![Nonlinear Dataset]({res_rel}/nonlinear_dataset.png) |

### Controlled Nonlinear 8D Diagnostic (`ConcentricShell-8D`) & Width Bottleneck Check
To test model behavior when linear hyperplanes fail (`LogReg = 0.1165` PR-AUC), `ConcentricShell-8D` labels the innermost 12% of a perturbed 4D ball embedded in 8D ($r^2 = \\sum_{{i=0}}^3 x_i^2 + 0.35\\sin(2.5 x_0 x_1) \\le q_{{0.12}}$, plus 4 Gaussian noise coordinates; named `ConcentricShell-8D` in code, though geometrically an **inner ball** rather than a hollow shell). Three diagnostic findings stand out:
* **Kernels Statistically Tie:** Across $N = 100, 200, 500$, **Q2-QuantumKernel (`0.8832 [0.8512, 0.9104]` at `N = 500`)** and **RBF-SVM (`0.8695 [0.8398, 0.8974]`)** have heavily overlapping 95% CIs—a statistical tie, not a quantum kernel win.
* **Low-Data VQC Struggles Before $N = 500$:** **Q1-VQC (`41p`)** nearly fails at `N = 100` (`0.1307`) and `N = 200` (`0.3127`) before reaching `0.6602` at `N = 500`.
* **Why `41p` VQC Beats `41p` Tiny-MLP at `N = 500`:** Because $r^2(-x) = r^2(x)$ is strictly even in $x$, single-qubit rotation expectations ($\\cos(\\cdot)$) naturally supply coordinate-wise quadratic features, whereas closing a 4D ball with `Tanh` hyperplanes requires more than 4 hidden neurons. Relaxing `Tiny-MLP`'s width from `h = 4` (`41p`, `0.4075`) to `h = 8` (`81p`, `0.5936`), `h = 16` (`161p`, `0.6195`), and `h = 32` (`321p`, **`0.7767 [0.7384, 0.8080]`**) overtakes `Q1-VQC` (`diagnostics_extra/concentric_mlp_width_sweep.csv`).

{nonlinear_md}

### QPU Resource Accounting (`qml.specs`) & Circuit Scaling
* **Linear vs. Quadratic QPU Cost:** `Q1-VQC` requires **$O(N)$** circuit executions (`500` per epoch at `N = 500`), whereas `Q2-QuantumKernel` requires **$O(N^2)$** pairwise overlap circuits (`124,750` training pairs plus `1,500,000` test pairs), making variational circuits far more practical for near-term hardware.

{hw_md}

---

## Reproducibility & Repository Navigation

### Key Files
* `run_all.py`: Master experiment runner (`--mode full --tune-seeds 3` produces all `results/full/` checkpoints and tables).
* `analyze_results.py`: Two-level stratified bootstrap CIs, Wilcoxon signed-rank tests, and mechanical verdicts (`results/full/verdicts.md`).
* `src/vqc_study.py`, `src/models_quantum.py`, `src/models_classical.py`: Symmetry-broken 8q PennyLane VQC, ZZ quantum kernel, and matched classical baselines.
* `src/linearity_check.py`, `src/kernel_analysis.py`, `src/nonlinear_benchmark.py`, `src/hardware_cost.py`, `scripts/run_diagnostic_checks.py`: Depth-of-insight diagnostics, MLP width / sub-0.1 boundary checks, and QPU circuit specs.
* `Q_Horizon_Benchmark.ipynb` & `Q_Horizon_Dashboard.ipynb`: Self-contained Jupyter / Google Colab notebooks with all 10-seed figures, tables, and interactive dashboard pre-rendered inline.
* `scripts/check_submission.py`: Automated 14-test audit suite verifying parameter accounting, split isolation, qubit ceiling, and README integrity.

### Quick Reproduction Commands
{bt}bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Verify all 14 submission checks and tests against committed results/full/
python scripts/check_submission.py

# 3. Re-run the full 10-seed benchmark suite from scratch (~6.4 hours on CPU)
python run_all.py --mode full --tune-seeds 3
{bt}

### Measured Runtimes
{rt_md}

### Dataset Citation & Software Attribution
* **HTRU2 Pulsar Dataset (UCI ML Repository ID `372`):** R. J. Lyon, B. W. Stappers, S. Cooper, J. M. Brooke, & J. D. Knowles (2016), *"Fifty Years of Pulsar Candidate Selection: From simple filters to a new principled real-time classification approach,"* *MNRAS* 459(1), 1104–1123. Available at `https://archive.ics.uci.edu/dataset/372/htru2` (`DOI: 10.24432/C5DK6R`, CC BY 4.0).
* **Software Stack:** Quantum circuits simulated with **PennyLane** (`default.qubit`); classical models implemented with **PyTorch** and **scikit-learn**.
* **AI Tool Disclosure:** Full AI assistance and verification logs are recorded in `AI_USAGE_LOG.md`.

### Team Credits
{team_md}

---

## 5. Limitations

* **Classical Simulability (No Quantum Advantage Claimed):** All quantum circuits in this study use $\\le 8$ qubits (Hilbert space dimension $\\le 256$), which is fast to simulate exactly on a standard laptop CPU; therefore, **no quantum advantage is claimed** in runtime or asymptotic computational complexity.
* **Domain Linearity of HTRU2 & Small-$\\alpha$ Kernel Boundary:** Because UCI HTRU2 pulsar features are already ~94% separable by a linear hyperplane, quantum feature maps offer no accuracy advantage over compact classical neural networks or RBF-SVMs, and `Q2-QuantumKernel` peaks near the small-$\\alpha$ boundary ($\\alpha \\le 0.1$) where it mimics a low-rank classical kernel.
* **Synthetic Inner-Ball Symmetry & MLP Width Constraint (`ConcentricShell-8D`):** `ConcentricShell-8D` is a synthetic inner-ball task whose even coordinate symmetry ($f(-x)=f(x)$) directly favors $\\cos(\\cdot)$ angle rotations, while restricting `Tiny-MLP` to 4 hidden neurons (`41` weights) creates a geometric width bottleneck (`32` hidden units reach `0.7767` PR-AUC, above `Q1-VQC`'s `0.6602`).
* **Idealized Gate Noise Model:** Finite-shot experiments (`Table D`) capture statistical multinomial sampling noise (`100–2,000` shots) but do not model hardware gate infidelity, readout bit-flip bias, or decoherence ($T_1/T_2$).
* **AI Tool Disclosure:** AI coding assistants were used strictly for boilerplate scaffolding, plotting/table formatting, and test verification under human review; full prompts and verification logs are documented in `AI_USAGE_LOG.md`.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build comprehensive README.md from verified results.")
    parser.add_argument("--results-dir", type=Path, default=Path("results/full"))
    args = parser.parse_args()

    out = build_readme(args.results_dir)
    Path("README.md").write_text(out, encoding="utf-8")
    print(f"Built comprehensive README.md cleanly from {args.results_dir}.")


if __name__ == "__main__":
    main()