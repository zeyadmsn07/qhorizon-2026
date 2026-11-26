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

### Plain-English Summary: What We Tested & What We Found

1. **The Problem & Why Accuracy Is a Trap:** The UCI HTRU2 pulsar dataset (R. J. Lyon et al., 2016) contains **17,898** radio-telescope candidate signals, of which only **9.16%** are real pulsars. A trivial model that predicts *"not a pulsar"* every time achieves **90.83% accuracy** while catching zero pulsars. We therefore evaluate all models using **PR-AUC** (Precision-Recall Area Under the Curve) and **MCC** (Matthews Correlation Coefficient) across **10 independent evaluation seeds (`100..109`)** after tuning hyperparameters strictly on **3 disjoint validation seeds (`200..202`)**.
2. **Classical Models Win on HTRU2—Because Pulsar Data Is Mostly Linear:** On `N = 500` training samples, our weight-matched **Tiny-MLP (41 weights)** achieves **`0.9246 ± 0.0079` PR-AUC**, edging out both **Q2-QuantumKernel (8 qubits, `0.9116 ± 0.0057`)** and **Q1-VQC (41 weights, `0.9102 ± 0.0025`)**, though both quantum models outperform **Random Forest (`0.9016 ± 0.0089`)**. Full-pool diagnostics show that a simple straight-line **Logistic Regression (`0.9378` PR-AUC)** nearly matches a neural network (`0.9382`), meaning pulsar signals do not need complex quantum feature spaces.
3. **Engineering Fixes Eliminated VQC Training Collapse:** Our initial ring-entangled VQC (`Ring L=2, Fragile v1`) suffered from an even-parity symmetry trap ($f(-x) = f(x)$ from $\\langle Z \\rangle = \\cos(\\theta)$), stalling at `0.8860 ± 0.0062`. Adding a $\\pi/2$ phase offset to break symmetry, reading out all 8 qubits linearly, and switching to sparse **bipartite entanglement** cut two-qubit CNOT gates in half (**8 vs. 16**), reduced circuit depth by $3\\times$ (**7 vs. 21**), achieved **0/10 training collapses**, and boosted PR-AUC by **`+0.0242`**.
4. **The Reversal on Genuinely Nonlinear Data (`ConcentricShell-8D`):** When we remove linear separability on an 8D spherical-shell dataset where Logistic Regression fails (`0.1165` PR-AUC), the exact same **41-weight Q1-VQC (`0.6602` PR-AUC)** beats the **41-weight Tiny-MLP (`0.4075` PR-AUC)** by **`+0.2527`**, and **Q2-QuantumKernel (`0.8832`)** beats **RBF-SVM (`0.8695`)**.

---

## 1. Fair Benchmark Protocol & Main Results (`N_train = 500`)

### How We Kept Every Comparison Fair
* **Exact Weight Parity:** `Q1-VQC (Bipartite_SignLinear_41p)` and `Tiny-MLP (41p)` both have **41 trainable scalar weights** and train with minibatch AdamW + binary cross-entropy early stopping.
* **Identical Kernel Solver:** `Q2-QuantumKernel (8q)` and `RBF-SVM` both use the exact same dual Support Vector Machine solver (`0` trainable weights); only the kernel matrix changes.
* **Zero Data Leakage:** Yeo-Johnson power scaling and standardization are fit **only on the training split** inside each seed, evaluated against a locked `3,000`-sample test set and `1,500`-sample validation set.

![Main Benchmark]({res_rel}/main_benchmark.png)

> **Reading the Main Benchmark Chart:** Bars and error markers show mean PR-AUC and 95% two-level bootstrap confidence intervals across 10 seeds (`100..109`). Classical `Tiny-MLP (41p)` and `RBF-SVM` sit practically tied at the top (~`0.923–0.925`), followed closely by `LogReg` (`0.9184`), while `Q2-QuantumKernel` (`0.9116`) and `Q1-VQC` (`0.9102`) beat `RandomForest` (`0.9016`) with zero training collapses.

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
* **H4a (Low-Data Myth):** At $N = 100$ training samples, `RBF-SVM` beats `Q2-QuantumKernel` by `0.0460` PR-AUC, refuting the claim of small-sample quantum kernel superiority on HTRU2 (**Contradicted**).
* **H4b (Measurement Shot Noise):** Running `2,000` or `500` shots per circuit stays within `0.0011` PR-AUC of exact statevector simulation (**Practically tied**), while `100` shots begins showing mild sampling variance (`-0.0068`, **Inconclusive**).

---

## 3. Focused Ablations & VQC Stability Diagnostics

### A. Fixing VQC Parity Collapse & Entanglement Topology (Gold A)
* **Why VQC v1 Struggled:** Using `RY` feature rotations with standard $\\langle Z \\rangle = \\cos(\\theta)$ readout creates an even symmetry ($f(-x) = f(x)$) around zero. Shifting the circuit phase by $\\pi/2$ (turning $\\cos$ into $\\sin$ around zero) and reading out a linear combination of all 8 qubits eliminated training collapse (`0/10`) and lowered seed-to-seed standard deviation by $2.5\\times$ (`0.0062` down to `0.0025`).
* **Why Bipartite Beats Ring:** Connecting pairs of qubits in a shallow bipartite layer (`depth 7`, `8` two-qubit gates) avoids barren-plateau / optimization drag seen in deeper ring circuits (`depth 21`, `16` two-qubit gates).

| VQC Training Stability & Loss Curves | Entanglement Topology & Depth (Gold A) |
| :---: | :---: |
| ![VQC Loss Diagnostics]({res_rel}/vqc_loss_diagnostics.png) | ![Gold A Entanglement]({res_rel}/gold_a_entanglement.png) |

![VQC Stability Ablation]({res_rel}/vqc_stability_ablation.png)

### B. Feature Encoding (Gold B), Sample Efficiency (Gold C) & Shot Noise (Table D)
* **Encoding & Radial Norm (Gold B):** 8-qubit angle encoding (`2 arctan(alpha * x)`) preserves each feature on its own wire without deep state-preparation circuits. When compressing all 8 features into 3 qubits via amplitude encoding, L2 normalization destroys vector magnitude (`0.65` PR-AUC); adding a 4th qubit for the radial norm recovers `+0.2450` PR-AUC.
* **Sample Sweep (Gold C) & Shot Robustness (Table D):** Across $N \\in \\{{100, 200, 500, 1000\\}}$, classical `Tiny-MLP` and `RBF-SVM` maintain a steady lead on HTRU2, while `Q1-VQC` remains stable under realistic NISQ shot budgets (`500–2,000` shots).

| Feature Encoding & Radial Norm (Gold B) | Sample Size Efficiency Sweep (Gold C) |
| :---: | :---: |
| ![Gold B Encoding]({res_rel}/gold_b_encoding.png) | ![Gold C Sweep]({res_rel}/gold_c_sweep.png) |

| Finite Measurement Shot Noise (Table D) | Full-Pool Linearity Ceiling Check |
| :---: | :---: |
| ![Table D Shots]({res_rel}/table_d_shots.png) | ![Linearity Check]({res_rel}/linearity_check.png) |

---

## 4. Deep-Dive Insights, Nonlinear 8D Reversal & Hardware Costs

### Why Classical Wins HTRU2: Linearity & Kernel Concentration
1. **Linearity Ceiling (`linearity_check.png`):** On the full training pool (`N = 13,398`), 9-weight **Logistic Regression (`0.9378` PR-AUC)** matches 41-weight **Tiny-MLP (`0.9382`)**.
2. **Quantum Kernel Concentration (`kernel_analysis.png`):** The 8-qubit ZZ quantum kernel achieves its best HTRU2 score (`0.9199`) only at small input scaling $\\alpha = 0.1$, where its Gram matrix has effective rank `1.84` and **`0.9913` correlation** with a tuned classical RBF kernel. Increasing $\\alpha$ toward `2.5` causes exponential kernel concentration (off-diagonal similarity collapses from `0.86` to `0.008`), dropping PR-AUC to `0.8424`.

| Quantum Kernel Spectrum & Concentration | ConcentricShell-8D Nonlinear Generalization |
| :---: | :---: |
| ![Kernel Analysis]({res_rel}/kernel_analysis.png) | ![Nonlinear Dataset]({res_rel}/nonlinear_dataset.png) |

### Controlled Nonlinear 8D Diagnostic (`ConcentricShell-8D`)
To verify that our 41-parameter VQC is not simply under-parameterized, we evaluated the exact same models on an 8D spherical-shell classification task where straight hyperplanes fail completely (`LogReg = 0.1165` PR-AUC). With an equal 41-weight budget at `N = 500`, **Q1-VQC (`0.6602`) outperforms Tiny-MLP (`0.4075`) by `+0.2527` PR-AUC**, and **Q2-QuantumKernel (`0.8832`) outperforms RBF-SVM (`0.8695`)**:

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
* `src/linearity_check.py`, `src/kernel_analysis.py`, `src/nonlinear_benchmark.py`, `src/hardware_cost.py`: Depth-of-insight diagnostics and QPU circuit specs.
* `Q_Horizon_Benchmark.ipynb`: Self-contained Jupyter / Google Colab notebook with all 10-seed figures and tables pre-rendered inline.
* `scripts/check_submission.py`: Automated 14-test audit suite verifying parameter accounting, split isolation, qubit ceiling, and README integrity.

### Quick Reproduction Commands
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Verify all 14 submission checks and tests against committed results/full/
python scripts/check_submission.py

# 3. Re-run the full 10-seed benchmark suite from scratch (~6.4 hours on CPU)
python run_all.py --mode full --tune-seeds 3
```

### Measured Runtimes
{rt_md}

### Team Credits
{team_md}

---

## 5. Limitations

* **Classical Simulability (No Quantum Advantage Claimed):** All quantum circuits in this study use $\\le 8$ qubits (Hilbert space dimension $\\le 256$), which is fast to simulate exactly on a standard laptop CPU; therefore, **no quantum advantage is claimed** in runtime or asymptotic computational complexity.
* **Domain Linearity of HTRU2:** Because UCI HTRU2 pulsar features are already ~94% separable by a linear hyperplane, quantum feature maps offer no accuracy advantage over compact classical neural networks or RBF-SVMs on this dataset.
* **Synthetic Nature of `ConcentricShell-8D`:** The nonlinear 8D reversal demonstrates geometric inductive bias under matched parameter budgets, not real-world astrophysical data.
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