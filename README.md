# Q//Horizon 2026: Fair Quantum vs. Classical Pulsar Classification

> **Active Artifact Directory:** `results/fast` (`--mode fast`, `eval_seeds=[100, 101, 102]`). Running `python -m src.build_final_readme --results-dir results/full` updates all numbers from `results/full/` automatically.

## 1. Research Question & Why PR-AUC Matters
How do compact (`<= 8` qubit) variational quantum classifiers and quantum fidelity kernels compare against equal-budget classical baselines on imbalanced radio-astronomy tabular data, and when do quantum feature maps actually help?

We benchmark on the **UCI HTRU2 Pulsar Candidate Dataset** (R. J. Lyon, 2017, UCI ID `372`, CC BY 4.0; offline copy in `data_cache/htru2.csv`, see [`DATA_LICENSE.md`](DATA_LICENSE.md)): `17,898` candidates, `8` summary statistics (`4` integrated pulse-profile + `4` DM-SNR curve), and **`9.16%` true pulsars**. Predicting majority noise (`0`) everywhere yields **`90.83%` accuracy** while finding zero pulsars (`0.0917` PR-AUC, `0.0000` MCC), so we rank models by **PR-AUC** alongside **MCC** and **F1**.

## 2. Models & Design Rationale
- **Train-Only Yeo-Johnson + Bounded Angle Encoding:** Fit `PowerTransformer` strictly on `X_train` to tame pulsar skewness/kurtosis tails, then map features via `2 * atan(alpha * x)` into `(-pi, pi)` with tunable bandwidth `alpha` so outliers never wrap modulo `2*pi`.
- **Circuit Topology & Norm Restoration:** We test separable (`none`), `ring`, and domain-coupled `bipartite` entanglement (`0-3 <-> 4-7`), plus 3-qubit amplitude compression (`L2` normalization discards signal magnitude) vs. 4-qubit norm restoration on wire 3.
- **Readout & Symmetry Breaking ([`docs/vqc_stability.md`](docs/vqc_stability.md)):** Standard `RY(theta)|0>` yields `<Z> = cos(theta)`, an *even* function blind to feature sign near `0.05 * randn` init—explaining `2/10` seed collapses in v1 `Ring L=2`. We test `pi/2` sign offsets, 8-wire linear readout (`41` weights, matching `Tiny-MLP`), and restarts, selecting headline `Q1-VQC (Bipartite_SignLinear_41p)` strictly by `Highest mean validation PR-AUC across disjoint tune_seeds (tie-break: lower tune val std); test set never consulted.`.
- **Quantum Fidelity Kernel (`Q2-QuantumKernel`):** 8-qubit `ZZ` feature map with `K_ij = |<psi_i | psi_j>|^2` fed to a dual SVM (exact state-vector product in CPU simulation; `O(N^2)` circuit runs on hardware).

## 3. Fairness Protocol & Statistical Results (`N_train = 500`)
Locked stratified holdouts (`3,000` test, `1,500` val), disjoint tuning seeds (`tune_seeds = [200]`, `eval_seeds = [100, 101, 102]`), equal candidate budgets, validation-only thresholds, and separate complexity columns ([`results/fast/summary_tables.md`](results/fast/summary_tables.md)). Verdicts use a mechanical `margin = 0.01` PR-AUC rule on 95% two-level bootstrap CIs ([`results/fast/verdicts.md`](results/fast/verdicts.md)):

| Model | Test PR-AUC | 95% Two-Level CI | Collapse (<0.70) | Delta vs. `Tiny-MLP (41p)` (95% CI) | Verdict vs. Best |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Dummy** | `0.0917` | `[0.0917, 0.0917]` | 3/3 | `-0.8319` (`[-0.8557, -0.8039]`) | Contradicted |
| **LogReg** | `0.9220` | `[0.8898, 0.9485]` | 0/3 | `-0.0016` (`[-0.0126, +0.0072]`) | Inconclusive |
| **RBF-SVM** | `0.9023` | `[0.8731, 0.9277]` | 0/3 | `-0.0213` (`[-0.0355, -0.0047]`) | Contradicted |
| **RandomForest** | `0.9021` | `[0.8701, 0.9317]` | 0/3 | `-0.0214` (`[-0.0372, -0.0056]`) | Contradicted |
| **Tiny-MLP (41p)** | `0.9235` | `[0.8955, 0.9473]` | 0/3 | `+0.0000` (`[+0.0000, +0.0000]`) | Reference |
| **Q1-VQC (Bipartite_SignLinear_41p)** | `0.9104` | `[0.8821, 0.9353]` | 0/3 | `-0.0131` (`[-0.0239, +0.0008]`) | Inconclusive |
| **Q1-VQC (Ring L=2, Fragile v1)** | `0.8678` | `[0.8285, 0.9060]` | 0/3 | `-0.0558` (`[-0.0834, -0.0303]`) | Contradicted |
| **Q2-QuantumKernel (8q)** | `0.8772` | `[0.8397, 0.9104]` | 0/3 | `-0.0463` (`[-0.0661, -0.0280]`) | Contradicted |

| ID | Hypothesis Comparison | Delta PR-AUC | 95% Two-Level CI | Mechanical Verdict |
| :--- | :--- | :---: | :---: | :---: |
| **H1a** | `Tiny-MLP (41p) vs. Q2-QuantumKernel (8q)` | `+0.0463` | `[+0.0280, +0.0661]` | **Supported** |
| **H1b** | `Tiny-MLP (41p) vs. Q1-VQC (Bipartite_SignLinear_41p)` | `+0.0131` | `[-0.0008, +0.0239]` | **Inconclusive** |
| **H1c** | `Q2-QuantumKernel (8q) vs. RandomForest` | `-0.0249` | `[-0.0422, -0.0104]` | **Contradicted** |
| **H2a** | `VQC_bipartite_L2 vs. VQC_none_L2` | `-0.0365` | `[-0.0612, -0.0125]` | **Contradicted** |
| **H2b** | `VQC_bipartite_L2 vs. VQC_ring_L2` | `-0.0163` | `[-0.0441, +0.0175]` | **Inconclusive** |
| **H3** | `4q_AmplitudePlusNorm_L4 vs. 3q_Amplitude_L5` | `+0.1536` | `[+0.0275, +0.2511]` | **Supported** |
| **H4a** | `N=100 Q2-QuantumKernel vs. RBF-SVM (tied)` | `-0.0805` | `[-0.1540, -0.0215]` | **Contradicted** |
| **H4b_2000** | `Shots=2000 vs. Analytic (tied)` | `-0.0009` | `[-0.0023, +0.0004]` | **Practically tied** |
| **H4b_500** | `Shots=500 vs. Analytic (tied)` | `-0.0021` | `[-0.0048, +0.0005]` | **Practically tied** |
| **H4b_100** | `Shots=100 vs. Analytic (tied)` | `-0.0046` | `[-0.0110, +0.0017]` | **Inconclusive** |

![Main Benchmark](results/fast/main_benchmark.png)

## 4. Depth of Insight ([`docs/insights.md`](docs/insights.md))
1. **Why Quantum Doesn't Win on HTRU2 (Near-Linearity):** Even on all `13,398` pool samples, 9-parameter `LogReg` (`0.9378` PR-AUC) nearly matches `RBF-SVM` (`0.9050`), leaving `< 0.01` PR-AUC headroom for nonlinear feature maps.
2. **Kernel Concentration & Alignment:** Across `alpha`, `Q2-QuantumKernel` peaks at `alpha=0.1`, `d=1` (`0.9199` PR-AUC, `KTA=0.4261`, rank `1.8`), while larger `alpha` concentrates off-diagonal overlaps.
3. **Harder Nonlinear 8D Benchmark (`ConcentricShell-8D`):** Pre-screened with `RBF - LogReg = +0.7889 >= 0.05`, linear `LogReg` fails (`0.1125`), whereas `Q2-QuantumKernel` (`0.7900`) and `Q1-VQC` (`0.1302`) are evaluated alongside `Tiny-MLP` (`0.3457`) and `RBF-SVM` (`0.8741`).
4. **Hardware Cost ([`results/fast/hardware_cost_table.csv`](results/fast/hardware_cost_table.csv)):** `Q1-VQC` scales as `O(N)` (`3,000` test circuits), while `Q2-QuantumKernel` needs `124,750` train pairs + `1.5M` test pairs. At `<= 8` qubits, all circuits are CPU-simulable; **no quantum advantage is claimed**.

## 5. Limitations
Noiseless `default.qubit` state-vector simulation (shot noise only at inference; no depolarizing gate noise), `O(N^2)` QPU kernel scaling, `N <= 1000` quantum training subsamples, one `1,500`-sample validation split shared across tuning/early-stopping/thresholding, and `~275` test pulsars.

## 6. Reproducibility, Credits, AI Disclosure & Team
```bash
python run_all.py --mode smoke          # ~1.5 min sanity check -> results/smoke/
python run_all.py --mode fast           # fast dev check -> results/fast/
python run_all.py --mode full           # 10-seed benchmark -> results/full/
python -m src.make_tables --results-dir results/fast
python analyze_results.py --results-dir results/fast
python -m src.plot_results --results-dir results/fast
python scripts/check_submission.py      # automated submission audit
```
- `--mode smoke`: 449.98 s
- `--mode fast`: 749.21 s (12.49 min)
- `v1 10-seed sweep`: 9231.16 s (153.85 min)

- **Sources & Credits:** PennyLane (`0.45.1`), PyTorch, scikit-learn, SciPy, NumPy, pandas, matplotlib; Dataset: UCI HTRU2 (`id=372`, R. J. Lyon, 2017, CC BY 4.0). Q&A rationale: [`docs/design_rationale.md`](docs/design_rationale.md).
- **AI Tool Disclosure:** Every AI-assisted step and human verification check is logged in [`AI_USAGE_LOG.md`](AI_USAGE_LOG.md) (rules archived in [`docs/ai/AGENTS.md`](docs/ai/AGENTS.md)).

| Member | Role | Verified Deliverables |
| :--- | :--- | :--- |
| **Zeyad Mohamed Nada** | Team Lead & QML Benchmark Architect | Verified train-only Yeo-Johnson splits (`src/data.py`), disjoint multi-seed tuning (`src/tuning.py`), VQC symmetry/readout ablations (`src/vqc_study.py`), two-level bootstrap CIs (`src/stats.py`), linearity/kernel/nonlinear-8D analyses, and clean-venv reproducibility |
