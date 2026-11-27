# Phase 7 Depth-of-Insight Summary (`results/full`)

## 1. HTRU2 Near-Linear Separability & Full-Data Ceiling (`linearity_check.csv`)
On the full `13,398`-sample training pool with validation-tuned hyperparameters, 9-parameter linear `LogReg` achieves `0.9377` PR-AUC alongside `Tiny-MLP (41p)` (`0.9394`), `Poly2-LogReg (45p)` (`0.9411`), and `RBF-SVM` (`0.9309`). Because HTRU2 is ~94% separable by a single hyperplane after Yeo-Johnson standardization, any nonlinear feature map (quantum or classical) has less than `~0.004` PR-AUC headroom above linear/quadratic baselines.

## 2. Quantum Fidelity Kernel Geometry vs. Classical RBF (`kernel_analysis.csv`)
Across the benchmark grid `alpha in [0.1, 2.5]` and depths `1, 2`, the 8-qubit ZZ fidelity kernel peaks at the lower boundary `alpha = 0.1`, `depth = 1` (`test PR-AUC = 0.9199`, `KTA = 0.4261`, effective rank `1.84`, off-diagonal correlation `0.991` with tuned RBF-SVM `0.9255`). Probing `alpha = 0.05 / 0.02` (`diagnostics_extra/kernel_sub01_check.csv`) reaches `0.9281 / 0.9287` only by collapsing into a rank-`1.05` (`off_diag_mean = 0.994`) near-linear kernel, whereas larger `alpha` drives exponential off-diagonal concentration (`mean -> 0.008`).

## 3. Controlled Nonlinear 8D Inner-Ball Diagnostic (`nonlinear_dataset_summary.csv`)
Pre-screened with `RBF-SVM - LogReg = +0.7889 >= 0.05` at `N=500`, `ConcentricShell-8D` labels an even-symmetry ($f(-x)=f(x)$) perturbed 4D inner ball in 8D where linear `LogReg` fails (`0.1165` PR-AUC). `Q2-QuantumKernel` (`0.8832 [0.8512, 0.9104]`) and `RBF-SVM` (`0.8695 [0.8398, 0.8974]`) statistically tie. While 41-weight `Q1-VQC` (`0.6602`, after `0.1307` at `N=100` and `0.3127` at `N=200`) exceeds width-4 `Tiny-MLP` (`0.4075`), our width sweep (`diagnostics_extra/concentric_mlp_width_sweep.csv`) shows this reflects `cos(.)` symmetry alignment plus a 4-neuron bottleneck: widening `Tiny-MLP` to `8 / 16 / 32` units reaches `0.5936 / 0.6195 / 0.7767` PR-AUC.

## 4. Hardware Execution Cost & Classical Simulability (`hardware_cost_table.csv`)
Using `qml.specs`, `Q1-VQC` requires `O(N)` QPU executions (`3,000` test circuits), whereas `Q2-QuantumKernel` requires `124,750` training pairs (`O(N^2)`) plus `1,500,000` test pairs at `~10,000` shots/pair. At `<= 8` qubits (`dim <= 256`), all circuits are cheaply simulable on classical CPUs and **no quantum advantage is claimed**.
