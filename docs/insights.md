# Phase 7 Depth-of-Insight Summary (`results/full`)

## 1. HTRU2 Near-Linear Separability & Full-Data Ceiling (`linearity_check.csv`)
Even on the full `13,398`-sample training pool, 9-parameter linear `LogReg` achieves `0.9378` PR-AUC versus `0.9050` for `RBF-SVM` (a gap of `-0.0328`). Because HTRU2 is nearly linearly separable after Yeo-Johnson standardization, any nonlinear feature map (quantum or classical) has less than `~0.01` PR-AUC headroom above a linear hyperplane.

## 2. Quantum Fidelity Kernel Geometry vs. Classical RBF (`kernel_analysis.csv`)
Across bandwidths `alpha in [0.1, 2.5]` and depths `1, 2`, the 8-qubit ZZ fidelity kernel peaks at `alpha = 0.1`, `depth = 1` (`test PR-AUC = 0.9199`, `KTA = 0.4261`, effective rank `1.8`, off-diagonal correlation `0.991` with tuned RBF-SVM `0.9255`). Larger `alpha` or deeper ZZ layers drive off-diagonal entries toward concentration (`mean -> 0.008`), shrinking kernel alignment and margin generalization.

## 3. Controlled Nonlinear 8D Benchmark (`nonlinear_dataset_summary.csv`)
Pre-screened with `RBF-SVM - LogReg = +0.7889 >= 0.05` at `N=500`, `ConcentricShell-8D` breaks linear `LogReg` (`0.1165` PR-AUC). Nonlinear models all separate the radial shell (`RBF-SVM = 0.8695`, `Tiny-MLP = 0.4075`, `Q2-QuantumKernel = 0.8832`, `Q1-VQC = 0.6602`), confirming that both quantum models capture genuine nonlinear structure when linear boundaries fail, though classical RBF-SVM/MLP remain stronger or faster in 8D.

## 4. Hardware Execution Cost & Classical Simulability (`hardware_cost_table.csv`)
Using `qml.specs`, `Q1-VQC` requires `O(N)` QPU executions (`3,000` test circuits), whereas `Q2-QuantumKernel` requires `124,750` training pairs (`O(N^2)`) plus `1,500,000` test pairs at `~10,000` shots/pair. At `<= 8` qubits (`dim <= 256`), all circuits are cheaply simulable on classical CPUs and **no quantum advantage is claimed**.
