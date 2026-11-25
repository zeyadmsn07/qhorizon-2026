# Known Issues & v1 Baseline Audit

## v1 Original 10-Seed Baseline (N_train = 500, Test PR-AUC mean +/- std)
- RBF-SVM: 0.9227 +/- 0.0036
- LogReg (9 params): 0.9192 +/- 0.0142
- Tiny-MLP (41 params): 0.9187 +/- 0.0084
- Q2-QuantumKernel (8q): 0.9023 +/- 0.0230
- RandomForest: 0.9015 +/- 0.0156
- Q1-VQC (8q Ring L=2): 0.7800 +/- 0.1437 (2/10 seeds collapsed: seed 105 -> 0.44, seed 109 -> 0.59)
- Gold A VQC variants: none_L1 0.8945, bipartite_L2 0.8981, bipartite_L4 0.8990
- Gold B Encoding: 3q_Amplitude 0.6185, 4q_AmplitudePlusNorm 0.8431, 8q_Angle_Reupload 0.8662

## Issue Register (Addressed Across Phases 1-8)
- **Issue A (Fragile Headline VQC):** Ring L=2 suffered 2/10 seed collapses due to single-seed tuning on evaluation seed 100, even-function parity symmetry of <Z> = cos(2*atan(alpha*x)) near zero initialization, and uniform wire-mean readout.
- **Issue B (Overstated Verdicts):** Small differences (e.g., 0.9023 vs 0.9015 and 0.8990 vs 0.8945) were labeled Confirmed without confidence intervals, and hypotheses were labeled pre-registered despite iterative development.
- **Issue C (Test-Set Uncertainty):** All 10 seeds share one 3,000-sample test split (~275 pulsars), requiring two-level bootstrap CIs and paired seed comparisons.
- **Issue D (Incompatible Parameter Counts):** Hyperparameters (SVM C), support vectors, tree nodes, and gradient-trained weights were mixed into a single parameter column.
- **Issue E (Near-Linear Separability Unexplained):** LogReg (0.9192) nearly matches RBF-SVM (0.9227), showing HTRU2 leaves minimal headroom for nonlinear feature maps.
- **Issue F (Reproducibility Overwrite Hazard):** Running FAST_MODE=True overwrote results/*.csv in place and relied on live network downloads via ucimlrepo.
- **Issue G (Repository Hygiene):** Root-level AI workflow files, placeholder team credits, and generic verification notes require cleanup and plain-English design rationale docs.
