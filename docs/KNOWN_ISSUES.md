# Known Issues & v1 Baseline Audit

## v1 Original 10-Seed Baseline (N_train = 500, Test PR-AUC mean +/- std)
- RBF-SVM: 0.9227 +/- 0.0036 | LogReg (9p): 0.9192 +/- 0.0142 | Tiny-MLP (41p): 0.9187 +/- 0.0084
- Q2-QuantumKernel (8q): 0.9023 +/- 0.0230 | RandomForest: 0.9015 +/- 0.0156
- Q1-VQC (8q Ring L=2): 0.7800 +/- 0.1437 (seeds 105 -> 0.4406 and 109 -> 0.5940 collapsed; init_grad_norm ~0.43)
- Gold A stable VQCs: VQC_none_L1 0.8945 +/- 0.0010, VQC_bipartite_L2 0.8981 +/- 0.0057, VQC_bipartite_L4 0.8990 +/- 0.0113
- Gold B Encoding: 3q_Amplitude 0.6185 +/- 0.0473, 4q_AmplitudePlusNorm 0.8431 +/- 0.0151, 8q_Angle_Reupload 0.8662 +/- 0.0194

## Issues Addressed Across Improvement Phases
- Issue A: Single-seed tuning on eval seed 100 and even-parity cos(theta) init symmetry caused 2/10 collapses on Ring L=2.
- Issue B: Tiny gaps (0.0008 and 0.0045 PR-AUC) were labeled Confirmed without CIs; remove 'pre-registered' phrasing.
- Issue C: Single 3,000-sample test split hides test-sampling variance; add two-level bootstrap CIs and paired deltas.
- Issue D: Single n_params column mixed SVM C, support vectors, tree nodes, and gradient-trained weights.
- Issue E: LogReg (0.9192) nearly ties RBF-SVM (0.9227), showing HTRU2 is nearly linear after Yeo-Johnson scaling.
- Issue F & G: Isolate smoke/fast/full dirs via CLI, cache htru2.csv offline, and move AI workflow docs to docs/ai/.
