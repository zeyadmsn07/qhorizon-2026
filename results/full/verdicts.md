# Statistical Verdicts & Two-Level Bootstrap CIs (`results/full`)

Rule (`margin = 0.01` PR-AUC, 95% CI across seeds x stratified test resamples): **Supported** (`CI > 0` and `|delta| > 0.01`), **Practically tied** (`CI` inside `[-0.01, +0.01]`), **Contradicted** (`CI < 0` and `|delta| > 0.01`), **Inconclusive** otherwise.

## 1. Hypothesis Verdicts
| ID | Comparison (A vs. B) | Delta PR-AUC | 95% Two-Level CI | Wilcoxon p (min p) | Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **H1a** | Tiny-MLP (41p) vs. Q2-QuantumKernel (8q) | `+0.0130` | `[+0.0047, +0.0217]` | 0.0039 (min 0.0020) | **Supported** |
| **H1b** | Tiny-MLP (41p) vs. Q1-VQC (Bipartite_SignLinear_41p) | `+0.0144` | `[+0.0060, +0.0232]` | 0.0020 (min 0.0020) | **Supported** |
| **H1c** | Q2-QuantumKernel (8q) vs. RandomForest | `+0.0101` | `[+0.0027, +0.0189]` | 0.0020 (min 0.0020) | **Supported** |
| **H2a** | VQC_bipartite_L2 vs. VQC_none_L2 | `+0.0110` | `[-0.0006, +0.0227]` | 0.0371 (min 0.0020) | **Inconclusive** |
| **H2b** | VQC_bipartite_L2 vs. VQC_ring_L2 | `+0.0120` | `[+0.0025, +0.0225]` | 0.0020 (min 0.0020) | **Supported** |
| **H3** | 4q_AmplitudePlusNorm_L4 vs. 3q_Amplitude_L5 | `+0.2450` | `[+0.1909, +0.2885]` | 0.0020 (min 0.0020) | **Supported** |
| **H4a** | N=100 Q2-QuantumKernel vs. RBF-SVM (tied) | `-0.0460` | `[-0.0847, -0.0142]` | 0.0098 (min 0.0020) | **Contradicted** |
| **H4b_2000** | Shots=2000 vs. Analytic (tied) | `-0.0007` | `[-0.0015, +0.0000]` | 0.0059 (min 0.0020) | **Practically tied** |
| **H4b_500** | Shots=500 vs. Analytic (tied) | `-0.0011` | `[-0.0028, +0.0005]` | 0.1602 (min 0.0020) | **Practically tied** |
| **H4b_100** | Shots=100 vs. Analytic (tied) | `-0.0068` | `[-0.0112, -0.0031]` | 0.0020 (min 0.0020) | **Inconclusive** |

## 2. Pairwise Comparison vs. Best Classical (`Tiny-MLP (41p)`)
| Model | Mean PR-AUC | 95% Two-Level CI | Collapse | Delta vs. Best Classical | Paired 95% CI | Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy** | `0.0917` | `[0.0917, 0.0917]` | 10/10 | `-0.8330` | `[-0.8573, -0.8067]` | Contradicted |
| **LogReg** | `0.9184` | `[0.8891, 0.9453]` | 0/10 | `-0.0062` | `[-0.0187, +0.0031]` | Inconclusive |
| **RBF-SVM** | `0.9227` | `[0.8940, 0.9459]` | 0/10 | `-0.0020` | `[-0.0079, +0.0039]` | Practically tied |
| **RandomForest** | `0.9016` | `[0.8721, 0.9293]` | 0/10 | `-0.0231` | `[-0.0363, -0.0110]` | Contradicted |
| **Tiny-MLP (41p)** | `0.9246` | `[0.8983, 0.9490]` | 0/10 | `+0.0000` | `[+0.0000, +0.0000]` | Reference |
| **Q1-VQC (Bipartite_SignLinear_41p)** | `0.9102` | `[0.8817, 0.9356]` | 0/10 | `-0.0144` | `[-0.0232, -0.0060]` | Contradicted |
| **Q1-VQC (Ring L=2, Fragile v1)** | `0.8860` | `[0.8521, 0.9169]` | 0/10 | `-0.0386` | `[-0.0541, -0.0229]` | Contradicted |
| **Q2-QuantumKernel (8q)** | `0.9116` | `[0.8819, 0.9370]` | 0/10 | `-0.0130` | `[-0.0217, -0.0047]` | Contradicted |
