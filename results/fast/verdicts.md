# Statistical Verdicts & Two-Level Bootstrap CIs (`results/fast`)

Rule (`margin = 0.01` PR-AUC, 95% CI across seeds x stratified test resamples): **Supported** (`CI > 0` and `|delta| > 0.01`), **Practically tied** (`CI` inside `[-0.01, +0.01]`), **Contradicted** (`CI < 0` and `|delta| > 0.01`), **Inconclusive** otherwise.

## 1. Hypothesis Verdicts
| ID | Comparison (A vs. B) | Delta PR-AUC | 95% Two-Level CI | Wilcoxon p (min p) | Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **H1a** | Tiny-MLP (41p) vs. Q2-QuantumKernel (8q) | `+0.0463` | `[+0.0280, +0.0661]` | n/a (<5 seeds) | **Supported** |
| **H1b** | Tiny-MLP (41p) vs. Q1-VQC (Bipartite_SignLinear_41p) | `+0.0131` | `[-0.0008, +0.0239]` | n/a (<5 seeds) | **Inconclusive** |
| **H1c** | Q2-QuantumKernel (8q) vs. RandomForest | `-0.0249` | `[-0.0422, -0.0104]` | n/a (<5 seeds) | **Contradicted** |
| **H2a** | VQC_bipartite_L2 vs. VQC_none_L2 | `-0.0365` | `[-0.0612, -0.0125]` | n/a (<5 seeds) | **Contradicted** |
| **H2b** | VQC_bipartite_L2 vs. VQC_ring_L2 | `-0.0163` | `[-0.0441, +0.0175]` | n/a (<5 seeds) | **Inconclusive** |
| **H3** | 4q_AmplitudePlusNorm_L4 vs. 3q_Amplitude_L5 | `+0.1536` | `[+0.0275, +0.2511]` | n/a (<5 seeds) | **Supported** |
| **H4a** | N=100 Q2-QuantumKernel vs. RBF-SVM (tied) | `-0.0805` | `[-0.1540, -0.0215]` | n/a (<5 seeds) | **Contradicted** |
| **H4b_2000** | Shots=2000 vs. Analytic (tied) | `-0.0009` | `[-0.0023, +0.0004]` | n/a (<5 seeds) | **Practically tied** |
| **H4b_500** | Shots=500 vs. Analytic (tied) | `-0.0021` | `[-0.0048, +0.0005]` | n/a (<5 seeds) | **Practically tied** |
| **H4b_100** | Shots=100 vs. Analytic (tied) | `-0.0046` | `[-0.0110, +0.0017]` | n/a (<5 seeds) | **Inconclusive** |

## 2. Pairwise Comparison vs. Best Classical (`Tiny-MLP (41p)`)
| Model | Mean PR-AUC | 95% Two-Level CI | Collapse | Delta vs. Best Classical | Paired 95% CI | Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy** | `0.0917` | `[0.0917, 0.0917]` | 3/3 | `-0.8319` | `[-0.8557, -0.8039]` | Contradicted |
| **LogReg** | `0.9220` | `[0.8898, 0.9485]` | 0/3 | `-0.0016` | `[-0.0126, +0.0072]` | Inconclusive |
| **RBF-SVM** | `0.9023` | `[0.8731, 0.9277]` | 0/3 | `-0.0213` | `[-0.0355, -0.0047]` | Contradicted |
| **RandomForest** | `0.9021` | `[0.8701, 0.9317]` | 0/3 | `-0.0214` | `[-0.0372, -0.0056]` | Contradicted |
| **Tiny-MLP (41p)** | `0.9235` | `[0.8955, 0.9473]` | 0/3 | `+0.0000` | `[+0.0000, +0.0000]` | Reference |
| **Q1-VQC (Bipartite_SignLinear_41p)** | `0.9104` | `[0.8821, 0.9353]` | 0/3 | `-0.0131` | `[-0.0239, +0.0008]` | Inconclusive |
| **Q1-VQC (Ring L=2, Fragile v1)** | `0.8678` | `[0.8285, 0.9060]` | 0/3 | `-0.0558` | `[-0.0834, -0.0303]` | Contradicted |
| **Q2-QuantumKernel (8q)** | `0.8772` | `[0.8397, 0.9104]` | 0/3 | `-0.0463` | `[-0.0661, -0.0280]` | Contradicted |
