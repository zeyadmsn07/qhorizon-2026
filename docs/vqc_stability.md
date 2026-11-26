# VQC Stability & Symmetry-Breaking Study (`C:/Users/MSI/Desktop/qhorizon-2026/results/full`)

**Pre-stated headline selection rule:** Highest mean validation PR-AUC across disjoint tune_seeds (tie-break: lower tune val std); test set never consulted.

**Validation-selected headline VQC:** `Bipartite_SignLinear_41p` (`tune val PR-AUC = 0.9022`)

| Variant | Tune Val PR-AUC | Test PR-AUC | 95% Bootstrap CI | Collapse Rate | Init Grad Norm | Trainable Params |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `Ring_L2_Baseline` | `0.8611` | `0.8860` | `[0.8518, 0.9155]` | 0/10 | `0.963` | 34 |
| `Init_WideNormal_0.5` | `0.8845` | `0.8894` | `[0.8529, 0.9183]` | 0/10 | `0.603` | 34 |
| `Init_Uniform_Pi` | `0.8670` | `0.8598` | `[0.8158, 0.8943]` | 0/10 | `0.260` | 34 |
| `SignAware_Offset_Pi2` | `0.8627` | `0.8854` | `[0.8486, 0.9143]` | 0/10 | `0.291` | 34 |
| `Readout_Linear8_41p` | `0.8725` | `0.8869` | `[0.8512, 0.9156]` | 0/10 | `1.659` | 41 |
| `Readout_SingleZ0` | `0.8707` | `0.8782` | `[0.8402, 0.9127]` | 0/10 | `0.399` | 34 |
| `Epochs100_Patience20` | `0.8677` | `0.8869` | `[0.8492, 0.9161]` | 0/10 | `1.012` | 34 |
| `Restarts_k3_ValSelected` | `0.8621` | `0.8884` | `[0.8545, 0.9186]` | 0/10 | `0.976` | 34 |
| `Bipartite_SignLinear_41p` | `0.9022` | `0.9102` | `[0.8813, 0.9349]` | 0/10 | `0.566` | 41 |
