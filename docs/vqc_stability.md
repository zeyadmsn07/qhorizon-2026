# VQC Stability & Symmetry-Breaking Study (`C:/Users/MSI/Desktop/qhorizon-2026/results/fast`)

**Pre-stated headline selection rule:** Highest mean validation PR-AUC across disjoint tune_seeds (tie-break: lower tune val std); test set never consulted.

**Validation-selected headline VQC:** `Bipartite_SignLinear_41p` (`tune val PR-AUC = 0.9078`)

| Variant | Tune Val PR-AUC | Test PR-AUC | 95% Bootstrap CI | Collapse Rate | Init Grad Norm | Trainable Params |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `Ring_L2_Baseline` | `0.8501` | `0.8678` | `[0.8279, 0.9039]` | 0/3 | `1.017` | 34 |
| `Init_WideNormal_0.5` | `0.8762` | `0.9036` | `[0.8733, 0.9302]` | 0/3 | `0.750` | 34 |
| `Init_Uniform_Pi` | `0.8524` | `0.8509` | `[0.8053, 0.8963]` | 0/3 | `0.237` | 34 |
| `SignAware_Offset_Pi2` | `0.8668` | `0.8807` | `[0.8473, 0.9082]` | 0/3 | `0.294` | 34 |
| `Readout_Linear8_41p` | `0.8486` | `0.8553` | `[0.8112, 0.8933]` | 0/3 | `1.796` | 41 |
| `Readout_SingleZ0` | `0.8294` | `0.8668` | `[0.8313, 0.9004]` | 0/3 | `1.082` | 34 |
| `Epochs100_Patience20` | `0.8392` | `0.8737` | `[0.8252, 0.9143]` | 0/3 | `1.017` | 34 |
| `Restarts_k3_ValSelected` | `0.8460` | `0.8647` | `[0.8279, 0.8976]` | 0/3 | `1.019` | 34 |
| `Bipartite_SignLinear_41p` | `0.9078` | `0.9104` | `[0.8833, 0.9352]` | 0/3 | `0.571` | 41 |
