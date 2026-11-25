# Q//Horizon 2026: Quantum vs. Classical Pulsar Classification

**Team Repository | Q//Horizon 2026 Quantum Computing Competition (AUC / Afretec Network)**

## Overview
This repository benchmarks Quantum Machine Learning (QML) classifiers against classical machine learning baselines on real-world radio-telescope pulsar candidate data (**UCI HTRU2**, CC BY 4.0) to answer a practical question:

> *Can a compact quantum model ($\le 10$ qubits) learn to classify real-world imbalanced signals, and under what data or architectural regimes is it actually worth using?*

## Project Structure (In Progress)
* `AGENTS.md` - Project constitution, coding standards, and simulation constraints.
* `WORKFLOW.md` - Step-by-step engineering plan and branch workflow.
* `requirements.txt` - Pinned Python dependencies (`pennylane==0.45.1`, `torch`, `scikit-learn`).
* `src/` - Leakage-free preprocessing, classical baselines, batched quantum classifiers, and equal-budget validation tuning.
* `run_all.py` - End-to-end experiment runner supporting `FAST_MODE` (quick verification) and full $10$-seed evaluation.
* `results/` - Saved benchmark CSVs and error-bar plots.

## Quickstart
```bash
pip install -r requirements.txt
python run_all.py
```

*(Full experimental hypotheses, methodology, results tables, and AI usage disclosure will be populated upon completion of the benchmark suite.)*
