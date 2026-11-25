# AI Usage Log

All major AI-assisted steps are logged here for transparency, as required by Q//Horizon 2026 rules.

| Date | Phase/Branch | AI Tool | Purpose | Human Verification |
|------|-------------|---------|---------|-------------------|
| 2026-11-25 | Phase 0 & Phase 1 / `main` | Gemini, Claude Opus & Antigravity | Architecture planning, fairness rules design (`AGENTS.md`), repository configuration, portable `requirements.txt`, and leakage-free HTRU2 data pipeline (`src/config.py`, `src/data.py`) | Verified train-only Yeo-Johnson fit, locked stratified test/val splits, ~9.16% pulsar ratio, and `.venv/` exclusion in `.gitignore` |
| 2026-11-25 | Phase 2 / `classical-baselines` | Claude Opus & Antigravity | Fair evaluation metrics (`src/metrics.py`), 5 classical baselines (`src/models_classical.py`), threshold constant-score fix, and smoke test (`verify_baselines.py`) | Verified Dummy achieves expected majority-noise baseline (acc 0.9083, F1 0.0, MCC 0.0) alongside LogReg (PR-AUC 0.9383), RBF-SVM (PR-AUC 0.8630), RandomForest (PR-AUC 0.9027), and Tiny-MLP (PR-AUC 0.9296) |
