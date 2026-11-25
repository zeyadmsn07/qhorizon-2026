# AI Usage Log

All major AI-assisted steps are logged here for transparency, as required by Q//Horizon 2026 rules.

| Date | Phase/Branch | AI Tool | Purpose | Human Verification |
|------|-------------|---------|---------|-------------------|
| 2026-11-25 | Phase 0 / `main` | Antigravity AI Assistant | Repository configuration, `.gitignore`, pinned `requirements.txt`, `AI_USAGE_LOG.md`, `WORKFLOW.md` | Verified clean pinned dependencies matching `.venv`, `.venv/` exclusion in `.gitignore`, and branch readiness |
| 2026-11-25 | Phase 1 / `classical-baselines` | Antigravity AI Assistant | Leakage-free HTRU2 data pipeline (`src/config.py`, `src/data.py`), evaluation metrics (`src/metrics.py`), classical baselines (`src/models_classical.py`), and verification runner (`verify_phase1.py`) | Verified zero test-set leakage, positive pulsar ratio (0.0920 train, 0.0917 test), and smoke-tested Dummy (PR-AUC 0.0917), LogReg (PR-AUC 0.9383), RBF-SVM (PR-AUC 0.8630), RandomForest (PR-AUC 0.9027), Tiny-MLP (PR-AUC 0.9296) with saved CSV metrics |
