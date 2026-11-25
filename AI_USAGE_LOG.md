# AI Usage Log

All major AI-assisted steps are logged here for transparency, as required by Q//Horizon 2026 rules.

| Date | Phase/Branch | AI Tool | Purpose | Human Verification |
|------|-------------|---------|---------|-------------------|
| 2026-11-25 | Phase 0 / `main` | Antigravity AI Assistant | Repository configuration, `.gitignore`, pinned `requirements.txt`, `AI_USAGE_LOG.md`, `WORKFLOW.md` | Verified clean pinned dependencies matching `.venv`, `.venv/` exclusion in `.gitignore`, and branch readiness |
| 2026-11-25 | Phase 1 / `classical-baselines` | Antigravity AI Assistant | Leakage-free HTRU2 data pipeline (`src/config.py`, `src/data.py`), evaluation metrics (`src/metrics.py`), and classical baselines (`src/models_classical.py`) | Verified zero test-set leakage, positive pulsar ratio (~9.16%), and smoke-tested Dummy, LogReg, RBF-SVM, Tiny-MLP |
