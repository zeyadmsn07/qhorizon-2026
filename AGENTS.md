# Project Constitution & Engineering Rules (`AGENTS.md`)

All contributors and AI coding agents working in this repository MUST follow every rule below without exception.

## 1. Git Branching & Push Protocol
* **Base Setup on `main`:** Commit and push **Phase 0** (repo config, `AGENTS.md`, `AI_USAGE_LOG.md`, `requirements.txt`) and **Phase 1** (`src/__init__.py`, `src/config.py`, `src/data.py`) directly to `main` after running a quick smoke test.
* **Descriptive Feature Branches:** For every subsequent bulk of work, create, commit to, and push a branch with a simple, plain-English name describing its contents (never use task-numbered names like `phase-2` or `task-3`):
  * `classical-baselines` - Fair evaluation metrics and the 5 classical models.
  * `quantum-models` - Batched variational quantum classifier and state-vector fidelity kernel SVM.
  * `gold-experiments` - Equal-budget validation tuner and benchmark runner.
  * `plots-and-readme` - Error-bar figures, master notebook, and final elaborated `README.md`.
* **Final Merge:** Merge all verified branches cleanly into `main` before final submission.

## 2. Human Comment Standard
* **No AI Filler:** Never write generic comments that just narrate obvious syntax (for example, do NOT write `# Initialize model`, `# Split data`, `# Loop over epochs`, or `# Import libraries`).
* **Explain Technical Decisions Only:** Write short, plain-English comments only where a reader needs to know *why* a choice was made (such as why Yeo-Johnson is fit strictly on `X_train`, why all wires are measured when entanglement is disabled, or why state-vector kernels avoid $O(N^2)$ circuit evaluations in simulation).

## 3. Quantum Simulation & Hardware Constraints
* **Qubit Ceiling:** Every `qml.device` MUST use $\le 10$ wires (`wires <= 8` for HTRU2).
* **Batched Vectorization Only:** Use `qml.device("default.qubit", wires=n_wires)` with `@qml.qnode(dev, interface="torch", diff_method="backprop")` for training. Pass 2D batched tensors shaped `(batch_size, n_features)` directly into gates.
* **Banned Patterns:** NEVER loop sample-by-sample in Python (`for x in X:`) during training, and NEVER combine `default.mixed` with batched PyTorch `backprop`. Apply finite measurement shots (`qml.set_shots`) at inference time only with `diff_method=None`.
* **Fast Exact Kernel Trick:** Compute batched state vectors $\Psi$ via `qml.state()` once and compute the fidelity Gram matrix $K = |\Psi \Psi^\dagger|^2$ via matrix multiplication rather than $O(N^2)$ `kernel_matrix` calls. Disclose in documentation that real hardware requires $O(N^2)$ circuit runs.

## 4. Statistical Fairness & Zero Data Leakage
* **Train-Only Preprocessing:** Fit `PowerTransformer` and any scalers strictly on `X_train`, then transform `X_val` and `X_test`.
* **Validation-Only Tuning & Thresholds:** Every tunable model family must share the exact same validation random-search budget (`N_TUNING_CONFIGS = 20` in full mode, `3` in `FAST_MODE`). Select classification thresholds on `y_val` only, never on `y_test`.

## 5. AI Transparency Log
* Log every major AI-assisted generation, debugging pass, or architectural update in `AI_USAGE_LOG.md` so our final 4-page report and `README.md` accurately credit AI tool usage as required by Q//Horizon 2026 rules.