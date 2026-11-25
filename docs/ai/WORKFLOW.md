# Q//Horizon 2026: Detailed Sub-Tasks, Optimizations, GitHub Workflow & Antigravity Blueprint

> **HOW TO USE THIS FILE WITH GOOGLE ANTIGRAVITY (FREE TIER):**
> 1. Clone your GitHub repo (containing the starter `README.md`) via PowerShell and open the folder in Antigravity.
> 2. Drop this `WORKFLOW.md` file (and `AGENTS.md` if you want it pre-placed) into the root of your repo folder.
> 3. **Save your free-tier rate limits:** Ask Antigravity to execute **one bulk of work at a time**, verify it, and push it to GitHub before moving to the next branch. Run the heavy $10$-seed `python run_all.py` sweep directly in your terminal rather than inside an agent loop.

---

## Copy-Paste Kickoff Prompt for Antigravity (Phase 0 & Phase 1 Only)

```text
Read WORKFLOW.md carefully. Execute ONLY Phase 0 and Phase 1 directly on `main`: create `AGENTS.md` (if not already present), `AI_USAGE_LOG.md`, `requirements.txt`, and `src/` (`__init__.py`, `config.py`, `data.py`), install dependencies, and run a smoke test verifying UCI HTRU2 downloads, caches, and preprocesses with zero data leakage. Follow every rule in AGENTS.md, especially writing short human comments that explain technical decisions rather than generic filler comments. Once verified, commit and push `main` to GitHub, create and switch to a new branch named `classical-baselines`, and STOP there without starting Phase 2.
```

---

## Git Branching & Push Workflow

Antigravity has terminal access and must commit and push verified work at each checkpoint:

1. **Base Setup on `main`:** Commit and push **Phase 0** and **Phase 1** directly to `main` so every branch shares the locked data pipeline and rules.
2. **Work in Descriptive Feature Branches:** Use simple, plain-English branch names (never task numbers):
   * `classical-baselines` - Fair evaluation metrics + the 5 classical models (`Phase 2`).
   * `quantum-models` - Batched VQC and state-vector fidelity kernel SVM (`Phase 3`).
   * `gold-experiments` - Equal-budget validation tuner + benchmark runner (`Phase 4`).
   * `plots-and-readme` - Error-bar figures, master notebook, and upgraded `README.md` (`Phase 5`).
3. **Final Merge:** Once all branches are verified and pushed, merge them cleanly back into `main` before submission.

---

## Phase 0: GitHub Repository Setup, Dependencies & AI Guardrails (On `main`)

### Task 0.1 - GitHub Repository Layout (No Docker)
* Pure Python packages (`pennylane`, `torch`, `scikit-learn`) install cleanly via `pip` on Windows, macOS, Linux, and Google Colab without container overhead.
* **Target Directory Layout:**
  ```text
  q-horizon-2026/
  |-- AGENTS.md                  # AI coding rules and comment style standard
  |-- AI_USAGE_LOG.md            # Mandatory log of AI tools & prompts used
  |-- WORKFLOW.md                # This engineering blueprint
  |-- README.md                  # Starter README now -> upgraded on `plots-and-readme`
  |-- requirements.txt           # Pinned Colab-compatible packages
  |-- Q_Horizon_Benchmark.ipynb  # Clean master notebook for judges
  |-- run_all.py                 # End-to-end experiment runner
  |-- src/
  |   |-- __init__.py
  |   |-- config.py              # Global seeds, FAST_MODE switch, paths
  |   |-- data.py                # UCI HTRU2 loader, locked splits, Yeo-Johnson
  |   |-- metrics.py             # PR-AUC, MCC, F1, validation-only threshold
  |   |-- models_classical.py    # 5 classical baselines (Dummy -> Tiny-MLP)
  |   |-- models_quantum.py      # Batched VQC (Q1) & State-Vector Kernel (Q2)
  |   `-- tuning.py              # Equal 20-config validation search
  `-- results/                   # Pre-computed CSVs & high-DPI plots
  ```
* Also add `data/` and `__pycache__/` (plus `.venv/` if using a virtual environment) to `.gitignore` so large raw caches and bytecode do not clutter commits, while keeping `results/*.csv` and `results/*.png` tracked in Git.

### Task 0.2 - `requirements.txt` (Pinned Colab-Compatible Versions)
Create `requirements.txt` in the repo root (`pennylane==0.45.1` requires NumPy 2+):
```text
pennylane==0.45.1
numpy>=2.0.0
torch>=2.3.0
scikit-learn>=1.5.0
scipy>=1.13.0
pandas>=2.2.0
matplotlib>=3.8.0
ucimlrepo>=0.0.7
tqdm>=4.66.0
```

### Task 0.3 - `AGENTS.md` (Ensure Present in Root)
If `AGENTS.md` is not yet in the root directory, write it with these exact rules:
1. **Git Workflow:** Lay Phase 0 and Phase 1 on `main` and push. Build subsequent work on descriptive branches (`classical-baselines`, `quantum-models`, `gold-experiments`, `plots-and-readme`) and merge into `main` at the end.
2. **Human Comment Standard:** Never write AI filler comments that restate syntax (`# Initialize model`, `# Loop over epochs`). Write concise plain-English comments only to explain *why* a technical decision was made.
3. **Qubit Ceiling:** Every `qml.device` MUST use $\le 10$ wires (`wires <= 8` for HTRU2).
4. **Vectorized Simulation Only:** Use `qml.device("default.qubit", wires=n_wires)` with `@qml.qnode(dev, interface="torch", diff_method="backprop")`. Pass 2D batched tensors `(batch_size, n_features)` directly into gates. Never loop sample-by-sample in Python during training, and never use `default.mixed` with batched PyTorch `backprop`.
5. **Zero Data Leakage:** Fit all preprocessors (`PowerTransformer`) strictly on `X_train` and select all decision thresholds strictly on `y_val`, never on `X_test` / `y_test`.
6. **Fast Exact Kernel Rule:** Compute batched state vectors $\Psi$ via `qml.state()` once and compute $K = |\Psi \Psi^\dagger|^2$ via matrix multiplication instead of $O(N^2)$ `kernel_matrix` calls.
7. **Equal Tuning Budget:** Every tunable model family must use the shared validation random search (`N_TUNING_CONFIGS = 20` in full mode, `3` in `FAST_MODE`).

### Task 0.4 - Initialize `AI_USAGE_LOG.md`
Create `AI_USAGE_LOG.md` with a clean markdown table (`Date | Phase/Branch | AI Tool | Purpose | Human Verification`) logging Gemini / Claude for architecture and fairness design, and Google Antigravity for repository scaffolding, debugging, and Git operations.

---

## Phase 1: Configuration & Leakage-Free Data Pipeline (On `main`)

### Task 1.1 - `src/__init__.py` & `src/config.py`
Create an empty `src/__init__.py` and write `src/config.py`:

```python
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Keep True while developing and for judge reruns (< 3 mins); set False for the full 10-seed sweep
FAST_MODE = True

MASTER_SPLIT_SEED = 2026
ALL_SEEDS = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
SEEDS = ALL_SEEDS[:2] if FAST_MODE else ALL_SEEDS

N_TUNING_CONFIGS = 3 if FAST_MODE else 20
N_TEST = 3000
N_VAL = 1500
SAMPLE_SWEEP = [100, 200, 500] if FAST_MODE else [100, 200, 500, 1000]
```

### Task 1.2 - `src/data.py` (UCI HTRU2 + Heavy-Tail Yeo-Johnson Preprocessing)

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from ucimlrepo import fetch_ucirepo

from src.config import DATA_DIR, MASTER_SPLIT_SEED, N_TEST, N_VAL

FEATURE_NAMES = [
    "IP_Mean", "IP_Std", "IP_Kurtosis", "IP_Skewness",
    "DM_Mean", "DM_Std", "DM_Kurtosis", "DM_Skewness",
]


def load_htru2_raw() -> tuple[np.ndarray, np.ndarray]:
    # Cache locally so repeated runs do not depend on UCI network availability
    cache_path = DATA_DIR / "htru2_cached.csv"
    if cache_path.exists():
        df = pd.read_csv(cache_path)
    else:
        dataset = fetch_ucirepo(id=372)  # Official UCI HTRU2 record (CC BY 4.0)
        X_df = dataset.data.features.copy()
        X_df.columns = FEATURE_NAMES
        y_ser = dataset.data.targets.iloc[:, 0].astype(int)
        df = X_df.copy()
        df["target"] = y_ser.values
        df.to_csv(cache_path, index=False)

    X = df[FEATURE_NAMES].to_numpy(dtype=np.float64)
    y = df["target"].to_numpy(dtype=np.int64)
    return X, y


def get_locked_splits() -> dict:
    # Lock Test (3000) and Val (1500) once so every sample-size sweep evaluates against identical holdouts
    X, y = load_htru2_raw()
    X_rem, X_test, y_rem, y_test = train_test_split(
        X, y, test_size=N_TEST, stratify=y, random_state=MASTER_SPLIT_SEED
    )
    X_pool, X_val, y_pool, y_val = train_test_split(
        X_rem, y_rem, test_size=N_VAL, stratify=y_rem, random_state=MASTER_SPLIT_SEED
    )
    return {
        "X_pool": X_pool, "y_pool": y_pool,
        "X_val": X_val,   "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
    }


def sample_and_preprocess(splits: dict, n_train: int, seed: int) -> dict:
    X_train_raw, _, y_train, _ = train_test_split(
        splits["X_pool"], splits["y_pool"],
        train_size=n_train, stratify=splits["y_pool"], random_state=seed
    )

    # Skewness and kurtosis have extreme tails that crush MinMax angles near zero;
    # Yeo-Johnson tames outliers and is fit strictly on X_train to prevent test leakage
    pt = PowerTransformer(method="yeo-johnson", standardize=True)
    X_train = pt.fit_transform(X_train_raw)
    X_val = pt.transform(splits["X_val"])
    X_test = pt.transform(splits["X_test"])

    return {
        "X_train": X_train.astype(np.float32),
        "y_train": y_train,
        "X_val": X_val.astype(np.float32),
        "y_val": splits["y_val"],
        "X_test": X_test.astype(np.float32),
        "y_test": splits["y_test"],
    }
```

**Checkpoint (End of Phase 1):** Run a one-line Python check verifying `get_locked_splits()` and `sample_and_preprocess(..., n_train=500, seed=100)` return finite `float32` arrays with zero NaNs and ~9.16% positive ratio. Commit Phase 0 + Phase 1 to `main`, push `main` to GitHub, create branch `classical-baselines`, and stop until instructed to run Phase 2.

---

## Phase 2: Fair Metrics & The 5 Classical Baselines (Branch: `classical-baselines`)

### Task 2.1 - `src/metrics.py`

```python
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
)


def select_threshold_on_val(y_val: np.ndarray, val_scores: np.ndarray) -> float:
    # Pick the decision threshold on Validation only so Test metrics stay completely unbiased
    if np.allclose(val_scores, val_scores[0]):
        return float(val_scores[0])

    candidates = np.unique(np.quantile(val_scores, np.linspace(0.05, 0.99, 101)))
    best_tau = float(candidates[0])
    best_f1 = -1.0
    best_mcc = -1.0

    for tau in candidates:
        preds = (val_scores >= tau).astype(int)
        f1 = f1_score(y_val, preds, pos_label=1, zero_division=0)
        mcc = matthews_corrcoef(y_val, preds)
        if (f1 > best_f1) or (np.isclose(f1, best_f1) and mcc > best_mcc):
            best_f1 = f1
            best_mcc = mcc
            best_tau = float(tau)

    return best_tau


def evaluate_predictions(
    y_val: np.ndarray,
    val_scores: np.ndarray,
    y_test: np.ndarray,
    test_scores: np.ndarray,
    n_params: int,
    train_time_sec: float,
) -> dict:
    # PR-AUC is our primary metric because only 9.16% of HTRU2 candidates are real pulsars
    val_pr_auc = float(average_precision_score(y_val, val_scores))
    test_pr_auc = float(average_precision_score(y_test, test_scores))

    tau = select_threshold_on_val(y_val, val_scores)
    test_preds = (test_scores >= tau).astype(int)

    return {
        "val_pr_auc": val_pr_auc,
        "test_pr_auc": test_pr_auc,
        "test_mcc": float(matthews_corrcoef(y_test, test_preds)),
        "test_f1": float(f1_score(y_test, test_preds, pos_label=1, zero_division=0)),
        "test_acc": float(accuracy_score(y_test, test_preds)),
        "threshold": tau,
        "n_params": int(n_params),
        "train_time_sec": float(train_time_sec),
    }
```

### Task 2.2 - `src/models_classical.py`

```python
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


class TinyMLP(nn.Module):
    # hidden_dim=4 gives 41 trainable parameters, matching our 8-qubit L=2 quantum circuit (34 params)
    def __init__(self, in_features: int = 8, hidden_dim: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def count_torch_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fit_predict_classical(
    model_name: str,
    params: dict,
    data: dict,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, X_test = data["X_val"], data["X_test"]
    t0 = time.perf_counter()

    if model_name == "Dummy":
        # Sets the 90.84% majority-noise accuracy baseline and 0.0916 PR-AUC floor
        clf = DummyClassifier(strategy="most_frequent")
        clf.fit(X_train, y_train)
        val_scores = clf.predict_proba(X_val)[:, 1]
        test_scores = clf.predict_proba(X_test)[:, 1]
        n_params = 0

    elif model_name == "LogReg":
        clf = LogisticRegression(
            C=params["C"],
            class_weight=params["class_weight"],
            max_iter=1000,
            random_state=seed,
        )
        clf.fit(X_train, y_train)
        val_scores = clf.decision_function(X_val)
        test_scores = clf.decision_function(X_test)
        n_params = clf.coef_.size + clf.intercept_.size

    elif model_name == "RBF-SVM":
        clf = SVC(
            kernel="rbf",
            C=params["C"],
            gamma=params["gamma"],
            class_weight=params["class_weight"],
            random_state=seed,
        )
        clf.fit(X_train, y_train)
        val_scores = clf.decision_function(X_val)
        test_scores = clf.decision_function(X_test)
        n_params = clf.support_vectors_.size

    elif model_name == "RandomForest":
        clf = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            class_weight=params["class_weight"],
            random_state=seed,
            n_jobs=-1,
        )
        clf.fit(X_train, y_train)
        val_scores = clf.predict_proba(X_val)[:, 1]
        test_scores = clf.predict_proba(X_test)[:, 1]
        n_params = sum(est.tree_.node_count for est in clf.estimators_)

    elif model_name == "Tiny-MLP":
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = TinyMLP(in_features=X_train.shape[1], hidden_dim=params.get("hidden_dim", 4))
        n_params = count_torch_params(model)

        pos_ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        pos_weight = torch.tensor([pos_ratio * params["pos_weight_mult"]], dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"]
        )

        Xt = torch.tensor(X_train, dtype=torch.float32)
        yt = torch.tensor(y_train, dtype=torch.float32)
        Xv = torch.tensor(X_val, dtype=torch.float32)
        yv = torch.tensor(data["y_val"], dtype=torch.float32)
        Xte = torch.tensor(X_test, dtype=torch.float32)

        best_val_loss = float("inf")
        best_state = None
        patience_left = 15

        for _ in range(120):
            model.train()
            perm = torch.randperm(len(Xt))
            for i in range(0, len(Xt), params["batch_size"]):
                idx = perm[i : i + params["batch_size"]]
                optimizer.zero_grad()
                loss = criterion(model(Xt[idx]), yt[idx])
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                v_loss = criterion(model(Xv), yv).item()
            if v_loss < best_val_loss - 1e-5:
                best_val_loss = v_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_left = 15
            else:
                patience_left -= 1
                if patience_left == 0:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            val_scores = model(Xv).numpy()
            test_scores = model(Xte).numpy()
    else:
        raise ValueError(f"Unknown classical model: {model_name}")

    fit_sec = time.perf_counter() - t0
    return val_scores, test_scores, n_params, fit_sec
```

**Checkpoint:** Smoke-test all 5 classical models on one seed, commit, and push branch `classical-baselines`.

---

## Phase 3: Optimized Quantum Circuits (Branch: `quantum-models`)

Create branch `quantum-models` from `main` (or merge `classical-baselines` forward if needed for testing).

### Task 3.1 - `src/models_quantum.py`

```python
import time
import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
from sklearn.svm import SVC


def _apply_entanglement(n_wires: int, topology: str, layer_idx: int) -> None:
    if topology == "none" or n_wires < 2:
        return
    elif topology == "ring":
        for i in range(n_wires):
            qml.CNOT(wires=[i, (i + 1) % n_wires])
    elif topology == "bipartite" and n_wires == 8:
        # Tests our domain hypothesis linking Pulse Profile wires (0..3) to DM-SNR wires (4..7)
        if layer_idx % 2 == 0:
            for k in range(4):
                qml.CNOT(wires=[k, k + 4])
        else:
            for k in range(4):
                qml.CNOT(wires=[k + 4, (k + 1) % 4])
    else:
        for i in range(n_wires - 1):
            qml.CNOT(wires=[i, i + 1])


class VariationalQuantumClassifier(nn.Module):
    def __init__(
        self,
        encoding: str = "angle",       # "angle" (8q), "amplitude" (3q), "amplitude_norm" (4q)
        entanglement: str = "ring",    # "none", "ring", "bipartite"
        n_layers: int = 2,
        alpha: float = 1.0,
        reupload: bool = False,
    ):
        super().__init__()
        self.encoding = encoding
        self.entanglement = entanglement
        self.n_layers = n_layers
        self.alpha = float(alpha)
        self.reupload = reupload

        if encoding == "angle":
            self.n_wires = 8
        elif encoding == "amplitude":
            self.n_wires = 3
        elif encoding == "amplitude_norm":
            self.n_wires = 4
        else:
            raise ValueError(f"Unsupported encoding: {encoding}")

        # Small initial weights start optimization in a smooth region with healthy gradients
        self.weights = nn.Parameter(0.05 * torch.randn(n_layers, self.n_wires, 2))
        # Expectation averages lie in [-1, 1]; learnable scale and bias map them cleanly to BCE logits
        self.logit_scale = nn.Parameter(torch.tensor(4.0))
        self.logit_bias = nn.Parameter(torch.tensor(0.0))

        dev = qml.device("default.qubit", wires=self.n_wires)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(x_batch, weights):
            for l in range(self.n_layers):
                if l == 0 or (self.reupload and self.encoding == "angle"):
                    self._encode_inputs(x_batch)
                for w in range(self.n_wires):
                    qml.RY(weights[l, w, 0], wires=w)
                    qml.RZ(weights[l, w, 1], wires=w)
                _apply_entanglement(self.n_wires, self.entanglement, l)
            # Average Pauli-Z across all wires so unentangled ("none") circuits still use all features
            return [qml.expval(qml.PauliZ(w)) for w in range(self.n_wires)]

        self.circuit = circuit

    def _encode_inputs(self, x_batch: torch.Tensor) -> None:
        if self.encoding == "angle":
            # Smooth arctan bounds angles to (-pi, pi) so unseen test outliers never wrap modulo 2*pi
            angles = 2.0 * torch.atan(self.alpha * x_batch)
            qml.AngleEmbedding(angles, wires=range(8), rotation="Y")
        elif self.encoding == "amplitude":
            # Normalize in float64 to satisfy PennyLane's unit-norm check while compressing 8 features to 3 qubits
            x64 = x_batch.to(torch.float64)
            norms = torch.linalg.norm(x64, dim=-1, keepdim=True).clamp_min(1e-12)
            qml.AmplitudeEmbedding(x64 / norms, wires=range(3), normalize=False)
        elif self.encoding == "amplitude_norm":
            # Re-injects the discarded L2 norm on wire 3 to test if radial magnitude loss hurts amplitude encoding
            x64 = x_batch.to(torch.float64)
            norms = torch.linalg.norm(x64, dim=-1, keepdim=True).clamp_min(1e-12)
            qml.AmplitudeEmbedding(x64 / norms, wires=range(3), normalize=False)
            norm_angle = 2.0 * torch.atan(self.alpha * norms.squeeze(-1))
            qml.RY(norm_angle, wires=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cast QNode float64 outputs back to x.dtype so PyTorch BCEWithLogitsLoss never hits dtype mismatches
        expvals = torch.stack(self.circuit(x, self.weights), dim=-1).to(dtype=x.dtype)
        z_bar = expvals.mean(dim=-1)
        return self.logit_scale * z_bar + self.logit_bias

    @torch.no_grad()
    def predict_shots(self, x: torch.Tensor, shots: int | None) -> np.ndarray:
        # Evaluate finite shots at inference only (diff_method=None) to avoid default.mixed batching bugs
        dev = qml.device("default.qubit", wires=self.n_wires)

        @qml.qnode(dev, interface="torch", diff_method=None)
        def infer_circuit(x_b, weights):
            for l in range(self.n_layers):
                if l == 0 or (self.reupload and self.encoding == "angle"):
                    self._encode_inputs(x_b)
                for w in range(self.n_wires):
                    qml.RY(weights[l, w, 0], wires=w)
                    qml.RZ(weights[l, w, 1], wires=w)
                _apply_entanglement(self.n_wires, self.entanglement, l)
            return [qml.expval(qml.PauliZ(w)) for w in range(self.n_wires)]

        if shots is not None:
            infer_circuit = qml.set_shots(infer_circuit, shots=shots)

        expvals = torch.stack(infer_circuit(x, self.weights), dim=-1).to(dtype=x.dtype)
        logits = self.logit_scale * expvals.mean(dim=-1) + self.logit_bias
        return logits.cpu().numpy()


def train_predict_vqc(
    params: dict,
    data: dict,
    seed: int,
    return_model: bool = False,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    t0 = time.perf_counter()

    model = VariationalQuantumClassifier(
        encoding=params.get("encoding", "angle"),
        entanglement=params.get("entanglement", "ring"),
        n_layers=params.get("n_layers", 2),
        alpha=params.get("alpha", 1.0),
        reupload=params.get("reupload", False),
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    Xt = torch.tensor(data["X_train"], dtype=torch.float32)
    yt = torch.tensor(data["y_train"], dtype=torch.float32)
    Xv = torch.tensor(data["X_val"], dtype=torch.float32)
    yv = torch.tensor(data["y_val"], dtype=torch.float32)
    Xte = torch.tensor(data["X_test"], dtype=torch.float32)

    pos_ratio = (data["y_train"] == 0).sum() / max((data["y_train"] == 1).sum(), 1)
    pos_weight = torch.tensor([pos_ratio * params.get("pos_weight_mult", 1.0)], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params.get("lr", 0.02),
        weight_decay=params.get("weight_decay", 1e-4),
    )

    # Record step-0 gradient norm to diagnose whether deeper circuits suffer from vanishing gradients
    optimizer.zero_grad()
    init_loss = criterion(model(Xt), yt)
    init_loss.backward()
    init_grad_norm = float(
        torch.sqrt(sum(p.grad.pow(2).sum() for p in model.parameters() if p.grad is not None))
    )
    optimizer.zero_grad()

    best_val_loss = float("inf")
    best_state = None
    patience = 12
    batch_size = params.get("batch_size", 64)

    for _ in range(params.get("max_epochs", 75)):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), batch_size):
            idx = perm[i : i + batch_size]
            optimizer.zero_grad()
            loss = criterion(model(Xt[idx]), yt[idx])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            v_loss = criterion(model(Xv), yv).item()
        if v_loss < best_val_loss - 1e-5:
            best_val_loss = v_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience = 12
        else:
            patience -= 1
            if patience == 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        val_scores = model(Xv).numpy()
        test_scores = model(Xte).numpy()

    fit_sec = time.perf_counter() - t0
    if return_model:
        return val_scores, test_scores, n_params, fit_sec, init_grad_norm, model
    return val_scores, test_scores, n_params, fit_sec


def train_predict_quantum_kernel_svm(
    params: dict,
    data: dict,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    t0 = time.perf_counter()
    n_wires = 8
    alpha = float(params.get("alpha", 1.0))
    depth = int(params.get("depth", 1))
    dev = qml.device("default.qubit", wires=n_wires)

    @qml.qnode(dev, interface="torch", diff_method=None)
    def feature_state(x_batch):
        # IQP-style ZZ feature map capturing pairwise feature products
        angles = 2.0 * torch.atan(alpha * x_batch)
        for _ in range(depth):
            for w in range(n_wires):
                qml.Hadamard(wires=w)
                qml.RZ(angles[:, w], wires=w)
            for w in range(n_wires):
                w_next = (w + 1) % n_wires
                qml.CNOT(wires=[w, w_next])
                qml.RZ(angles[:, w] * angles[:, w_next], wires=w_next)
                qml.CNOT(wires=[w, w_next])
        return qml.state()

    Xt = torch.tensor(data["X_train"], dtype=torch.float32)
    Xv = torch.tensor(data["X_val"], dtype=torch.float32)
    Xte = torch.tensor(data["X_test"], dtype=torch.float32)

    # Simulation shortcut: batch-compute state vectors once and evaluate K = |Psi Psi^dagger|^2
    # via matrix multiplication instead of O(N^2) circuit runs (real hardware requires O(N^2) runs)
    with torch.no_grad():
        psi_tr = feature_state(Xt)
        psi_va = feature_state(Xv)
        psi_te = feature_state(Xte)

        K_tr = torch.abs(psi_tr @ psi_tr.conj().T).pow(2).cpu().numpy()
        K_va = torch.abs(psi_va @ psi_tr.conj().T).pow(2).cpu().numpy()
        K_te = torch.abs(psi_te @ psi_tr.conj().T).pow(2).cpu().numpy()

    clf = SVC(
        kernel="precomputed",
        C=params.get("C", 1.0),
        class_weight=params.get("class_weight", "balanced"),
        random_state=seed,
    )
    clf.fit(K_tr, data["y_train"])
    val_scores = clf.decision_function(K_va)
    test_scores = clf.decision_function(K_te)

    n_params = 1
    fit_sec = time.perf_counter() - t0
    return val_scores, test_scores, n_params, fit_sec
```

**Checkpoint:** Smoke-test `VariationalQuantumClassifier` (all 3 encodings + finite-shot `predict_shots`) and `train_predict_quantum_kernel_svm`, commit, and push branch `quantum-models`.

---

## Phase 4: Equal-Budget Tuner & Experiment Runner (Branch: `gold-experiments`)

Create branch `gold-experiments` and merge `classical-baselines` and `quantum-models` into it so all models are present.

### Task 4.1 - `src/tuning.py`

```python
import numpy as np
from sklearn.model_selection import ParameterSampler

from src.config import N_TUNING_CONFIGS, SEEDS
from src.data import sample_and_preprocess
from src.metrics import evaluate_predictions
from src.models_classical import fit_predict_classical
from src.models_quantum import train_predict_quantum_kernel_svm, train_predict_vqc

SEARCH_SPACES = {
    "Dummy": [{}],
    "LogReg": {
        "C": np.logspace(-3, 3, 50),
        "class_weight": [None, "balanced"],
    },
    "RBF-SVM": {
        "C": np.logspace(-2, 3, 50),
        "gamma": np.logspace(-3, 1, 50),
        "class_weight": [None, "balanced", {0: 1, 1: 5}, {0: 1, 1: 10}],
    },
    "RandomForest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 8, None],
        "min_samples_leaf": [1, 3, 8],
        "class_weight": [None, "balanced", "balanced_subsample"],
    },
    "Tiny-MLP": {
        "hidden_dim": [4],  # Fixed at 41 params for an equal parameter-budget comparison against VQC
        "lr": [1e-3, 3e-3, 1e-2, 3e-2],
        "weight_decay": [0.0, 1e-4, 1e-2],
        "pos_weight_mult": [0.5, 1.0, 1.5],
        "batch_size": [32, 64],
    },
    "Q1-VQC": {
        "alpha": [0.25, 0.5, 1.0, 1.5, 2.5],  # Input angle bandwidth is the dominant QML performance lever
        "lr": [0.005, 0.01, 0.02, 0.04],
        "weight_decay": [0.0, 1e-4, 1e-3],
        "pos_weight_mult": [0.5, 1.0, 1.5],
        "batch_size": [64],
    },
    "Q2-QuantumKernel": {
        "alpha": np.logspace(-1.0, 0.5, 30),
        "depth": [1, 2],
        "C": np.logspace(-2, 3, 40),
        "class_weight": [None, "balanced", {0: 1, 1: 5}, {0: 1, 1: 10}],
    },
}


def _run_once(family: str, params: dict, data: dict, seed: int) -> dict:
    if family in {"Dummy", "LogReg", "RBF-SVM", "RandomForest", "Tiny-MLP"}:
        vs, ts, npa, sec = fit_predict_classical(family, params, data, seed)
    elif family == "Q1-VQC":
        vs, ts, npa, sec = train_predict_vqc(params, data, seed)
    elif family == "Q2-QuantumKernel":
        vs, ts, npa, sec = train_predict_quantum_kernel_svm(params, data, seed)
    else:
        raise ValueError(f"Unknown family: {family}")
    return evaluate_predictions(data["y_val"], vs, data["y_test"], ts, npa, sec)


def tune_and_evaluate_10_seeds(
    splits: dict,
    n_train: int,
    family: str,
    fixed_overrides: dict | None = None,
    label: str | None = None,
) -> list[dict]:
    # Give every tunable model the exact same validation random-search budget before test evaluation
    fixed_overrides = fixed_overrides or {}
    label = label or family

    space = SEARCH_SPACES[family]
    if isinstance(space, list):
        candidates = [{**c, **fixed_overrides} for c in space]
    else:
        sampler = ParameterSampler(space, n_iter=N_TUNING_CONFIGS, random_state=42)
        candidates = [{**c, **fixed_overrides} for c in sampler]

    ref_data = sample_and_preprocess(splits, n_train=n_train, seed=SEEDS[0])
    best_cfg = candidates[0]
    best_val_pr = -1.0

    for cfg in candidates:
        res = _run_once(family, cfg, ref_data, seed=SEEDS[0])
        if res["val_pr_auc"] > best_val_pr:
            best_val_pr = res["val_pr_auc"]
            best_cfg = cfg

    records = []
    for s in SEEDS:
        d = sample_and_preprocess(splits, n_train=n_train, seed=s)
        metrics = _run_once(family, best_cfg, d, seed=s)
        metrics.update({"model": label, "family": family, "n_train": n_train, "seed": s})
        records.append(metrics)
    return records
```

### Task 4.2 - `run_all.py`

```python
import pandas as pd
import torch
from tqdm import tqdm

from src.config import RESULTS_DIR, SAMPLE_SWEEP, SEEDS
from src.data import get_locked_splits, sample_and_preprocess
from src.metrics import evaluate_predictions
from src.models_quantum import train_predict_vqc
from src.tuning import tune_and_evaluate_10_seeds


def main():
    splits = get_locked_splits()

    # Main benchmark at N=500 across all 5 classical baselines and 2 quantum models
    main_rows = []
    specs = [
        ("Dummy", {}, "Dummy"),
        ("LogReg", {}, "LogReg"),
        ("RBF-SVM", {}, "RBF-SVM"),
        ("RandomForest", {}, "RandomForest"),
        ("Tiny-MLP", {}, "Tiny-MLP (41p)"),
        ("Q1-VQC", {"encoding": "angle", "entanglement": "ring", "n_layers": 2}, "Q1-VQC (8q Ring L=2)"),
        ("Q2-QuantumKernel", {}, "Q2-QuantumKernel (8q)"),
    ]
    for fam, ov, lbl in tqdm(specs, desc="Main Benchmark"):
        main_rows.extend(tune_and_evaluate_10_seeds(splits, 500, fam, ov, lbl))
    pd.DataFrame(main_rows).to_csv(RESULTS_DIR / "main_benchmark.csv", index=False)

    # Gold A: isolate whether entanglement topology and depth actually help on tabular data
    gold_a_rows = []
    for ent in ["none", "ring", "bipartite"]:
        for L in [1, 2, 4]:
            lbl = f"VQC_{ent}_L{L}"
            ov = {"encoding": "angle", "entanglement": ent, "n_layers": L}
            gold_a_rows.extend(tune_and_evaluate_10_seeds(splits, 500, "Q1-VQC", ov, lbl))
    pd.DataFrame(gold_a_rows).to_csv(RESULTS_DIR / "gold_a_entanglement.csv", index=False)

    # Gold B: compare 8-qubit angle encoding vs 3-qubit amplitude compression and norm restoration
    gold_b_specs = [
        ("8q_Angle_L2", {"encoding": "angle", "entanglement": "ring", "n_layers": 2, "reupload": False}),
        ("8q_Angle_Reupload_L2", {"encoding": "angle", "entanglement": "ring", "n_layers": 2, "reupload": True}),
        ("3q_Amplitude_L5", {"encoding": "amplitude", "entanglement": "ring", "n_layers": 5}),
        ("4q_AmplitudePlusNorm_L4", {"encoding": "amplitude_norm", "entanglement": "ring", "n_layers": 4}),
    ]
    gold_b_rows = []
    for lbl, ov in tqdm(gold_b_specs, desc="Gold B Encoding"):
        gold_b_rows.extend(tune_and_evaluate_10_seeds(splits, 500, "Q1-VQC", ov, lbl))
    pd.DataFrame(gold_b_rows).to_csv(RESULTS_DIR / "gold_b_encoding.csv", index=False)

    # Gold C: test low-data sample efficiency across N = 100..1000
    gold_c_rows = []
    sweep_models = [
        ("Dummy", {}, "Dummy"),
        ("RBF-SVM", {}, "RBF-SVM"),
        ("Tiny-MLP", {}, "Tiny-MLP (41p)"),
        ("Q1-VQC", {"encoding": "angle", "entanglement": "ring", "n_layers": 2}, "Q1-VQC (8q)"),
        ("Q2-QuantumKernel", {}, "Q2-QuantumKernel (8q)"),
    ]
    for N in SAMPLE_SWEEP:
        for fam, ov, lbl in sweep_models:
            gold_c_rows.extend(tune_and_evaluate_10_seeds(splits, N, fam, ov, lbl))
    pd.DataFrame(gold_c_rows).to_csv(RESULTS_DIR / "gold_c_sweep.csv", index=False)

    # Table D: check inference-only shot noise degradation on trained weights
    shot_rows = []
    for s in SEEDS:
        d = sample_and_preprocess(splits, n_train=500, seed=s)
        _, _, npa, sec, grad_norm, model = train_predict_vqc(
            {"encoding": "angle", "entanglement": "ring", "n_layers": 2, "alpha": 1.0},
            d, seed=s, return_model=True,
        )
        Xv = torch.tensor(d["X_val"], dtype=torch.float32)
        Xte = torch.tensor(d["X_test"], dtype=torch.float32)
        for shots in [None, 2000, 500, 100]:
            vs = model.predict_shots(Xv, shots=shots)
            ts = model.predict_shots(Xte, shots=shots)
            m = evaluate_predictions(d["y_val"], vs, d["y_test"], ts, npa, sec)
            m.update({"shots": "analytic" if shots is None else str(shots), "seed": s, "init_grad_norm": grad_norm})
            shot_rows.append(m)
    pd.DataFrame(shot_rows).to_csv(RESULTS_DIR / "table_d_shots.csv", index=False)


if __name__ == "__main__":
    main()
```

**Checkpoint:** Have Antigravity verify `python run_all.py` with `FAST_MODE = True` (generating all 5 CSVs in $< 3$ minutes), commit, and push branch `gold-experiments`. Then switch `FAST_MODE = False` locally in your terminal, run `python run_all.py` once to save the full $10$-seed CSVs, set `FAST_MODE = True` back for judges, and commit the full CSVs.

---

## Phase 5: Plots, Elaborate README & Final Merge (Branch: `plots-and-readme` -> Merge to `main`)

### Task 5.1 - Error-Bar Plots (`src/plot_results.py`) & Master Notebook (`Q_Horizon_Benchmark.ipynb`)
* Write `src/plot_results.py` to generate clean 300-DPI PNG figures plotting $\text{mean} \pm \text{std}$ across the $10$ seeds (`main_benchmark.png`, `gold_a_entanglement.png`, `gold_b_encoding.png`, `gold_c_sweep.png`).
* Build `Q_Horizon_Benchmark.ipynb` calling `src/*`, displaying the saved tables/figures, and providing concise plain-English *"Why did this happen?"* takeaways under each experiment.

### Task 5.2 - Upgrade Starter `README.md` into the Elaborate Competition `README.md`
Replace the starter `README.md` with the final competition documentation containing:
1. **Research Question & Dataset Citation:** Cite official UCI HTRU2 (`ucimlrepo` ID `372`, R. J. Lyon, 2017, CC BY 4.0) and explain why the $\approx 9.16\%$ pulsar rate makes naive accuracy ($90.84\%$ by guessing all noise) misleading.
2. **Pre-Registered Hypotheses (`H1`-`H4`) & Verdict Table:** State each hypothesis alongside whether the $10$-seed benchmark confirmed or refuted it within error bars.
3. **Fairness & Simulation Transparency:** Document the locked $3,000$-sample test split, $1,500$-sample validation split, train-only Yeo-Johnson transform, equal $20$-config validation search, validation-only thresholds, and explicitly state that $K = |\Psi \Psi^\dagger|^2$ uses exact state-vector simulation (requiring $O(N^2)$ circuit runs on physical quantum hardware).
4. **Results Summary & Embedded Plots:** Embed `results/*.png` and a clean $\text{mean} \pm \text{std}$ summary table directly in the README.
5. **Reproducibility Instructions:** Explain how judges can verify everything in $< 3$ minutes with `FAST_MODE = True` or reproduce the full $10$-seed sweep with `FAST_MODE = False`.
6. **Credits, AI Usage Disclosure & Team Roles:** Credit PennyLane, PyTorch, and scikit-learn, summarize `AI_USAGE_LOG.md`, and list each team member's contributions.

### Task 5.3 - Final Merge to `main` & Submission Check
* Ensure `FAST_MODE = True` in `src/config.py`, commit and push `plots-and-readme`, merge all branches cleanly into `main`, and push `main`.
* Run `Q_Horizon_Benchmark.ipynb` from scratch in a clean Google Colab environment to confirm zero missing imports.
* Submit the GitHub repo, 4-page PDF report, and presentation slides before **Saturday, November 28, 2026 at 23:59 Cairo time**.