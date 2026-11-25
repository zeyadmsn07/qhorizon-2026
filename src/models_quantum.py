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
    def predict_shots(
        self,
        x: torch.Tensor,
        shots: int | None,
        seed: int = 2026,
        chunk_size: int = 512,
    ) -> np.ndarray:
        # Evaluate finite shots at inference only (diff_method=None) to avoid default.mixed batching bugs.
        # Fixed device seed ensures reproducibility across multi-shot evaluations.
        dev = qml.device("default.qubit", wires=self.n_wires, seed=seed)

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

        # Slice inputs into chunks to keep memory usage low on 3000-sample test evaluations
        chunks = []
        for i in range(0, len(x), chunk_size):
            x_chunk = x[i : i + chunk_size]
            expvals = torch.stack(infer_circuit(x_chunk, self.weights), dim=-1).to(dtype=x.dtype)
            logits = self.logit_scale * expvals.mean(dim=-1) + self.logit_bias
            chunks.append(logits.cpu().numpy())

        return np.concatenate(chunks, axis=0) if len(chunks) > 0 else np.array([], dtype=np.float32)


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
