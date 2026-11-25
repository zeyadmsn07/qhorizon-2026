import json
import time
from pathlib import Path
import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
from sklearn.svm import SVC


def _apply_entanglement(n_wires: int, topology: str, layer_idx: int) -> None:
    if topology == "none" or n_wires < 2:
        return
    if topology == "ring":
        for i in range(n_wires):
            qml.CNOT(wires=[i, (i + 1) % n_wires])
    elif topology == "bipartite" and n_wires == 8:
        # Couples Pulse Profile wires (0..3) with DM-SNR wires (4..7)
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
        encoding: str = "angle",
        entanglement: str = "ring",
        n_layers: int = 2,
        alpha: float = 1.0,
        reupload: bool = False,
        init_mode: str = "small_normal",
        sign_offset: bool = False,
        readout: str = "mean_z",
    ):
        super().__init__()
        self.encoding = encoding
        self.entanglement = entanglement
        self.n_layers = int(n_layers)
        self.alpha = float(alpha)
        self.reupload = bool(reupload)
        self.init_mode = init_mode
        self.sign_offset = bool(sign_offset)
        self.readout = readout

        self.n_wires = {"angle": 8, "amplitude": 3, "amplitude_norm": 4}[encoding]
        assert self.n_wires <= 10, "Qubit ceiling exceeded"

        if init_mode == "small_normal":
            w_init = 0.05 * torch.randn(self.n_layers, self.n_wires, 2)
        elif init_mode == "wide_normal":
            w_init = 0.50 * torch.randn(self.n_layers, self.n_wires, 2)
        elif init_mode == "uniform_pi":
            w_init = torch.empty(self.n_layers, self.n_wires, 2).uniform_(-np.pi, np.pi)
        else:
            raise ValueError(f"Unknown init_mode: {init_mode}")
        self.weights = nn.Parameter(w_init)

        if readout in {"mean_z", "single_z0"}:
            self.logit_scale = nn.Parameter(torch.tensor(4.0))
            self.logit_bias = nn.Parameter(torch.tensor(0.0))
            self.readout_linear = None
        elif readout == "linear_z":
            # Per-wire linear readout (+9 params on 8q -> 41 total params at L=2, matching Tiny-MLP)
            self.readout_linear = nn.Linear(self.n_wires, 1)
            nn.init.constant_(self.readout_linear.weight, 4.0 / self.n_wires)
            nn.init.zeros_(self.readout_linear.bias)
        else:
            raise ValueError(f"Unknown readout: {readout}")

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
            return [qml.expval(qml.PauliZ(w)) for w in range(self.n_wires)]

        self.circuit = circuit

    def _encode_inputs(self, x_batch: torch.Tensor) -> None:
        if self.encoding == "angle":
            angles = 2.0 * torch.atan(self.alpha * x_batch)
            if self.sign_offset:
                # Shifting RY by pi/2 turns <Z> from cos(theta) (even) into -sin(theta) (odd in feature sign)
                angles = angles + (0.5 * torch.pi)
            qml.AngleEmbedding(angles, wires=range(8), rotation="Y")
        elif self.encoding == "amplitude":
            x64 = x_batch.to(torch.float64)
            norms = torch.linalg.norm(x64, dim=-1, keepdim=True).clamp_min(1e-12)
            qml.AmplitudeEmbedding(x64 / norms, wires=range(3), normalize=False)
        elif self.encoding == "amplitude_norm":
            x64 = x_batch.to(torch.float64)
            norms = torch.linalg.norm(x64, dim=-1, keepdim=True).clamp_min(1e-12)
            qml.AmplitudeEmbedding(x64 / norms, wires=range(3), normalize=False)
            qml.RY(2.0 * torch.atan(self.alpha * norms.squeeze(-1)), wires=3)

    def _apply_readout(self, expvals: torch.Tensor) -> torch.Tensor:
        if self.readout == "mean_z":
            return self.logit_scale * expvals.mean(dim=-1) + self.logit_bias
        if self.readout == "single_z0":
            return self.logit_scale * expvals[:, 0] + self.logit_bias
        return self.readout_linear(expvals).squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expvals = torch.stack(self.circuit(x, self.weights), dim=-1).to(dtype=x.dtype)
        return self._apply_readout(expvals)

    @torch.no_grad()
    def predict_shots(self, x: torch.Tensor, shots: int | None, seed: int = 2026, chunk_size: int = 512) -> np.ndarray:
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
        chunks = []
        for i in range(0, len(x), chunk_size):
            xc = x[i : i + chunk_size]
            ev = torch.stack(infer_circuit(xc, self.weights), dim=-1).to(dtype=x.dtype)
            chunks.append(self._apply_readout(ev).cpu().numpy())
        return np.concatenate(chunks, axis=0) if chunks else np.array([], dtype=np.float32)

def _grad_norm(model: nn.Module) -> float:
    sq = [p.grad.pow(2).sum() for p in model.parameters() if p.grad is not None]
    return float(torch.sqrt(sum(sq))) if sq else 0.0


def train_predict_vqc(
    params: dict,
    data: dict,
    seed: int,
    return_model: bool = False,
    diag_path: Path | None = None,
):
    t0 = time.perf_counter()
    Xt = torch.tensor(data["X_train"], dtype=torch.float32)
    yt = torch.tensor(data["y_train"], dtype=torch.float32)
    Xv = torch.tensor(data["X_val"], dtype=torch.float32)
    yv = torch.tensor(data["y_val"], dtype=torch.float32)
    Xte = torch.tensor(data["X_test"], dtype=torch.float32)

    pos_ratio = (data["y_train"] == 0).sum() / max((data["y_train"] == 1).sum(), 1)
    pos_weight = torch.tensor([pos_ratio * params.get("pos_weight_mult", 1.0)], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    n_restarts = int(params.get("n_restarts", 1))
    max_epochs = int(params.get("max_epochs", 55))
    batch_size = int(params.get("batch_size", 128))
    val_every = int(params.get("val_every", 2))
    patience = int(params.get("patience", 12))

    winner = None
    for r_idx in range(n_restarts):
        r_seed = seed + 10000 * r_idx
        torch.manual_seed(r_seed)
        np.random.seed(r_seed)
        model = VariationalQuantumClassifier(
            encoding=params.get("encoding", "angle"),
            entanglement=params.get("entanglement", "ring"),
            n_layers=params.get("n_layers", 2),
            alpha=params.get("alpha", 1.0),
            reupload=params.get("reupload", False),
            init_mode=params.get("init_mode", "small_normal"),
            sign_offset=params.get("sign_offset", False),
            readout=params.get("readout", "mean_z"),
        )
        opt = torch.optim.AdamW(model.parameters(), lr=params.get("lr", 0.02), weight_decay=params.get("weight_decay", 1e-4))
        opt.zero_grad()
        criterion(model(Xt), yt).backward()
        init_gn = _grad_norm(model)
        opt.zero_grad()

        best_vl, best_ep, best_state = float("inf"), 0, None
        p_left = max(2, patience // val_every)
        ep_log, tr_log, va_log = [], [], []
        early_stopped, stopped_ep = False, max_epochs - 1

        for ep in range(max_epochs):
            model.train()
            perm = torch.randperm(len(Xt))
            ep_losses = []
            for i in range(0, len(Xt), batch_size):
                idx = perm[i : i + batch_size]
                opt.zero_grad()
                loss = criterion(model(Xt[idx]), yt[idx])
                loss.backward()
                opt.step()
                ep_losses.append(float(loss.item()))
            if (ep % val_every == 0) or (ep == max_epochs - 1):
                model.eval()
                with torch.no_grad():
                    vl = float(criterion(model(Xv), yv).item())
                ep_log.append(ep); tr_log.append(float(np.mean(ep_losses))); va_log.append(vl)
                if vl < best_vl - 1e-5:
                    best_vl, best_ep = vl, ep
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    p_left = max(2, patience // val_every)
                else:
                    p_left -= 1
                    if p_left == 0:
                        early_stopped, stopped_ep = True, ep
                        break

        if best_state is not None:
            model.load_state_dict(best_state)
        opt.zero_grad()
        criterion(model(Xt), yt).backward()
        final_gn = _grad_norm(model)
        opt.zero_grad()

        # Select among random restarts strictly by validation BCE loss (never test data)
        if (winner is None) or (best_vl < winner["best_val_loss"]):
            winner = {
                "model": model, "best_val_loss": best_vl, "best_epoch": best_ep,
                "early_stopped": early_stopped, "stopped_epoch": stopped_ep,
                "init_grad_norm": init_gn, "final_grad_norm": final_gn,
                "chosen_restart": r_idx, "epochs_logged": ep_log,
                "train_losses": tr_log, "val_losses": va_log,
            }

    best_model = winner["model"]
    best_model.eval()
    with torch.no_grad():
        val_scores = best_model(Xv).numpy()
        test_scores = best_model(Xte).numpy()

    n_params = sum(p.numel() for p in best_model.parameters() if p.requires_grad)
    if best_model.readout_linear is not None:
        l_scale = float(best_model.readout_linear.weight.norm().item())
        l_bias = float(best_model.readout_linear.bias.item())
    else:
        l_scale = float(best_model.logit_scale.item())
        l_bias = float(best_model.logit_bias.item())

    if diag_path is not None:
        diag_path.parent.mkdir(parents=True, exist_ok=True)
        diag = {k: v for k, v in winner.items() if k != "model"}
        diag.update({
            "seed": seed, "n_restarts": n_restarts, "alpha": float(params.get("alpha", 1.0)),
            "lr": float(params.get("lr", 0.02)), "weight_decay": float(params.get("weight_decay", 1e-4)),
            "init_mode": params.get("init_mode", "small_normal"), "sign_offset": bool(params.get("sign_offset", False)),
            "readout": params.get("readout", "mean_z"), "logit_scale": l_scale, "logit_bias": l_bias,
        })
        diag_path.write_text(json.dumps(diag, indent=2), encoding="utf-8")

    complexity = {"n_trainable_params": int(n_params), "n_tuned_hyperparams": 4, "model_size_values": int(n_params), "n_support_vectors": float("nan")}
    fit_sec = time.perf_counter() - t0
    if return_model:
        return val_scores, test_scores, complexity, fit_sec, winner["init_grad_norm"], best_model
    return val_scores, test_scores, complexity, fit_sec


def train_predict_quantum_kernel_svm(params: dict, data: dict, seed: int) -> tuple[np.ndarray, np.ndarray, dict, float]:
    t0 = time.perf_counter()
    n_wires, alpha, depth = 8, float(params.get("alpha", 1.0)), int(params.get("depth", 1))
    dev = qml.device("default.qubit", wires=n_wires)

    @qml.qnode(dev, interface="torch", diff_method=None)
    def feature_state(x_batch):
        angles = 2.0 * torch.atan(alpha * x_batch)
        for _ in range(depth):
            for w in range(n_wires):
                qml.Hadamard(wires=w); qml.RZ(angles[:, w], wires=w)
            for w in range(n_wires):
                wn = (w + 1) % n_wires
                qml.CNOT(wires=[w, wn]); qml.RZ(angles[:, w] * angles[:, wn], wires=wn); qml.CNOT(wires=[w, wn])
        return qml.state()

    Xt, Xv, Xte = torch.tensor(data["X_train"]), torch.tensor(data["X_val"]), torch.tensor(data["X_test"])
    with torch.no_grad():
        ptr, pva, pte = feature_state(Xt), feature_state(Xv), feature_state(Xte)
        K_tr = torch.abs(ptr @ ptr.conj().T).pow(2).cpu().numpy()
        K_va = torch.abs(pva @ ptr.conj().T).pow(2).cpu().numpy()
        K_te = torch.abs(pte @ ptr.conj().T).pow(2).cpu().numpy()
    clf = SVC(kernel="precomputed", C=params.get("C", 1.0), class_weight=params.get("class_weight", "balanced"), random_state=seed)
    clf.fit(K_tr, data["y_train"])
    n_sv = int(len(clf.support_))
    comp = {"n_trainable_params": 0, "n_tuned_hyperparams": 4, "model_size_values": int(n_sv * data["X_train"].shape[1]), "n_support_vectors": n_sv}
    return clf.decision_function(K_va), clf.decision_function(K_te), comp, time.perf_counter() - t0
