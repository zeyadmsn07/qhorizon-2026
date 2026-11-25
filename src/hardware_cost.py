import argparse
from collections.abc import Mapping
import pandas as pd
import pennylane as qml
import torch
from src.config import get_mode_config, get_results_dir
from src.models_quantum import VariationalQuantumClassifier


def _get(obj, key, default=0):
    val = obj.get(key, default) if isinstance(obj, Mapping) else getattr(obj, key, default)
    return default if val is None else val


def _specs_dict(qnode, *args) -> dict:
    sp = qml.specs(qnode)(*args)
    res = _get(sp, "resources", sp)
    wires = int(_get(res, "num_wires", 0))
    depth = int(_get(res, "depth", 0))
    total_gates = int(_get(res, "num_gates", 0))
    gsizes = _get(res, "gate_sizes", {})
    gtypes = _get(res, "gate_types", {})
    two_q = int(_get(gsizes, 2, _get(gtypes, "CNOT", 0)))

    if total_gates == 0 or wires == 0:
        tape = qml.tape.make_qscript(qnode.func)(*args)
        wires = len(tape.wires)
        total_gates = len(tape.operations)
        two_q = sum(1 for op in tape.operations if len(op.wires) == 2)
        depth = int(tape.graph.get_depth())

    assert 1 <= wires <= 10, f"Invalid qubit count {wires} (ceiling is 10)"
    return {"qubits": wires, "depth": depth, "total_gates": total_gates, "two_qubit_gates": two_q}


def run_hardware_cost(cfg_name: str | None = None) -> pd.DataFrame:
    cfg = get_mode_config(cfg_name)
    res_dir = get_results_dir(cfg.name)
    x_dummy = torch.ones((2, 8), dtype=torch.float32)

    rows = []
    vqc_variants = [
        ("Q1-VQC (8q Bipartite Sign+Linear, 41p)", dict(encoding="angle", entanglement="bipartite", n_layers=2, sign_offset=True, readout="linear_z")),
        ("Q1-VQC (8q Ring L=2, 34p)", dict(encoding="angle", entanglement="ring", n_layers=2)),
        ("Q1-VQC (4q Amplitude+Norm L=4, 34p)", dict(encoding="amplitude_norm", entanglement="ring", n_layers=4)),
        ("Q1-VQC (3q Amplitude L=5, 32p)", dict(encoding="amplitude", entanglement="ring", n_layers=5)),
    ]
    for name, kw in vqc_variants:
        m = VariationalQuantumClassifier(**kw)
        s = _specs_dict(m.circuit, x_dummy, m.weights)
        np_w = sum(p.numel() for p in m.parameters() if p.requires_grad)
        rows.append({**s, "model": name, "trainable_params": np_w, "qpu_circuits_train_N500": "500/epoch (O(N))", "qpu_circuits_test_3000": "3,000 (O(N_test))", "shots_per_eval": "500-2,000 shots", "simulability": "<=8q (dim<=256), CPU-simulable"})

    dev = qml.device("default.qubit", wires=8)
    @qml.qnode(dev, interface="torch", diff_method=None)
    def zz_inv(xa, xb):
        for x_in in (xa, xb):
            for w in range(8):
                qml.Hadamard(wires=w); qml.RZ(x_in[:, w], wires=w)
            for w in range(8):
                wn = (w + 1) % 8
                qml.CNOT(wires=[w, wn]); qml.RZ(x_in[:, w] * x_in[:, wn], wires=wn); qml.CNOT(wires=[w, wn])
        return qml.probs(wires=range(8))

    sk = _specs_dict(zz_inv, x_dummy, x_dummy)
    rows.append({**sk, "model": "Q2-QuantumKernel (8q ZZ d=1, Inversion Test)", "trainable_params": 0, "qpu_circuits_train_N500": "124,750 pairs (N*(N-1)/2 = O(N^2))", "qpu_circuits_test_3000": "1,500,000 pairs (N*N_test)", "shots_per_eval": "~10,000 shots/pair for +-0.01 K_ij", "simulability": "8q (dim=256), state-vector shortcut in sim"})
    rows.append({"model": "RBF-SVM (Classical Reference)", "qubits": 0, "depth": 0, "total_gates": 0, "two_qubit_gates": 0, "trainable_params": 0, "qpu_circuits_train_N500": "0 QPU runs", "qpu_circuits_test_3000": "0 QPU runs", "shots_per_eval": "Exact float64", "simulability": "Classical baseline (no quantum advantage claimed)"})

    df = pd.DataFrame(rows)[["model", "qubits", "depth", "total_gates", "two_qubit_gates", "trainable_params", "qpu_circuits_train_N500", "qpu_circuits_test_3000", "shots_per_eval", "simulability"]]
    df.to_csv(res_dir / "hardware_cost_table.csv", index=False)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "fast", "full"], default=None)
    print(run_hardware_cost(ap.parse_args().mode).to_string(index=False))