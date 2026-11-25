import time
import numpy as np
import pandas as pd
import torch

from src.config import get_results_dir
RESULTS_DIR = get_results_dir('smoke')
from src.data import get_locked_splits, sample_and_preprocess
from src.metrics import evaluate_predictions
from src.models_quantum import train_predict_quantum_kernel_svm, train_predict_vqc


def main():
    print("=== [1/4] Loading Locked HTRU2 Data (N=500, Seed=100) ===")
    splits = get_locked_splits()
    data = sample_and_preprocess(splits, n_train=500, seed=100)
    print(f"Data ready: Train={data['X_train'].shape}, Val={data['X_val'].shape}, Test={data['X_test'].shape}")

    results = []

    print("\n=== [2/4] Testing Q1-VQC (8q Angle Ring L=2, Full 75 Epochs) ===")
    vqc_params = {
        "encoding": "angle",
        "entanglement": "ring",
        "n_layers": 2,
        "alpha": 1.0,
        "max_epochs": 75,
    }
    val_scores, test_scores, n_params, fit_sec, grad_norm, model = train_predict_vqc(
        vqc_params, data, seed=100, return_model=True
    )
    assert not np.isnan(val_scores).any(), "VQC val_scores contain NaNs"
    assert not np.isnan(test_scores).any(), "VQC test_scores contain NaNs"
    assert grad_norm > 0, f"Expected positive init_grad_norm, got {grad_norm}"
    print(f"Initial gradient norm: {grad_norm:.6f} (healthy non-zero gradient)")

    # Test analytic parity between forward() and predict_shots(shots=None)
    Xv = torch.tensor(data["X_val"], dtype=torch.float32)
    Xte = torch.tensor(data["X_test"], dtype=torch.float32)
    with torch.no_grad():
        val_fwd = model(Xv).numpy()
    val_analytic = model.predict_shots(Xv, shots=None, chunk_size=512)
    test_analytic = model.predict_shots(Xte, shots=None, chunk_size=512)
    assert np.allclose(val_scores, val_analytic, atol=1e-5), "Analytic predict_shots differs from forward() scores"
    print("Analytic predict_shots verified: exact parity with forward() pass")

    # Evaluate analytic predictions
    m_vqc = evaluate_predictions(data["y_val"], val_scores, data["y_test"], test_scores, n_params, fit_sec)
    m_vqc["model"] = "Q1-VQC (8q Ring L=2)"
    results.append(m_vqc)
    print(f"[{m_vqc['model']:>26}] PR-AUC: {m_vqc['test_pr_auc']:.4f} | F1: {m_vqc['test_f1']:.4f} | MCC: {m_vqc['test_mcc']:.4f} | Acc: {m_vqc['test_acc']:.4f} | Time: {m_vqc['train_time_sec']:.2f}s")

    # Test finite shots inference (shots=200) with deterministic chunked execution
    t0_shots = time.perf_counter()
    val_shots200 = model.predict_shots(Xv, shots=200, seed=2026, chunk_size=512)
    test_shots200 = model.predict_shots(Xte, shots=200, seed=2026, chunk_size=512)
    shots_sec = time.perf_counter() - t0_shots
    assert not np.isnan(val_shots200).any(), "Shots val_scores contain NaNs"
    assert not np.isnan(test_shots200).any(), "Shots test_scores contain NaNs"
    m_shots = evaluate_predictions(data["y_val"], val_shots200, data["y_test"], test_shots200, n_params, shots_sec)
    m_shots["model"] = "Q1-VQC (8q Ring L=2, shots=200)"
    results.append(m_shots)
    print(f"[{m_shots['model']:>26}] PR-AUC: {m_shots['test_pr_auc']:.4f} | F1: {m_shots['test_f1']:.4f} | MCC: {m_shots['test_mcc']:.4f} | Acc: {m_shots['test_acc']:.4f} | Time: {m_shots['train_time_sec']:.2f}s")

    print("\n=== [3/4] Fast Architectural Smoke Tests (15 Epochs) ===")
    arch_configs = [
        ("Q1-VQC (8q None L=2)", {"encoding": "angle", "entanglement": "none", "n_layers": 2, "alpha": 1.0, "max_epochs": 15}),
        ("Q1-VQC (8q Bipartite L=2)", {"encoding": "angle", "entanglement": "bipartite", "n_layers": 2, "alpha": 1.0, "max_epochs": 15}),
        ("Q1-VQC (8q Reupload L=2)", {"encoding": "angle", "entanglement": "ring", "n_layers": 2, "alpha": 1.0, "reupload": True, "max_epochs": 15}),
        ("Q1-VQC (3q Amplitude L=2)", {"encoding": "amplitude", "entanglement": "ring", "n_layers": 2, "alpha": 1.0, "max_epochs": 15}),
        ("Q1-VQC (4q AmplitudeNorm L=2)", {"encoding": "amplitude_norm", "entanglement": "ring", "n_layers": 2, "alpha": 1.0, "max_epochs": 15}),
    ]
    for lbl, p in arch_configs:
        vs, ts, np_count, sec = train_predict_vqc(p, data, seed=100)
        assert not np.isnan(vs).any(), f"{lbl} val_scores contain NaNs"
        assert not np.isnan(ts).any(), f"{lbl} test_scores contain NaNs"
        m = evaluate_predictions(data["y_val"], vs, data["y_test"], ts, np_count, sec)
        m["model"] = lbl
        results.append(m)
        print(f"[{lbl:>26}] PR-AUC: {m['test_pr_auc']:.4f} | F1: {m['test_f1']:.4f} | MCC: {m['test_mcc']:.4f} | Acc: {m['test_acc']:.4f} | Time: {m['train_time_sec']:.2f}s")

    print("\n=== [4/4] Testing Q2-QuantumKernel (8q ZZ Depth=1) ===")
    kernel_params = {"alpha": 1.0, "depth": 1, "C": 1.0, "class_weight": "balanced"}
    vs_k, ts_k, np_k, sec_k = train_predict_quantum_kernel_svm(kernel_params, data, seed=100)
    assert not np.isnan(vs_k).any(), "QuantumKernel val_scores contain NaNs"
    assert not np.isnan(ts_k).any(), "QuantumKernel test_scores contain NaNs"
    m_k = evaluate_predictions(data["y_val"], vs_k, data["y_test"], ts_k, np_k, sec_k)
    m_k["model"] = "Q2-QuantumKernel (8q)"
    results.append(m_k)
    print(f"[{m_k['model']:>26}] PR-AUC: {m_k['test_pr_auc']:.4f} | F1: {m_k['test_f1']:.4f} | MCC: {m_k['test_mcc']:.4f} | Acc: {m_k['test_acc']:.4f} | Time: {m_k['train_time_sec']:.2f}s")

    # Format and save verification results
    df = pd.DataFrame(results)
    cols = ["model", "test_pr_auc", "test_f1", "test_mcc", "test_acc", "threshold", "n_trainable_params", "n_tuned_hyperparams", "model_size_values", "n_support_vectors", "train_time_sec"]
    df = df[cols]
    output_path = RESULTS_DIR / "quantum_models_verification.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved quantum verification metrics to: {output_path}")
    print("\nSummary Table:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
