import sys
from pathlib import Path
import pandas as pd
import numpy as np

from src.config import get_results_dir
RESULTS_DIR = get_results_dir('smoke')
from src.data import get_locked_splits, sample_and_preprocess
from src.metrics import evaluate_predictions
from src.models_classical import fit_predict_classical


def main():
    print("=== [1/3] Testing HTRU2 Data Pipeline & Leakage-Free Preprocessing ===")
    splits = get_locked_splits()
    print(f"Locked splits obtained: Pool={splits['X_pool'].shape[0]}, Val={splits['X_val'].shape[0]}, Test={splits['X_test'].shape[0]}")
    assert splits['X_test'].shape[0] == 3000, "Expected 3000 test samples"
    assert splits['X_val'].shape[0] == 1500, "Expected 1500 val samples"

    data = sample_and_preprocess(splits, n_train=500, seed=100)
    assert not np.isnan(data['X_train']).any(), "X_train contains NaNs"
    assert not np.isnan(data['X_val']).any(), "X_val contains NaNs"
    assert not np.isnan(data['X_test']).any(), "X_test contains NaNs"
    assert data['X_train'].dtype == np.float32, "X_train must be float32"

    train_pos_ratio = (data['y_train'] == 1).mean()
    test_pos_ratio = (data['y_test'] == 1).mean()
    print(f"Train positive ratio: {train_pos_ratio:.4f} (expected ~0.0916)")
    print(f"Test positive ratio:  {test_pos_ratio:.4f} (expected ~0.0916)")
    assert 0.08 <= train_pos_ratio <= 0.11, "Pulsar positive ratio out of expected range"

    print("\n=== [2/3] Running Classical Baseline Smoke Tests (N=500, Seed=100) ===")
    test_configs = [
        ("Dummy", {}),
        ("LogReg", {"C": 1.0, "class_weight": "balanced"}),
        ("RBF-SVM", {"C": 1.0, "gamma": "scale", "class_weight": "balanced"}),
        ("RandomForest", {"n_estimators": 50, "max_depth": 5, "min_samples_leaf": 3, "class_weight": "balanced"}),
        ("Tiny-MLP", {"hidden_dim": 4, "lr": 0.01, "weight_decay": 1e-4, "pos_weight_mult": 1.0, "batch_size": 64}),
    ]

    results = []
    for model_name, params in test_configs:
        val_scores, test_scores, n_params, train_time = fit_predict_classical(
            model_name=model_name,
            params=params,
            data=data,
            seed=100,
        )
        metrics = evaluate_predictions(
            y_val=data["y_val"],
            val_scores=val_scores,
            y_test=data["y_test"],
            test_scores=test_scores,
            n_params=n_params,
            train_time_sec=train_time,
        )
        metrics["model"] = model_name
        results.append(metrics)
        print(f"[{model_name:>12}] PR-AUC: {metrics['test_pr_auc']:.4f} | F1: {metrics['test_f1']:.4f} | MCC: {metrics['test_mcc']:.4f} | Acc: {metrics['test_acc']:.4f} | Params: {metrics['n_params']:>4} | Time: {metrics['train_time_sec']:.3f}s")

    # Verify Dummy baseline accurately captures the majority-class noise baseline
    dummy_res = next(r for r in results if r["model"] == "Dummy")
    assert dummy_res["test_acc"] > 0.90, f"Expected Dummy accuracy > 0.90, got {dummy_res['test_acc']:.4f}"
    assert dummy_res["test_f1"] == 0.0, f"Expected Dummy F1 == 0.0, got {dummy_res['test_f1']:.4f}"
    assert dummy_res["test_mcc"] == 0.0, f"Expected Dummy MCC == 0.0, got {dummy_res['test_mcc']:.4f}"

    print("\n=== [3/3] Saving Verification Metrics CSV ===")
    df = pd.DataFrame(results)
    cols = ["model", "test_pr_auc", "test_f1", "test_mcc", "test_acc", "threshold", "n_params", "train_time_sec"]
    df = df[cols]
    output_path = RESULTS_DIR / "classical_baselines_verification.csv"
    df.to_csv(output_path, index=False)
    print(f"Metrics saved cleanly to: {output_path}")
    print("\nSummary Table:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
