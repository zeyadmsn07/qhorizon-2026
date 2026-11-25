import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, matthews_corrcoef


def select_threshold_on_val(y_val: np.ndarray, val_scores: np.ndarray) -> float:
    if np.allclose(val_scores, val_scores[0]):
        return float(np.max(val_scores)) + 1.0
    candidates = np.unique(np.quantile(val_scores, np.linspace(0.05, 0.99, 101)))
    best_tau, best_f1, best_mcc = float(candidates[0]), -1.0, -1.0
    for tau in candidates:
        preds = (val_scores >= tau).astype(int)
        f1 = f1_score(y_val, preds, pos_label=1, zero_division=0)
        mcc = matthews_corrcoef(y_val, preds)
        if (f1 > best_f1) or (np.isclose(f1, best_f1) and mcc > best_mcc):
            best_f1, best_mcc, best_tau = f1, mcc, float(tau)
    return best_tau


def evaluate_predictions(
    y_val: np.ndarray,
    val_scores: np.ndarray,
    y_test: np.ndarray,
    test_scores: np.ndarray,
    complexity: dict,
    train_time_sec: float,
) -> dict:
    # Separate gradient weights, tuned hyperparams, inference size, and SVs for like-for-like fairness
    tau = select_threshold_on_val(y_val, val_scores)
    test_preds = (test_scores >= tau).astype(int)
    return {
        "val_pr_auc": float(average_precision_score(y_val, val_scores)),
        "test_pr_auc": float(average_precision_score(y_test, test_scores)),
        "test_mcc": float(matthews_corrcoef(y_test, test_preds)),
        "test_f1": float(f1_score(y_test, test_preds, pos_label=1, zero_division=0)),
        "test_acc": float(accuracy_score(y_test, test_preds)),
        "threshold": tau,
        "n_trainable_params": int(complexity["n_trainable_params"]),
        "n_tuned_hyperparams": int(complexity["n_tuned_hyperparams"]),
        "model_size_values": int(complexity["model_size_values"]),
        "n_support_vectors": float(complexity["n_support_vectors"]),
        "train_time_sec": float(train_time_sec),
    }
