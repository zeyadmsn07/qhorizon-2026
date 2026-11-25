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
