import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score


def stratified_bootstrap_pr_auc(
    y_true: np.ndarray, scores: np.ndarray, n_boot: int = 1000, seed: int = 2026
) -> tuple[float, float, float]:
    # Resample positive and negative test indices separately to preserve pulsar prevalence
    rng = np.random.default_rng(seed)
    pos_idx = np.flatnonzero(y_true == 1)
    neg_idx = np.flatnonzero(y_true == 0)
    point = float(average_precision_score(y_true, scores))
    boots = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = np.concatenate([
            rng.choice(pos_idx, size=len(pos_idx), replace=True),
            rng.choice(neg_idx, size=len(neg_idx), replace=True),
        ])
        boots[b] = average_precision_score(y_true[idx], scores[idx])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def two_level_bootstrap_pr_auc(
    y_test: np.ndarray, per_seed_scores: list[np.ndarray], n_boot: int = 1000, seed: int = 2026
) -> tuple[float, float, float]:
    # Jointly resample evaluation seeds and stratified test indices to capture both variance sources
    rng = np.random.default_rng(seed)
    mat = np.asarray(per_seed_scores, dtype=np.float64)
    n_s = len(mat)
    point = float(np.mean([average_precision_score(y_test, s) for s in mat]))
    pos_idx = np.flatnonzero(y_test == 1)
    neg_idx = np.flatnonzero(y_test == 0)
    boots = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        s_ids = rng.integers(0, n_s, size=n_s)
        t_idx = np.concatenate([
            rng.choice(pos_idx, size=len(pos_idx), replace=True),
            rng.choice(neg_idx, size=len(neg_idx), replace=True),
        ])
        yb = y_test[t_idx]
        boots[b] = np.mean([average_precision_score(yb, mat[i, t_idx]) for i in s_ids])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def paired_delta_pr_auc(
    y_test: np.ndarray, scores_a: list[np.ndarray], scores_b: list[np.ndarray], n_boot: int = 1000, seed: int = 2026
) -> dict:
    rng = np.random.default_rng(seed)
    ma, mb = np.asarray(scores_a, dtype=np.float64), np.asarray(scores_b, dtype=np.float64)
    n_s = len(ma)
    deltas = np.array([average_precision_score(y_test, ma[i]) - average_precision_score(y_test, mb[i]) for i in range(n_s)])
    pos_idx, neg_idx = np.flatnonzero(y_test == 1), np.flatnonzero(y_test == 0)
    boots = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        s_ids = rng.integers(0, n_s, size=n_s)
        t_idx = np.concatenate([rng.choice(pos_idx, len(pos_idx), True), rng.choice(neg_idx, len(neg_idx), True)])
        yb = y_test[t_idx]
        boots[b] = np.mean([average_precision_score(yb, ma[i, t_idx]) - average_precision_score(yb, mb[i, t_idx]) for i in s_ids])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    try:
        pval = float(wilcoxon(deltas).pvalue) if n_s >= 5 else float("nan")
    except ValueError:
        pval = 1.0
    return {"mean_delta": float(np.mean(deltas)), "ci_lo": float(lo), "ci_hi": float(hi), "wilcoxon_p": pval, "min_attainable_p": float(2.0 / (2 ** n_s)), "n_seeds": int(n_s)}


def classify_verdict(mean_delta: float, ci_lo: float, ci_hi: float, direction: str = "greater", margin: float = 0.01) -> str:
    if direction == "tied":
        if ci_lo >= -margin and ci_hi <= margin:
            return "Practically tied"
        if ci_lo > margin or ci_hi < -margin:
            return "Contradicted"
        return "Inconclusive"
    if ci_lo > 0.0 and mean_delta > margin:
        return "Supported"
    if ci_lo >= -margin and ci_hi <= margin:
        return "Practically tied"
    if ci_hi < 0.0 and mean_delta < -margin:
        return "Contradicted"
    return "Inconclusive"