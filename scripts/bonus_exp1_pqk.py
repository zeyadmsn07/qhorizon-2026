import os
import sys
import time
import numpy as np
import pandas as pd
import pennylane as qml
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.getcwd())
from scripts.bonus_common import get_aligned_split

OUT_DIR = os.path.join("results", "full", "bonus")
os.makedirs(OUT_DIR, exist_ok=True)

# Precompute Z_i Z_{i+1} eigenvalues (+/-1) on all 256 basis states for 8-qubit IsingZZ ring
BASIS_BITS = ((np.arange(256)[:, None] >> np.arange(7, -1, -1)) & 1).astype(np.float64)
BASIS_Z = 1.0 - 2.0 * BASIS_BITS  # (256, 8) in {+1, -1}
ZZ_RING_SIGNS = np.stack([BASIS_Z[:, i] * BASIS_Z[:, (i + 1) % 8] for i in range(8)], axis=0)  # (8, 256)

def rot_yz_batch(theta):
    c, s = np.cos(0.5 * theta), np.sin(0.5 * theta)
    ep, em = np.exp(0.5j * theta), np.exp(-0.5j * theta)
    U = np.zeros((len(theta), 2, 2), dtype=np.complex128)
    U[:, 0, 0] = em * c
    U[:, 0, 1] = -em * s
    U[:, 1, 0] = ep * s
    U[:, 1, 1] = ep * c
    return U

def simulate_feature_map(X, alpha):
    x_s = alpha * X
    B = len(x_s)
    psi = np.full((B, 256), 1.0 / 16.0, dtype=np.complex128)  # H^{\otimes 8} |0>^{\otimes 8}
    for i in range(8):
        U = rot_yz_batch(x_s[:, i])
        psi = np.einsum("zia,zlar->zlir", U, psi.reshape(B, 1 << i, 2, 1 << (7 - i))).reshape(B, 256)
    zz_angles = np.stack([x_s[:, i] * x_s[:, (i + 1) % 8] for i in range(8)], axis=1)  # (B, 8)
    phases = np.exp(-0.5j * (zz_angles @ ZZ_RING_SIGNS))
    psi = psi * phases

    feats = []
    for i in range(8):
        pr = psi.reshape(B, 1 << i, 2, 1 << (7 - i))
        p0, p1 = pr[:, :, 0, :], pr[:, :, 1, :]
        ov = np.sum(p0.conj() * p1, axis=(1, 2))
        ex = 2.0 * np.real(ov)
        ey = 2.0 * np.imag(ov)
        ez = np.sum(np.abs(p0) ** 2 - np.abs(p1) ** 2, axis=(1, 2))
        feats.extend([ex, ey, ez])
    return psi, np.column_stack(feats)
# 1-sample verification against PennyLane default.qubit
dev_chk = qml.device("default.qubit", wires=8)
@qml.qnode(dev_chk)
def _qml_chk(x):
    for i in range(8):
        qml.Hadamard(wires=i)
        qml.RY(x[i], wires=i)
        qml.RZ(x[i], wires=i)
    for i in range(8):
        qml.IsingZZ(x[i] * x[(i + 1) % 8], wires=[i, (i + 1) % 8])
    return [obs for i in range(8) for obs in (qml.expval(qml.PauliX(i)), qml.expval(qml.PauliY(i)), qml.expval(qml.PauliZ(i)))]

_x0 = np.linspace(-0.7, 0.9, 8)
_, _f0 = simulate_feature_map(_x0[None, :], 1.0)
assert np.allclose(_f0[0], np.asarray(_qml_chk(_x0)), atol=1e-10), "PennyLane check failed!"

alphas = [0.25, 0.5, 1.0, 2.0, 4.0]
records = []

for seed in range(100, 110):
    t_seed = time.perf_counter()
    X_tr_raw, y_tr, X_va_raw, y_va, X_te_raw, y_te = get_aligned_split(500, seed)
    scaler = StandardScaler()
    X_tr = np.clip(scaler.fit_transform(X_tr_raw), -np.pi, np.pi)
    X_va = np.clip(scaler.transform(X_va_raw), -np.pi, np.pi)
    X_te = np.clip(scaler.transform(X_te_raw), -np.pi, np.pi)
    off_mask = ~np.eye(len(X_tr), dtype=bool)

    for alpha in alphas:
        t0 = time.perf_counter()
        psi_tr, phi_tr = simulate_feature_map(X_tr, alpha)
        psi_va, phi_va = simulate_feature_map(X_va, alpha)
        psi_te, phi_te = simulate_feature_map(X_te, alpha)
        pqk_sec = time.perf_counter() - t0

        base_g = 1.0 / max(phi_tr.shape[1] * np.var(phi_tr), 1e-8)
        best_sc, best_p = -1.0, (1.0, base_g)
        for C in [0.5, 2.0, 10.0]:
            for gm in [0.25, 1.0, 4.0]:
                clf = SVC(C=C, kernel="rbf", gamma=base_g * gm, class_weight="balanced").fit(phi_tr, y_tr)
                sc = average_precision_score(y_va, clf.decision_function(phi_va))
                if sc > best_sc:
                    best_sc, best_p = sc, (C, base_g * gm)

        C_opt, g_opt = best_p
        pqk_scores = SVC(C=C_opt, kernel="rbf", gamma=g_opt, class_weight="balanced").fit(phi_tr, y_tr).decision_function(phi_te)
        K_pqk = np.exp(-g_opt * np.sum((phi_tr[:, None, :] - phi_tr[None, :, :]) ** 2, axis=-1))

        K_fqk_tr = np.abs(psi_tr @ psi_tr.conj().T) ** 2
        K_fqk_va = np.abs(psi_va @ psi_tr.conj().T) ** 2
        K_fqk_te = np.abs(psi_te @ psi_tr.conj().T) ** 2
        best_fc = max([0.5, 2.0, 10.0], key=lambda C: average_precision_score(y_va, SVC(C=C, kernel="precomputed", class_weight="balanced").fit(K_fqk_tr, y_tr).decision_function(K_fqk_va)))
        fqk_scores = SVC(C=best_fc, kernel="precomputed", class_weight="balanced").fit(K_fqk_tr, y_tr).decision_function(K_fqk_te)

        n_tr, n_te = len(X_tr), len(X_te)
        records.append({
            "seed": seed, "alpha": alpha,
            "pqk_pr_auc": average_precision_score(y_te, pqk_scores),
            "fqk_pr_auc": average_precision_score(y_te, fqk_scores),
            "pqk_roc_auc": roc_auc_score(y_te, pqk_scores),
            "fqk_roc_auc": roc_auc_score(y_te, fqk_scores),
            "pqk_offdiag_std": float(np.std(K_pqk[off_mask])),
            "fqk_offdiag_std": float(np.std(K_fqk_tr[off_mask])),
            "pqk_circuits": n_tr + n_te,
            "fqk_pairwise_circuits": n_tr * (n_tr - 1) // 2 + n_te * n_tr,
            "pqk_sec": pqk_sec,
        })
    print(f"[Exp 1] Seed {seed}/109 done ({time.perf_counter() - t_seed:.2f}s)")

df = pd.DataFrame(records)
df.to_csv(os.path.join(OUT_DIR, "exp1_pqk_runs.csv"), index=False)
summary = df.groupby("alpha").agg(
    pqk_pr_mean=("pqk_pr_auc", "mean"), pqk_pr_std=("pqk_pr_auc", "std"),
    fqk_pr_mean=("fqk_pr_auc", "mean"), fqk_pr_std=("fqk_pr_auc", "std"),
    pqk_offdiag_std=("pqk_offdiag_std", "mean"), fqk_offdiag_std=("fqk_offdiag_std", "mean"),
    pqk_circuits=("pqk_circuits", "first"), fqk_circuits=("fqk_pairwise_circuits", "first"),
).reset_index()
summary.to_csv(os.path.join(OUT_DIR, "exp1_pqk_summary.csv"), index=False)
print("\n=== EXPERIMENT 1: PROJECTED QUANTUM KERNEL (PQK) VS FIDELITY KERNEL (10 SEEDS, N=500) ===")
print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
