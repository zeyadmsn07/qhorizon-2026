import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from src.data import get_locked_splits, sample_and_preprocess
from src.models_quantum import VariationalQuantumClassifier

class TestBaselineSmoke(unittest.TestCase):
    def test_splits_and_no_nans(self):
        splits = get_locked_splits()
        self.assertEqual(len(splits["X_test"]), 3000)
        self.assertEqual(len(splits["X_val"]), 1500)
        s = sample_and_preprocess(splits, n_train=500, seed=100)
        for k in ("X_train", "y_train", "X_val", "y_val", "X_test", "y_test"):
            self.assertFalse(np.isnan(s[k]).any())

    def test_dummy_accuracy_trap(self):
        df = pd.read_csv(Path("results/v1_original_10seed/main_benchmark.csv"))
        self.assertGreater(df.loc[df["model"] == "Dummy", "test_acc"].mean(), 0.90)

    def test_vqc_predict_shots_none_matches_forward(self):
        torch.manual_seed(42)
        m = VariationalQuantumClassifier().eval()
        x = torch.randn(6, 8, dtype=torch.float32)
        with torch.no_grad():
            fwd = m(x).cpu().numpy()
        shots_none = m.predict_shots(x, shots=None)
        self.assertTrue(np.allclose(shots_none, fwd, atol=1e-5))

    def test_offline_cache_and_determinism(self):
        self.assertTrue(Path("data_cache/htru2.csv").exists())
        from src.models_classical import fit_predict_classical
        from src.models_quantum import train_predict_vqc
        splits = get_locked_splits()
        d = sample_and_preprocess(splits, n_train=200, seed=100)
        mlp_cfg = {"hidden_dim": 4, "lr": 0.01, "weight_decay": 1e-4, "pos_weight_mult": 1.0, "batch_size": 64}
        v1, t1, _, _ = fit_predict_classical("Tiny-MLP", mlp_cfg, d, seed=100)
        v2, t2, _, _ = fit_predict_classical("Tiny-MLP", mlp_cfg, d, seed=100)
        self.assertTrue(np.allclose(v1, v2, atol=1e-7) and np.allclose(t1, t2, atol=1e-7))
        q_cfg = {"encoding": "angle", "entanglement": "ring", "n_layers": 1, "alpha": 1.0, "max_epochs": 3}
        qv1, qt1, _, _ = train_predict_vqc(q_cfg, d, seed=100)
        qv2, qt2, _, _ = train_predict_vqc(q_cfg, d, seed=100)
        self.assertTrue(np.allclose(qv1, qv2, atol=1e-7) and np.allclose(qt1, qt2, atol=1e-7))
