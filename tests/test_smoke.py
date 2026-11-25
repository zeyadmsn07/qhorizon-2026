import inspect
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.dummy import DummyClassifier

from src.data import get_locked_splits, sample_and_preprocess
import src.models_quantum as mq


class TestBaselineSmoke(unittest.TestCase):
    def test_splits_and_no_nans(self):
        splits = get_locked_splits()
        self.assertEqual(len(splits["X_test"]), 3000)
        self.assertEqual(len(splits["X_val"]), 1500)
        sample = sample_and_preprocess(splits, n_train=500, seed=100)
        for k in ("X_train", "y_train", "X_val", "y_val", "X_test", "y_test"):
            self.assertFalse(np.isnan(sample[k]).any(), f"NaN found in {k}")

    def test_dummy_accuracy_trap(self):
        df = pd.read_csv(Path("results/v1_original_10seed/main_benchmark.csv"))
        dummy_acc = df.loc[df["model"] == "Dummy", "test_acc"].mean()
        self.assertGreater(dummy_acc, 0.90)
        splits = get_locked_splits()
        s = sample_and_preprocess(splits, n_train=500, seed=100)
        clf = DummyClassifier(strategy="most_frequent").fit(s["X_train"], s["y_train"])
        self.assertGreater(clf.score(s["X_test"], s["y_test"]), 0.90)

    def test_vqc_predict_shots_none_matches_forward(self):
        vqc_cls = next(
            obj for _, obj in inspect.getmembers(mq, inspect.isclass)
            if issubclass(obj, torch.nn.Module) and hasattr(obj, "predict_shots")
        )
        torch.manual_seed(42)
        model = vqc_cls()
        model.eval()
        dtype = next(model.parameters()).dtype
        x = torch.randn(6, 8, dtype=dtype)
        with torch.no_grad():
            fwd = model(x)
            shots_none = model.predict_shots(x, shots=None)
        matches = torch.allclose(shots_none, fwd, atol=1e-5) or torch.allclose(
            shots_none, torch.sigmoid(fwd), atol=1e-5
        )
        self.assertTrue(matches, "predict_shots(shots=None) did not match forward()")


if __name__ == "__main__":
    unittest.main()
