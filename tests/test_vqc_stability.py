import json, tempfile, unittest
from pathlib import Path
import numpy as np
import torch
from src.data import get_locked_splits, sample_and_preprocess
from src.models_quantum import VariationalQuantumClassifier, train_predict_vqc

class TestVqcStabilityExtensions(unittest.TestCase):
    def test_sign_symmetry_and_linear_readout_parity(self):
        m_even = VariationalQuantumClassifier(sign_offset=False, readout="mean_z").eval()
        m_odd = VariationalQuantumClassifier(sign_offset=True, readout="linear_z").eval()
        nn_zero = lambda m: m.weights.data.zero_()
        nn_zero(m_even); nn_zero(m_odd)
        x = torch.ones(2, 8, dtype=torch.float32); x[1] = -1.0
        with torch.no_grad():
            out_even = m_even(x).numpy()
            out_odd = m_odd(x).numpy()
        self.assertAlmostEqual(float(out_even[0]), float(out_even[1]), places=5)
        self.assertGreater(abs(float(out_odd[0] - out_odd[1])), 1.0)
        self.assertEqual(sum(p.numel() for p in m_odd.parameters() if p.requires_grad), 41)
        self.assertTrue(np.allclose(out_odd, m_odd.predict_shots(x, shots=None), atol=1e-5))

    def test_diagnostics_json_written(self):
        d = sample_and_preprocess(get_locked_splits(), n_train=100, seed=100)
        with tempfile.TemporaryDirectory() as td:
            dp = Path(td) / "diag.json"
            train_predict_vqc({"max_epochs": 4, "n_restarts": 2, "sign_offset": True}, d, seed=100, diag_path=dp)
            payload = json.loads(dp.read_text(encoding="utf-8"))
            self.assertIn("final_grad_norm", payload)
            self.assertIn("val_losses", payload)