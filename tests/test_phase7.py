import unittest
from pathlib import Path
import pandas as pd
from src.hardware_cost import run_hardware_cost

class TestPhase7Insights(unittest.TestCase):
    def test_hardware_table_and_selection_rule(self):
        hw = run_hardware_cost("smoke")
        self.assertLessEqual(int(hw["qubits"].max()), 10)
        self.assertGreater(int(hw["total_gates"].max()), 10)
        nl_p = Path("results/fast/nonlinear_dataset_summary.csv")
        if nl_p.exists():
            nl = pd.read_csv(nl_p)
            self.assertGreaterEqual(float(nl["precheck_rbf_minus_logreg"].iloc[0]), 0.05)