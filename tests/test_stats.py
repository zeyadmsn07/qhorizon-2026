import unittest
import numpy as np
from src.stats import classify_verdict, paired_delta_pr_auc, stratified_bootstrap_pr_auc, two_level_bootstrap_pr_auc

class TestBootstrapStats(unittest.TestCase):
    def test_synthetic_bootstrap_and_verdicts(self):
        rng = np.random.default_rng(7)
        y = np.array([0] * 900 + [1] * 100)
        good = [y * 2.5 + rng.normal(0, 0.5, size=len(y)) for _ in range(6)]
        near = [g + rng.normal(0, 0.005, size=len(y)) for g in good]
        weak = [y * 0.8 + rng.normal(0, 0.9, size=len(y)) for _ in range(6)]
        pt, lo, hi = stratified_bootstrap_pr_auc(y, good[0], n_boot=300)
        self.assertTrue(lo <= pt <= hi and lo > 0.80)
        pt2, lo2, hi2 = two_level_bootstrap_pr_auc(y, good, n_boot=300)
        self.assertTrue(lo2 <= pt2 <= hi2)
        d_win = paired_delta_pr_auc(y, good, weak, n_boot=300)
        self.assertEqual(classify_verdict(d_win["mean_delta"], d_win["ci_lo"], d_win["ci_hi"], "greater"), "Supported")
        d_tie = paired_delta_pr_auc(y, good, near, n_boot=300)
        self.assertEqual(classify_verdict(d_tie["mean_delta"], d_tie["ci_lo"], d_tie["ci_hi"], "greater"), "Practically tied")
        d_rev = paired_delta_pr_auc(y, weak, good, n_boot=300)
        self.assertEqual(classify_verdict(d_rev["mean_delta"], d_rev["ci_lo"], d_rev["ci_hi"], "greater"), "Contradicted")