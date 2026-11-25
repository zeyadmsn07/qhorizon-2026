import ast, json, sys, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class TestFinalSubmissionAudit(unittest.TestCase):
    def test_qubit_ceiling_across_all_modules(self):
        for py in Path("src").glob("*.py"):
            tree = ast.parse(py.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "device":
                    for kw in node.keywords:
                        if kw.arg == "wires" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, int):
                            self.assertLessEqual(kw.value.value, 10, f"{py} exceeds 10-qubit ceiling")

    def test_no_fast_mode_leftovers_or_root_result_leaks(self):
        for p in list(Path("src").glob("*.py")) + [Path("run_all.py"), Path("README.md"), Path("Q_Horizon_Benchmark.ipynb")]:
            self.assertNotIn("FAST_MODE", p.read_text(encoding="utf-8"), f"FAST_MODE leftover in {p}")
        loose_csvs = list(Path("results").glob("*.csv"))
        self.assertEqual(len(loose_csvs), 0, f"Found un-isolated CSVs in results/: {loose_csvs}")
        self.assertTrue(Path("results/v1_original_10seed/main_benchmark.csv").exists())

    def test_readme_matches_verdicts_and_credits(self):
        res_dir = Path("results/full") if Path("results/full/verdicts.json").exists() else Path("results/fast")
        v = json.loads((res_dir / "verdicts.json").read_text(encoding="utf-8"))
        readme = Path("README.md").read_text(encoding="utf-8")
        for hid, h in v["hypotheses"].items():
            self.assertIn(f"{h['mean_delta']:+.4f}", readme, f"README missing exact delta for {hid}")
        for req in ["Lyon", "372", "PennyLane", "PyTorch", "scikit-learn", "AI_USAGE_LOG.md", "no quantum advantage is claimed"]:
            self.assertIn(req, readme, f"README missing required credit/disclosure: {req}")
        team_txt = Path("docs/TEAM.md").read_text(encoding="utf-8")
        self.assertIn("Zeyad Mohamed Nada", team_txt)
        self.assertNotIn("Teammate 2 Name", team_txt)


def main() -> int:
    loader = unittest.defaultTestLoader
    suite = unittest.TestSuite([
        loader.discover(str(ROOT / "tests"), top_level_dir=str(ROOT)),
        loader.loadTestsFromTestCase(TestFinalSubmissionAudit),
    ])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        return 1
    print("\nALL SUBMISSION CHECKS PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())