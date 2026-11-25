import unittest
from pathlib import Path

class TestReadmeHonesty(unittest.TestCase):
    def test_no_overclaims_and_limitations_present(self):
        txt = Path("README.md").read_text(encoding="utf-8")
        self.assertNotIn("Pre-Registered", txt)
        self.assertNotIn("| **Confirmed** |", txt)
        self.assertIn("## 2. Hypotheses and What the Data Showed", txt)
        self.assertIn("## 4. Limitations", txt)
        self.assertIn("no quantum advantage is claimed", txt)