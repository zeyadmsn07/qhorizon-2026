"""Create Q_Horizon_Dashboard.ipynb: the dashboard embedded in a notebook, code cells collapsed.

    python dashboard/make_notebook.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(True)}


def code(src):
    return {"cell_type": "code", "metadata": {"hide_input": True, "jupyter": {"source_hidden": True}, "tags": ["hide_input"]},
            "execution_count": None, "outputs": [], "source": src.splitlines(True)}


cells = [
    md("# Q//HORIZON 2026: Interactive Results\n\nRun the cells below. Code is collapsed; click the bar on the left of a cell (or *View > Collapse*) to expand it. "
       "The same page is available outside Jupyter via `python dashboard/serve.py`."),
    code('import subprocess, sys\nfrom pathlib import Path\nROOT = Path.cwd()\nif not (ROOT / "dashboard").exists():\n    ROOT = ROOT.parent\nsubprocess.run([sys.executable, str(ROOT / "dashboard" / "build_data.py")], check=True)'),
    md("## 1. Interactive dashboard"),
    code('import html\nfrom IPython.display import HTML, display\npage = (ROOT / "dashboard" / "qhorizon_dashboard.html").read_text(encoding="utf-8")\n'
         'display(HTML(f\'<iframe srcdoc="{html.escape(page, quote=True)}" width="100%" height="900" style="border:1px solid #E2E8F0;border-radius:12px;background:#F8FAFC"></iframe>\'))'),
    md("## 2. Audit tables (10-seed averages, N = 500)"),
    code('import pandas as pd\nfrom IPython.display import display\n'
         'df = pd.read_csv(ROOT / "results" / "full" / "main_benchmark.csv")\ndf = df[df.n_train == 500]\n'
         'T = df.groupby("model").agg(PR_AUC_mean=("test_pr_auc", "mean"), PR_AUC_std=("test_pr_auc", "std"), MCC=("test_mcc", "mean"), F1=("test_f1", "mean"), weights=("n_trainable_params", "first"), collapsed=("collapsed", "sum")).sort_values("PR_AUC_mean", ascending=False)\n'
         'def tint(row):\n    c = "#E9EEFC" if row.name.startswith(("LogReg", "RBF", "Random", "Tiny")) else ("#F3EBFC" if row.name.startswith(("Q1", "Q2")) else "#F1F5F9")\n    return [f"background-color:{c}"] * len(row)\n'
         'display(T.style.apply(tint, axis=1).format({"PR_AUC_mean": "{:.4f}", "PR_AUC_std": "{:.4f}", "MCC": "{:.3f}", "F1": "{:.3f}"}))'),
    code('import json\nv = json.load(open(ROOT / "results" / "full" / "verdicts.json"))["hypotheses"]\n'
         'V = pd.DataFrame([{"ID": k, "Comparison": h["comparison"], "Delta": round(h["mean_delta"], 4), "CI low": round(h["ci_lo"], 4), "CI high": round(h["ci_hi"], 4), "Verdict": h["verdict"]} for k, h in v.items()])\n'
         'colors = {"Supported": "#DCFCE7", "Contradicted": "#FEE2E2", "Inconclusive": "#FEF3C7", "Practically tied": "#E0E7FF"}\n'
         'display(V.style.map(lambda x: f"background-color:{colors.get(x, \'\')}", subset=["Verdict"]).hide(axis="index"))'),
]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out = ROOT / "Q_Horizon_Dashboard.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out)