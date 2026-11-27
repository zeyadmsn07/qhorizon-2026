"""Build dashboard/data.js (and a standalone HTML bundle) from results/full/*.csv.

Usage (from the repo root):
    python dashboard/build_data.py

Only pandas/numpy are needed (both already in requirements.txt). Nothing is
re-run or re-trained: every number shown in the dashboard is read straight from
the committed 10-seed result files.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = ROOT / "results" / "full"
BONUS = RES / "bonus"
HTRU2 = ROOT / "data_cache" / "htru2.csv"

CLASSICAL_PREFIXES = ("LogReg", "RBF-SVM", "RandomForest", "Tiny-MLP", "FQK-baseline")
GRAY = ("Dummy",)


def family(name: str) -> str:
    if name.startswith(GRAY):
        return "baseline"
    if name.startswith(CLASSICAL_PREFIXES):
        return "classical"
    return "quantum"


def pretty(name: str) -> str:
    """Short, readable label (full name stays available in the tooltip)."""
    table = {
        "Q1-VQC (Bipartite_SignLinear_41p)": "Q1-VQC (41p)",
        "Q1-VQC (Ring L=2, Fragile v1)": "Q1-VQC Ring v1",
        "Q1-VQC (Headline)": "Q1-VQC (41p)",
    }
    return table.get(name, name)


def agg(df: pd.DataFrame, by, col="test_pr_auc") -> pd.DataFrame:
    g = df.groupby(by)
    out = g[col].agg(mean="mean", std="std", n="count")
    out["collapse"] = g["collapsed"].sum() if "collapsed" in df else 0
    for extra in ("test_mcc", "test_f1", "n_trainable_params", "train_time_sec"):
        if extra in df:
            out[extra] = g[extra].mean()
    return out.reset_index()


def rows(df: pd.DataFrame, name_col="model", label_map=None) -> list[dict]:
    res = []
    for r in df.to_dict("records"):
        nm = str(r[name_col])
        item = {
            "name": nm,
            "label": (label_map or pretty)(nm),
            "family": family(nm),
            "mean": float(r["mean"]),
            "std": float(0 if pd.isna(r["std"]) else r["std"]),
            "n": int(r["n"]),
            "collapse": int(r.get("collapse", 0)),
        }
        for k_src, k_dst in (("test_mcc", "mcc"), ("test_f1", "f1"),
                             ("n_trainable_params", "params"), ("train_time_sec", "sec")):
            if k_src in r and not pd.isna(r[k_src]):
                item[k_dst] = float(r[k_src])
        res.append(item)
    return res


def main() -> None:
    main_df = pd.read_csv(RES / "main_benchmark.csv")
    main_df = main_df[main_df["n_train"] == 500]
    main_tbl = agg(main_df, "model")

    # ---- HTRU2 class balance (hero numbers) --------------------------------
    htru = pd.read_csv(HTRU2)
    n_signals = int(len(htru))
    pulsar_pct = float(htru["target"].mean() * 100)

    # ---- Concentric Shell-8D ----------------------------------------------
    shell = pd.read_csv(RES / "nonlinear_dataset_summary.csv")
    shell_models = []
    for r in shell[shell["n_train"] == 500].to_dict("records"):
        shell_models.append({
            "name": r["model"], "label": pretty(r["model"]), "family": family(r["model"]),
            "mean": float(r["mean_pr_auc"]), "lo": float(r["ci_lo"]), "hi": float(r["ci_hi"]),
        })
    shell_by_n = {}
    for n, sub in shell.groupby("n_train"):
        shell_by_n[int(n)] = [{
            "name": r["model"], "label": pretty(r["model"]), "family": family(r["model"]),
            "mean": float(r["mean_pr_auc"]), "lo": float(r["ci_lo"]), "hi": float(r["ci_hi"]),
        } for r in sub.to_dict("records")]

    # ---- Experiment 1: wiring & depth -------------------------------------------
    ga = agg(pd.read_csv(RES / "gold_a_entanglement.csv"), "model")
    hw = pd.read_csv(RES / "hardware_cost_table.csv")
    hw_rows = hw.to_dict("records")

    # ---- Experiment 2: encoding ---------------------------------------------------
    gb = agg(pd.read_csv(RES / "gold_b_encoding.csv"), "model")

    # ---- Experiment 3: low-data sweep -------------------------------------------
    gc_raw = pd.read_csv(RES / "gold_c_sweep.csv")
    gc_raw = gc_raw[gc_raw["model"] != "Dummy"]
    gc = agg(gc_raw, ["model", "n_train"])
    gold_c = {}
    for model, sub in gc.groupby("model"):
        sub = sub.sort_values("n_train")
        gold_c[model] = {
            "label": pretty(model), "family": family(model),
            "points": [{"n": int(r.n_train), "mean": float(r["mean"]), "std": float(r["std"])}
                       for _, r in sub.iterrows()],
        }

    # ---- Experiment 4: shots ------------------------------------------------------
    td = pd.read_csv(RES / "table_d_shots.csv")
    td["shots"] = td["shots"].astype(str)
    gd = agg(td, "shots")
    order = {"100": 0, "500": 1, "2000": 2, "analytic": 3}
    gd["o"] = gd["shots"].map(order)
    gd = gd.sort_values("o")
    gold_d = [{
        "shots": r["shots"], "label": "Exact (analytic)" if r["shots"] == "analytic" else f"{int(r['shots']):,} shots",
        "mean": float(r["mean"]), "std": float(r["std"]), "n": int(r["n"]),
    } for _, r in gd.iterrows()]

    # ---- Experiment 5: PQK -------------------------------------------------------
    pqk = pd.read_csv(BONUS / "exp1_pqk_summary.csv")
    pqk_circ = int(pqk["pqk_circuits"].iloc[0])
    fqk_circ = int(pqk["fqk_circuits"].iloc[0])
    pqk_points = [{
        "alpha": float(r.alpha),
        "pqk": float(r.pqk_pr_mean), "pqk_std": float(r.pqk_pr_std),
        "fqk": float(r.fqk_pr_mean), "fqk_std": float(r.fqk_pr_std),
    } for r in pqk.itertuples()]

    # ---- Experiment 6: hybrid blend ---------------------------------------------
    hyb = pd.read_csv(BONUS / "exp3_disagreement_summary.csv")
    hybrid = [{
        "n": int(r.N), "vqc": float(r.vqc_pr), "mlp": float(r.mlp_pr),
        "blend": float(r.prob_blend_pr), "rank_blend": float(r.rank_blend_pr),
    } for r in hyb.itertuples()]

    # ---- Verdicts -----------------------------------------------------------
    verd = json.loads((RES / "verdicts.json").read_text(encoding="utf-8"))
    verdicts = [{
        "id": k, "comparison": v["comparison"], "delta": v["mean_delta"],
        "lo": v["ci_lo"], "hi": v["ci_hi"], "p": v["wilcoxon_p"], "verdict": v["verdict"],
    } for k, v in verd["hypotheses"].items()]

    data = {
        "meta": {
            "n_signals": n_signals,
            "pulsar_pct": round(pulsar_pct, 2),
            "noise_pct": round(100 - pulsar_pct, 2),
            "eval_seeds": "100..109", "n_seeds": 10, "n_train": 500,
            "pqk_circuits": pqk_circ, "fqk_circuits": fqk_circ,
            "circuit_ratio": round(fqk_circ / pqk_circ),
            "weights": 41,
        },
        "htru2": rows(main_tbl),
        "shell": shell_models,
        "shell_by_n": shell_by_n,
        "gold_a": rows(ga),
        "hardware": hw_rows,
        "gold_b": rows(gb),
        "gold_c": gold_c,
        "gold_d": gold_d,
        "pqk": pqk_points,
        "hybrid": hybrid,
        "verdicts": verdicts,
    }

    js = "window.QH_DATA = " + json.dumps(data, indent=1, ensure_ascii=False) + ";\n"
    (HERE / "data.js").write_text(js, encoding="utf-8")
    print(f"wrote {HERE / 'data.js'}  ({len(js):,} bytes)")

    # ---- Standalone single-file bundle -------------------------------------
    html = (HERE / "index.html").read_text(encoding="utf-8")
    css = (HERE / "style.css").read_text(encoding="utf-8")
    app = (HERE / "app.js").read_text(encoding="utf-8")
    html = html.replace('<link rel="stylesheet" href="style.css">', f"<style>\n{css}\n</style>")
    html = html.replace('<script src="data.js"></script>', f"<script>\n{js}</script>")
    html = html.replace('<script src="app.js"></script>', f"<script>\n{app}\n</script>")
    out = HERE / "qhorizon_dashboard.html"
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out}  ({len(html):,} bytes)  <- single file, opens offline")


if __name__ == "__main__":
    main()