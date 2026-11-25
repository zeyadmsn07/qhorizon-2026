import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from src.config import COLLAPSE_PR_AUC, RESULTS_DIR
from src.stats import classify_verdict, paired_delta_pr_auc, two_level_bootstrap_pr_auc


def _load_scores(res_dir: Path, sub: pd.DataFrame) -> tuple[np.ndarray, list[np.ndarray]]:
    ordered = sub.sort_values("seed")
    y_t = np.load(res_dir / str(ordered["pred_file"].iloc[0]))["y_test"]
    sc = [np.load(res_dir / str(p))["test_scores"] for p in ordered["pred_file"]]
    return y_t, sc


def generate_verdicts(res_dir: Path, n_boot: int = 1000) -> dict:
    main_df = pd.read_csv(res_dir / "main_benchmark.csv")
    ga_df = pd.read_csv(res_dir / "gold_a_entanglement.csv")
    gb_df = pd.read_csv(res_dir / "gold_b_encoding.csv")
    gc_df = pd.read_csv(res_dir / "gold_c_sweep.csv")
    td_df = pd.read_csv(res_dir / "table_d_shots.csv")
    td_df["shots"] = td_df["shots"].astype(str)

    vqc_all = [m for m in main_df["model"].unique() if m.startswith("Q1-VQC")]
    non_fragile = [m for m in vqc_all if "Fragile" not in m]
    headline_vqc = non_fragile[0] if non_fragile else vqc_all[0]
    c_models = [m for m in main_df["model"].unique() if m not in vqc_all and not m.startswith("Q2-") and m != "Dummy"]
    best_cls = max(c_models, key=lambda m: main_df[main_df["model"] == m]["test_pr_auc"].mean())

    hyps = {}
    md = [
        f"# Statistical Verdicts & Two-Level Bootstrap CIs (`{res_dir.as_posix()}`)",
        "",
        "Rule (`margin = 0.01` PR-AUC, 95% CI across seeds x stratified test resamples): **Supported** (`CI > 0` and `|delta| > 0.01`), **Practically tied** (`CI` inside `[-0.01, +0.01]`), **Contradicted** (`CI < 0` and `|delta| > 0.01`), **Inconclusive** otherwise.",
        "",
        "## 1. Hypothesis Verdicts",
        "| ID | Comparison (A vs. B) | Delta PR-AUC | 95% Two-Level CI | Wilcoxon p (min p) | Verdict |",
        "| :--- | :--- | :---: | :---: | :---: | :---: |",
    ]

    def rec(hid: str, desc: str, y_t, sa, sb, direction: str = "greater"):
        r = paired_delta_pr_auc(y_t, sa, sb, n_boot=n_boot)
        v = classify_verdict(r["mean_delta"], r["ci_lo"], r["ci_hi"], direction=direction)
        r.update({"comparison": desc, "verdict": v, "direction": direction})
        hyps[hid] = r
        ps = "n/a (<5 seeds)" if np.isnan(r["wilcoxon_p"]) else f"{r['wilcoxon_p']:.4f} (min {r['min_attainable_p']:.4f})"
        md.append(f"| **{hid}** | {desc} | `{r['mean_delta']:+.4f}` | `[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]` | {ps} | **{v}** |")

    yt, sa_cls = _load_scores(res_dir, main_df[main_df["model"] == best_cls])
    _, sb_qk = _load_scores(res_dir, main_df[main_df["model"] == "Q2-QuantumKernel (8q)"])
    _, sb_vqc = _load_scores(res_dir, main_df[main_df["model"] == headline_vqc])
    _, sb_mlp = _load_scores(res_dir, main_df[main_df["model"] == "Tiny-MLP (41p)"])
    _, sb_rf = _load_scores(res_dir, main_df[main_df["model"] == "RandomForest"])
    rec("H1a", f"{best_cls} vs. Q2-QuantumKernel (8q)", yt, sa_cls, sb_qk, "greater")
    rec("H1b", f"Tiny-MLP (41p) vs. {headline_vqc}", yt, sb_mlp, sb_vqc, "greater")
    rec("H1c", "Q2-QuantumKernel (8q) vs. RandomForest", yt, sb_qk, sb_rf, "greater")

    _, sa_bip = _load_scores(res_dir, ga_df[ga_df["model"] == "VQC_bipartite_L2"])
    _, sb_none = _load_scores(res_dir, ga_df[ga_df["model"] == "VQC_none_L2"])
    _, sb_ring = _load_scores(res_dir, ga_df[ga_df["model"] == "VQC_ring_L2"])
    rec("H2a", "VQC_bipartite_L2 vs. VQC_none_L2", yt, sa_bip, sb_none, "greater")
    rec("H2b", "VQC_bipartite_L2 vs. VQC_ring_L2", yt, sa_bip, sb_ring, "greater")

    _, sa_norm = _load_scores(res_dir, gb_df[gb_df["model"] == "4q_AmplitudePlusNorm_L4"])
    _, sb_amp = _load_scores(res_dir, gb_df[gb_df["model"] == "3q_Amplitude_L5"])
    rec("H3", "4q_AmplitudePlusNorm_L4 vs. 3q_Amplitude_L5", yt, sa_norm, sb_amp, "greater")

    gc100 = gc_df[gc_df["n_train"] == 100]
    _, sa_qk100 = _load_scores(res_dir, gc100[gc100["model"] == "Q2-QuantumKernel (8q)"])
    _, sb_svm100 = _load_scores(res_dir, gc100[gc100["model"] == "RBF-SVM"])
    rec("H4a", "N=100 Q2-QuantumKernel vs. RBF-SVM (tied)", yt, sa_qk100, sb_svm100, "tied")

    _, sa_an = _load_scores(res_dir, td_df[td_df["shots"] == "analytic"])
    for sh in ["2000", "500", "100"]:
        _, sb_sh = _load_scores(res_dir, td_df[td_df["shots"] == sh])
        rec(f"H4b_{sh}", f"Shots={sh} vs. Analytic (tied)", yt, sb_sh, sa_an, "tied")

    md += ["", f"## 2. Pairwise Comparison vs. Best Classical (`{best_cls}`)", "| Model | Mean PR-AUC | 95% Two-Level CI | Collapse | Delta vs. Best Classical | Paired 95% CI | Verdict |", "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |"]
    models_summary = {}
    for m_name in main_df["model"].unique():
        sub = main_df[main_df["model"] == m_name]
        _, sm = _load_scores(res_dir, sub)
        pt, lo, hi = two_level_bootstrap_pr_auc(yt, sm, n_boot=n_boot)
        cr = f"{int((sub['test_pr_auc'] < COLLAPSE_PR_AUC).sum())}/{len(sub)}"
        if m_name == best_cls:
            d, dlo, dhi, st = 0.0, 0.0, 0.0, "Reference"
        else:
            dr = paired_delta_pr_auc(yt, sm, sa_cls, n_boot=n_boot)
            d, dlo, dhi = dr["mean_delta"], dr["ci_lo"], dr["ci_hi"]
            st = classify_verdict(d, dlo, dhi, "greater")
        models_summary[m_name] = {"mean": pt, "ci_lo": lo, "ci_hi": hi, "collapse": cr, "delta_vs_best": d, "delta_lo": dlo, "delta_hi": dhi, "status": st}
        md.append(f"| **{m_name}** | `{pt:.4f}` | `[{lo:.4f}, {hi:.4f}]` | {cr} | `{d:+.4f}` | `[{dlo:+.4f}, {dhi:+.4f}]` | {st} |")

    payload = {"best_classical": best_cls, "headline_vqc": headline_vqc, "hypotheses": hyps, "models": models_summary}
    (res_dir / "verdicts.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (res_dir / "verdicts.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    d_dir = RESULTS_DIR / "full" if (RESULTS_DIR / "full" / "main_benchmark.csv").exists() else RESULTS_DIR / "fast"
    ap.add_argument("--results-dir", type=Path, default=d_dir)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    generate_verdicts(args.results_dir, n_boot=args.n_boot)
    print((args.results_dir / "verdicts.md").read_text(encoding="utf-8"))