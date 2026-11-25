import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from src.config import COLLAPSE_PR_AUC, RESULTS_DIR

PM = "\u00b1"
DASH = "\u2014"


def _fmt_ms(s: pd.Series) -> str:
    std = 0.0 if len(s) < 2 else float(s.std(ddof=1))
    return f"{s.mean():.4f} {PM} {std:.4f}"


def _fmt_med_iqr(s: pd.Series) -> str:
    q25, q75 = np.percentile(s, [25, 75])
    return f"{float(np.median(s)):.4f} ({float(q75 - q25):.4f})"


def _summarize_group(df: pd.DataFrame, group_cols: list[str]) -> str:
    has_p3 = "n_trainable_params" in df.columns
    headers = group_cols + ["PR-AUC (Mean+/-Std)", "Median (IQR)", "Collapse", "Test F1", "Test MCC", "Test Acc"]
    if has_p3:
        headers += ["Trainable", "Tuned HP", "Model Size", "SVs", "Sim/CPU s*"]
    elif "n_params" in df.columns:
        headers += ["Params (v1)", "Time s"]

    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join([":---"] * len(headers)) + " |"]
    for keys, sub in df.groupby(group_cols, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        pr = sub["test_pr_auc"]
        n_col = int((pr < COLLAPSE_PR_AUC).sum())
        row = [str(k) for k in keys] + [
            _fmt_ms(pr),
            _fmt_med_iqr(pr),
            f"{n_col}/{len(pr)}",
            _fmt_ms(sub["test_f1"]),
            _fmt_ms(sub["test_mcc"]),
            _fmt_ms(sub["test_acc"]),
        ]
        if has_p3:
            sv = sub["n_support_vectors"]
            sv_txt = DASH if sv.isna().all() else f"{sv.mean():.1f}"
            row += [
                f"{int(round(sub['n_trainable_params'].mean()))}",
                f"{int(round(sub['n_tuned_hyperparams'].mean()))}",
                f"{sub['model_size_values'].mean():.0f}",
                sv_txt,
                f"{sub['train_time_sec'].mean():.2f}",
            ]
        elif "n_params" in df.columns:
            row += [f"{sub['n_params'].mean():.0f}", f"{sub['train_time_sec'].mean():.2f}"]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def generate_tables(res_dir: Path) -> str:
    sections = [f"# Regenerated Benchmark Tables (`{res_dir.as_posix()}`)\n"]
    if "n_trainable_params" in pd.read_csv(res_dir / "main_benchmark.csv").columns:
        sections.append("""## Fair Comparison Pairs & Parameter Accounting
- `Trainable`: gradient-optimized or directly fitted weight scalars (`0` for non-parametric SVM/RF/Kernel).
- `Tuned HP`: hyperparameter dimensions searched under the shared validation budget; `Model Size`: inference scalars (`SVs x features`, tree nodes, or weights).

| Comparison Pair | Quantum / Target Model | Classical Counterpart | Why This Pair Is Fair |
| :--- | :--- | :--- | :--- |
| Parametric Weight-Matched | Q1-VQC (34 weights) | Tiny-MLP (21 / 41 / 81 weights) | Both use minibatch AdamW + BCE early stopping with matched parameter counts |
| Non-Parametric Kernel | Q2-QuantumKernel (0 weights, 8q ZZ) | RBF-SVM (0 weights, Gaussian) | Identical dual SVM solver; only the kernel Gram matrix differs |
| Linear Separability Floor | All nonlinear models | LogReg (9 weights) | Quantifies how much pulsar separation is already linear |
| Tree Ensemble Reference | Q1-VQC & Q2-QuantumKernel | RandomForest (0 weights) | Standard non-parametric tabular ensemble baseline |
""")

    specs = [
        ("Main Benchmark (N=500)", "main_benchmark.csv", ["model"]),
        ("MLP Capacity Ablation (N=500)", "mlp_ablation.csv", ["model"]),
        ("Gold A: Entanglement & Depth", "gold_a_entanglement.csv", ["model"]),
        ("Gold B: Encoding & Norm Restoration", "gold_b_encoding.csv", ["model"]),
        ("Gold C: Sample Efficiency Sweep", "gold_c_sweep.csv", ["n_train", "model"]),
        ("Table D: Finite Shot Noise", "table_d_shots.csv", ["model", "shots"]),
    ]
    for title, fn, gcols in specs:
        p = res_dir / fn
        if p.exists():
            df = pd.read_csv(p)
            if "shots" in df.columns:
                df["shots"] = df["shots"].astype(str)
            sections.append(f"## {title}\n" + _summarize_group(df, gcols) + "\n")
    sections.append("*Note: Quantum fit times measure CPU state-vector simulation (`default.qubit`), not physical QPU hardware execution.*\n")
    out_md = "\n".join(sections)
    (res_dir / "summary_tables.md").write_text(out_md, encoding="utf-8")
    return out_md


def main() -> None:
    ap = argparse.ArgumentParser()
    default_dir = RESULTS_DIR / "full" if (RESULTS_DIR / "full" / "main_benchmark.csv").exists() else RESULTS_DIR / "v1_original_10seed"
    ap.add_argument("--results-dir", type=Path, default=default_dir)
    args = ap.parse_args()
    md = generate_tables(args.results_dir)
    print(md.split("## Gold A")[0])
    print(f"Saved full tables to {args.results_dir / 'summary_tables.md'}")


if __name__ == "__main__":
    main()
