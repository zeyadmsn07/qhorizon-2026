import argparse, json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from src.config import COLLAPSE_PR_AUC, ModeConfig, RESULTS_DIR, get_mode_config, get_results_dir
from src.data import get_locked_splits, sample_and_preprocess
from src.models_quantum import train_predict_vqc
from src.stats import two_level_bootstrap_pr_auc
from src.tuning import tune_and_evaluate_10_seeds


VQC_ABLATION_SPECS = [
    ("Ring_L2_Baseline", {"encoding": "angle", "entanglement": "ring", "n_layers": 2}),
    ("Init_WideNormal_0.5", {"encoding": "angle", "entanglement": "ring", "n_layers": 2, "init_mode": "wide_normal"}),
    ("Init_Uniform_Pi", {"encoding": "angle", "entanglement": "ring", "n_layers": 2, "init_mode": "uniform_pi"}),
    ("SignAware_Offset_Pi2", {"encoding": "angle", "entanglement": "ring", "n_layers": 2, "sign_offset": True}),
    ("Readout_Linear8_41p", {"encoding": "angle", "entanglement": "ring", "n_layers": 2, "readout": "linear_z"}),
    ("Readout_SingleZ0", {"encoding": "angle", "entanglement": "ring", "n_layers": 2, "readout": "single_z0"}),
    ("Epochs100_Patience20", {"encoding": "angle", "entanglement": "ring", "n_layers": 2, "max_epochs": 100, "patience": 20}),
    ("Restarts_k3_ValSelected", {"encoding": "angle", "entanglement": "ring", "n_layers": 2, "n_restarts": 3}),
    ("Bipartite_SignLinear_41p", {"encoding": "angle", "entanglement": "bipartite", "n_layers": 2, "sign_offset": True, "readout": "linear_z"}),
]


def plot_collapse_diagnostics(splits: dict, out_dir: Path, max_epochs: int = 55) -> None:
    # Compare fragile Ring L=2 (alpha=0.25, even cos symmetry) vs SignAware+Linear on seed 100 (good) and seed 105 (v1 collapse)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), sharey=True)
    cfgs = [
        ("Fragile Ring L=2 (even cos init)", {"encoding": "angle", "entanglement": "ring", "n_layers": 2, "alpha": 0.25, "lr": 0.01, "max_epochs": max_epochs, "val_every": 1}),
        ("Sign-Aware + Linear8 (odd sin init)", {"encoding": "angle", "entanglement": "bipartite", "n_layers": 2, "alpha": 0.5, "lr": 0.02, "sign_offset": True, "readout": "linear_z", "max_epochs": max_epochs, "val_every": 1}),
    ]
    for ax, (title, params) in zip(axes, cfgs):
        for s, col in [(100, "#1B9E77"), (105, "#D95F02")]:
            dp = out_dir / "diagnostics" / f"curve__{ s }_{ 'fragile' if 'Fragile' in title else 'stable' }.json"
            train_predict_vqc(params, sample_and_preprocess(splits, 500, s), seed=s, diag_path=dp)
            d = json.loads(dp.read_text(encoding="utf-8"))
            ax.plot(d["epochs_logged"], d["train_losses"], linestyle="--", alpha=0.65, color=col, label=f"Seed {s} Train")
            ax.plot(d["epochs_logged"], d["val_losses"], linewidth=2.0, color=col, label=f"Seed {s} Val (init |g|={d['init_grad_norm']:.2f})")
        ax.set_title(title, fontsize=10.5, fontweight="bold")
        ax.set_xlabel("Epoch"); ax.grid(True, alpha=0.25, linestyle="--")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Weighted BCE Loss")
    fig.suptitle("VQC Optimization Diagnostics: Seed 100 vs. Collapsed Seed 105", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "vqc_loss_diagnostics.png", dpi=300)
    plt.close(fig)


def run_vqc_study(cfg: ModeConfig, out_dir: Path, splits: dict, n_tune_seeds: int | None = None, cache: dict | None = None, tlog: list | None = None) -> dict:
    cache = cache if cache is not None else {}
    tlog = tlog if tlog is not None else []
    plot_collapse_diagnostics(splits, out_dir, max_epochs=min(55, max(25, cfg.vqc_max_epochs)))

    rows = []
    for lbl, ov in VQC_ABLATION_SPECS:
        eff_ov = dict(ov)
        if cfg.name != "full" and "max_epochs" in eff_ov:
            eff_ov["max_epochs"] = cfg.vqc_max_epochs + 15
        rows.extend(tune_and_evaluate_10_seeds(splits, 500, "Q1-VQC", eff_ov, lbl, cache=cache, mode_cfg=cfg, out_dir=out_dir, n_tune_seeds=n_tune_seeds, tuning_log=tlog))

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "vqc_stability_ablation.csv", index=False)

    # Pre-stated selection rule: highest mean validation PR-AUC across tune_seeds (never test data!)
    ablation_labels = {lbl: ov for lbl, ov in VQC_ABLATION_SPECS}
    vqc_entries = [e for e in tlog if e["family"] == "Q1-VQC" and e["label"] in ablation_labels]
    best_entry = max(vqc_entries, key=lambda e: (e["chosen_mean_val_pr_auc"], -e["chosen_std_val_pr_auc"]))
    chosen_lbl = best_entry["label"]
    chosen_ov = ablation_labels[chosen_lbl]

    sel = {
        "rule": "Highest mean validation PR-AUC across disjoint tune_seeds (tie-break: lower tune val std); test set never consulted.",
        "chosen_label": chosen_lbl,
        "chosen_overrides": chosen_ov,
        "chosen_tune_mean_val_pr_auc": best_entry["chosen_mean_val_pr_auc"],
        "chosen_tune_std_val_pr_auc": best_entry["chosen_std_val_pr_auc"],
        "candidates_tune_val": [{k: e[k] for k in ("label", "chosen_mean_val_pr_auc", "chosen_std_val_pr_auc")} for e in vqc_entries],
    }
    (out_dir / "headline_vqc_selection.json").write_text(json.dumps(sel, indent=2), encoding="utf-8")

    # Write empirical docs/vqc_stability.md summary
    import numpy as np
    lines = [f"# VQC Stability & Symmetry-Breaking Study (`{out_dir.as_posix()}`)\n", f"**Pre-stated headline selection rule:** {sel['rule']}\n", f"**Validation-selected headline VQC:** `{chosen_lbl}` (`tune val PR-AUC = {sel['chosen_tune_mean_val_pr_auc']:.4f}`)\n", "| Variant | Tune Val PR-AUC | Test PR-AUC | 95% Bootstrap CI | Collapse Rate | Init Grad Norm | Trainable Params |", "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |"]
    for lbl, _ in VQC_ABLATION_SPECS:
        sub = df[df["model"] == lbl].sort_values("seed")
        yt = np.load(out_dir / str(sub["pred_file"].iloc[0]))["y_test"]
        sc = [np.load(out_dir / str(p))["test_scores"] for p in sub["pred_file"]]
        pt, lo, hi = two_level_bootstrap_pr_auc(yt, sc, n_boot=400)
        tv = next(e["chosen_mean_val_pr_auc"] for e in vqc_entries if e["label"] == lbl)
        cr = f"{int((sub['test_pr_auc'] < COLLAPSE_PR_AUC).sum())}/{len(sub)}"
        lines.append(f"| `{lbl}` | `{tv:.4f}` | `{pt:.4f}` | `[{lo:.4f}, {hi:.4f}]` | {cr} | `{sub['init_grad_norm'].mean():.3f}` | {int(sub['n_trainable_params'].iloc[0])} |")
    Path("docs/vqc_stability.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return sel


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "fast", "full"], default=None)
    ap.add_argument("--tune-seeds", type=int, choices=[1, 3], default=None)
    args = ap.parse_args()
    c = get_mode_config(args.mode)
    run_vqc_study(c, get_results_dir(c.name), get_locked_splits(), args.tune_seeds)
    print("Completed VQC stability study and wrote docs/vqc_stability.md.")