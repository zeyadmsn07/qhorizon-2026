import argparse, json, time
import pandas as pd
import torch
from src.config import COLLAPSE_PR_AUC, TUNE_SEEDS, get_mode_config, get_results_dir
from src.data import get_locked_splits, sample_and_preprocess
from src.metrics import evaluate_predictions
from src.models_quantum import train_predict_vqc
from src.tuning import get_best_config, save_pred_npz, tune_and_evaluate_10_seeds
from src.vqc_study import run_vqc_study


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "fast", "full"], default=None)
    ap.add_argument("--tune-seeds", type=int, choices=[1, 3], default=None)
    args = ap.parse_args()

    cfg = get_mode_config(args.mode)
    out_dir = get_results_dir(cfg.name)
    active_tune = list(TUNE_SEEDS[: args.tune_seeds]) if args.tune_seeds else list(cfg.tune_seeds)
    print(f"=== Q//Horizon Runner | mode={cfg.name} | tune_seeds={active_tune} | eval_seeds={list(cfg.eval_seeds)} ===", flush=True)
    if cfg.name != "full":
        print(f"NOTE: --mode {cfg.name} writes only to {out_dir} and never overwrites results/full/.", flush=True)

    t0 = time.perf_counter()
    splits = get_locked_splits()
    cache, tlog = {}, []
    kw = dict(cache=cache, mode_cfg=cfg, out_dir=out_dir, n_tune_seeds=args.tune_seeds, tuning_log=tlog)

    print("[0/5] Running Phase 5 VQC Stability Study & Validation-Only Headline Selection...", flush=True)
    sel = run_vqc_study(cfg, out_dir, splits, n_tune_seeds=args.tune_seeds, cache=cache, tlog=tlog)
    hl_ov = sel["chosen_overrides"]
    hl_lbl = f"Q1-VQC ({sel['chosen_label']})"
    frag_ov = {"encoding": "angle", "entanglement": "ring", "n_layers": 2}
    print(f"   -> Selected Headline VQC on tune_seeds: {hl_lbl} ({time.perf_counter() - t0:.1f}s)", flush=True)

    specs = [
        ("Dummy", {}, "Dummy"), ("LogReg", {}, "LogReg"), ("RBF-SVM", {}, "RBF-SVM"),
        ("RandomForest", {}, "RandomForest"), ("Tiny-MLP", {}, "Tiny-MLP (41p)"),
        ("Q1-VQC", hl_ov, hl_lbl),
        ("Q1-VQC", frag_ov, "Q1-VQC (Ring L=2, Fragile v1)"),
        ("Q2-QuantumKernel", {}, "Q2-QuantumKernel (8q)"),
    ]
    pd.DataFrame([r for f, o, l in specs for r in tune_and_evaluate_10_seeds(splits, 500, f, o, l, **kw)]).to_csv(out_dir / "main_benchmark.csv", index=False)
    print(f"[1/5] Saved main_benchmark.csv ({time.perf_counter() - t0:.1f}s elapsed)", flush=True)

    mlp_specs = [("Tiny-MLP", {"hidden_dim": 2}, "Tiny-MLP (h=2, 21p)"), ("Tiny-MLP", {"hidden_dim": 4}, "Tiny-MLP (h=4, 41p)"), ("Tiny-MLP", {"hidden_dim": 8}, "Tiny-MLP (h=8, 81p)")]
    pd.DataFrame([r for f, o, l in mlp_specs for r in tune_and_evaluate_10_seeds(splits, 500, f, o, l, **kw)]).to_csv(out_dir / "mlp_ablation.csv", index=False)

    ga = [r for ent in ["none", "ring", "bipartite"] for L in [1, 2, 4] for r in tune_and_evaluate_10_seeds(splits, 500, "Q1-VQC", {"encoding": "angle", "entanglement": ent, "n_layers": L}, f"VQC_{ent}_L{L}", **kw)]
    pd.DataFrame(ga).to_csv(out_dir / "gold_a_entanglement.csv", index=False)
    print(f"[2/5] Saved gold_a_entanglement.csv ({time.perf_counter() - t0:.1f}s elapsed)", flush=True)

    gb_specs = [
        ("8q_Angle_L2", {"encoding": "angle", "entanglement": "ring", "n_layers": 2, "reupload": False}),
        ("8q_Angle_Reupload_L2", {"encoding": "angle", "entanglement": "ring", "n_layers": 2, "reupload": True}),
        ("3q_Amplitude_L5", {"encoding": "amplitude", "entanglement": "ring", "n_layers": 5}),
        ("4q_AmplitudePlusNorm_L4", {"encoding": "amplitude_norm", "entanglement": "ring", "n_layers": 4}),
    ]
    pd.DataFrame([r for l, o in gb_specs for r in tune_and_evaluate_10_seeds(splits, 500, "Q1-VQC", o, l, **kw)]).to_csv(out_dir / "gold_b_encoding.csv", index=False)
    print(f"[3/5] Saved gold_b_encoding.csv ({time.perf_counter() - t0:.1f}s elapsed)", flush=True)

    sw = [("Dummy", {}, "Dummy"), ("RBF-SVM", {}, "RBF-SVM"), ("Tiny-MLP", {}, "Tiny-MLP (41p)"), ("Q1-VQC", hl_ov, hl_lbl), ("Q2-QuantumKernel", {}, "Q2-QuantumKernel (8q)")]
    pd.DataFrame([r for N in cfg.sample_sweep for f, o, l in sw for r in tune_and_evaluate_10_seeds(splits, N, f, o, l, **kw)]).to_csv(out_dir / "gold_c_sweep.csv", index=False)
    print(f"[4/5] Saved gold_c_sweep.csv ({time.perf_counter() - t0:.1f}s elapsed)", flush=True)

    best_vqc = dict(get_best_config(cache, "Q1-VQC", 500, hl_ov))
    best_vqc.setdefault("max_epochs", cfg.vqc_max_epochs)
    shots_rows = []
    for s in cfg.eval_seeds:
        d = sample_and_preprocess(splits, 500, s)
        _, _, comp, sec, gn, model = train_predict_vqc(best_vqc, d, seed=s, return_model=True)
        Xv, Xte = torch.tensor(d["X_val"], dtype=torch.float32), torch.tensor(d["X_test"], dtype=torch.float32)
        for sh in [None, 2000, 500, 100]:
            lbl_sh = "analytic" if sh is None else str(sh)
            vs, ts = model.predict_shots(Xv, sh, seed=s), model.predict_shots(Xte, sh, seed=s + 1000)
            m = evaluate_predictions(d["y_val"], vs, d["y_test"], ts, comp, sec)
            rel = save_pred_npz(out_dir, "TableD", f"HeadlineVQC_shots_{lbl_sh}", 500, s, d["y_val"], vs, d["y_test"], ts, m["threshold"])
            m.update({"shots": lbl_sh, "seed": s, "init_grad_norm": gn, "collapsed": int(m["test_pr_auc"] < COLLAPSE_PR_AUC), "model": hl_lbl, "pred_file": rel})
            shots_rows.append(m)
    pd.DataFrame(shots_rows).to_csv(out_dir / "table_d_shots.csv", index=False)

    elapsed = time.perf_counter() - t0
    meta = {"mode": cfg.name, "headline_vqc": hl_lbl, "tune_seeds": active_tune, "eval_seeds": list(cfg.eval_seeds), "wall_clock_sec": round(elapsed, 2)}
    (out_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[5/5] Completed --mode {cfg.name} in {elapsed:.2f}s -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()