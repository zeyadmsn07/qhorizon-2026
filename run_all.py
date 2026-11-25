import argparse, json, time
import pandas as pd
import torch
from src.config import get_mode_config, get_results_dir
from src.data import get_locked_splits, sample_and_preprocess
from src.metrics import evaluate_predictions
from src.models_quantum import train_predict_vqc
from src.tuning import get_best_config, tune_and_evaluate_10_seeds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "fast", "full"], default=None)
    args = ap.parse_args()
    cfg = get_mode_config(args.mode)
    out_dir = get_results_dir(cfg.name)
    if cfg.name != "full":
        print(f"NOTE: --mode {cfg.name} writes to {out_dir} and will NOT overwrite results/full/ or reproduce 10-seed README tables.")

    t0 = time.perf_counter()
    splits = get_locked_splits()
    cache = {}

    specs = [
        ("Dummy", {}, "Dummy"), ("LogReg", {}, "LogReg"), ("RBF-SVM", {}, "RBF-SVM"),
        ("RandomForest", {}, "RandomForest"), ("Tiny-MLP", {}, "Tiny-MLP (41p)"),
        ("Q1-VQC", {"encoding": "angle", "entanglement": "ring", "n_layers": 2}, "Q1-VQC (8q Ring L=2)"),
        ("Q2-QuantumKernel", {}, "Q2-QuantumKernel (8q)"),
    ]
    pd.DataFrame([r for f, o, l in specs for r in tune_and_evaluate_10_seeds(splits, 500, f, o, l, cache, cfg)]).to_csv(out_dir / "main_benchmark.csv", index=False)

    ga = [r for ent in ["none", "ring", "bipartite"] for L in [1, 2, 4] for r in tune_and_evaluate_10_seeds(splits, 500, "Q1-VQC", {"encoding": "angle", "entanglement": ent, "n_layers": L}, f"VQC_{ent}_L{L}", cache, cfg)]
    pd.DataFrame(ga).to_csv(out_dir / "gold_a_entanglement.csv", index=False)

    gb_specs = [
        ("8q_Angle_L2", {"encoding": "angle", "entanglement": "ring", "n_layers": 2, "reupload": False}),
        ("8q_Angle_Reupload_L2", {"encoding": "angle", "entanglement": "ring", "n_layers": 2, "reupload": True}),
        ("3q_Amplitude_L5", {"encoding": "amplitude", "entanglement": "ring", "n_layers": 5}),
        ("4q_AmplitudePlusNorm_L4", {"encoding": "amplitude_norm", "entanglement": "ring", "n_layers": 4}),
    ]
    pd.DataFrame([r for l, o in gb_specs for r in tune_and_evaluate_10_seeds(splits, 500, "Q1-VQC", o, l, cache, cfg)]).to_csv(out_dir / "gold_b_encoding.csv", index=False)

    sw = [("Dummy", {}, "Dummy"), ("RBF-SVM", {}, "RBF-SVM"), ("Tiny-MLP", {}, "Tiny-MLP (41p)"), ("Q1-VQC", {"encoding": "angle", "entanglement": "ring", "n_layers": 2}, "Q1-VQC (8q)"), ("Q2-QuantumKernel", {}, "Q2-QuantumKernel (8q)")]
    pd.DataFrame([r for N in cfg.sample_sweep for f, o, l in sw for r in tune_and_evaluate_10_seeds(splits, N, f, o, l, cache, cfg)]).to_csv(out_dir / "gold_c_sweep.csv", index=False)

    best_vqc = dict(get_best_config(cache, "Q1-VQC", 500, {"encoding": "angle", "entanglement": "ring", "n_layers": 2}), max_epochs=cfg.vqc_max_epochs)
    shots_rows = []
    for s in cfg.eval_seeds:
        d = sample_and_preprocess(splits, 500, s)
        _, _, npa, sec, gn, model = train_predict_vqc(best_vqc, d, seed=s, return_model=True)
        Xv, Xte = torch.tensor(d["X_val"], dtype=torch.float32), torch.tensor(d["X_test"], dtype=torch.float32)
        for sh in [None, 2000, 500, 100]:
            m = evaluate_predictions(d["y_val"], model.predict_shots(Xv, sh, seed=s), d["y_test"], model.predict_shots(Xte, sh, seed=s + 1000), npa, sec)
            m.update({"shots": "analytic" if sh is None else str(sh), "seed": s, "init_grad_norm": gn, "model": "Q1-VQC (8q Ring L=2)"})
            shots_rows.append(m)
    pd.DataFrame(shots_rows).to_csv(out_dir / "table_d_shots.csv", index=False)

    elapsed = time.perf_counter() - t0
    (out_dir / "run_metadata.json").write_text(json.dumps({"mode": cfg.name, "wall_clock_sec": round(elapsed, 2)}, indent=2), encoding="utf-8")
    print(f"Completed --mode {cfg.name} in {elapsed:.2f}s -> {out_dir}")


if __name__ == "__main__":
    main()
