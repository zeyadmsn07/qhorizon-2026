import time
import pandas as pd
import torch
from tqdm import tqdm

from src.config import RESULTS_DIR, SAMPLE_SWEEP, SEEDS
from src.data import get_locked_splits, sample_and_preprocess
from src.metrics import evaluate_predictions
from src.models_quantum import train_predict_vqc
from src.tuning import get_best_config, tune_and_evaluate_10_seeds


def main():
    t_start = time.perf_counter()
    splits = get_locked_splits()
    run_cache = {}

    # Section 1: Main benchmark at N=500 across all 5 classical baselines and 2 quantum models
    print("\n" + "=" * 70)
    print(" SECTION 1: Main Benchmark (N=500, Classical vs. Quantum)")
    print("=" * 70)
    t0_main = time.perf_counter()
    main_rows = []
    specs = [
        ("Dummy", {}, "Dummy"),
        ("LogReg", {}, "LogReg"),
        ("RBF-SVM", {}, "RBF-SVM"),
        ("RandomForest", {}, "RandomForest"),
        ("Tiny-MLP", {}, "Tiny-MLP (41p)"),
        ("Q1-VQC", {"encoding": "angle", "entanglement": "ring", "n_layers": 2}, "Q1-VQC (8q Ring L=2)"),
        ("Q2-QuantumKernel", {}, "Q2-QuantumKernel (8q)"),
    ]
    for fam, ov, lbl in tqdm(specs, desc="Main Benchmark"):
        main_rows.extend(tune_and_evaluate_10_seeds(splits, 500, fam, ov, lbl, cache=run_cache))

    main_df = pd.DataFrame(main_rows)
    main_path = RESULTS_DIR / "main_benchmark.csv"
    main_df.to_csv(main_path, index=False)
    sec_main = time.perf_counter() - t0_main
    print(f"-> Saved {main_path.name} in {sec_main:.2f}s ({len(main_df)} rows)")

    # Section 2: Gold A - isolate whether entanglement topology and depth help on tabular data
    print("\n" + "=" * 70)
    print(" SECTION 2: Gold A - Entanglement Topology and Depth Ablation")
    print("=" * 70)
    t0_gold_a = time.perf_counter()
    gold_a_rows = []
    for ent in ["none", "ring", "bipartite"]:
        for L in [1, 2, 4]:
            lbl = f"VQC_{ent}_L{L}"
            ov = {"encoding": "angle", "entanglement": ent, "n_layers": L}
            gold_a_rows.extend(tune_and_evaluate_10_seeds(splits, 500, "Q1-VQC", ov, lbl, cache=run_cache))

    gold_a_df = pd.DataFrame(gold_a_rows)
    gold_a_path = RESULTS_DIR / "gold_a_entanglement.csv"
    gold_a_df.to_csv(gold_a_path, index=False)
    sec_gold_a = time.perf_counter() - t0_gold_a
    print(f"-> Saved {gold_a_path.name} in {sec_gold_a:.2f}s ({len(gold_a_df)} rows)")

    # Section 3: Gold B - 8-qubit angle encoding vs 3-qubit amplitude compression and norm restoration
    print("\n" + "=" * 70)
    print(" SECTION 3: Gold B - Encoding Architecture & Compression Ablation")
    print("=" * 70)
    t0_gold_b = time.perf_counter()
    gold_b_specs = [
        ("8q_Angle_L2", {"encoding": "angle", "entanglement": "ring", "n_layers": 2, "reupload": False}),
        ("8q_Angle_Reupload_L2", {"encoding": "angle", "entanglement": "ring", "n_layers": 2, "reupload": True}),
        ("3q_Amplitude_L5", {"encoding": "amplitude", "entanglement": "ring", "n_layers": 5}),
        ("4q_AmplitudePlusNorm_L4", {"encoding": "amplitude_norm", "entanglement": "ring", "n_layers": 4}),
    ]
    gold_b_rows = []
    for lbl, ov in tqdm(gold_b_specs, desc="Gold B Encoding"):
        gold_b_rows.extend(tune_and_evaluate_10_seeds(splits, 500, "Q1-VQC", ov, lbl, cache=run_cache))

    gold_b_df = pd.DataFrame(gold_b_rows)
    gold_b_path = RESULTS_DIR / "gold_b_encoding.csv"
    gold_b_df.to_csv(gold_b_path, index=False)
    sec_gold_b = time.perf_counter() - t0_gold_b
    print(f"-> Saved {gold_b_path.name} in {sec_gold_b:.2f}s ({len(gold_b_df)} rows)")

    # Section 4: Gold C - sample efficiency sweep across N
    print("\n" + "=" * 70)
    print(" SECTION 4: Gold C - Sample Efficiency Sweep")
    print("=" * 70)
    t0_gold_c = time.perf_counter()
    gold_c_rows = []
    sweep_models = [
        ("Dummy", {}, "Dummy"),
        ("RBF-SVM", {}, "RBF-SVM"),
        ("Tiny-MLP", {}, "Tiny-MLP (41p)"),
        ("Q1-VQC", {"encoding": "angle", "entanglement": "ring", "n_layers": 2}, "Q1-VQC (8q)"),
        ("Q2-QuantumKernel", {}, "Q2-QuantumKernel (8q)"),
    ]
    for N in SAMPLE_SWEEP:
        for fam, ov, lbl in sweep_models:
            gold_c_rows.extend(tune_and_evaluate_10_seeds(splits, N, fam, ov, lbl, cache=run_cache))

    gold_c_df = pd.DataFrame(gold_c_rows)
    gold_c_path = RESULTS_DIR / "gold_c_sweep.csv"
    gold_c_df.to_csv(gold_c_path, index=False)
    sec_gold_c = time.perf_counter() - t0_gold_c
    print(f"-> Saved {gold_c_path.name} in {sec_gold_c:.2f}s ({len(gold_c_df)} rows)")

    # Section 5: Table D - inference-only shot noise degradation on trained weights
    print("\n" + "=" * 70)
    print(" SECTION 5: Table D - Finite Measurement Shot Noise Degradation")
    print("=" * 70)
    t0_table_d = time.perf_counter()
    shot_rows = []
    best_vqc_cfg = get_best_config(run_cache, "Q1-VQC", 500, {"encoding": "angle", "entanglement": "ring", "n_layers": 2})

    for s in SEEDS:
        d = sample_and_preprocess(splits, n_train=500, seed=s)
        _, _, npa, sec, grad_norm, model = train_predict_vqc(
            best_vqc_cfg,
            d,
            seed=s,
            return_model=True,
        )
        Xv = torch.tensor(d["X_val"], dtype=torch.float32)
        Xte = torch.tensor(d["X_test"], dtype=torch.float32)
        for shots in [None, 2000, 500, 100]:
            # Distinct device seeds per set guarantee independent, reproducible shot sampling
            vs = model.predict_shots(Xv, shots=shots, seed=s)
            ts = model.predict_shots(Xte, shots=shots, seed=s + 1000)
            m = evaluate_predictions(d["y_val"], vs, d["y_test"], ts, npa, sec)
            m.update({
                "shots": "analytic" if shots is None else str(shots),
                "seed": s,
                "init_grad_norm": grad_norm,
                "model": "Q1-VQC (8q Ring L=2)",
            })
            shot_rows.append(m)

    table_d_df = pd.DataFrame(shot_rows)
    table_d_path = RESULTS_DIR / "table_d_shots.csv"
    table_d_df.to_csv(table_d_path, index=False)
    sec_table_d = time.perf_counter() - t0_table_d
    print(f"-> Saved {table_d_path.name} in {sec_table_d:.2f}s ({len(table_d_df)} rows)")

    total_sec = time.perf_counter() - t_start
    print("\n" + "=" * 70)
    print(f" ALL BENCHMARKS COMPLETED in {total_sec:.2f}s ({total_sec / 60:.2f} min)")
    print("=" * 70)


if __name__ == "__main__":
    main()
