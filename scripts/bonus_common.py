import glob
import importlib
import inspect
import os
import pkgutil
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.getcwd())
PRED_DIR = os.path.join("results", "full", "predictions")

df_raw = pd.read_csv(os.path.join("data_cache", "htru2.csv"))
target_col = "target" if "target" in df_raw.columns else df_raw.columns[-1]
X_ALL = df_raw.drop(columns=[target_col]).values.astype(np.float64)
Y_ALL = df_raw[target_col].values.astype(int)

def _unpack_candidate(obj, n_train, y_te_ref, y_va_ref):
    n_te = len(y_te_ref)
    if isinstance(obj, dict):
        d = {k.lower(): v for k, v in obj.items()}
        if "y_test" in d and "x_test" in d and "x_train" in d and "y_train" in d:
            yt = np.asarray(d["y_test"]).ravel().astype(int)
            if len(yt) == n_te and np.array_equal(yt, y_te_ref):
                xva = np.asarray(d.get("x_val", d["x_train"][:50]))
                yva = np.asarray(d.get("y_val", d["y_train"][:50])).ravel().astype(int)
                return np.asarray(d["x_train"]), np.asarray(d["y_train"]).ravel().astype(int), xva, yva, np.asarray(d["x_test"]), yt
    elif hasattr(obj, "__dict__"):
        return _unpack_candidate(vars(obj), n_train, y_te_ref, y_va_ref)
    elif isinstance(obj, (tuple, list)) and len(obj) >= 4:
        arrs = [np.asarray(a) for a in obj]
        x2d = [a for a in arrs if a.ndim == 2]
        y1d = [a.ravel().astype(int) for a in arrs if a.ndim == 1]
        if len(x2d) >= 2 and len(y1d) >= 2:
            if len(y1d[-1]) == n_te and np.array_equal(y1d[-1], y_te_ref) and len(x2d[0]) == n_train:
                xva = x2d[1] if len(x2d) >= 3 else x2d[0][:50]
                yva = y1d[1] if len(y1d) >= 3 else y1d[0][:50]
                return x2d[0], y1d[0], xva, yva, x2d[-1], y1d[-1]
    return None
_SPLIT_MODE_LOGGED = False

def get_aligned_split(n_train, seed):
    global _SPLIT_MODE_LOGGED
    ref_files = sorted(glob.glob(os.path.join(PRED_DIR, f"Q1-VQC__*Bipartite*__N{n_train}__seed{seed}.npz")))
    ref_d = np.load(ref_files[0], allow_pickle=True)
    y_te_ref = np.asarray(ref_d["y_test"]).ravel().astype(int)
    y_va_ref = np.asarray(ref_d["y_val"]).ravel().astype(int) if "y_val" in ref_d.keys() else None
    n_te = len(y_te_ref)
    n_va = len(y_va_ref) if y_va_ref is not None else max(int(0.2 * n_train), 50)

    if os.path.isdir("src"):
        for _, mod_name, _ in pkgutil.iter_modules(["src"]):
            try:
                mod = importlib.import_module(f"src.{mod_name}")
            except Exception:
                continue
            for _, fn in inspect.getmembers(mod, inspect.isfunction):
                try:
                    sig = inspect.signature(fn)
                except Exception:
                    continue
                kw = {}
                ok = True
                for p, param in sig.parameters.items():
                    pl = p.lower()
                    if pl in ("n_train", "train_size", "n_samples", "sample_size", "subset_size", "n"):
                        kw[p] = n_train
                    elif pl in ("seed", "random_state"):
                        kw[p] = seed
                    elif pl in ("df", "data"):
                        kw[p] = df_raw
                    elif pl == "x":
                        kw[p] = X_ALL
                    elif pl == "y":
                        kw[p] = Y_ALL
                    elif param.default is inspect.Parameter.empty:
                        ok = False
                        break
                if ok and kw:
                    try:
                        res = _unpack_candidate(fn(**kw), n_train, y_te_ref, y_va_ref)
                        if res is not None:
                            return res
                    except Exception:
                        pass

    for rs in [seed, 42, 0, 1, 2026, seed + 1000]:
        for hold_sz in [n_te + n_va, n_te]:
            if hold_sz >= len(Y_ALL) - n_train:
                continue
            X_pool, X_hold, y_pool, y_hold = train_test_split(X_ALL, Y_ALL, test_size=hold_sz, random_state=rs, stratify=Y_ALL)
            if hold_sz == n_te and np.array_equal(y_hold, y_te_ref):
                X_tr, X_va, y_tr, y_va = train_test_split(X_pool, y_pool, train_size=n_train, test_size=n_va, random_state=seed, stratify=y_pool)
                return X_tr, y_tr, X_va, y_va, X_hold, y_hold
            if hold_sz == n_te + n_va:
                for rs2 in [rs, seed, 42, 0]:
                    X_va, X_te, y_va, y_te = train_test_split(X_hold, y_hold, test_size=n_te, random_state=rs2, stratify=y_hold)
                    if np.array_equal(y_te, y_te_ref):
                        X_tr, _, y_tr, _ = train_test_split(X_pool, y_pool, train_size=n_train, random_state=seed, stratify=y_pool)
                        return X_tr, y_tr, X_va, y_va, X_te, y_te

    if not _SPLIT_MODE_LOGGED:
        print(f"[bonus_common] Using exact-cardinality stratified split (N_tr={n_train}, N_va={n_va}, N_te={n_te})")
        _SPLIT_MODE_LOGGED = True
    X_pool, X_hold, y_pool, y_hold = train_test_split(X_ALL, Y_ALL, test_size=n_te + n_va, random_state=seed, stratify=Y_ALL)
    X_va, X_te, y_va, y_te = train_test_split(X_hold, y_hold, test_size=n_te, random_state=seed, stratify=y_hold)
    X_tr, _, y_tr, _ = train_test_split(X_pool, y_pool, train_size=n_train, random_state=seed, stratify=y_pool)
    return X_tr, y_tr, X_va, y_va, X_te, y_te
