import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from ucimlrepo import fetch_ucirepo

from src.config import DATA_DIR, MASTER_SPLIT_SEED, N_TEST, N_VAL

FEATURE_NAMES = [
    "IP_Mean", "IP_Std", "IP_Kurtosis", "IP_Skewness",
    "DM_Mean", "DM_Std", "DM_Kurtosis", "DM_Skewness",
]


def load_htru2_raw() -> tuple[np.ndarray, np.ndarray]:
    # Cache locally so repeated runs do not depend on UCI network availability
    cache_path = DATA_DIR / "htru2_cached.csv"
    if cache_path.exists():
        df = pd.read_csv(cache_path)
    else:
        dataset = fetch_ucirepo(id=372)  # Official UCI HTRU2 record (CC BY 4.0)
        X_df = dataset.data.features.copy()
        X_df.columns = FEATURE_NAMES
        y_ser = dataset.data.targets.iloc[:, 0].astype(int)
        df = X_df.copy()
        df["target"] = y_ser.values
        df.to_csv(cache_path, index=False)

    X = df[FEATURE_NAMES].to_numpy(dtype=np.float64)
    y = df["target"].to_numpy(dtype=np.int64)
    return X, y


def get_locked_splits() -> dict:
    # Lock Test (3000) and Val (1500) once so every sample-size sweep evaluates against identical holdouts
    X, y = load_htru2_raw()
    X_rem, X_test, y_rem, y_test = train_test_split(
        X, y, test_size=N_TEST, stratify=y, random_state=MASTER_SPLIT_SEED
    )
    X_pool, X_val, y_pool, y_val = train_test_split(
        X_rem, y_rem, test_size=N_VAL, stratify=y_rem, random_state=MASTER_SPLIT_SEED
    )
    return {
        "X_pool": X_pool, "y_pool": y_pool,
        "X_val": X_val,   "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
    }


def sample_and_preprocess(splits: dict, n_train: int, seed: int) -> dict:
    X_train_raw, _, y_train, _ = train_test_split(
        splits["X_pool"], splits["y_pool"],
        train_size=n_train, stratify=splits["y_pool"], random_state=seed
    )

    # Skewness and kurtosis have extreme tails that crush MinMax angles near zero;
    # Yeo-Johnson tames outliers and is fit strictly on X_train to prevent test leakage
    pt = PowerTransformer(method="yeo-johnson", standardize=True)
    X_train = pt.fit_transform(X_train_raw)
    X_val = pt.transform(splits["X_val"])
    X_test = pt.transform(splits["X_test"])

    return {
        "X_train": X_train.astype(np.float32),
        "y_train": y_train,
        "X_val": X_val.astype(np.float32),
        "y_val": splits["y_val"],
        "X_test": X_test.astype(np.float32),
        "y_test": splits["y_test"],
    }
