import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


class TinyMLP(nn.Module):
    # hidden_dim=4 gives 41 trainable parameters, matching our 8-qubit L=2 quantum circuit (34 params)
    def __init__(self, in_features: int = 8, hidden_dim: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def count_torch_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fit_predict_classical(
    model_name: str,
    params: dict,
    data: dict,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, X_test = data["X_val"], data["X_test"]
    t0 = time.perf_counter()

    if model_name == "Dummy":
        # Sets the 90.84% majority-noise accuracy baseline and 0.0916 PR-AUC floor
        clf = DummyClassifier(strategy="most_frequent")
        clf.fit(X_train, y_train)
        val_scores = clf.predict_proba(X_val)[:, 1]
        test_scores = clf.predict_proba(X_test)[:, 1]
        n_params = 0

    elif model_name == "LogReg":
        clf = LogisticRegression(
            C=params["C"],
            class_weight=params["class_weight"],
            max_iter=1000,
            random_state=seed,
        )
        clf.fit(X_train, y_train)
        val_scores = clf.decision_function(X_val)
        test_scores = clf.decision_function(X_test)
        n_params = clf.coef_.size + clf.intercept_.size

    elif model_name == "RBF-SVM":
        clf = SVC(
            kernel="rbf",
            C=params["C"],
            gamma=params["gamma"],
            class_weight=params["class_weight"],
            random_state=seed,
        )
        clf.fit(X_train, y_train)
        val_scores = clf.decision_function(X_val)
        test_scores = clf.decision_function(X_test)
        n_params = clf.support_vectors_.size

    elif model_name == "RandomForest":
        clf = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            class_weight=params["class_weight"],
            random_state=seed,
            n_jobs=-1,
        )
        clf.fit(X_train, y_train)
        val_scores = clf.predict_proba(X_val)[:, 1]
        test_scores = clf.predict_proba(X_test)[:, 1]
        n_params = sum(est.tree_.node_count for est in clf.estimators_)

    elif model_name == "Tiny-MLP":
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = TinyMLP(in_features=X_train.shape[1], hidden_dim=params.get("hidden_dim", 4))
        n_params = count_torch_params(model)

        pos_ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        pos_weight = torch.tensor([pos_ratio * params["pos_weight_mult"]], dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"]
        )

        Xt = torch.tensor(X_train, dtype=torch.float32)
        yt = torch.tensor(y_train, dtype=torch.float32)
        Xv = torch.tensor(X_val, dtype=torch.float32)
        yv = torch.tensor(data["y_val"], dtype=torch.float32)
        Xte = torch.tensor(X_test, dtype=torch.float32)

        best_val_loss = float("inf")
        best_state = None
        patience_left = 15

        for _ in range(120):
            model.train()
            perm = torch.randperm(len(Xt))
            for i in range(0, len(Xt), params["batch_size"]):
                idx = perm[i : i + params["batch_size"]]
                optimizer.zero_grad()
                loss = criterion(model(Xt[idx]), yt[idx])
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                v_loss = criterion(model(Xv), yv).item()
            if v_loss < best_val_loss - 1e-5:
                best_val_loss = v_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_left = 15
            else:
                patience_left -= 1
                if patience_left == 0:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            val_scores = model(Xv).numpy()
            test_scores = model(Xte).numpy()
    else:
        raise ValueError(f"Unknown classical model: {model_name}")

    fit_sec = time.perf_counter() - t0
    return val_scores, test_scores, n_params, fit_sec
