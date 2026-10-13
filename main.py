import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree

warnings.filterwarnings("ignore")
SEED = 42
HERE = Path(__file__).resolve().parent
DISCLAIMER = ("This project is an academic machine-learning experiment using a historical dataset.")


# ------------------------------------------------------------------ data
def load_data():
    path = next((p for p in (HERE / "data" / "arrhythmia.data", HERE / "arrhythmia.data") if p.is_file()), None)
    if path is None:
        raise SystemExit("arrhythmia.data not found. Put it next to main.py (or in a data/ folder).")
    df = pd.read_csv(path, header=None, na_values="?")
    assert df.shape == (452, 280), f"unexpected shape {df.shape}; is this the UCI arrhythmia.data file?"
    X, y = df.iloc[:, :-1], df.iloc[:, -1].astype(int)       # last column = class label (never in X)
    X.columns = [f"f{i}" for i in range(1, 280)]
    return X, y


class DropMostlyMissing(BaseEstimator, TransformerMixin):
    """Drop features whose share of missing values in the TRAINING rows exceeds `threshold`."""
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.keep_ = X.columns[X.isna().mean() <= self.threshold]
        return self

    def transform(self, X):
        return pd.DataFrame(X)[self.keep_]


def make_pipeline(model):
    return Pipeline([("drop", DropMostlyMissing()), ("impute", SimpleImputer(strategy="median")), ("model", model)])


# ------------------------------------------------------------------ main
def main():
    X, y = load_data()
    print(f"Loaded {X.shape[0]} patients x {X.shape[1]} features; classes present: {[int(c) for c in sorted(y.unique())]}")
    print(f"Missing values: {int(X.isna().sum().sum())} cells; most-missing feature f14 = {X['f14'].isna().mean():.1%}")
    print(f"Majority-class baseline (always predict class 1): {(y == 1).mean():.1%}\n")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)
    cv5 = StratifiedKFold(5, shuffle=True, random_state=SEED)

    # --- CART with pruning (grid search on TRAINING rows only)
    grid = {"model__criterion": ["gini", "entropy"], "model__max_depth": [4, 6, 8, None],
            "model__min_samples_leaf": [1, 3, 5], "model__ccp_alpha": [0.0, 0.005, 0.01, 0.02]}
    cart_search = GridSearchCV(make_pipeline(DecisionTreeClassifier(random_state=SEED)), grid, cv=cv5, scoring="accuracy")
    cart_search.fit(X_tr, y_tr)
    cart = cart_search.best_estimator_
    print("CART best settings (chosen by 5-fold CV on training rows):",
          {k.replace("model__", ""): v for k, v in cart_search.best_params_.items()})

    # --- Random Forest (many trees voting)
    rf = make_pipeline(RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)).fit(X_tr, y_tr)

    # --- Evaluation: held-out test set (used once) + 10-fold CV on all rows (paper protocol)
    rows, lines = {}, []
    cv10 = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=SEED)
    for name, model in (("CART (pruned)", cart), ("Random Forest", rf)):
        pred = model.predict(X_te)
        cv_acc = cross_val_score(make_pipeline(model.named_steps["model"]), X, y, cv=cv10, scoring="accuracy").mean()
        rows[name] = {"Test accuracy": accuracy_score(y_te, pred), "Test balanced acc.": balanced_accuracy_score(y_te, pred),
                      "Test macro-F1": f1_score(y_te, pred, average="macro"), "10-fold CV accuracy (all rows)": cv_acc}
    table = pd.DataFrame(rows).T.round(3)
    print("\n" + table.to_string() + "\n")
    print("Reference: the paper reports ~80% (R/rpart, its own protocol) - NOT our result.\n")

    # --- Figures (Random Forest = best model; CART tree shown for interpretability)
    pred_rf = rf.predict(X_te)
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay.from_predictions(y_te, pred_rf, ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Random Forest - confusion matrix (held-out test set)")
    fig.savefig(HERE / "confusion_matrix.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    kept = rf.named_steps["drop"].keep_
    imp = pd.Series(rf.named_steps["model"].feature_importances_, index=kept).nlargest(15)[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(imp.index, imp.values, color="#2b6cb0")
    ax.set_title("Random Forest - top 15 feature importances\n(describes the model, not causation)")
    fig.savefig(HERE / "feature_importance.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(22, 9))
    plot_tree(cart.named_steps["model"], max_depth=3, feature_names=list(cart.named_steps["drop"].keep_),
              class_names=[str(c) for c in cart.classes_], filled=True, rounded=True, fontsize=8, ax=ax)
    ax.set_title("CART decision tree (top 3 levels shown)")
    fig.savefig(HERE / "decision_tree.png", dpi=130, bbox_inches="tight"); plt.close(fig)

    (HERE / "results.txt").write_text(table.to_string() + "\n\n" + DISCLAIMER + "\n", encoding="utf-8")
    print("Saved: results.txt, confusion_matrix.png, feature_importance.png, decision_tree.png\n")
    print(DISCLAIMER)


if __name__ == "__main__":
    main()
