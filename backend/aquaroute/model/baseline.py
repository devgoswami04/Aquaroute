"""Baseline flood-susceptibility model (Module 4).

RandomForest / XGBoost over the per-segment feature vectors. This is the
*comparison bar* the Flood Response Function must beat — a susceptibility
classifier with no notion of time (no onset/peak/clearance). Reuse scikit-learn /
XGBoost; we only own the feature contract (brief §2.1).

Public API
----------
FEATURES : list[str]           # the model's input columns
train_baseline(X, y, kind)     -> fitted sklearn Pipeline
save_model(model, path) / load_model(path)
feature_importances(model)     -> dict
"""
from __future__ import annotations

from pathlib import Path

# Order matters: the pipeline is trained and queried on exactly these columns.
FEATURES = [
    "length_m", "elevation", "slope", "twi", "depression_depth",
    "upstream_area", "imperviousness", "is_underpass",
    "intensity_max", "total_mm", "duration_wet_h", "antecedent_6h", "peak_time_frac",
]


def _make_classifier(kind: str):
    if kind == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=300, n_jobs=-1, class_weight="balanced", random_state=42,
        )
    if kind in ("xgboost", "xgb"):
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.08,
            subsample=0.8, colsample_bytree=0.8, tree_method="hist",
            eval_metric="logloss", n_jobs=-1, random_state=42,
        )
    raise ValueError(f"unknown baseline kind: {kind!r} (use 'xgboost' or 'rf')")


def train_baseline(X, y, kind: str = "xgboost"):
    """Fit a median-imputed classifier on the FEATURES columns.

    ``y`` may be a Series of the flooded label or the (flooded, depth_proxy)
    DataFrame from ``assemble_training_set`` — we take the ``flooded`` column.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    y_flood = y["flooded"] if hasattr(y, "columns") else y
    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", _make_classifier(kind)),
    ])
    pipe.fit(X[FEATURES], y_flood)
    pipe._aqua_kind = kind  # tag for save/reporting
    return pipe


def feature_importances(model) -> dict:
    clf = model.named_steps["clf"]
    if not hasattr(clf, "feature_importances_"):
        return {}
    imp = dict(zip(FEATURES, (float(v) for v in clf.feature_importances_)))
    return dict(sorted(imp.items(), key=lambda kv: kv[1], reverse=True))


def save_model(model, path: str | Path) -> Path:
    import joblib
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(path: str | Path):
    import joblib
    return joblib.load(path)
