"""Reusable classification workflow for custom StatMate datasets."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def eligible_targets(data):
    """Return columns suitable for a basic classification target."""
    targets = []
    for column in data.columns:
        values = data[column].dropna()
        n_classes = values.nunique()
        if 2 <= n_classes <= min(20, max(2, len(values) // 2)):
            targets.append(column)
    return targets


def prepare_classification_data(data, target, predictors):
    """Validate target/predictors and return X, y plus feature type lists."""
    if target not in data.columns:
        raise ValueError("The selected target does not exist.")
    if not predictors:
        raise ValueError("Select at least one predictor.")
    if target in predictors:
        raise ValueError("The target cannot also be a predictor.")
    missing = [column for column in predictors if column not in data.columns]
    if missing:
        raise ValueError("Unknown predictor(s): " + ", ".join(map(str, missing)))

    selected = data[[target, *predictors]].replace([np.inf, -np.inf], np.nan)
    selected = selected.dropna(subset=[target])
    if selected.empty:
        raise ValueError("No observations remain after removing missing target values.")

    y = selected[target]
    counts = y.value_counts()
    if len(counts) < 2:
        raise ValueError("Classification requires at least two target classes.")
    if counts.min() < 2:
        raise ValueError("Every target class needs at least two observations.")

    X = selected[predictors].copy()
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = [column for column in predictors if column not in numeric]

    unusable = [column for column in predictors if X[column].dropna().nunique() < 2]
    if unusable:
        raise ValueError(
            "Predictors must contain at least two observed values. Check: "
            + ", ".join(map(str, unusable))
        )
    return X, y, numeric, categorical


def build_preprocessor(numeric, categorical):
    """Create leakage-safe preprocessing fitted inside each CV fold."""
    transformers = []
    if numeric:
        transformers.append((
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            numeric,
        ))
    if categorical:
        transformers.append((
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical,
        ))
    return ColumnTransformer(transformers=transformers)


def evaluate_classifiers(data, target, predictors, random_state=42):
    """Compare four classifiers with stratified cross-validation."""
    X, y, numeric, categorical = prepare_classification_data(
        data, target, predictors
    )
    min_class = int(y.value_counts().min())
    n_splits = min(5, min_class)
    if n_splits < 2:
        raise ValueError("Not enough observations per class for cross-validation.")

    cv = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    estimators = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=max(1, min(5, len(y) - len(y) // n_splits))
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=random_state
        ),
    }
    # zero_division=0 makes macro metrics deterministic when a model does not
    # predict one of the classes in a validation fold (common in small or
    # imbalanced datasets) while keeping the limitation visible in the score.
    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(
            precision_score, average="macro", zero_division=0
        ),
        "recall": make_scorer(
            recall_score, average="macro", zero_division=0
        ),
        "f1": make_scorer(
            f1_score, average="macro", zero_division=0
        ),
    }

    rows = []
    for name, estimator in estimators.items():
        model = Pipeline([
            ("preprocess", build_preprocessor(numeric, categorical)),
            ("model", estimator),
        ])
        scores = cross_validate(
            model, X, y, cv=cv, scoring=scoring, error_score="raise"
        )
        rows.append({
            "Model": name,
            "Accuracy": scores["test_accuracy"].mean(),
            "Precision": scores["test_precision"].mean(),
            "Recall": scores["test_recall"].mean(),
            "F1": scores["test_f1"].mean(),
            "Accuracy Std": scores["test_accuracy"].std(),
            "CV Folds": n_splits,
        })
    return pd.DataFrame(rows)


def _choose(prompt, columns, allow_many=False):
    print()
    for index, column in enumerate(columns, start=1):
        print(f"{index}. {column}")
    answer = input(prompt).strip()
    if not answer:
        return None
    try:
        indices = [int(value.strip()) for value in answer.split(",")]
    except ValueError:
        return None
    if not allow_many and len(indices) != 1:
        return None
    if not indices or any(index < 1 or index > len(columns) for index in indices):
        return None
    chosen = [columns[index - 1] for index in indices]
    return chosen if allow_many else chosen[0]


def custom_classification_analysis(data):
    """Interactive custom-dataset classification and model comparison."""
    print("\n" + "=" * 60)
    print("CUSTOM DATASET CLASSIFICATION")
    print("=" * 60)

    targets = eligible_targets(data)
    if not targets:
        print("\nNo suitable classification target was detected.")
        return None

    target = _choose("\nChoose target number (Enter to cancel): ", targets)
    if target is None:
        print("\nClassification cancelled.")
        return None

    predictors = [column for column in data.columns if column != target]
    chosen = _choose(
        "\nChoose predictor number(s), separated by commas (Enter to cancel): ",
        predictors,
        allow_many=True,
    )
    if not chosen:
        print("\nClassification cancelled.")
        return None

    try:
        results = evaluate_classifiers(data, target, chosen)
    except ValueError as error:
        print(f"\nWarning: {error}")
        return None

    display = results.copy()
    for column in ["Accuracy", "Precision", "Recall", "F1", "Accuracy Std"]:
        display[column] = (display[column] * 100).round(2)

    print(f"\nTarget: {target}")
    print("Predictors: " + ", ".join(map(str, chosen)))
    print("\nStratified cross-validation results (%):")
    print(display.to_string(index=False))

    Path("results").mkdir(exist_ok=True)
    results.to_csv("results/custom_classification_results.csv", index=False)
    print("\nResults saved to: results/custom_classification_results.csv")
    return results
