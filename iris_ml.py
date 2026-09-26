"""Built-in Iris machine-learning workflows for StatMate."""

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc


def machine_learning_analysis(data):
    print("\n" + "=" * 60)
    print("MACHINE LEARNING - IRIS CLASSIFICATION")
    print("=" * 60)

    # Features
    X = data[
        [
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)"
        ]
    ]

    # Target
    y = data["target"]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(f"\nTraining observations: {len(X_train)}")
    print(f"Testing observations: {len(X_test)}")

    # Standardize the features
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create and train model
    model = LogisticRegression(max_iter=1000)

    model.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred = model.predict(X_test_scaled)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nModel Accuracy: {accuracy:.4f}")
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    print("\nConfusion Matrix:")
    print(cm)

    # Classification report
    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "setosa",
                "versicolor",
                "virginica"
            ]
        )
    )

    plt.figure(figsize=(7, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=["setosa", "versicolor", "virginica"],
        yticklabels=["setosa", "versicolor", "virginica"]
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Iris Classification Confusion Matrix")

    os.makedirs("figures", exist_ok=True)

    plt.savefig(
        "figures/confusion_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("\nConfusion matrix saved to:")
    print("figures/confusion_matrix.png")

def compare_models(data):
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    features = [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)"
    ]

    X = data[features]
    y = data["target"]

    # --------------------------------------------------
    # Define models
    # --------------------------------------------------

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000))
        ]),

        "K-Nearest Neighbors": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=5))
        ]),

        "Decision Tree": DecisionTreeClassifier(
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
    }

    # --------------------------------------------------
    # Cross-validation setup
    # --------------------------------------------------

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    results = []

    # --------------------------------------------------
    # Evaluate each model
    # --------------------------------------------------

    for name, model in models.items():

        accuracy_scores = cross_val_score(
            model,
            X,
            y,
            cv=cv,
            scoring="accuracy"
        )

        precision_scores = cross_val_score(
            model,
            X,
            y,
            cv=cv,
            scoring="precision_macro"
        )

        recall_scores = cross_val_score(
            model,
            X,
            y,
            cv=cv,
            scoring="recall_macro"
        )

        f1_scores = cross_val_score(
            model,
            X,
            y,
            cv=cv,
            scoring="f1_macro"
        )

        results.append({
            "Model": name,
            "Accuracy": accuracy_scores.mean(),
            "Precision": precision_scores.mean(),
            "Recall": recall_scores.mean(),
            "F1": f1_scores.mean(),
            "Accuracy Std": accuracy_scores.std()
        })

    # --------------------------------------------------
    # Results table
    # --------------------------------------------------

    results_df = pd.DataFrame(results)

    print("\nCross-Validation Results:")
    print("-" * 60)

    display_df = results_df.copy()

    display_df["Accuracy"] = (
        display_df["Accuracy"] * 100
    ).round(2)

    display_df["Precision"] = (
        display_df["Precision"] * 100
    ).round(2)

    display_df["Recall"] = (
        display_df["Recall"] * 100
    ).round(2)

    display_df["F1"] = (
        display_df["F1"] * 100
    ).round(2)

    display_df["Accuracy Std"] = (
        display_df["Accuracy Std"] * 100
    ).round(2)

    print(
        display_df[
            [
                "Model",
                "Accuracy",
                "Precision",
                "Recall",
                "F1",
                "Accuracy Std"
            ]
        ].to_string(index=False)
    )

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------

    os.makedirs("results", exist_ok=True)

    results_df.to_csv(
        "results/model_comparison.csv",
        index=False
    )

    print("\nResults saved to:")
    print("results/model_comparison.csv")

    # --------------------------------------------------
    # Create comparison chart
    # --------------------------------------------------

    chart_data = display_df[
        ["Model", "Accuracy", "Precision", "Recall", "F1"]
    ]

    chart_data.set_index("Model").plot(
        kind="bar",
        figsize=(10, 6)
    )

    plt.ylabel("Score (%)")
    plt.xlabel("Model")
    plt.title("Model Performance Using 5-Fold Cross-Validation")

    plt.ylim(0, 100)

    plt.xticks(rotation=20)

    plt.legend(
        title="Metric"
    )

    plt.tight_layout()

    plt.savefig(
        "figures/model_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("\nPerformance chart saved to:")
    print("figures/model_comparison.png")

def roc_curve_analysis(data):

    print("\n" + "=" * 60)
    print("ROC-AUC ANALYSIS")
    print("=" * 60)

    features = [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)"
    ]

    X = data[features]
    y = data["target"]

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000))
        ]),

        "K-Nearest Neighbors": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=5))
        ]),

        "Decision Tree": DecisionTreeClassifier(
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    class_labels = sorted(y.unique())

    plt.figure(figsize=(10, 7))

    results = []

    for name, model in models.items():

        # Store out-of-fold probabilities
        oof_probabilities = pd.DataFrame(
            index=X.index,
            columns=class_labels,
            dtype=float
        )

        # Generate out-of-fold predictions
        for train_index, test_index in cv.split(X, y):

            X_train = X.iloc[train_index]
            X_test = X.iloc[test_index]

            y_train = y.iloc[train_index]

            model.fit(X_train, y_train)

            probabilities = model.predict_proba(X_test)

            model_classes = model.classes_

            for i, class_label in enumerate(model_classes):

                oof_probabilities.loc[
                    X_test.index,
                    class_label
                ] = probabilities[:, i]

        # Calculate one-vs-rest ROC curves
        # and macro-average AUC
        all_fpr = [0.0, 1.0]
        class_curves = []

        for class_label in class_labels:

            binary_y = (
                y == class_label
            ).astype(int)

            probabilities = oof_probabilities[
                class_label
            ].values

            fpr, tpr, _ = roc_curve(
                binary_y,
                probabilities
            )

            roc_auc = auc(
                fpr,
                tpr
            )

            class_curves.append(
                (fpr, tpr, roc_auc)
            )

            all_fpr.extend(fpr)

        # Create common FPR grid
        mean_fpr = sorted(set(all_fpr))

        mean_tpr = []

        for fpr_value in mean_fpr:

            tpr_values = []

            for fpr, tpr, _ in class_curves:

                tpr_values.append(
                    __import__("numpy").interp(
                        fpr_value,
                        fpr,
                        tpr
                    )
                )

            mean_tpr.append(
                sum(tpr_values) / len(tpr_values)
            )

        macro_auc = auc(
            mean_fpr,
            mean_tpr
        )

        plt.plot(
            mean_fpr,
            mean_tpr,
            label=f"{name} (AUC = {macro_auc:.3f})"
        )

        results.append({
            "Model": name,
            "Macro AUC": macro_auc
        })

    # Random classifier
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title(
        "Macro-Average ROC Curves Using 5-Fold Cross-Validation"
    )

    plt.legend(
        loc="lower right"
    )

    plt.tight_layout()

    os.makedirs("figures", exist_ok=True)

    plt.savefig(
        "figures/roc_curves.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # Save AUC results
    results_df = pd.DataFrame(results)

    os.makedirs("results", exist_ok=True)

    results_df.to_csv(
        "results/roc_auc_results.csv",
        index=False
    )

    print("\nMacro-Average ROC-AUC Results:")
    print("-" * 60)

    display_df = results_df.copy()

    display_df["Macro AUC"] = (
        display_df["Macro AUC"]
        .round(3)
    )

    print(
        display_df.to_string(
            index=False
        )
    )

    print("\nROC curve saved to:")
    print("figures/roc_curves.png")

    print("\nAUC results saved to:")
    print("results/roc_auc_results.csv")

def feature_importance_analysis(data):

    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 60)

    features = [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)"
    ]

    X = data[features]
    y = data["target"]

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X, y)

    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    print("\nFeature Importance:")
    print("-" * 60)

    display_df = importance_df.copy()

    display_df["Importance"] = (
        display_df["Importance"].round(4)
    )

    print(display_df.to_string(index=False))

    os.makedirs("results", exist_ok=True)

    importance_df.to_csv(
        "results/feature_importance.csv",
        index=False
    )

    plt.figure(figsize=(10, 6))

    plt.barh(
        importance_df["Feature"],
        importance_df["Importance"]
    )

    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Random Forest Feature Importance")

    plt.gca().invert_yaxis()

    plt.tight_layout()

    os.makedirs("figures", exist_ok=True)

    plt.savefig(
        "figures/feature_importance.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("\nFeature importance chart saved to:")
    print("figures/feature_importance.png")

    print("\nFeature importance results saved to:")
    print("results/feature_importance.csv")

def train_prediction_model(data):

    features = [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)"
    ]

    X = data[features]
    y = data["target"]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    model.fit(X, y)

    return model, features

def predict_new_flower(data):
    print("\n" + "=" * 60)
    print("PREDICT A NEW IRIS FLOWER")
    print("=" * 60)

    # Features used by the model
    features = [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)"
    ]

    model, features = train_prediction_model(data)

    print("\nEnter the measurements of the flower.")

    try:
        sepal_length = float(
            input("Sepal length (cm): ")
        )

        sepal_width = float(
            input("Sepal width (cm): ")
        )

        petal_length = float(
            input("Petal length (cm): ")
        )

        petal_width = float(
            input("Petal width (cm): ")
        )

    except ValueError:
        print("\nInvalid input. Please enter numbers only.")
        return
        # Validate measurement ranges
    if not (4.0 <= sepal_length <= 8.0):
        print("\n⚠️ Sepal length should be between 4.0 and 8.0 cm.")
        return

    if not (2.0 <= sepal_width <= 5.0):
        print("\n⚠️ Sepal width should be between 2.0 and 5.0 cm.")
        return

    if not (1.0 <= petal_length <= 7.0):
        print("\n⚠️ Petal length should be between 1.0 and 7.0 cm.")
        return

    if not (0.1 <= petal_width <= 3.0):
        print("\n⚠️ Petal width should be between 0.1 and 3.0 cm.")
        return

    # Create a DataFrame for the new flower
    new_flower = pd.DataFrame(
        [[
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]],
        columns=features
    )

    # Scale the new observation
    prediction = model.predict(new_flower)

    predicted_species = prediction[0]

    print("\n" + "-" * 60)
    print(f"Predicted Species: {predicted_species.upper()}")
    print("-" * 60)

