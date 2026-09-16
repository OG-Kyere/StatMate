import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from scipy.stats import pearsonr, shapiro, f_oneway

from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm

from sklearn.datasets import load_iris
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold
)

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)


def load_dataset():
    """Load the Iris dataset from Scikit-learn."""

    iris = load_iris(as_frame=True)

    data = iris.frame

    # Convert target numbers into species names
    data["target"] = data["target"].map(
        dict(enumerate(iris.target_names))
    )

    return data


def explore_data(data):
    """Display basic information about the dataset."""

    print("\n" + "=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)

    print(f"\nNumber of observations: {data.shape[0]}")
    print(f"Number of variables: {data.shape[1]}")

    print("\nVariables:")

    for column in data.columns:
        print(f" - {column}")


def descriptive_statistics(data):
    """Display descriptive statistics."""

    print("\n" + "=" * 60)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 60)

    numerical_data = data.select_dtypes(
        include="number"
    )

    print(
        numerical_data.describe().round(2)
    )


def correlation_analysis(data):
    """Perform Pearson correlation analysis."""

    print("\n" + "=" * 60)
    print("CORRELATION ANALYSIS")
    print("=" * 60)

    numerical_data = data.select_dtypes(
        include="number"
    )

    correlation_matrix = numerical_data.corr()

    print("\nPearson Correlation Matrix:")

    print(
        correlation_matrix.round(3)
    )

    print("\nSignificance Tests:")
    print("-" * 60)

    columns = numerical_data.columns

    for i in range(len(columns)):

        for j in range(i + 1, len(columns)):

            variable_1 = columns[i]
            variable_2 = columns[j]

            x = numerical_data[variable_1]
            y = numerical_data[variable_2]

            r, p_value = pearsonr(x, y)

            if p_value < 0.05:
                significance = "Statistically significant"
            else:
                significance = "Not statistically significant"

            print(
                f"{variable_1} vs {variable_2}"
            )

            print(
                f"  Pearson r = {r:.3f}"
            )

            print(
                f"  p-value   = {p_value:.5f}"
            )

            print(
                f"  Result    = {significance}\n"
            )


def statistical_tests(data):
    """Perform statistical hypothesis tests."""

    print("\n" + "=" * 60)
    print("STATISTICAL HYPOTHESIS TESTING")
    print("=" * 60)

    alpha = 0.05

    # ------------------------------------------------
    # Shapiro-Wilk Normality Test
    # ------------------------------------------------

    print("\n1. SHAPIRO-WILK NORMALITY TEST")
    print("-" * 60)

    species = data["target"].unique()

    for group in species:

        group_data = data[
            data["target"] == group
        ]["petal length (cm)"]

        statistic, p_value = shapiro(
            group_data
        )

        if p_value < alpha:
            result = "Reject H0 → Data may not be normally distributed"
        else:
            result = "Fail to reject H0 → No strong evidence of non-normality"

        print(f"\nSpecies: {group}")
        print(f"Statistic = {statistic:.4f}")
        print(f"p-value   = {p_value:.5f}")
        print(f"Result    = {result}")

    # ------------------------------------------------
    # One-Way ANOVA
    # ------------------------------------------------

    print("\n\n2. ONE-WAY ANOVA")
    print("-" * 60)

    setosa = data[
        data["target"] == "setosa"
    ]["petal length (cm)"]

    versicolor = data[
        data["target"] == "versicolor"
    ]["petal length (cm)"]

    virginica = data[
        data["target"] == "virginica"
    ]["petal length (cm)"]

    statistic, p_value = f_oneway(
        setosa,
        versicolor,
        virginica
    )

    print("\nTest variable: Petal Length")
    print("Groups: Setosa, Versicolor, Virginica")

    print(f"\nF-statistic = {statistic:.4f}")
    print(f"p-value     = {p_value:.10f}")

    print("\nHypotheses:")
    print("H0: The mean petal length is equal across all species.")
    print("H1: At least one species has a different mean petal length.")

    if p_value < alpha:

        print(
            "\nDecision: Reject H0."
        )

        print(
            "Conclusion: There is statistically significant "
            "evidence that mean petal length differs among "
            "the three species."
        )

    else:

        print(
            "\nDecision: Fail to reject H0."
        )

        print(
            "Conclusion: There is not enough statistical evidence "
            "to conclude that the mean petal lengths differ."
        )

def post_hoc_analysis(data):
    print("\n" + "=" * 60)
    print("TUKEY HSD POST-HOC ANALYSIS")
    print("=" * 60)

    # Variable being compared
    variable = "petal length (cm)"

    # Perform Tukey HSD
    tukey = pairwise_tukeyhsd(
        endog=data[variable],
        groups=data["target"],
        alpha=0.05
    )

    print("\nPairwise comparisons:")
    print(tukey)

    print("\nInterpretation:")

    # Extract Tukey results
    results = pd.DataFrame(
        data=tukey._results_table.data[1:],
        columns=tukey._results_table.data[0]
    )

    for _, row in results.iterrows():

        group1 = row["group1"]
        group2 = row["group2"]
        p_value = float(row["p-adj"])
        reject = row["reject"]

        if reject:
            print(
                f"- {group1} vs {group2}: "
                f"Significant difference (p = {p_value:.4f})"
            )
        else:
            print(
                f"- {group1} vs {group2}: "
                f"No significant difference (p = {p_value:.4f})"
            )

def regression_analysis(data):
    print("\n" + "=" * 60)
    print("MULTIPLE LINEAR REGRESSION")
    print("=" * 60)

    # Define dependent variable
    y = data["petal length (cm)"]

    # Define independent variables
    X = data[
        [
            "sepal length (cm)",
            "sepal width (cm)",
            "petal width (cm)"
        ]
    ]

    # Add intercept
    X = sm.add_constant(X)

    # Fit regression model
    model = sm.OLS(y, X).fit()

    # Display regression results
    print("\nRegression Summary:")
    print(model.summary())

    # Model statistics
    print("\nKey Results:")
    print(f"R-squared: {model.rsquared:.4f}")
    print(f"Adjusted R-squared: {model.rsquared_adj:.4f}")
    print(f"F-statistic p-value: {model.f_pvalue:.6f}")

    # Coefficients
    print("\nCoefficients:")

    for variable in model.params.index:
        coefficient = model.params[variable]
        p_value = model.pvalues[variable]

        print(
            f"{variable}: "
            f"coefficient = {coefficient:.4f}, "
            f"p-value = {p_value:.6f}"
        )

    # Interpretation
    print("\nInterpretation:")

    if model.f_pvalue < 0.05:
        print(
            "The overall regression model is statistically significant "
            "at the 5% significance level."
        )
    else:
        print(
            "The overall regression model is not statistically significant "
            "at the 5% significance level."
        )

    print(
        f"The model explains approximately "
        f"{model.rsquared * 100:.2f}% of the variation "
        f"in petal length."
    )

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

def create_visualizations(data):
    """Create automatic visualizations."""

    print("\n" + "=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)

    os.makedirs(
        "figures",
        exist_ok=True
    )

    numerical_data = data.select_dtypes(
        include="number"
    )

    # Histograms
    numerical_data.hist(
        figsize=(10, 8),
        bins=15
    )

    plt.suptitle(
        "Distribution of Numerical Variables"
    )

    plt.tight_layout()

    plt.savefig(
        "figures/distributions.png"
    )

    plt.close()

    # Boxplots
    plt.figure(figsize=(10, 6))

    sns.boxplot(
        data=numerical_data
    )

    plt.title(
        "Boxplots of Numerical Variables"
    )

    plt.xticks(rotation=30)

    plt.tight_layout()

    plt.savefig(
        "figures/boxplots.png"
    )

    plt.close()

    # Correlation heatmap
    plt.figure(figsize=(8, 6))

    correlation = numerical_data.corr()

    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )

    plt.title(
        "Correlation Heatmap"
    )

    plt.tight_layout()

    plt.savefig(
        "figures/correlation_heatmap.png"
    )

    plt.close()

    # Petal length by species
    plt.figure(figsize=(8, 6))

    sns.boxplot(
        data=data,
        x="target",
        y="petal length (cm)"
    )

    plt.title(
        "Petal Length by Iris Species"
    )

    plt.xlabel("Species")
    plt.ylabel("Petal Length (cm)")

    plt.tight_layout()

    plt.savefig(
        "figures/petal_length_by_species.png"
    )

    plt.close()

    print("\nVisualizations created successfully!")

def generate_report(data):
    print("\n" + "=" * 60)
    print("GENERATING STATMATE REPORT")
    print("=" * 60)

    os.makedirs("reports", exist_ok=True)

    report_path = "reports/statmate_report.txt"

    with open(report_path, "w", encoding="utf-8") as report:

        report.write("=" * 70 + "\n")
        report.write("STATMATE — STATISTICAL ANALYSIS REPORT\n")
        report.write("=" * 70 + "\n\n")

        # --------------------------------------------------
        # DATASET INFORMATION
        # --------------------------------------------------

        report.write("1. DATASET INFORMATION\n")
        report.write("-" * 70 + "\n")

        report.write(
            f"Number of observations: {data.shape[0]}\n"
        )

        report.write(
            f"Number of variables: {data.shape[1]}\n\n"
        )

        report.write("Variables:\n")

        for column in data.columns:
            report.write(f"- {column}\n")

        # --------------------------------------------------
        # DESCRIPTIVE STATISTICS
        # --------------------------------------------------

        report.write("\n\n2. DESCRIPTIVE STATISTICS\n")
        report.write("-" * 70 + "\n")

        report.write(
            data.describe().to_string()
        )

        # --------------------------------------------------
        # CORRELATION ANALYSIS
        # --------------------------------------------------

        report.write("\n\n3. CORRELATION MATRIX\n")
        report.write("-" * 70 + "\n")

        numerical_data = data.drop(
            columns=["target"]
        ).select_dtypes(include="number")

        report.write(
            numerical_data.corr().round(3).to_string()
        )

        # --------------------------------------------------
        # MACHINE LEARNING RESULTS
        # --------------------------------------------------

        report.write("\n\n4. MACHINE LEARNING MODEL RESULTS\n")
        report.write("-" * 70 + "\n")

        model_results_path = (
            "results/model_comparison.csv"
        )

        if os.path.exists(model_results_path):

            model_results = pd.read_csv(
                model_results_path
            )

            display_results = model_results.copy()

            for column in [
                "Accuracy",
                "Precision",
                "Recall",
                "F1",
                "Accuracy Std"
            ]:

                if column in display_results.columns:
                    display_results[column] = (
                        display_results[column] * 100
                    ).round(2)

            report.write(
                display_results.to_string(
                    index=False
                )
            )

        else:

            report.write(
                "Model comparison results have not "
                "been generated yet.\n"
            )

        # --------------------------------------------------
        # ROC-AUC RESULTS
        # --------------------------------------------------

        report.write("\n\n5. ROC-AUC RESULTS\n")
        report.write("-" * 70 + "\n")

        roc_results_path = (
            "results/roc_auc_results.csv"
        )

        if os.path.exists(roc_results_path):

            roc_results = pd.read_csv(
                roc_results_path
            )

            roc_results["Macro AUC"] = (
                roc_results["Macro AUC"]
                .round(3)
            )

            report.write(
                roc_results.to_string(
                    index=False
                )
            )

        else:

            report.write(
                "ROC-AUC results have not "
                "been generated yet.\n"
            )

        # --------------------------------------------------
        # FEATURE IMPORTANCE
        # --------------------------------------------------

        report.write("\n\n6. FEATURE IMPORTANCE\n")
        report.write("-" * 70 + "\n")

        importance_path = (
            "results/feature_importance.csv"
        )

        if os.path.exists(importance_path):

            importance_results = pd.read_csv(
                importance_path
            )

            importance_results["Importance"] = (
                importance_results["Importance"]
                .round(4)
            )

            report.write(
                importance_results.to_string(
                    index=False
                )
            )

        else:

            report.write(
                "Feature importance results have "
                "not been generated yet.\n"
            )

        # --------------------------------------------------
        # REPORT FOOTER
        # --------------------------------------------------

        report.write("\n\n")
        report.write("=" * 70 + "\n")
        report.write("END OF STATMATE REPORT\n")
        report.write("=" * 70 + "\n")

    print("\nReport successfully generated!")
    print("\nReport saved to:")
    print(report_path)

def generate_html_report(data):

    print("\n" + "=" * 60)
    print("GENERATING HTML REPORT")
    print("=" * 60)

    os.makedirs("reports", exist_ok=True)

    report_path = "reports/statmate_report.html"

    # Dataset information
    observations = data.shape[0]
    variables = data.shape[1]

    # Descriptive statistics
    descriptive = data.describe().round(3).to_html(
        classes="data-table"
    )

    # Correlation matrix
    numerical_data = data.drop(
        columns=["target"]
    ).select_dtypes(include="number")

    correlation = numerical_data.corr().round(3).to_html(
        classes="data-table"
    )

    # Model comparison
    model_results_path = "results/model_comparison.csv"

    if os.path.exists(model_results_path):

        model_results = pd.read_csv(
            model_results_path
        )

        display_models = model_results.copy()

        for column in [
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "Accuracy Std"
        ]:

            if column in display_models.columns:
                display_models[column] = (
                    display_models[column] * 100
                ).round(2)

        model_table = display_models.to_html(
            index=False,
            classes="data-table"
        )

    else:

        model_table = (
            "<p>Model comparison results "
            "are not available.</p>"
        )

    # ROC-AUC
    roc_results_path = "results/roc_auc_results.csv"

    if os.path.exists(roc_results_path):

        roc_results = pd.read_csv(
            roc_results_path
        )

        roc_results["Macro AUC"] = (
            roc_results["Macro AUC"]
            .round(3)
        )

        roc_table = roc_results.to_html(
            index=False,
            classes="data-table"
        )

    else:

        roc_table = (
            "<p>ROC-AUC results "
            "are not available.</p>"
        )

    # Feature importance
    importance_path = (
        "results/feature_importance.csv"
    )

    if os.path.exists(importance_path):

        importance_results = pd.read_csv(
            importance_path
        )

        importance_results["Importance"] = (
            importance_results["Importance"]
            .round(4)
        )

        importance_table = (
            importance_results.to_html(
                index=False,
                classes="data-table"
            )
        )

    else:

        importance_table = (
            "<p>Feature importance results "
            "are not available.</p>"
        )

    # Create HTML document
    html = f"""
<!DOCTYPE html>

<html lang="en">

<head>

<meta charset="UTF-8">

<meta name="viewport"
      content="width=device-width, initial-scale=1.0">

<title>StatMate Statistical Analysis Report</title>

<style>

body {{
    font-family: Arial, sans-serif;
    margin: 40px;
    background-color: #f5f7fa;
    color: #222;
}}

.container {{
    max-width: 1100px;
    margin: auto;
    background: white;
    padding: 40px;
    border-radius: 10px;
}}

h1 {{
    text-align: center;
    margin-bottom: 10px;
}}

.subtitle {{
    text-align: center;
    color: #666;
    margin-bottom: 40px;
}}

h2 {{
    border-bottom: 2px solid #ddd;
    padding-bottom: 8px;
    margin-top: 40px;
}}

.data-table {{
    border-collapse: collapse;
    width: 100%;
    margin-top: 15px;
    margin-bottom: 30px;
}}

.data-table th,
.data-table td {{
    border: 1px solid #ddd;
    padding: 10px;
    text-align: center;
}}

.data-table th {{
    background-color: #eee;
}}

.info-box {{
    padding: 20px;
    margin: 20px 0;
    border-radius: 8px;
    background-color: #f0f2f5;
}}

img {{
    max-width: 100%;
    display: block;
    margin: 20px auto;
}}

.footer {{
    text-align: center;
    margin-top: 50px;
    color: #777;
    font-size: 14px;
}}

</style>

</head>

<body>

<div class="container">

<h1>StatMate</h1>

<div class="subtitle">
Statistical Analysis Report
</div>

<h2>1. Dataset Information</h2>

<div class="info-box">

<p>
<strong>Observations:</strong>
{observations}
</p>

<p>
<strong>Variables:</strong>
{variables}
</p>

</div>

<h2>2. Descriptive Statistics</h2>

{descriptive}

<h2>3. Correlation Matrix</h2>

{correlation}

<h2>4. Machine Learning Model Comparison</h2>

{model_table}

<img src="../figures/model_comparison.png"
     alt="Model Comparison">

<h2>5. ROC-AUC Analysis</h2>

{roc_table}

<img src="../figures/roc_curves.png"
     alt="ROC Curves">

<h2>6. Feature Importance</h2>

{importance_table}

<img src="../figures/feature_importance.png"
     alt="Feature Importance">

<div class="footer">

<p>
Generated automatically by StatMate
</p>

<p>
Statistical Analysis Assistant
</p>

</div>

</div>

</body>

</html>
"""

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as report:

        report.write(html)

    print("\nHTML report successfully generated!")

    print("\nReport saved to:")
    print(report_path)

def main():

    # Load the dataset once
    data = load_dataset()

    while True:

        print("\n" + "=" * 60)
        print("                    STATMATE")
        print("          Statistical Analysis Assistant")
        print("=" * 60)

        print("\n1. Explore Dataset")
        print("2. Descriptive Statistics")
        print("3. Correlation Analysis")
        print("4. Statistical Tests")
        print("5. Post-Hoc Analysis")
        print("6. Regression Analysis")
        print("7. Machine Learning")
        print("8. Model Comparison")
        print("9. ROC-AUC Analysis")
        print("10. Feature Importance")
        print("11. Predict New Flower")
        print("12. Run Complete Analysis")
        print("13. Generate Statistical Report")
        print("14. Generate HTML Report")
        print("0. Exit")

        choice = input("\nEnter your choice: ").strip()

        if choice == "1":
            explore_data(data)

        elif choice == "2":
            descriptive_statistics(data)

        elif choice == "3":
            correlation_analysis(data)

        elif choice == "4":
            statistical_tests(data)

        elif choice == "5":
            post_hoc_analysis(data)

        elif choice == "6":
            regression_analysis(data)

        elif choice == "7":
            machine_learning_analysis(data)

        elif choice == "8":
            compare_models(data)

        elif choice == "9":
            roc_curve_analysis(data)

        elif choice == "10":
            feature_importance_analysis(data)

        elif choice == "11":
            predict_new_flower(data)

        elif choice == "12":
            explore_data(data)
            descriptive_statistics(data)
            correlation_analysis(data)
            statistical_tests(data)
            post_hoc_analysis(data)
            regression_analysis(data)
            machine_learning_analysis(data)
            compare_models(data)
            roc_curve_analysis(data)
            feature_importance_analysis(data)
            create_visualizations(data)
        elif choice == "13":
            generate_report(data)
        elif choice == "14":
            generate_html_report(data)
        elif choice == "0":
            print("\nThank you for using StatMate!")
            print("Goodbye 👋")
            break

        else:
            print("\n⚠️ Invalid choice. Please enter a number from 0 to 14.")



if __name__ == "__main__":
    main()