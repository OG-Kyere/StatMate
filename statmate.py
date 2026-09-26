import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from scipy.stats import pearsonr, shapiro, f_oneway, levene, ttest_ind, mannwhitneyu, kruskal
from statsmodels.stats.oneway import anova_oneway
from statsmodels.stats.multitest import multipletests
from itertools import combinations

from statsmodels.stats.multicomp import pairwise_tukeyhsd

from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
import statsmodels.formula.api as smf
import statsmodels.api as sm

from custom_ml import (
    custom_classification_analysis,
    custom_model_comparison,
    custom_roc_auc_analysis,
    custom_feature_importance_analysis,
    custom_prediction,
)

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


from data_loader import (
    load_dataset as _load_dataset,
    validate_dataset as _validate_dataset,
    print_dataset_profile as _print_dataset_profile,
    load_custom_dataset as _load_custom_dataset,
)


def load_dataset():
    """Compatibility wrapper for the built-in dataset loader."""
    return _load_dataset()


def validate_dataset(data):
    """Compatibility wrapper for dataset validation."""
    return _validate_dataset(data)


def print_dataset_profile(data, preview_rows=5):
    """Compatibility wrapper for dataset profiling."""
    return _print_dataset_profile(data, preview_rows=preview_rows)


def load_custom_dataset():
    """Compatibility wrapper for interactive custom dataset loading."""
    return _load_custom_dataset()


def explore_data(data):
    """Display basic information about the dataset."""

    print("\n" + "=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)

    print_dataset_profile(data)


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


from statistics_analysis import (
    calculate_correlations as _calculate_correlations,
    correlation_analysis as _correlation_analysis,
    select_hypothesis_variables as _select_hypothesis_variables,
    compare_groups as _compare_groups,
    custom_hypothesis_analysis as _custom_hypothesis_analysis,
)


def calculate_correlations(data):
    return _calculate_correlations(data)


def correlation_analysis(data):
    return _correlation_analysis(data)


def select_hypothesis_variables(data):
    return _select_hypothesis_variables(data)


def compare_groups(data, response, grouping):
    return _compare_groups(data, response, grouping)


def custom_hypothesis_analysis(data, post_hoc=False):
    return _custom_hypothesis_analysis(data, post_hoc=post_hoc)


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

from regression_analysis import (
    print_regression_results as _print_regression_results,
    fit_linear_regression as _fit_linear_regression,
    calculate_vif as _calculate_vif,
    print_vif_guidance as _print_vif_guidance,
    has_recommended_sample_size as _has_recommended_sample_size,
    print_sample_size_guidance as _print_sample_size_guidance,
    calculate_regression_diagnostics as _calculate_regression_diagnostics,
    save_custom_regression_diagnostics as _save_custom_regression_diagnostics,
    save_custom_regression_results as _save_custom_regression_results,
    select_custom_regression_variables as _select_custom_regression_variables,
    custom_regression_analysis as _custom_regression_analysis,
    regression_analysis as _regression_analysis
)


def print_regression_results(model, outcome_name):
    return _print_regression_results(model, outcome_name)

def fit_linear_regression(data, outcome, predictors):
    return _fit_linear_regression(data, outcome, predictors)

def calculate_vif(data, outcome, predictors):
    return _calculate_vif(data, outcome, predictors)

def print_vif_guidance(vif_results):
    return _print_vif_guidance(vif_results)

def has_recommended_sample_size(complete_rows, predictor_count):
    return _has_recommended_sample_size(complete_rows, predictor_count)

def print_sample_size_guidance(complete_rows, predictor_count):
    return _print_sample_size_guidance(complete_rows, predictor_count)

def calculate_regression_diagnostics(model):
    return _calculate_regression_diagnostics(model)

def save_custom_regression_diagnostics(model):
    return _save_custom_regression_diagnostics(model)

def save_custom_regression_results(model, outcome, predictors, complete_rows, vif_results, diagnostics, output_directory="results"):
    return _save_custom_regression_results(model, outcome, predictors, complete_rows, vif_results, diagnostics, output_directory=output_directory)

def select_custom_regression_variables(data):
    return _select_custom_regression_variables(data)

def custom_regression_analysis(data):
    return _custom_regression_analysis(data)

def regression_analysis(data):
    return _regression_analysis(data)

from iris_ml import (
    machine_learning_analysis as _machine_learning_analysis,
    compare_models as _compare_models,
    roc_curve_analysis as _roc_curve_analysis,
    feature_importance_analysis as _feature_importance_analysis,
    train_prediction_model as _train_prediction_model,
    predict_new_flower as _predict_new_flower,
)


def machine_learning_analysis(data):
    return _machine_learning_analysis(data)


def compare_models(data):
    return _compare_models(data)


def roc_curve_analysis(data):
    return _roc_curve_analysis(data)


def feature_importance_analysis(data):
    return _feature_importance_analysis(data)


def train_prediction_model(data):
    return _train_prediction_model(data)


def predict_new_flower(data):
    return _predict_new_flower(data)


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
        # REGRESSION DIAGNOSTICS
        # --------------------------------------------------

        report.write("\n\n4. REGRESSION ASSUMPTION DIAGNOSTICS\n")
        report.write("-" * 70 + "\n")

        vif_path = "results/vif_results.csv"

        predictors = [
            "sepal length (cm)",
            "sepal width (cm)",
            "petal width (cm)"
        ]

        X = data[predictors]
        y = data["petal length (cm)"]

        X_with_constant = sm.add_constant(X)

        regression_model = sm.OLS(
            y,
            X_with_constant
        ).fit()

        residuals = regression_model.resid

        # Normality
        shapiro_stat, shapiro_p = shapiro(residuals)

        report.write("\n4.1 Normality of Residuals\n")
        report.write("-" * 40 + "\n")
        report.write(
            f"Shapiro-Wilk statistic: {shapiro_stat:.4f}\n"
        )
        report.write(
            f"p-value: {shapiro_p:.4f}\n"
        )

        if shapiro_p > 0.05:
            report.write(
                "Conclusion: Residuals do not show significant "
                "evidence of non-normality.\n"
            )
        else:
            report.write(
                "Conclusion: Residuals show evidence of "
                "non-normality.\n"
            )

        # Homoscedasticity
        bp_test = het_breuschpagan(
            residuals,
            X_with_constant
        )

        bp_stat = bp_test[0]
        bp_p = bp_test[1]

        report.write("\n4.2 Homoscedasticity\n")
        report.write("-" * 40 + "\n")
        report.write(
            f"Breusch-Pagan statistic: {bp_stat:.4f}\n"
        )
        report.write(
            f"p-value: {bp_p:.4f}\n"
        )

        if bp_p > 0.05:
            report.write(
                "Conclusion: There is no significant evidence "
                "of heteroscedasticity.\n"
            )
        else:
            report.write(
                "Conclusion: There is evidence of "
                "heteroscedasticity.\n"
            )

        # Multicollinearity
        vif_data = pd.DataFrame()
        vif_data["Feature"] = X.columns
        vif_data["VIF"] = [
            variance_inflation_factor(
                X.values,
                i
            )
            for i in range(X.shape[1])
        ]

        report.write("\n4.3 Multicollinearity\n")
        report.write("-" * 40 + "\n")
        report.write(
            vif_data.round(3).to_string(index=False)
        )
        report.write(
            "\n\nVIF values around 1 indicate little "
            "multicollinearity. Larger values may indicate "
            "multicollinearity concerns.\n"
        )

        # Durbin-Watson
        dw_stat = durbin_watson(residuals)

        report.write("\n4.4 Autocorrelation\n")
        report.write("-" * 40 + "\n")
        report.write(
            f"Durbin-Watson statistic: {dw_stat:.4f}\n"
        )
        report.write(
            "Values around 2 generally indicate little "
            "first-order autocorrelation.\n"
        )
        report.write(
            "Note: The Iris observations do not have a natural "
            "time-based ordering, so this statistic is included "
            "as a general diagnostic demonstration.\n"
        )

        report.write("\nDiagnostic plots:\n")
        report.write(
            "- figures/regression_qq_plot.png\n"
        )
        report.write(
            "- figures/residuals_vs_fitted.png\n"
        )

        if os.path.exists(vif_path):
            report.write(
                "- results/vif_results.csv\n"
            )

        # --------------------------------------------------
        # MACHINE LEARNING RESULTS
        # --------------------------------------------------

        report.write("\n\n5. MACHINE LEARNING MODEL RESULTS\n")
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

        report.write("\n\n6. ROC-AUC RESULTS\n")
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

        report.write("\n\n7. FEATURE IMPORTANCE\n")
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

    # --------------------------------------------------
    # REGRESSION DIAGNOSTICS
    # --------------------------------------------------

    predictors = [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal width (cm)"
    ]

    X = data[predictors]
    y = data["petal length (cm)"]

    X_with_constant = sm.add_constant(X)

    regression_model = sm.OLS(
        y,
        X_with_constant
    ).fit()

    residuals = regression_model.resid

    # Normality
    shapiro_stat, shapiro_p = shapiro(residuals)

    normality_conclusion = (
        "No significant evidence of non-normality."
        if shapiro_p > 0.05
        else
        "Evidence of non-normality was detected."
    )

    # Homoscedasticity
    bp_test = het_breuschpagan(
        residuals,
        X_with_constant
    )

    bp_stat = bp_test[0]
    bp_p = bp_test[1]

    homoscedasticity_conclusion = (
        "No significant evidence of heteroscedasticity."
        if bp_p > 0.05
        else
        "Evidence of heteroscedasticity was detected."
    )

    # VIF
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [
        variance_inflation_factor(
            X.values,
            i
        )
        for i in range(X.shape[1])
    ]

    vif_data["VIF"] = vif_data["VIF"].round(3)

    vif_table = vif_data.to_html(
        index=False,
        classes="data-table"
    )

    # Durbin-Watson
    dw_stat = durbin_watson(residuals)

    diagnostics_table = f"""
    <table class="data-table">
        <tr>
            <th>Diagnostic</th>
            <th>Statistic</th>
            <th>p-value</th>
            <th>Interpretation</th>
        </tr>

        <tr>
            <td>Shapiro-Wilk</td>
            <td>{shapiro_stat:.4f}</td>
            <td>{shapiro_p:.4f}</td>
            <td>{normality_conclusion}</td>
        </tr>

        <tr>
            <td>Breusch-Pagan</td>
            <td>{bp_stat:.4f}</td>
            <td>{bp_p:.4f}</td>
            <td>{homoscedasticity_conclusion}</td>
        </tr>

        <tr>
            <td>Durbin-Watson</td>
            <td>{dw_stat:.4f}</td>
            <td>N/A</td>
            <td>Values around 2 generally indicate little first-order autocorrelation.</td>
        </tr>
    </table>
    """

    # --------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------

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

    # --------------------------------------------------
    # ROC-AUC
    # --------------------------------------------------

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

    # --------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------

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

    # --------------------------------------------------
    # CREATE HTML DOCUMENT
    # --------------------------------------------------

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

h3 {{
    margin-top: 30px;
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

.note-box {{
    padding: 15px;
    margin: 20px 0;
    border-radius: 8px;
    background-color: #f8f8f8;
    border-left: 4px solid #888;
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

<h2>4. Regression Assumption Diagnostics</h2>

{diagnostics_table}

<h3>Variance Inflation Factors</h3>

{vif_table}

<div class="note-box">

<strong>Note:</strong>
Durbin-Watson is primarily intended for assessing
first-order autocorrelation when observations have a
meaningful ordering. The Iris dataset does not have a
natural time-based ordering, so this statistic is included
as a general diagnostic demonstration.

</div>

<h3>Q-Q Plot of Regression Residuals</h3>

<img src="../figures/regression_qq_plot.png"
     alt="Regression Q-Q Plot">

<h3>Residuals vs Fitted Values</h3>

<img src="../figures/residuals_vs_fitted.png"
     alt="Residuals vs Fitted Values">

<h2>5. Machine Learning Model Comparison</h2>

{model_table}

<img src="../figures/model_comparison.png"
     alt="Model Comparison">

<h2>6. ROC-AUC Analysis</h2>

{roc_table}

<img src="../figures/roc_curves.png"
     alt="ROC Curves">

<h2>7. Feature Importance</h2>

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

def regression_diagnostics(data):

    print("\n" + "=" * 60)
    print("REGRESSION ASSUMPTION DIAGNOSTICS")
    print("=" * 60)

    predictors = [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal width (cm)"
    ]

    X = data[predictors]
    y = data["petal length (cm)"]

    X_with_constant = sm.add_constant(X)

    model = sm.OLS(
        y,
        X_with_constant
    ).fit()

    residuals = model.resid
    fitted_values = model.fittedvalues

    # --------------------------------------------------
    # NORMALITY
    # --------------------------------------------------

    shapiro_stat, shapiro_p = shapiro(
        residuals
    )

    print("\n1. NORMALITY OF RESIDUALS")
    print("-" * 60)

    print(
        f"Shapiro-Wilk statistic: "
        f"{shapiro_stat:.4f}"
    )

    print(
        f"p-value: {shapiro_p:.4f}"
    )

    if shapiro_p > 0.05:

        print(
            "Decision: Fail to reject H0"
        )

        print(
            "Conclusion: Residuals do not show "
            "significant evidence of non-normality."
        )

    else:

        print(
            "Decision: Reject H0"
        )

        print(
            "Conclusion: Residuals show evidence "
            "of non-normality."
        )

    # --------------------------------------------------
    # HOMOSCEDASTICITY
    # --------------------------------------------------

    bp_test = het_breuschpagan(
        residuals,
        X_with_constant
    )

    bp_stat = bp_test[0]
    bp_p = bp_test[1]

    print("\n2. HOMOSCEDASTICITY")
    print("-" * 60)

    print(
        f"Breusch-Pagan statistic: "
        f"{bp_stat:.4f}"
    )

    print(
        f"p-value: {bp_p:.4f}"
    )

    if bp_p > 0.05:

        print(
            "Decision: Fail to reject H0"
        )

        print(
            "Conclusion: There is no significant "
            "evidence of heteroscedasticity."
        )

    else:

        print(
            "Decision: Reject H0"
        )

        print(
            "Conclusion: There is evidence "
            "of heteroscedasticity."
        )

    # --------------------------------------------------
    # MULTICOLLINEARITY
    # --------------------------------------------------

    vif_data = pd.DataFrame()

    vif_data["Feature"] = X.columns

    vif_data["VIF"] = [
        variance_inflation_factor(
            X.values,
            i
        )
        for i in range(X.shape[1])
    ]

    print("\n3. MULTICOLLINEARITY")
    print("-" * 60)

    print(
        vif_data.round(3).to_string(
            index=False
        )
    )

    print(
        "\nGeneral guideline:"
    )

    print(
        "VIF values around 1 indicate little "
        "multicollinearity."
    )

    print(
        "Large VIF values may indicate "
        "multicollinearity concerns."
    )

    # --------------------------------------------------
    # DURBIN-WATSON
    # --------------------------------------------------

    dw_stat = durbin_watson(
        residuals
    )

    print("\n4. AUTOCORRELATION")
    print("-" * 60)

    print(
        f"Durbin-Watson statistic: "
        f"{dw_stat:.4f}"
    )

    print(
        "Values around 2 generally indicate "
        "little first-order autocorrelation."
    )

    # --------------------------------------------------
    # Q-Q PLOT
    # --------------------------------------------------

    os.makedirs(
        "figures",
        exist_ok=True
    )

    plt.figure(
        figsize=(8, 6)
    )

    sm.qqplot(
        residuals,
        line="45",
        fit=True
    )

    plt.title(
        "Q-Q Plot of Regression Residuals"
    )

    plt.tight_layout()

    plt.savefig(
        "figures/regression_qq_plot.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------
    # RESIDUALS VS FITTED
    # --------------------------------------------------

    plt.figure(
        figsize=(8, 6)
    )

    plt.scatter(
        fitted_values,
        residuals
    )

    plt.axhline(
        y=0,
        linestyle="--"
    )

    plt.xlabel(
        "Fitted Values"
    )

    plt.ylabel(
        "Residuals"
    )

    plt.title(
        "Residuals vs Fitted Values"
    )

    plt.tight_layout()

    plt.savefig(
        "figures/residuals_vs_fitted.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------
    # SAVE VIF RESULTS
    # --------------------------------------------------

    os.makedirs(
        "results",
        exist_ok=True
    )

    vif_data.to_csv(
        "results/vif_results.csv",
        index=False
    )

    print(
        "\nDiagnostic plots saved to:"
    )

    print(
        "figures/regression_qq_plot.png"
    )

    print(
        "figures/residuals_vs_fitted.png"
    )

    print(
        "\nVIF results saved to:"
    )

    print(
        "results/vif_results.csv"
    )

def main():
    """Run the StatMate command-line interface."""
    import sys
    from statmate_cli import run_cli

    # Pass the current module as the analysis API. This preserves the public
    # statmate.py functions used by existing users and tests while separating
    # menu/state orchestration from analysis code.
    run_cli(sys.modules[__name__])


if __name__ == "__main__":
    main()
