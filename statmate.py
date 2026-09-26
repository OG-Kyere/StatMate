# Backwards-compatible public symbols used by existing integrations and tests.
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import shapiro, f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

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


from reporting import (
    create_visualizations as _create_visualizations,
    generate_report as _generate_report,
    generate_html_report as _generate_html_report,
)
from iris_diagnostics import regression_diagnostics as _regression_diagnostics


def create_visualizations(data):
    return _create_visualizations(data)


def generate_report(data):
    return _generate_report(data)


def generate_html_report(data):
    return _generate_html_report(data)


def regression_diagnostics(data):
    return _regression_diagnostics(data)


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
