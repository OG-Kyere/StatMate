"""Linear regression and diagnostic workflows for StatMate."""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import shapiro
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson


def print_regression_results(model, outcome_name):
    """Print a consistent OLS summary for Iris and custom datasets."""

    print("\nRegression Summary:")
    print(model.summary())

    print("\nKey Results:")
    print(f"R-squared: {model.rsquared:.4f}")
    print(f"Adjusted R-squared: {model.rsquared_adj:.4f}")
    print(f"F-statistic p-value: {model.f_pvalue:.6f}")

    print("\nCoefficients:")
    for variable in model.params.index:
        print(
            f"{variable}: "
            f"coefficient = {model.params[variable]:.4f}, "
            f"p-value = {model.pvalues[variable]:.6f}"
        )

    print("\nInterpretation:")
    if model.f_pvalue < 0.05:
        print("The overall regression model is statistically significant at the 5% significance level.")
    else:
        print("The overall regression model is not statistically significant at the 5% significance level.")

    print(
        f"The model explains approximately {model.rsquared * 100:.2f}% "
        f"of the variation in {outcome_name}."
    )


def fit_linear_regression(data, outcome, predictors):
    """Fit OLS after removing rows missing values in selected variables."""

    selected_columns = [outcome, *predictors]
    analysis_data = (
        data[selected_columns]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    required_rows = len(predictors) + 2

    if len(analysis_data) < required_rows:
        raise ValueError(
            f"Regression needs at least {required_rows} complete rows for the selected variables; "
            f"only {len(analysis_data)} are available."
        )
    if analysis_data[outcome].nunique() < 2:
        raise ValueError("The outcome variable must contain at least two distinct values.")

    X = sm.add_constant(analysis_data[predictors], has_constant="add")
    if any(analysis_data[predictor].nunique() < 2 for predictor in predictors):
        raise ValueError("Predictor variables must each contain at least two distinct values.")
    if np.linalg.matrix_rank(X.to_numpy(dtype=float)) < X.shape[1]:
        raise ValueError(
            "The selected predictors are perfectly collinear. Choose a different combination of predictors."
        )

    return sm.OLS(analysis_data[outcome], X).fit(), len(analysis_data)


def calculate_vif(data, outcome, predictors):
    """Calculate VIF values using the same complete rows as the regression."""

    analysis_data = (
        data[[outcome, *predictors]]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    predictor_data = sm.add_constant(analysis_data[predictors], has_constant="add")

    return pd.DataFrame(
        {
            "Predictor": predictors,
            "VIF": [
                variance_inflation_factor(predictor_data.to_numpy(dtype=float), index + 1)
                for index in range(len(predictors))
            ],
        }
    )


def print_vif_guidance(vif_results):
    """Display actionable collinearity guidance for custom regression."""

    print("\nCollinearity check (Variance Inflation Factor):")
    print(vif_results.to_string(index=False, formatters={"VIF": "{:.2f}".format}))

    high_vif = vif_results[vif_results["VIF"] >= 10]
    moderate_vif = vif_results[(vif_results["VIF"] >= 5) & (vif_results["VIF"] < 10)]

    if not high_vif.empty:
        print(
            "\nWarning: High collinearity (VIF ≥ 10) was detected for: "
            + ", ".join(high_vif["Predictor"].astype(str))
            + ". Consider removing or replacing overlapping predictors, then run option 6 again."
        )
    elif not moderate_vif.empty:
        print(
            "\nNote: Moderate collinearity (VIF 5–10) was detected for: "
            + ", ".join(moderate_vif["Predictor"].astype(str))
            + ". Interpret these coefficients with care."
        )
    else:
        print("\nNo concerning collinearity was detected (all VIF values are below 5).")


def has_recommended_sample_size(complete_rows, predictor_count):
    """Return whether a model meets a 10-rows-per-predictor rule of thumb."""

    return complete_rows >= predictor_count * 10


def print_sample_size_guidance(complete_rows, predictor_count):
    """Explain when a fitted custom model has limited data for its complexity."""

    rows_per_predictor = complete_rows / predictor_count
    print(
        f"\nSample-size check: {complete_rows} complete rows for {predictor_count} predictor(s) "
        f"({rows_per_predictor:.1f} rows per predictor)."
    )
    if not has_recommended_sample_size(complete_rows, predictor_count):
        print(
            "Warning: This is below the common rule of thumb of 10 complete rows per predictor. "
            "Use fewer predictors or collect more data for more stable estimates."
        )


def calculate_regression_diagnostics(model):
    """Return assumption and influence statistics for a fitted OLS model."""

    residuals = model.resid
    bp_statistic, bp_p_value, _, _ = het_breuschpagan(residuals, model.model.exog)
    cook_distances = model.get_influence().cooks_distance[0]
    cook_threshold = 4 / len(residuals)
    shapiro_statistic, shapiro_p_value = shapiro(residuals)

    return {
        "shapiro_statistic": shapiro_statistic,
        "shapiro_p_value": shapiro_p_value,
        "breusch_pagan_statistic": bp_statistic,
        "breusch_pagan_p_value": bp_p_value,
        "durbin_watson": durbin_watson(residuals),
        "cook_threshold": cook_threshold,
        "influential_observations": int((cook_distances > cook_threshold).sum()),
    }


def save_custom_regression_diagnostics(model):
    """Print and save residual diagnostics for a custom regression model."""

    diagnostics = calculate_regression_diagnostics(model)
    residuals = model.resid
    fitted_values = model.fittedvalues

    print("\nRegression diagnostics:")
    print(
        f"Shapiro-Wilk p-value: {diagnostics['shapiro_p_value']:.4f} "
        "(values below 0.05 suggest non-normal residuals)"
    )
    print(
        f"Breusch-Pagan p-value: {diagnostics['breusch_pagan_p_value']:.4f} "
        "(values below 0.05 suggest unequal residual variance)"
    )
    print(
        f"Durbin-Watson: {diagnostics['durbin_watson']:.4f} "
        "(values near 2 suggest little first-order autocorrelation)"
    )
    print(
        f"Potentially influential observations: {diagnostics['influential_observations']} "
        f"(Cook's distance threshold: {diagnostics['cook_threshold']:.4f})"
    )

    os.makedirs("figures", exist_ok=True)

    figure, axis = plt.subplots(figsize=(8, 6))
    sm.qqplot(residuals, line="45", fit=True, ax=axis)
    axis.set_title("Custom Regression Q-Q Plot of Residuals")
    figure.tight_layout()
    figure.savefig("figures/custom_regression_qq_plot.png", dpi=300, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 6))
    axis.scatter(fitted_values, residuals)
    axis.axhline(y=0, linestyle="--")
    axis.set_xlabel("Fitted Values")
    axis.set_ylabel("Residuals")
    axis.set_title("Custom Regression Residuals vs Fitted Values")
    figure.tight_layout()
    figure.savefig("figures/custom_regression_residuals_vs_fitted.png", dpi=300, bbox_inches="tight")
    plt.close(figure)

    print("Diagnostic plots saved to:")
    print("figures/custom_regression_qq_plot.png")
    print("figures/custom_regression_residuals_vs_fitted.png")

    return diagnostics


def save_custom_regression_results(
    model, outcome, predictors, complete_rows, vif_results, diagnostics, output_directory="results"
):
    """Save reusable tables for the most recently fitted custom regression."""

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    confidence_intervals = model.conf_int()
    coefficient_results = pd.DataFrame(
        {
            "Variable": model.params.index,
            "Coefficient": model.params.values,
            "Standard Error": model.bse.values,
            "t Statistic": model.tvalues.values,
            "p Value": model.pvalues.values,
            "CI Lower": confidence_intervals.iloc[:, 0].values,
            "CI Upper": confidence_intervals.iloc[:, 1].values,
        }
    )
    coefficient_path = output_path / "custom_regression_coefficients.csv"
    coefficient_results.to_csv(coefficient_path, index=False)

    diagnostic_results = pd.DataFrame(
        {
            "Metric": [
                "Outcome",
                "Predictors",
                "Complete rows",
                "R-squared",
                "Adjusted R-squared",
                "Model F-test p-value",
                "Shapiro-Wilk p-value",
                "Breusch-Pagan p-value",
                "Durbin-Watson",
                "Influential observations",
            ],
            "Value": [
                outcome,
                ", ".join(map(str, predictors)),
                complete_rows,
                model.rsquared,
                model.rsquared_adj,
                model.f_pvalue,
                diagnostics["shapiro_p_value"],
                diagnostics["breusch_pagan_p_value"],
                diagnostics["durbin_watson"],
                diagnostics["influential_observations"],
            ],
        }
    )
    diagnostics_path = output_path / "custom_regression_diagnostics.csv"
    diagnostic_results.to_csv(diagnostics_path, index=False)

    vif_path = output_path / "custom_regression_vif.csv"
    vif_results.to_csv(vif_path, index=False)

    print("Custom regression results saved to:")
    print(coefficient_path)
    print(diagnostics_path)
    print(vif_path)


def select_custom_regression_variables(data):
    """Prompt the user to select numeric outcome and predictor columns."""

    numerical_columns = data.select_dtypes(include="number").columns.tolist()
    if len(numerical_columns) < 2:
        print("\nWarning: Custom regression needs at least two numeric columns.")
        return None, None

    print("\nNumeric variables:")
    for index, column in enumerate(numerical_columns, start=1):
        print(f"{index}. {column}")

    outcome_choice = input("Choose the numeric outcome variable (number, or press Enter to cancel): ").strip()
    if not outcome_choice:
        print("\nCustom regression cancelled.")
        return None, None

    try:
        outcome_index = int(outcome_choice) - 1
        if not 0 <= outcome_index < len(numerical_columns):
            raise IndexError
        outcome = numerical_columns[outcome_index]
    except (ValueError, IndexError):
        print("\nWarning: Please enter a valid outcome variable number.")
        return None, None

    predictor_choices = input(
        "Choose one or more predictor variables (comma-separated numbers, or press Enter to cancel): "
    ).strip()
    if not predictor_choices:
        print("\nCustom regression cancelled.")
        return None, None

    try:
        predictor_indices = [int(choice.strip()) - 1 for choice in predictor_choices.split(",")]
        if (
            not predictor_indices
            or len(set(predictor_indices)) != len(predictor_indices)
            or any(index < 0 or index >= len(numerical_columns) for index in predictor_indices)
        ):
            raise ValueError
        predictors = [numerical_columns[index] for index in predictor_indices]
    except (ValueError, IndexError):
        print("\nWarning: Enter unique, valid predictor variable numbers separated by commas.")
        return None, None

    if outcome in predictors:
        print("\nWarning: The outcome variable cannot also be a predictor.")
        return None, None

    return outcome, predictors


def custom_regression_analysis(data):
    """Run a user-configured linear regression on an uploaded dataset."""

    print("\n" + "=" * 60)
    print("CUSTOM MULTIPLE LINEAR REGRESSION")
    print("=" * 60)

    outcome, predictors = select_custom_regression_variables(data)
    if outcome is None:
        return

    try:
        model, complete_rows = fit_linear_regression(data, outcome, predictors)
    except (ValueError, TypeError) as error:
        print(f"\nWarning: Could not fit the regression model: {error}")
        return

    print(f"\nOutcome: {outcome}")
    print("Predictors: " + ", ".join(map(str, predictors)))
    print(f"Complete rows used: {complete_rows} of {len(data)}")
    print_sample_size_guidance(complete_rows, len(predictors))
    vif_results = calculate_vif(data, outcome, predictors)
    print_vif_guidance(vif_results)
    print_regression_results(model, outcome)
    diagnostics = save_custom_regression_diagnostics(model)
    save_custom_regression_results(
        model, outcome, predictors, complete_rows, vif_results, diagnostics
    )


def regression_analysis(data):
    print("\n" + "=" * 60)
    print("MULTIPLE LINEAR REGRESSION")
    print("=" * 60)

    model, _ = fit_linear_regression(
        data,
        "petal length (cm)",
        ["sepal length (cm)", "sepal width (cm)", "petal width (cm)"],
    )
    print_regression_results(model, "petal length")

