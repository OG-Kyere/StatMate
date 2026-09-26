"""Visualization and report generation for StatMate's built-in Iris workflow."""

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import shapiro
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson


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

