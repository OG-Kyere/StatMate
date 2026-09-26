"""Built-in Iris regression diagnostic workflow for StatMate."""

import os

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from scipy.stats import shapiro
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson


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

