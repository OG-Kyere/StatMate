"""Text and HTML reporting for arbitrary StatMate custom datasets."""

from html import escape
from pathlib import Path

import numpy as np
import pandas as pd


def build_custom_report_sections(data):
    """Build non-interactive report tables that are valid for any dataset."""
    numeric = data.select_dtypes(include="number")
    dtype_summary = pd.DataFrame({
        "Column": data.columns.astype(str),
        "Data Type": [str(dtype) for dtype in data.dtypes],
        "Missing": data.isna().sum().to_numpy(),
        "Missing (%)": (data.isna().mean().to_numpy() * 100).round(2),
        "Unique": [data[column].nunique(dropna=True) for column in data.columns],
    })

    descriptive = numeric.describe().T if not numeric.empty else pd.DataFrame()

    if numeric.shape[1] >= 2:
        clean_numeric = numeric.replace([np.inf, -np.inf], np.nan)
        correlation = clean_numeric.corr()
    else:
        correlation = pd.DataFrame()

    return {
        "dtype_summary": dtype_summary,
        "descriptive": descriptive,
        "correlation": correlation,
    }


def build_session_sections(session=None):
    """Convert stored analysis results into report-friendly tables."""
    sections = {}
    if session is None:
        return sections
    results = getattr(session, "results", session if isinstance(session, dict) else {})

    regression = results.get("regression")
    if regression:
        model = regression["model"]
        ci = model.conf_int()
        sections["regression_summary"] = pd.DataFrame([{
            "Outcome": regression["outcome"],
            "Predictors": ", ".join(map(str, regression["predictors"])),
            "Complete Rows": regression["complete_rows"],
            "R-squared": model.rsquared,
            "Adjusted R-squared": model.rsquared_adj,
            "Model p-value": model.f_pvalue,
        }])
        sections["regression_coefficients"] = pd.DataFrame({
            "Variable": model.params.index,
            "Coefficient": model.params.values,
            "Standard Error": model.bse.values,
            "t Statistic": model.tvalues.values,
            "p Value": model.pvalues.values,
            "CI Lower": ci.iloc[:, 0].values,
            "CI Upper": ci.iloc[:, 1].values,
        })
        sections["regression_vif"] = regression["vif"]
        d = regression["diagnostics"]
        sections["regression_diagnostics"] = pd.DataFrame([{
            "Shapiro-Wilk p-value": d["shapiro_p_value"],
            "Breusch-Pagan p-value": d["breusch_pagan_p_value"],
            "Durbin-Watson": d["durbin_watson"],
            "Influential Observations": d["influential_observations"],
        }])

    classification = results.get("classification")
    if isinstance(classification, pd.DataFrame):
        sections["classification"] = classification

    for key in ("hypothesis", "post_hoc"):
        result = results.get(key)
        if result:
            labels = list(map(str, result["labels"]))
            normality = [
                value if value is not None else np.nan
                for value in result["normality"]
            ]
            sections[f"{key}_summary"] = pd.DataFrame([{
                "Test": result["name"],
                "Statistic": result["statistic"],
                "p Value": result["p"],
                "Levene p Value": result["variance_p"],
                "Excluded Rows": result["dropped"],
            }])
            sections[f"{key}_groups"] = pd.DataFrame({
                "Group": labels,
                "n": [len(group) for group in result["groups"]],
                "Shapiro-Wilk p Value": normality,
            })

    for key in ("model_comparison", "roc_auc", "feature_importance"):
        result = results.get(key)
        if isinstance(result, pd.DataFrame):
            sections[key] = result

    return sections


def _table_html(frame):
    """Render a DataFrame as a compact HTML table."""
    if frame is None or frame.empty:
        return "<p>No applicable results.</p>"
    return frame.to_html(
        border=0,
        classes="dataframe",
        justify="center",
        escape=True,
    )


def _table_text(frame):
    """Render a DataFrame for the plain-text report."""
    if frame is None or frame.empty:
        return "No applicable results."
    return frame.to_string()


def generate_custom_text_report(
    data,
    output_path="reports/custom_statmate_report.txt",
    session=None,
):
    """Generate a plain-text statistical summary for a custom dataset."""
    sections = build_custom_report_sections(data)
    session_sections = build_session_sections(session)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    text = f"""STATMATE CUSTOM DATASET REPORT
{'=' * 60}

DATASET SUMMARY
Rows: {len(data)}
Columns: {data.shape[1]}
Numeric columns: {data.select_dtypes(include="number").shape[1]}

COLUMN AND MISSING-DATA SUMMARY
{'-' * 60}
{_table_text(sections["dtype_summary"])}

DESCRIPTIVE STATISTICS
{'-' * 60}
{_table_text(sections["descriptive"])}

NUMERIC CORRELATION MATRIX
{'-' * 60}
{_table_text(sections["correlation"])}

STORED HYPOTHESIS TEST RESULTS
{'-' * 60}
{_table_text(session_sections.get("hypothesis_summary"))}
{_table_text(session_sections.get("hypothesis_groups"))}

STORED POST-HOC WORKFLOW RESULTS
{'-' * 60}
{_table_text(session_sections.get("post_hoc_summary"))}
{_table_text(session_sections.get("post_hoc_groups"))}

STORED REGRESSION RESULTS
{'-' * 60}
{_table_text(session_sections.get("regression_summary"))}
{_table_text(session_sections.get("regression_coefficients"))}
{_table_text(session_sections.get("regression_vif"))}
{_table_text(session_sections.get("regression_diagnostics"))}

STORED CLASSIFICATION RESULTS
{'-' * 60}
{_table_text(session_sections.get("classification"))}

STORED MODEL COMPARISON RESULTS
{'-' * 60}
{_table_text(session_sections.get("model_comparison"))}

STORED ROC-AUC RESULTS
{'-' * 60}
{_table_text(session_sections.get("roc_auc"))}

STORED FEATURE IMPORTANCE RESULTS
{'-' * 60}
{_table_text(session_sections.get("feature_importance"))}

ANALYSIS SCOPE
{'-' * 60}
This automatic report does not guess an outcome, grouping variable, or
statistical research question. Outcome-dependent analyses remain available
through StatMate's interactive analysis options.
"""
    output.write_text(text, encoding="utf-8")
    print(f"\nCustom statistical report saved to: {output.as_posix()}")
    return str(output)


def generate_custom_html_report(
    data,
    output_path="reports/custom_statmate_report.html",
    session=None,
):
    """Generate a self-contained HTML summary for a custom dataset."""
    sections = build_custom_report_sections(data)
    session_sections = build_session_sections(session)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    title = "StatMate Custom Dataset Report"
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(title)}</title>
<style>
body {{ font-family: Arial, sans-serif; max-width: 1100px; margin: 40px auto; padding: 0 20px; line-height: 1.5; }}
h1, h2 {{ margin-top: 1.4em; }}
.summary {{ padding: 12px 16px; border: 1px solid #ccc; border-radius: 8px; }}
table {{ border-collapse: collapse; width: 100%; margin: 16px 0 28px; font-size: 0.92rem; }}
th, td {{ border: 1px solid #ddd; padding: 7px; text-align: right; }}
th {{ text-align: center; }}
.note {{ font-size: 0.92rem; }}
</style>
</head>
<body>
<h1>{escape(title)}</h1>
<div class="summary">
<strong>Rows:</strong> {len(data)}<br>
<strong>Columns:</strong> {data.shape[1]}<br>
<strong>Numeric columns:</strong> {data.select_dtypes(include="number").shape[1]}
</div>

<h2>Column and Missing-Data Summary</h2>
{_table_html(sections["dtype_summary"])}

<h2>Descriptive Statistics</h2>
{_table_html(sections["descriptive"])}

<h2>Numeric Correlation Matrix</h2>
{_table_html(sections["correlation"])}

<h2>Stored Hypothesis Test Results</h2>
{_table_html(session_sections.get("hypothesis_summary"))}
{_table_html(session_sections.get("hypothesis_groups"))}

<h2>Stored Post-Hoc Workflow Results</h2>
{_table_html(session_sections.get("post_hoc_summary"))}
{_table_html(session_sections.get("post_hoc_groups"))}

<h2>Stored Regression Results</h2>
{_table_html(session_sections.get("regression_summary"))}
{_table_html(session_sections.get("regression_coefficients"))}
{_table_html(session_sections.get("regression_vif"))}
{_table_html(session_sections.get("regression_diagnostics"))}

<h2>Stored Classification Results</h2>
{_table_html(session_sections.get("classification"))}

<h2>Stored Model Comparison Results</h2>
{_table_html(session_sections.get("model_comparison"))}

<h2>Stored ROC-AUC Results</h2>
{_table_html(session_sections.get("roc_auc"))}

<h2>Stored Feature Importance Results</h2>
{_table_html(session_sections.get("feature_importance"))}

<h2>Analysis Scope</h2>
<p class="note">
This automatic report does not guess an outcome, grouping variable, or statistical
research question. Outcome-dependent hypothesis tests, regression, classification,
ROC-AUC, feature importance, and prediction remain available through StatMate's
interactive analysis options.
</p>
</body>
</html>
"""
    output.write_text(html, encoding="utf-8")
    print(f"\nCustom HTML report saved to: {output.as_posix()}")
    return str(output)
