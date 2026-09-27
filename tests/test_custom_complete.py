"""Tests for the custom complete-analysis workflow."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

from custom_complete import custom_complete_analysis


class CustomCompleteAnalysisTests(unittest.TestCase):
    def test_runs_safe_automatic_analyses_without_guessing_outcome(self):
        data = pd.DataFrame({
            "x": [1, 2, 3, 4],
            "y": [2, 4, 6, 8],
            "group": ["a", "a", "b", "b"],
        })
        api = Mock()

        with patch("custom_complete.generate_custom_html_report", return_value="reports/custom_statmate_report.html") as report:
            result = custom_complete_analysis(data, api)

        api.explore_data.assert_called_once_with(data)
        api.descriptive_statistics.assert_called_once_with(data)
        api.correlation_analysis.assert_called_once_with(data)
        report.assert_called_once_with(data)
        self.assertIs(result["data"], data)
        self.assertEqual(result["report_path"], "reports/custom_statmate_report.html")

    def test_does_not_run_outcome_dependent_analyses_automatically(self):
        data = pd.DataFrame({"x": [1, 2, 3], "y": [3, 2, 1]})
        api = Mock()

        with patch("custom_complete.generate_custom_html_report", return_value="report.html"):
            custom_complete_analysis(data, api)

        api.custom_hypothesis_analysis.assert_not_called()
        api.custom_regression_analysis.assert_not_called()
        api.custom_classification_analysis.assert_not_called()


class CustomReportingTests(unittest.TestCase):
    def test_html_report_contains_generic_dataset_sections(self):
        from custom_reporting import generate_custom_html_report

        data = pd.DataFrame({
            "score": [10.0, 12.0, None, 18.0],
            "age": [20, 21, 22, 23],
            "group": ["A", "A", "B", "B"],
        })
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.html"
            path = generate_custom_html_report(data, output)
            html = Path(path).read_text(encoding="utf-8")

        self.assertIn("Column and Missing-Data Summary", html)
        self.assertIn("Descriptive Statistics", html)
        self.assertIn("Numeric Correlation Matrix", html)
        self.assertIn("score", html)
        self.assertIn("group", html)

    def test_html_report_handles_dataset_without_numeric_columns(self):
        from custom_reporting import generate_custom_html_report

        data = pd.DataFrame({
            "group": ["A", "B", "A"],
            "status": ["yes", "no", "yes"],
        })
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.html"
            path = generate_custom_html_report(data, output)
            html = Path(path).read_text(encoding="utf-8")

        self.assertIn("Numeric columns:</strong> 0", html)
        self.assertIn("No applicable results.", html)


    def test_text_report_contains_generic_dataset_sections(self):
        from custom_reporting import generate_custom_text_report

        data = pd.DataFrame({
            "score": [10.0, 12.0, None, 18.0],
            "age": [20, 21, 22, 23],
            "group": ["A", "A", "B", "B"],
        })
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.txt"
            path = generate_custom_text_report(data, output)
            report = Path(path).read_text(encoding="utf-8")

        self.assertIn("COLUMN AND MISSING-DATA SUMMARY", report)
        self.assertIn("DESCRIPTIVE STATISTICS", report)
        self.assertIn("NUMERIC CORRELATION MATRIX", report)
        self.assertIn("score", report)
        self.assertIn("group", report)


    def test_html_report_includes_stored_classification_results(self):
        from analysis_session import AnalysisSession
        from custom_reporting import generate_custom_html_report

        data = pd.DataFrame({"x": [1, 2, 3, 4], "group": ["A", "A", "B", "B"]})
        session = AnalysisSession()
        session.remember("classification", pd.DataFrame([{
            "Model": "Logistic Regression",
            "Accuracy": 0.90,
            "F1": 0.89,
        }]))

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.html"
            path = generate_custom_html_report(data, output, session=session)
            html = Path(path).read_text(encoding="utf-8")

        self.assertIn("Stored Classification Results", html)
        self.assertIn("Logistic Regression", html)
        self.assertIn("0.9", html)


    def test_html_report_includes_extended_session_results(self):
        from analysis_session import AnalysisSession
        from custom_reporting import generate_custom_html_report

        data = pd.DataFrame({
            "score": [10, 12, 14, 16, 18, 20],
            "group": ["A", "A", "A", "B", "B", "B"],
        })
        session = AnalysisSession()
        hypothesis = {
            "name": "Welch t-test",
            "statistic": 2.5,
            "p": 0.03,
            "normality": [0.20, 0.30],
            "variance_p": 0.10,
            "labels": ["A", "B"],
            "groups": [[10, 12, 14], [16, 18, 20]],
            "clean": data,
            "dropped": 0,
        }
        session.remember("hypothesis", hypothesis)
        session.remember("model_comparison", pd.DataFrame([{
            "Model": "Random Forest", "Accuracy": 0.95
        }]))
        session.remember("roc_auc", pd.DataFrame([{
            "Metric": "ROC-AUC", "AUC": 0.97
        }]))
        session.remember("feature_importance", pd.DataFrame([{
            "Feature": "score", "Importance": 1.0
        }]))

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.html"
            path = generate_custom_html_report(data, output, session=session)
            html = Path(path).read_text(encoding="utf-8")

        self.assertIn("Stored Hypothesis Test Results", html)
        self.assertIn("Welch t-test", html)
        self.assertIn("Stored Model Comparison Results", html)
        self.assertIn("Random Forest", html)
        self.assertIn("Stored ROC-AUC Results", html)
        self.assertIn("0.97", html)
        self.assertIn("Stored Feature Importance Results", html)
        self.assertIn("score", html)


if __name__ == "__main__":
    unittest.main()
