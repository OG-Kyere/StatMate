import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

from statmate import (
    calculate_regression_diagnostics,
    calculate_vif,
    fit_linear_regression,
    has_recommended_sample_size,
    load_dataset,
    save_custom_regression_results,
    select_custom_regression_variables,
)


class RegressionTests(unittest.TestCase):
    def test_custom_numeric_regression_uses_selected_columns_and_complete_rows(self):
        data = pd.DataFrame(
            {
                "outcome": [3.0, 5.0, 7.0, 9.0, None],
                "predictor_one": [1.0, 2.0, 3.0, 4.0, 5.0],
                "predictor_two": [2.0, 1.0, 2.0, 1.0, 2.0],
                "category": ["a", "b", "a", "b", "a"],
            }
        )

        model, complete_rows = fit_linear_regression(
            data, "outcome", ["predictor_one", "predictor_two"]
        )

        self.assertEqual(complete_rows, 4)
        self.assertEqual(model.nobs, 4)
        self.assertAlmostEqual(model.params["predictor_one"], 2.0)

    def test_iris_regression_columns_remain_available(self):
        data = load_dataset()
        model, complete_rows = fit_linear_regression(
            data,
            "petal length (cm)",
            ["sepal length (cm)", "sepal width (cm)", "petal width (cm)"],
        )

        self.assertEqual(complete_rows, 150)
        self.assertGreater(model.rsquared, 0.9)

    def test_custom_regression_excludes_non_finite_selected_values(self):
        data = pd.DataFrame(
            {"outcome": [3.0, 5.0, 7.0, 9.0, 11.0], "predictor": [1.0, 2.0, np.inf, 4.0, 5.0]}
        )

        model, complete_rows = fit_linear_regression(data, "outcome", ["predictor"])

        self.assertEqual(complete_rows, 4)
        self.assertEqual(model.nobs, 4)

    def test_custom_regression_rejects_perfectly_collinear_predictors(self):
        data = pd.DataFrame(
            {"outcome": [3.0, 5.0, 7.0, 9.0], "first": [1.0, 2.0, 3.0, 4.0], "second": [2.0, 4.0, 6.0, 8.0]}
        )

        with self.assertRaisesRegex(ValueError, "perfectly collinear"):
            fit_linear_regression(data, "outcome", ["first", "second"])

    def test_vif_identifies_high_near_collinearity(self):
        data = pd.DataFrame(
            {
                "outcome": [5, 7, 9, 11, 13, 15],
                "first": [1, 2, 3, 4, 5, 6],
                "second": [1.1, 1.9, 3.1, 3.9, 5.1, 5.9],
            }
        )

        vif_results = calculate_vif(data, "outcome", ["first", "second"])

        self.assertTrue((vif_results["VIF"] >= 10).any())

    def test_sample_size_rule_of_thumb_uses_ten_rows_per_predictor(self):
        self.assertTrue(has_recommended_sample_size(20, 2))
        self.assertFalse(has_recommended_sample_size(19, 2))

    def test_custom_diagnostics_return_assumption_and_influence_statistics(self):
        data = pd.DataFrame(
            {"outcome": [3.0, 5.0, 7.2, 8.8, 11.1, 13.0], "predictor": [1, 2, 3, 4, 5, 6]}
        )
        model, _ = fit_linear_regression(data, "outcome", ["predictor"])

        diagnostics = calculate_regression_diagnostics(model)

        self.assertIn("shapiro_p_value", diagnostics)
        self.assertIn("breusch_pagan_p_value", diagnostics)
        self.assertIn("durbin_watson", diagnostics)
        self.assertIn("influential_observations", diagnostics)
        self.assertGreaterEqual(diagnostics["influential_observations"], 0)

    def test_custom_regression_results_are_exported_to_csv(self):
        data = pd.DataFrame(
            {"outcome": [3.0, 5.0, 7.2, 8.8, 11.1, 13.0], "predictor": [1, 2, 3, 4, 5, 6]}
        )
        model, complete_rows = fit_linear_regression(data, "outcome", ["predictor"])
        diagnostics = calculate_regression_diagnostics(model)
        vif_results = calculate_vif(data, "outcome", ["predictor"])

        with TemporaryDirectory() as temporary_directory:
            save_custom_regression_results(
                model,
                "outcome",
                ["predictor"],
                complete_rows,
                vif_results,
                diagnostics,
                temporary_directory,
            )

            coefficients = pd.read_csv(Path(temporary_directory) / "custom_regression_coefficients.csv")
            self.assertIn("Coefficient", coefficients.columns)
            self.assertTrue((Path(temporary_directory) / "custom_regression_diagnostics.csv").is_file())
            self.assertTrue((Path(temporary_directory) / "custom_regression_vif.csv").is_file())

    def test_custom_variable_picker_accepts_numeric_outcome_and_predictors(self):
        data = pd.DataFrame(
            {"outcome": [1.0, 2.0], "predictor": [3.0, 4.0], "label": ["a", "b"]}
        )

        with patch("builtins.input", side_effect=["1", "2"]):
            outcome, predictors = select_custom_regression_variables(data)

        self.assertEqual(outcome, "outcome")
        self.assertEqual(predictors, ["predictor"])


if __name__ == "__main__":
    unittest.main()
