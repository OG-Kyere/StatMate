import unittest
from unittest.mock import patch
import contextlib
import io

import numpy as np
import pandas as pd

import custom_ml
import statmate as app


class CustomClassificationTests(unittest.TestCase):
    def data(self):
        rng = np.random.default_rng(42)
        n = 60
        return pd.DataFrame({
            "age": rng.normal(30, 5, n),
            "income": rng.normal(1000, 150, n),
            "region": np.tile(["north", "south", "east"], 20),
            "target": np.repeat(["no", "yes"], 30),
        })

    def test_mixed_predictors_and_four_models(self):
        results = custom_ml.evaluate_classifiers(
            self.data(), "target", ["age", "income", "region"]
        )
        self.assertEqual(len(results), 4)
        self.assertEqual(
            set(results["Model"]),
            {"Logistic Regression", "K-Nearest Neighbors", "Decision Tree", "Random Forest"},
        )
        self.assertTrue(results["Accuracy"].between(0, 1).all())

    def test_missing_predictors_are_imputed(self):
        data = self.data()
        data.loc[0:5, "age"] = np.nan
        data.loc[6:10, "region"] = None
        results = custom_ml.evaluate_classifiers(
            data, "target", ["age", "region"]
        )
        self.assertEqual(len(results), 4)

    def test_invalid_classification_inputs(self):
        data = self.data()
        with self.assertRaises(ValueError):
            custom_ml.evaluate_classifiers(data, "target", [])
        with self.assertRaises(ValueError):
            custom_ml.evaluate_classifiers(data, "target", ["target"])
        one_class = data.assign(target="yes")
        with self.assertRaises(ValueError):
            custom_ml.evaluate_classifiers(one_class, "target", ["age"])

    def test_cv_adapts_to_smallest_class(self):
        data = self.data().iloc[:12].copy()
        data["target"] = ["rare"] * 3 + ["common"] * 9
        results = custom_ml.evaluate_classifiers(data, "target", ["age", "region"])
        self.assertTrue((results["CV Folds"] == 3).all())

    def test_custom_menu_routes_option_seven(self):
        with patch("builtins.input", side_effect=["16", "7", "0"]),              patch.object(app, "load_custom_dataset", return_value=self.data()),              patch.object(app, "custom_classification_analysis") as custom:
            app.main()
        custom.assert_called_once()


if __name__ == "__main__":
    unittest.main()
