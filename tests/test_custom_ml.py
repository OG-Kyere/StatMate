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


    def test_custom_roc_auc_binary_and_multiclass(self):
        binary = custom_ml.custom_roc_auc_analysis
        with patch.object(custom_ml, "_classification_setup", return_value=("target", ["age", "region"])):
            with patch.object(custom_ml.Path, "mkdir"), patch.object(pd.DataFrame, "to_csv"):
                result = binary(self.data())
        self.assertTrue(result.loc[0, "AUC"] >= 0)
        multi = self.data()
        multi["target"] = np.repeat(["a", "b", "c"], 20)
        with patch.object(custom_ml, "_classification_setup", return_value=("target", ["age", "region"])):
            with patch.object(custom_ml.Path, "mkdir"), patch.object(pd.DataFrame, "to_csv"):
                result = custom_ml.custom_roc_auc_analysis(multi)
        self.assertEqual(result.loc[0, "Classes"], 3)

    def test_custom_feature_importance_preserves_transformed_names(self):
        with patch.object(custom_ml, "_classification_setup", return_value=("target", ["age", "region"])):
            with patch.object(custom_ml.Path, "mkdir"), patch.object(pd.DataFrame, "to_csv"):
                result = custom_ml.custom_feature_importance_analysis(self.data())
        self.assertAlmostEqual(result["Importance"].sum(), 1.0)
        self.assertTrue(result["Feature"].str.contains("region").any())

    def test_custom_prediction_returns_probabilities(self):
        with patch.object(custom_ml, "_classification_setup", return_value=("target", ["age", "region"])), \
             patch("builtins.input", side_effect=["30", "north"]):
            result = custom_ml.custom_prediction(self.data())
        self.assertIn(result["Prediction"], {"yes", "no"})
        self.assertAlmostEqual(sum(result["Probabilities"].values()), 1.0)

    def test_custom_menu_routes_options_eight_to_eleven(self):
        data = self.data()
        with patch("builtins.input", side_effect=["16", "8", "9", "10", "11", "0"]), \
             patch.object(app, "load_custom_dataset", return_value=data), \
             patch.object(app, "custom_model_comparison") as compare, \
             patch.object(app, "custom_roc_auc_analysis") as roc, \
             patch.object(app, "custom_feature_importance_analysis") as importance, \
             patch.object(app, "custom_prediction") as prediction:
            app.main()
        compare.assert_called_once()
        roc.assert_called_once()
        importance.assert_called_once()
        prediction.assert_called_once()


if __name__ == "__main__":
    unittest.main()
