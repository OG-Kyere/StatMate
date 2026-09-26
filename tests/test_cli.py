import contextlib
import io
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from statmate_cli import run_cli


class CliArchitectureTests(unittest.TestCase):
    def api(self):
        api = Mock()
        api.load_dataset.return_value = pd.DataFrame({"x": [1, 2]})
        api.load_custom_dataset.return_value = pd.DataFrame({"x": [3, 4]})
        return api

    def test_exit_does_not_run_analysis(self):
        api = self.api()
        with patch("builtins.input", side_effect=["0"]), contextlib.redirect_stdout(io.StringIO()):
            run_cli(api)
        api.load_dataset.assert_called_once()

    def test_custom_options_route_through_api(self):
        api = self.api()
        with patch("builtins.input", side_effect=["16", "7", "8", "9", "10", "11", "0"]), contextlib.redirect_stdout(io.StringIO()):
            run_cli(api)
        api.custom_classification_analysis.assert_called_once()
        api.custom_model_comparison.assert_called_once()
        api.custom_roc_auc_analysis.assert_called_once()
        api.custom_feature_importance_analysis.assert_called_once()
        api.custom_prediction.assert_called_once()

    def test_switch_back_restores_iris_routing(self):
        api = self.api()
        with patch("builtins.input", side_effect=["16", "17", "7", "0"]), contextlib.redirect_stdout(io.StringIO()):
            run_cli(api)
        self.assertEqual(api.load_dataset.call_count, 2)
        api.machine_learning_analysis.assert_called_once()


if __name__ == "__main__":
    unittest.main()
