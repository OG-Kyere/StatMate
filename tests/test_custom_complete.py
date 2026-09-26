"""Tests for the custom complete-analysis workflow."""

import unittest
from unittest.mock import Mock

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

        result = custom_complete_analysis(data, api)

        api.explore_data.assert_called_once_with(data)
        api.descriptive_statistics.assert_called_once_with(data)
        api.correlation_analysis.assert_called_once_with(data)
        self.assertIs(result, data)

    def test_does_not_run_outcome_dependent_analyses_automatically(self):
        data = pd.DataFrame({"x": [1, 2, 3], "y": [3, 2, 1]})
        api = Mock()

        custom_complete_analysis(data, api)

        api.custom_hypothesis_analysis.assert_not_called()
        api.custom_regression_analysis.assert_not_called()
        api.custom_classification_analysis.assert_not_called()


if __name__ == "__main__":
    unittest.main()
