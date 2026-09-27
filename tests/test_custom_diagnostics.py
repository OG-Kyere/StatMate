"""Tests for stored custom regression diagnostics."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from regression_analysis import custom_session_regression_diagnostics


class CustomSessionDiagnosticsTests(unittest.TestCase):
    def test_requires_regression_before_custom_diagnostics(self):
        session = SimpleNamespace(results={})

        with patch("builtins.print") as printer:
            result = custom_session_regression_diagnostics(session)

        self.assertIsNone(result)
        self.assertTrue(any("Run option 6 first" in str(call) for call in printer.call_args_list))

    def test_returns_stored_diagnostics_without_refitting(self):
        diagnostics = {
            "shapiro_p_value": 0.25,
            "breusch_pagan_p_value": 0.40,
            "durbin_watson": 1.95,
            "influential_observations": 1,
            "cook_threshold": 0.04,
        }
        session = SimpleNamespace(results={
            "regression": {
                "outcome": "score",
                "predictors": ["age", "income"],
                "complete_rows": 100,
                "diagnostics": diagnostics,
                "vif": pd.DataFrame({
                    "Predictor": ["age", "income"],
                    "VIF": [1.2, 1.4],
                }),
            }
        })

        result = custom_session_regression_diagnostics(session)

        self.assertIs(result, diagnostics)


if __name__ == "__main__":
    unittest.main()
