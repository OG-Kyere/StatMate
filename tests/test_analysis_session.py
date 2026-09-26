"""Tests for in-memory custom analysis session state."""

import unittest

from analysis_session import AnalysisSession


class AnalysisSessionTests(unittest.TestCase):
    def test_remember_stores_non_none_results(self):
        session = AnalysisSession()
        result = {"outcome": "score", "predictors": ["age"]}

        returned = session.remember("regression", result)

        self.assertIs(returned, result)
        self.assertIs(session.results["regression"], result)

    def test_remember_ignores_cancelled_analysis(self):
        session = AnalysisSession()
        session.remember("classification", None)

        self.assertNotIn("classification", session.results)

    def test_reset_clears_configuration_and_results(self):
        session = AnalysisSession(
            regression={"outcome": "y"},
            classification={"target": "group"},
            hypothesis={"response": "score"},
        )
        session.results["regression"] = {"ok": True}

        session.reset()

        self.assertIsNone(session.regression)
        self.assertIsNone(session.classification)
        self.assertIsNone(session.hypothesis)
        self.assertEqual(session.results, {})


if __name__ == "__main__":
    unittest.main()
