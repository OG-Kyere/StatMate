import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import data_loader


class DataLoaderTests(unittest.TestCase):
    def test_builtin_iris_has_named_target(self):
        data = data_loader.load_dataset()
        self.assertIn("target", data.columns)
        self.assertEqual(set(data["target"].unique()), {"setosa", "versicolor", "virginica"})

    def test_validation_rejects_empty_dataset(self):
        errors, warnings = data_loader.validate_dataset(pd.DataFrame())
        self.assertTrue(errors)
        self.assertEqual(warnings, [])

    def test_validation_warns_when_no_numeric_columns(self):
        data = pd.DataFrame({"group": ["a", "b", "c"]})
        errors, warnings = data_loader.validate_dataset(data)
        self.assertEqual(errors, [])
        self.assertTrue(any("No numerical columns" in item for item in warnings))

    def test_csv_loader_strips_column_names(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            pd.DataFrame({" age ": [20, 21], "group": ["a", "b"]}).to_csv(path, index=False)
            with patch("builtins.input", return_value=str(path)), contextlib.redirect_stdout(io.StringIO()):
                loaded = data_loader.load_custom_dataset()
        self.assertEqual(loaded.columns.tolist(), ["age", "group"])


if __name__ == "__main__":
    unittest.main()
