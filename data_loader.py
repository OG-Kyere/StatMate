"""Dataset loading, validation and profiling utilities for StatMate."""

from pathlib import Path

import pandas as pd
from sklearn.datasets import load_iris


def load_dataset():
    """Load the built-in Iris dataset from Scikit-learn."""
    iris = load_iris(as_frame=True)
    data = iris.frame
    data["target"] = data["target"].map(dict(enumerate(iris.target_names)))
    return data


def validate_dataset(data):
    """Return validation errors and warnings for an uploaded dataset."""
    errors, warnings = [], []
    if data.empty:
        errors.append("The dataset has no rows or no columns.")
        return errors, warnings
    if data.columns.duplicated().any():
        duplicates = data.columns[data.columns.duplicated()].tolist()
        errors.append("Column names must be unique. Duplicate column(s): " + ", ".join(map(str, duplicates)))
    if any(not str(column).strip() for column in data.columns):
        errors.append("Every column must have a name.")
    empty_columns = data.columns[data.isna().all()].tolist()
    if empty_columns:
        warnings.append("Entirely empty column(s): " + ", ".join(map(str, empty_columns)))
    if data.shape[0] < 2:
        warnings.append("The dataset has fewer than two rows; most analyses need more data.")
    if len(data.select_dtypes(include="number").columns) == 0:
        warnings.append("No numerical columns were detected; descriptive and correlation analyses will have limited output.")
    return errors, warnings


def print_dataset_profile(data, preview_rows=5):
    """Display a reusable overview for built-in and uploaded datasets."""
    print(f"\nNumber of observations: {data.shape[0]}")
    print(f"Number of variables: {data.shape[1]}")
    print("\nVariables and data types:")
    for column, dtype in data.dtypes.items():
        print(f" - {column}: {dtype}")
    missing_values = data.isna().sum()
    if missing_values.any():
        print("\nMissing values:")
        print(missing_values[missing_values > 0].to_string())
    else:
        print("\nMissing values: none")
    numerical_columns = data.select_dtypes(include="number").columns.tolist()
    categorical_columns = data.select_dtypes(exclude="number").columns.tolist()
    print(f"\nDetected numerical variables: {', '.join(map(str, numerical_columns)) or 'none'}")
    print(f"Detected categorical variables: {', '.join(map(str, categorical_columns)) or 'none'}")
    print(f"\nPreview (first {min(preview_rows, len(data))} rows):")
    print(data.head(preview_rows).to_string(index=False))


def load_custom_dataset():
    """Prompt for, load, validate and profile a CSV or Excel dataset."""
    print("\nLoad a CSV or Excel (.xlsx, .xls) dataset.")
    file_path = input("Enter the full path to the dataset (or press Enter to cancel): ").strip().strip('"')
    if not file_path:
        print("\nDataset loading cancelled.")
        return None
    path = Path(file_path).expanduser()
    if not path.is_file():
        print(f"\nWarning: File not found: {path}")
        return None
    suffix = path.suffix.lower()
    if suffix not in {".csv", ".xlsx", ".xls"}:
        print("\nWarning: Unsupported file type. Please choose a CSV, XLSX, or XLS file.")
        return None
    try:
        if suffix == ".csv":
            try:
                data = pd.read_csv(path)
            except UnicodeDecodeError:
                data = pd.read_csv(path, encoding="latin-1")
                print("\nNote: the CSV was read using Latin-1 encoding.")
        else:
            data = pd.read_excel(path)
    except (OSError, ValueError, pd.errors.ParserError, ImportError) as error:
        print(f"\nWarning: Could not load the dataset: {error}")
        return None
    data.columns = [str(column).strip() for column in data.columns]
    errors, warnings = validate_dataset(data)
    if errors:
        print("\nWarning: The dataset could not be used:")
        for error in errors:
            print(f" - {error}")
        return None
    print("\nDataset loaded successfully.")
    for warning in warnings:
        print(f"Warning: {warning}")
    print_dataset_profile(data)
    return data
