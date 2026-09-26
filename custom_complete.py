"""End-to-end analysis orchestration for custom StatMate datasets."""

def custom_complete_analysis(data, api):
    """Run the non-interactive core of a complete custom-data analysis.

    Analyses that require a user to choose an outcome/grouping variable remain
    available through their dedicated menu options. This workflow focuses on
    analyses that are valid without guessing the user's statistical question.
    """
    print("\n" + "=" * 60)
    print("CUSTOM DATASET COMPLETE ANALYSIS")
    print("=" * 60)

    print("\n1. Dataset overview")
    api.explore_data(data)

    print("\n2. Descriptive statistics")
    api.descriptive_statistics(data)

    print("\n3. Correlation analysis")
    api.correlation_analysis(data)

    print("\n" + "-" * 60)
    print("Complete automatic analysis finished.")
    print(
        "Outcome-dependent analyses were not guessed. "
        "Use options 4-11 to select variables for hypothesis tests, "
        "regression, classification, ROC-AUC, feature importance, "
        "or prediction."
    )
    return data
