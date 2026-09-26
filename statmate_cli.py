"""Command-line interface for StatMate.

Analysis functions remain importable from statmate.py for backwards compatibility.
This module owns dataset state, menu rendering, and command routing.
"""


def run_cli(api):
    data = api.load_dataset()
    dataset_name = "Built-in Iris dataset"
    iris_workflow = True

    while True:
        print("\n" + "=" * 60)
        print("                    STATMATE")
        print("          Statistical Analysis Assistant")
        print("=" * 60)
        print(f"Current dataset: {dataset_name}")

        print("\n1. Explore Dataset")
        print("2. Descriptive Statistics")
        print("3. Correlation Analysis")
        print("4. Statistical Tests")
        print("5. Post-Hoc Analysis")
        print("6. Regression Analysis")
        print("7. Machine Learning")
        print("8. Model Comparison")
        print("9. ROC-AUC Analysis")
        print("10. Feature Importance")
        print("11. Predict New Flower")
        print("12. Run Complete Analysis")
        print("13. Generate Statistical Report")
        print("14. Generate HTML Report")
        print("15. Regression Diagnostics")
        print("16. Load Custom CSV/Excel Dataset")
        print("17. Switch Back to Iris Dataset")
        print("0. Exit")

        choice = input("\nEnter your choice: ").strip()

        if choice == "15" and not iris_workflow:
            print(
                "\nWarning: This option currently uses Iris-specific variables. "
                "Custom datasets currently support options 1 through 14."
            )
        elif choice == "1":
            api.explore_data(data)
        elif choice == "2":
            api.descriptive_statistics(data)
        elif choice == "3":
            api.correlation_analysis(data)
        elif choice == "4":
            api.statistical_tests(data) if iris_workflow else api.custom_hypothesis_analysis(data)
        elif choice == "5":
            api.post_hoc_analysis(data) if iris_workflow else api.custom_hypothesis_analysis(data, post_hoc=True)
        elif choice == "6":
            api.regression_analysis(data) if iris_workflow else api.custom_regression_analysis(data)
        elif choice == "7":
            api.machine_learning_analysis(data) if iris_workflow else api.custom_classification_analysis(data)
        elif choice == "8":
            api.compare_models(data) if iris_workflow else api.custom_model_comparison(data)
        elif choice == "9":
            api.roc_curve_analysis(data) if iris_workflow else api.custom_roc_auc_analysis(data)
        elif choice == "10":
            api.feature_importance_analysis(data) if iris_workflow else api.custom_feature_importance_analysis(data)
        elif choice == "11":
            api.predict_new_flower(data) if iris_workflow else api.custom_prediction(data)
        elif choice == "12" and not iris_workflow:
            api.custom_complete_analysis(data)
        elif choice == "12":
            api.explore_data(data)
            api.descriptive_statistics(data)
            api.correlation_analysis(data)
            api.statistical_tests(data)
            api.post_hoc_analysis(data)
            api.regression_analysis(data)
            api.regression_diagnostics(data)
            api.machine_learning_analysis(data)
            api.compare_models(data)
            api.roc_curve_analysis(data)
            api.feature_importance_analysis(data)
            api.create_visualizations(data)
            api.generate_report(data)
            api.generate_html_report(data)
        elif choice == "13":
            api.generate_report(data) if iris_workflow else api.generate_custom_text_report(data)
        elif choice == "14":
            api.generate_html_report(data) if iris_workflow else api.generate_custom_html_report(data)
        elif choice == "15":
            api.regression_diagnostics(data)
        elif choice == "16":
            custom_data = api.load_custom_dataset()
            if custom_data is not None:
                data = custom_data
                dataset_name = "Custom dataset"
                iris_workflow = False
        elif choice == "17":
            data = api.load_dataset()
            dataset_name = "Built-in Iris dataset"
            iris_workflow = True
            print("\nSwitched back to the built-in Iris dataset.")
        elif choice == "0":
            print("\nThank you for using StatMate!")
            print("Goodbye!")
            break
        else:
            print("\nWarning: Invalid choice. Please enter a number from 0 to 17.")
