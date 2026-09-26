# StatMate — Statistical Analysis Assistant

StatMate is a Python CLI for statistics and machine learning. It starts with Scikit-learn's Iris dataset (150 flowers, four measurements, three species) and accepts custom CSV and Excel files.

## Installation

```sh
git clone https://github.com/OG-Kyere/StatMate.git
cd StatMate
python -m pip install -r requirements.txt
python statmate.py
```

On Windows, use `py` instead of `python` if needed. Excel loading supports `.xlsx` (openpyxl) and `.xls` (xlrd), reading the first worksheet. CSV decoding falls back to Latin-1 if UTF-8 fails.

## Menu

```text
1. Explore Dataset
2. Descriptive Statistics
3. Correlation Analysis
4. Statistical Tests
5. Post-Hoc Analysis
6. Regression Analysis
7. Machine Learning
8. Model Comparison
9. ROC-AUC Analysis
10. Feature Importance
11. Predict New Flower
12. Run Complete Analysis
13. Generate Statistical Report
14. Generate HTML Report
15. Regression Diagnostics
16. Load Custom CSV/Excel Dataset
17. Switch Back to Iris Dataset
0. Exit
```

Option 16 validates the file, empty data and column names, then profiles types, missing values and a preview. Option 17 restores Iris. Custom datasets support options 1–11. Options 7–11 provide reusable classification, model comparison, ROC-AUC, feature importance and prediction with mixed numeric/categorical predictors. Options 12–15 remain Iris-specific. Custom regression diagnostics run within option 6.

## Current features

### Correlation

Pearson analysis selects real numeric columns and removes missing values and infinities separately for each pair. It prints pair counts, coefficients and unadjusted p-values. At least three paired observations are required; constant variables (including those constant only after pairwise cleaning) and non-finite results are skipped with a reason. The matrix uses the same observations and displays `NaN` for unavailable results. Fewer than two numeric columns returns a helpful message. P-values are exploratory and are not corrected for multiple comparisons.

### Hypothesis testing

Iris retains its original petal-length Shapiro-Wilk, ANOVA and Tukey HSD workflow without extra prompts.

For custom data, options 4 and 5 offer numbered numeric-response and categorical-group selections. Text, category and boolean group columns are supported. Numeric group codes must be converted to text/category before loading. Enter cancels; invalid input returns to the menu.

Only rows missing a selected variable or containing non-finite responses are excluded. Every observed group must retain at least three values with within-group variation. An entirely missing group is not silently discarded.

The CLI reports group sizes, Shapiro-Wilk normality and median-centered Levene variance checks, using alpha 0.05:

- No detected non-normality: Welch t-test for two groups; ordinary ANOVA for three or more if Levene detects no variance difference, otherwise Welch ANOVA.
- Detected non-normality or unassessed normality: two-sided Mann-Whitney U for two groups, Kruskal-Wallis for three or more. This CLI requires at least five values per group for Kruskal-Wallis.
- Option 5 runs post-hoc tests only after a significant omnibus test with at least three groups: Tukey HSD after ordinary ANOVA, Holm-adjusted pairwise Welch tests after Welch ANOVA, or Holm-adjusted pairwise Mann-Whitney tests after Kruskal-Wallis.

Independent observations are required; these tests are not for paired/repeated measurements. Screening is guidance, not proof of assumptions, especially in small samples. Rank tests compare distributions, not necessarily means or medians; tied/small-sample rank p-values can be approximate. Shapiro screening is skipped above 5,000 values per group and rank tests are used because [SciPy documents reduced p-value accuracy above 5,000](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html). Welch ANOVA uses [Statsmodels' unequal-variance implementation](https://www.statsmodels.org/stable/generated/statsmodels.stats.oneway.anova_oneway.html).

### Custom classification

For uploaded datasets, option 7 can run a reusable classification workflow. Choose a detected target and one or more predictors; numeric predictors are median-imputed and standardized, while categorical predictors are most-frequent-imputed and one-hot encoded. All preprocessing is fitted inside each cross-validation fold to avoid data leakage.

StatMate compares Logistic Regression, K-Nearest Neighbors, Decision Tree and Random Forest using stratified cross-validation. The number of folds automatically drops below five when the smallest target class requires it. Accuracy, macro precision, macro recall, macro F1 and accuracy variability are saved to `results/custom_classification_results.csv`. Custom ROC-AUC and feature importance are exported to `results/custom_roc_auc.csv` and `results/custom_feature_importance.csv`.

StatMate 2.0 also supports out-of-fold binary or macro one-vs-rest multiclass ROC-AUC, Random Forest transformed-feature importance, and single-observation Logistic Regression prediction with class probabilities. These workflows reuse leakage-safe preprocessing. Regression-ML workflows remain future work.

### Regression, ML and reporting

Custom OLS offers numeric outcome/predictor selection, complete finite rows, constant and perfect-collinearity checks, VIF, sample-size guidance, coefficient exports, and diagnostics. Diagnostics include Shapiro-Wilk, Breusch-Pagan, Durbin-Watson, Cook's distance, Q-Q plots and residuals versus fitted values.

Iris supports Logistic Regression, KNN, Decision Tree and Random Forest; scaled pipelines; train/test evaluation; five-fold stratified cross-validation; accuracy, precision, recall and F1 comparisons; out-of-fold one-vs-rest and macro-average ROC-AUC; feature importance; and new-flower prediction.

Text and HTML reports already exist. Option 12 generates the complete Iris workflow and supporting files. Individual reports can use existing saved results; rerun the relevant analyses to refresh those files and HTML-linked figures.

## Project structure

```text
StatMate/
├── statmate.py              # Public analysis API: loading, statistics, regression, Iris ML, reporting
├── statmate_cli.py          # CLI menu, dataset state and command routing
├── data_loader.py           # Built-in/custom loading, validation and profiling
├── statistics_analysis.py   # Robust correlation and custom group inference
├── regression_analysis.py   # OLS fitting, VIF, diagnostics and custom regression
├── iris_ml.py               # Built-in Iris classification, CV, ROC-AUC and prediction
├── iris_diagnostics.py      # Built-in Iris regression assumption diagnostics
├── reporting.py             # Visualizations, TXT and HTML report generation
├── custom_ml.py             # Reusable custom-dataset classification engine
├── README.md
├── requirements.txt
├── .gitignore
├── tests/
│   ├── test_cli.py
│   ├── test_data_loader.py
│   ├── test_regression.py
│   ├── test_statistics.py
│   └── test_custom_ml.py
├── figures/                 # Generated PNGs
├── results/                 # Generated CSVs
└── reports/                 # Generated text and HTML
```

## Outputs

Paths are relative to the working directory. Rerunning an analysis overwrites its named files. Correlation and hypothesis results print to the terminal; they do not yet export files.

| Directory | Files |
| --- | --- |
| `results/` | `model_comparison.csv`, `roc_auc_results.csv`, `feature_importance.csv`, `vif_results.csv`, `custom_classification_results.csv` |
| `results/` (custom OLS) | `custom_regression_coefficients.csv`, `custom_regression_diagnostics.csv`, `custom_regression_vif.csv` |
| `figures/` | `distributions.png`, `boxplots.png`, `correlation_heatmap.png`, `petal_length_by_species.png`, `confusion_matrix.png`, `model_comparison.png`, `roc_curves.png`, `feature_importance.png`, `regression_qq_plot.png`, `residuals_vs_fitted.png` |
| `figures/` (custom OLS) | `custom_regression_qq_plot.png`, `custom_regression_residuals_vs_fitted.png` |
| `reports/` | `statmate_report.txt`, `statmate_report.html` |

## Tests

```sh
python -m unittest discover -s tests -v
```

Tests cover regression fitting/diagnostics/exports, pairwise correlation cleaning, degenerate data, group-test selection, post-hoc routing, custom classification preprocessing/model comparison, invalid inputs, custom menu routing and the original Iris tests.

## Future improvements

- Extend custom classification outputs with visualizations and automated reports.
- Continue the modular refactor by extracting remaining Iris diagnostics and reporting from the backwards-compatible `statmate.py` API.
- Add effect sizes, confidence intervals, optional correlation multiplicity correction and hypothesis exports.
- Support paired/repeated-measures designs and richer small-sample inference.
- Add hyperparameter tuning, regression ML models and feature selection.
- Add PDF export and interactive dashboards.

## Author

**Kyere Ofosu Gideon** — statistics graduate and aspiring data professional. StatMate is an ongoing learning and portfolio project combining classical statistics and machine learning.
