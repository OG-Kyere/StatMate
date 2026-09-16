# 📊 StatMate — Statistical Analysis Assistant

StatMate is a Python-based statistical analysis and machine-learning application built around the Iris dataset.

The project combines statistical methods with machine-learning techniques to explore data, test relationships, build predictive models, and evaluate classification performance from one interactive command-line application.

## What StatMate Does

StatMate currently provides:

### Statistical Analysis

* Dataset exploration
* Descriptive statistics
* Pearson correlation analysis
* Shapiro-Wilk normality testing
* One-way ANOVA
* Tukey HSD post-hoc analysis
* Multiple linear regression

### Machine Learning

* Train/test splitting
* Feature standardization
* Logistic Regression
* K-Nearest Neighbors
* Decision Tree
* Random Forest
* Five-fold stratified cross-validation
* Accuracy, precision, recall, and F1-score
* Confusion matrix
* New flower species prediction

### Data Visualization

StatMate automatically generates visualizations including:

* Feature distributions
* Boxplots
* Correlation heatmap
* Petal length by species
* Classification confusion matrix
* Model performance comparison

## Dataset

StatMate uses the Iris dataset provided directly through `scikit-learn`.

The dataset contains 150 observations across three Iris species:

* Setosa
* Versicolor
* Virginica

Four flower measurements are used:

* Sepal length
* Sepal width
* Petal length
* Petal width

No external CSV download is required.

## Project Structure

```text
StatMate/
│
├── statmate.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── figures/
│   ├── distributions.png
│   ├── boxplots.png
│   ├── correlation_heatmap.png
│   ├── petal_length_by_species.png
│   ├── confusion_matrix.png
│   └── model_comparison.png
│
└── results/
    └── model_comparison.csv
```

## Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd StatMate
```

Install the required packages:

```bash
py -m pip install -r requirements.txt
```

## Running StatMate

Run:

```bash
py statmate.py
```

The application opens an interactive menu where you can choose the analysis you want to perform.

```text
============================================================
                    STATMATE
          Statistical Analysis Assistant
============================================================

1. Explore Dataset
2. Descriptive Statistics
3. Correlation Analysis
4. Statistical Tests
5. Post-Hoc Analysis
6. Regression Analysis
7. Machine Learning
8. Model Comparison
9. Predict New Flower
10. Run Complete Analysis
0. Exit
```

## Statistical Workflow

The statistical analysis follows this general workflow:

```text
Dataset
   ↓
Exploration
   ↓
Descriptive Statistics
   ↓
Correlation Analysis
   ↓
Normality Testing
   ↓
ANOVA
   ↓
Tukey HSD
   ↓
Regression Analysis
```

## Machine-Learning Workflow

```text
Iris Dataset
     ↓
Feature Selection
     ↓
Stratified Cross-Validation
     ↓
Model Training
     ↓
Model Evaluation
     ↓
Performance Comparison
     ↓
New Flower Prediction
```

## Technologies

The project uses:

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* SciPy
* Scikit-learn
* Statsmodels

## Learning Objectives

This project was developed to strengthen practical skills in:

* Statistical analysis
* Statistical hypothesis testing
* Regression modelling
* Machine learning
* Model evaluation
* Data visualization
* Python programming
* Reproducible data analysis

## Future Improvements

Planned improvements include:

* A graphical user interface
* More datasets
* Automated statistical test selection
* Additional machine-learning algorithms
* Hyperparameter tuning
* ROC and precision-recall curves
* Feature importance analysis
* Interactive dashboards
* Automated statistical reports

## Author

**Kyere Ofosu Gideon**

Statistics graduate interested in statistical modelling, data analysis, machine learning, and practical applications of statistics.

---

*StatMate is a learning and portfolio project demonstrating the integration of statistical analysis and machine learning using Python.*
