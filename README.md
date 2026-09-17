# StatMate — Statistical Analysis Assistant

StatMate is a Python-based statistical analysis assistant designed to bring common statistical and machine learning workflows into one simple command-line application.

The project starts with the **Iris dataset** from Scikit-learn and demonstrates how statistical analysis, hypothesis testing, regression, machine learning, model evaluation, and visualization can be combined into a practical data-analysis workflow. It can also load your own CSV or Excel dataset for reusable data exploration.

---

## 🚀 Features

### 📊 Statistical Analysis

StatMate currently provides:

* Dataset exploration
* Descriptive statistics
* Pearson correlation analysis
* CSV and Excel dataset loading with validation and data profiling
* Shapiro-Wilk normality testing
* One-way ANOVA
* Tukey HSD post-hoc analysis
* Multiple linear regression using Ordinary Least Squares (OLS)

### 🤖 Machine Learning

StatMate includes several classification models:

* Logistic Regression
* K-Nearest Neighbors (KNN)
* Decision Tree
* Random Forest

The application evaluates models using:

* Accuracy
* Precision
* Recall
* F1-score
* Accuracy standard deviation
* 5-fold stratified cross-validation

### 📈 ROC-AUC Analysis

StatMate performs ROC-AUC analysis using **5-fold cross-validation** and out-of-fold predicted probabilities.

The analysis:

* Generates one-vs-rest ROC curves
* Calculates macro-average ROC curves
* Calculates macro-average AUC
* Compares the classification models
* Saves ROC curves as a PNG image
* Saves AUC results as a CSV file

### 🌳 Feature Importance

StatMate uses a Random Forest classifier to investigate the relative importance of the four flower measurements.

The analysis:

* Calculates feature importance scores
* Ranks the predictors
* Displays the results in the terminal
* Creates a feature-importance visualization
* Saves the results as a CSV file

### 🔮 New Flower Prediction

Users can enter measurements for a new flower:

* Sepal length
* Sepal width
* Petal length
* Petal width

StatMate then uses a Logistic Regression model to predict the flower species.

---

## 📉 Data Visualization

StatMate generates visualizations including:

* Feature distribution plots
* Boxplots
* Correlation heatmaps
* Petal-length comparisons by species
* Classification confusion matrices
* Model-performance comparisons
* ROC curves
* Random Forest feature importance

Generated visualizations are stored in the `figures/` directory.

---

## 🗂️ Dataset

StatMate uses the **Iris dataset** provided by Scikit-learn.

The dataset contains **150 observations** across three Iris species:

* Setosa
* Versicolor
* Virginica

Four numerical measurements are used:

| Feature      | Description                        |
| ------------ | ---------------------------------- |
| Sepal Length | Length of the sepal in centimetres |
| Sepal Width  | Width of the sepal in centimetres  |
| Petal Length | Length of the petal in centimetres |
| Petal Width  | Width of the petal in centimetres  |

The dataset is loaded directly through Scikit-learn:

```python
from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)
```

No external CSV download is required.

### Load your own dataset

From the StatMate menu, choose **16. Load Custom CSV/Excel Dataset** and enter the full path to a `.csv`, `.xlsx`, or `.xls` file. StatMate checks that the file exists, has a supported format, contains rows and uniquely named columns, then displays its data types, missing values, detected numerical/categorical variables, and a five-row preview.

Custom datasets support the reusable exploration options:

* Explore Dataset
* Descriptive Statistics
* Correlation Analysis
* Regression Analysis — choose one numeric outcome and one or more numeric predictors from a numbered list. Rows missing a selected value are excluded from that model only. StatMate displays a VIF collinearity check: values of 5–10 warrant care, while values of 10 or more suggest reselecting overlapping predictors.

The hypothesis testing, machine-learning, prediction, visualization, and report options remain Iris-specific during this phase, so StatMate prevents them from running against incompatible custom columns. Choose **17. Switch Back to Iris Dataset** at any time to restore the complete original workflow.

---

## 🔬 Statistical Workflow

StatMate follows a structured statistical-analysis workflow:

```text
Load Dataset
     ↓
Explore Data
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

This provides a progression from basic data exploration to inferential statistical analysis.

---

## 🤖 Machine Learning Workflow

The machine-learning workflow follows:

```text
Prepare Features
      ↓
Train/Test Split
      ↓
Feature Scaling
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Cross-Validation
      ↓
Model Comparison
      ↓
ROC-AUC Analysis
      ↓
Feature Importance
      ↓
New Flower Prediction
```

This demonstrates several important concepts used in applied machine learning.

---

## 📁 Project Structure

```text
StatMate/
│
├── statmate.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── figures/
│   ├── distributions.png
│   ├── boxplots.png
│   ├── correlation_heatmap.png
│   ├── petal_length_by_species.png
│   ├── confusion_matrix.png
│   ├── model_comparison.png
│   ├── roc_curves.png
│   └── feature_importance.png
│
└── results/
    ├── model_comparison.csv
    ├── roc_auc_results.csv
    └── feature_importance.csv
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/OG-Kyere/StatMate.git
```

Navigate into the project:

```bash
cd StatMate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running StatMate

Run the application with:

```bash
python statmate.py
```

On Windows, if the `python` command is unavailable, you can use:

```bash
py statmate.py
```

You will be presented with an interactive menu:

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
9. ROC-AUC Analysis
10. Feature Importance
11. Predict New Flower
12. Run Complete Analysis
0. Exit
```

---

## 📊 Output Files

StatMate automatically saves important analysis results.

### Model Comparison

```text
results/model_comparison.csv
```

Contains cross-validation performance metrics for the four machine-learning models.

### ROC-AUC Results

```text
results/roc_auc_results.csv
```

Contains macro-average AUC values for each classification model.

### Feature Importance

```text
results/feature_importance.csv
```

Contains Random Forest feature-importance scores.

Visualizations are saved in:

```text
figures/
```

---

## 🛠️ Technologies Used

* **Python**
* **Pandas** — data manipulation
* **NumPy** — numerical computation
* **Matplotlib** — visualization
* **Seaborn** — statistical visualization
* **SciPy** — statistical tests
* **Statsmodels** — statistical modelling
* **Scikit-learn** — machine learning

---

## 🎯 Learning Objectives

This project was developed to strengthen practical understanding of:

* Exploratory Data Analysis (EDA)
* Descriptive statistics
* Statistical hypothesis testing
* Correlation analysis
* Analysis of variance
* Post-hoc testing
* Linear regression
* Classification
* Feature scaling
* Cross-validation
* Model evaluation
* ROC-AUC analysis
* Feature importance
* Data visualization
* Python programming
* Git and GitHub

---

## 🔮 Future Improvements

Possible future versions of StatMate may include:

* Automated statistical report generation
* Additional statistical tests
* More classification algorithms
* Regression machine-learning models
* Hyperparameter tuning
* Automated model selection
* Interactive dashboards
* Support for user-uploaded datasets
* More advanced feature-selection techniques
* Exportable PDF/HTML analysis reports
* Improved command-line interface

---

## 👨‍💻 Author

**Kyere Ofosu Gideon**

Statistics graduate and aspiring data professional with interests in:

* Statistical modelling
* Data analysis
* Machine learning
* Data visualization
* Risk analytics
* Applied statistics

This project is part of my continued development in statistical programming and data science.

---

## 📌 Project Status

StatMate is an ongoing learning and portfolio project.

The current version combines **classical statistical analysis with machine-learning techniques**, providing a practical demonstration of an end-to-end statistical data-analysis workflow.

---

⭐ If you find the project useful, feel free to explore the repository and follow the development of StatMate.
