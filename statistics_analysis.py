"""Reusable correlation and independent-group inference for custom datasets."""

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, shapiro, f_oneway, levene, ttest_ind, mannwhitneyu, kruskal
from statsmodels.stats.oneway import anova_oneway
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def calculate_correlations(data):
    """Return a matrix and pairwise tests using finite, paired observations."""
    numeric = data.select_dtypes(include="number").select_dtypes(exclude="complex")
    numeric = numeric.astype(float).replace([np.inf, -np.inf], np.nan)
    matrix = pd.DataFrame(np.nan, index=numeric.columns, columns=numeric.columns)
    results = []
    for column in numeric:
        values = numeric[column].dropna()
        if len(values) >= 3 and values.nunique() > 1:
            matrix.loc[column, column] = 1.0
    for first, second in combinations(numeric.columns, 2):
        pair = numeric[[first, second]].dropna()
        result = dict(first=first, second=second, n=len(pair), r=np.nan, p=np.nan, reason="")
        if len(pair) < 3:
            result["reason"] = "at least 3 paired observations are required"
        elif (pair.nunique() < 2).any():
            result["reason"] = "a variable is constant in the paired observations"
        else:
            r, p_value = pearsonr(pair[first], pair[second])
            if np.isfinite(r) and np.isfinite(p_value):
                result.update(r=float(r), p=float(p_value))
                matrix.loc[first, second] = matrix.loc[second, first] = r
            else:
                result["reason"] = "numerical instability prevented a finite result"
        results.append(result)
    return matrix, results


def correlation_analysis(data):
    """Print pairwise Pearson coefficients, sample counts and unadjusted p-values."""
    print("\nCORRELATION ANALYSIS")
    matrix, results = calculate_correlations(data)
    if len(matrix) < 2:
        print("Warning: Correlation requires at least two real numeric columns.")
        return
    print("\nPearson Correlation Matrix (NaN means unavailable):")
    print(matrix.round(3))
    print("\nPairwise complete finite observations; p-values are unadjusted for multiple tests.")
    for result in results:
        print(f"{result['first']} vs {result['second']} (n={result['n']})")
        if result["reason"]:
            print(f"  Skipped: {result['reason']}.")
        else:
            print(f"  Pearson r = {result['r']:.3f}; p-value = {result['p']:.5f}")


def select_hypothesis_variables(data):
    """Select a real numeric response and an explicitly categorical grouping."""
    numeric = data.select_dtypes(include="number").select_dtypes(exclude="complex").columns.tolist()
    categorical = data.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist()
    if not numeric or not categorical:
        print("Warning: Choose a dataset with a numeric response and a categorical grouping column.")
        print("Numeric category codes must be converted to text/category before loading.")
        return None
    selected = []
    for label, columns in [("numeric response", numeric), ("categorical grouping", categorical)]:
        print(f"\nSelect {label}:")
        for index, column in enumerate(columns, 1):
            print(f"{index}. {column}")
        choice = input("Enter a number (Enter cancels): ").strip()
        if not choice:
            print("Analysis cancelled.")
            return None
        try:
            index = int(choice)
            if not 1 <= index <= len(columns):
                raise ValueError
        except ValueError:
            print("Warning: Invalid selection; returning to the menu.")
            return None
        selected.append(columns[index - 1])
    return tuple(selected)


def compare_groups(data, response, grouping):
    """Analyze independent groups; keep cleaning and test selection reusable."""
    numeric = data.select_dtypes(include="number").select_dtypes(exclude="complex").columns
    categorical = data.select_dtypes(include=["object", "string", "category", "bool"]).columns
    if response not in numeric or grouping not in categorical or response == grouping:
        raise ValueError("Select a real numeric response and a categorical grouping column.")
    clean = data[[response, grouping]].copy()
    clean[response] = clean[response].astype(float).replace([np.inf, -np.inf], np.nan)
    labels = clean[grouping].dropna().unique()
    clean = clean.dropna()
    groups = [clean.loc[clean[grouping] == label, response].to_numpy() for label in labels]
    if len(groups) < 2:
        raise ValueError("At least two observed groups are required.")
    if any(len(group) < 3 for group in groups):
        raise ValueError("Every observed group needs at least 3 finite responses; no groups were silently dropped.")
    if any(np.unique(group).size < 2 for group in groups):
        raise ValueError("A group has a constant response; group comparison was skipped.")
    normality = [float(shapiro(group).pvalue) if len(group) <= 5000 else None for group in groups]
    normal = all(p is not None and np.isfinite(p) and p >= 0.05 for p in normality)
    variance_p = float(levene(*groups, center="median").pvalue)
    if normal:
        if len(groups) == 2:
            name = "Welch t-test"
            statistic, p_value = ttest_ind(*groups, equal_var=False)
        elif np.isfinite(variance_p) and variance_p >= 0.05:
            name = "One-way ANOVA"
            statistic, p_value = f_oneway(*groups)
        else:
            name = "Welch ANOVA"
            test = anova_oneway(groups, use_var="unequal")
            statistic, p_value = test.statistic, test.pvalue
    elif len(groups) == 2:
        name = "Mann-Whitney U"
        statistic, p_value = mannwhitneyu(*groups, alternative="two-sided", method="auto")
    else:
        if any(len(group) < 5 for group in groups):
            raise ValueError("Kruskal-Wallis requires at least 5 finite responses per group in this CLI.")
        name = "Kruskal-Wallis"
        statistic, p_value = kruskal(*groups)
    if not np.isfinite(statistic) or not np.isfinite(p_value):
        raise ValueError("Numerical instability prevented a finite test result.")
    return dict(name=name, statistic=float(statistic), p=float(p_value), normality=normality,
                variance_p=variance_p, labels=labels, groups=groups, clean=clean,
                dropped=len(data) - len(clean))


def custom_hypothesis_analysis(data, post_hoc=False):
    selection = select_hypothesis_variables(data)
    if selection is None:
        return
    response, grouping = selection
    try:
        result = compare_groups(data, response, grouping)
    except ValueError as error:
        print(f"Warning: {error}")
        return
    print(f"\nResponse: {response}; grouping: {grouping}; excluded rows: {result['dropped']}")
    print("Independent observations are required. Assumption screening is guidance, not proof.")
    for label, group, p in zip(result["labels"], result["groups"], result["normality"]):
        normality = f"{p:.5f}" if p is not None else "not assessed (n > 5000); using ranks"
        print(f"{label}: n={len(group)}, Shapiro-Wilk p={normality}")
    print(f"Median-centered Levene p={result['variance_p']:.5f}")
    print(f"{result['name']}: statistic={result['statistic']:.4f}, p={result['p']:.5f}")
    print("Rank tests compare distributions, not necessarily means or medians; small-sample/tied p-values can be approximate.")
    print("Reject H0 at 0.05." if result["p"] < 0.05 else "Fail to reject H0 at 0.05.")
    if not post_hoc:
        return result
    if len(result["groups"]) < 3 or result["p"] >= 0.05:
        print("No post-hoc tests: need at least three groups and a significant omnibus test.")
        return result
    if result["name"] == "One-way ANOVA":
        print(pairwise_tukeyhsd(result["clean"][response], result["clean"][grouping].astype(str)))
    else:
        pairs = list(combinations(range(len(result["groups"])), 2))
        rank_test = result["name"] == "Kruskal-Wallis"
        p_values = []
        for i, j in pairs:
            a, b = result["groups"][i], result["groups"][j]
            test = mannwhitneyu(a, b, alternative="two-sided", method="auto") if rank_test else ttest_ind(a, b, equal_var=False)
            p_values.append(test.pvalue)
        if not np.isfinite(p_values).all():
            print("Warning: Post-hoc comparison produced a non-finite result; skipped.")
            return result
        adjusted = multipletests(p_values, method="holm")[1]
        print("Pairwise " + ("Mann-Whitney U" if rank_test else "Welch t-tests") + " with Holm correction:")
        for (i, j), p in zip(pairs, adjusted):
            print(f"{result['labels'][i]} vs {result['labels'][j]}: adjusted p={p:.5f}")
    return result


