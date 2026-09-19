import contextlib
import io
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, ttest_ind, kruskal
import statmate as app


class CorrelationTests(unittest.TestCase):
    def test_pairwise_cleaning_and_no_mutation(self):
        data = pd.DataFrame({'a': [1, 2, 3, 4, np.inf], 'b': [2, 5, 4, None, 8], 'c': [None, 1, 4, 6, 7]})
        original = data.copy(deep=True)
        matrix, rows = app.calculate_correlations(data)
        self.assertEqual([r['n'] for r in rows], [3, 3, 3])
        expected = pearsonr([1, 2, 3], [2, 5, 4])
        self.assertAlmostEqual(rows[0]['r'], expected.statistic)
        self.assertAlmostEqual(rows[0]['p'], expected.pvalue)
        self.assertEqual(matrix.loc['b', 'a'], matrix.loc['a', 'b'])
        pd.testing.assert_frame_equal(data, original)

    def test_invalid_pairs(self):
        for values, reason in [([1, 1, 1], 'constant'), ([1, 2, None], 'at least 3'), ([None]*3, 'at least 3')]:
            _, rows = app.calculate_correlations(pd.DataFrame({'a': [1, 2, 3], 'b': pd.Series(values, dtype=float)}))
            self.assertIn(reason, rows[0]['reason'])
            self.assertTrue(np.isnan(rows[0]['p']))

    def test_constant_only_after_pairwise_cleaning(self):
        _, rows = app.calculate_correlations(pd.DataFrame({'a': [1, 1, 1, 2], 'b': [1, 2, 3, None]}))
        self.assertIn('constant', rows[0]['reason'])

    def test_no_numeric_columns(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            app.correlation_analysis(pd.DataFrame({'label': ['a', 'b']}))
        self.assertIn('at least two', output.getvalue())


class HypothesisTests(unittest.TestCase):
    def data(self, groups=3):
        values = np.array([-1.5, -1, -.7, -.3, 0, .2, .5, .8, 1.1, 1.6])
        return pd.DataFrame({'score': np.concatenate([values + i*3 for i in range(groups)]),
                             'group': np.repeat(list('abc')[:groups], len(values))})

    def test_welch_two_groups(self):
        result = app.compare_groups(self.data(2), 'score', 'group')
        self.assertEqual(result['name'], 'Welch t-test')
        self.assertAlmostEqual(result['p'], ttest_ind(*result['groups'], equal_var=False).pvalue)

    def test_anova_and_welch_anova(self):
        data = self.data()
        self.assertEqual(app.compare_groups(data, 'score', 'group')['name'], 'One-way ANOVA')
        data.loc[data.group == 'c', 'score'] *= 20
        self.assertEqual(app.compare_groups(data, 'score', 'group')['name'], 'Welch ANOVA')

    def test_rank_tests(self):
        for count, expected in [(2, 'Mann-Whitney U'), (3, 'Kruskal-Wallis')]:
            data = self.data(count)
            data.loc[0, 'score'] = 1000
            result = app.compare_groups(data, 'score', 'group')
            self.assertEqual(result['name'], expected)
            if count == 3:
                self.assertAlmostEqual(result['p'], kruskal(*result['groups']).pvalue)

    def test_clean_selected_columns_only(self):
        data = self.data(2)
        data['irrelevant'] = np.nan
        data.loc[0, 'score'] = np.inf
        data.loc[10, 'group'] = None
        result = app.compare_groups(data, 'score', 'group')
        self.assertEqual(result['dropped'], 2)
        self.assertEqual([len(g) for g in result['groups']], [9, 9])

    def test_reject_invalid_data(self):
        cases = [pd.DataFrame({'score': [1, 2, 3], 'group': ['a']*3}),
                 pd.DataFrame({'score': [1, 2, 3, 4], 'group': ['a', 'a', 'b', 'b']}),
                 pd.DataFrame({'score': [1, 1, 1, 2, 3, 4], 'group': ['a']*3+['b']*3})]
        for data in cases:
            with self.subTest(data=data), self.assertRaises(ValueError):
                app.compare_groups(data, 'score', 'group')
        with self.assertRaises(ValueError):
            app.compare_groups(self.data(), 'missing', 'group')

    def test_group_with_all_missing_responses_is_not_dropped(self):
        data = self.data()
        data.loc[data.group == 'c', 'score'] = np.nan
        with self.assertRaisesRegex(ValueError, 'no groups were silently dropped'):
            app.compare_groups(data, 'score', 'group')

    def test_large_groups_skip_shapiro(self):
        data = pd.DataFrame({'score': np.tile(np.arange(5001), 2), 'group': np.repeat(['a', 'b'], 5001)})
        with patch.object(app, 'shapiro', side_effect=AssertionError('must skip')):
            result = app.compare_groups(data, 'score', 'group')
        self.assertEqual(result['normality'], [None, None])
        self.assertEqual(result['name'], 'Mann-Whitney U')

    def test_small_nonnormal_multigroup_is_rejected(self):
        data = pd.DataFrame({'score': [0, 0, 0, 100]*3, 'group': np.repeat(['a', 'b', 'c'], 4)})
        with self.assertRaisesRegex(ValueError, 'at least 5'):
            app.compare_groups(data, 'score', 'group')

    def test_nullable_numeric_and_unused_categories(self):
        data = self.data(2)
        data['score'] = data['score'].astype('Float64')
        data.loc[0, 'score'] = pd.NA
        data['group'] = pd.Categorical(data['group'], categories=['a', 'b', 'unused'])
        self.assertEqual(app.compare_groups(data, 'score', 'group')['dropped'], 1)

    def test_picker_invalid_cancel_and_valid(self):
        for answers in [[''], ['0'], ['-1'], ['abc'], ['99'], ['1', '2']]:
            with patch('builtins.input', side_effect=answers):
                self.assertIsNone(app.select_hypothesis_variables(self.data()))
        with patch('builtins.input', side_effect=['1', '1']):
            self.assertEqual(app.select_hypothesis_variables(self.data()), ('score', 'group'))

    def test_posthoc_paths(self):
        for method in ['anova', 'welch', 'rank']:
            data = self.data()
            if method == 'welch':
                data.loc[data.group == 'c', 'score'] *= 20
            if method == 'rank':
                data.loc[0, 'score'] = 1000
            with patch('builtins.input', side_effect=['1', '1']), contextlib.redirect_stdout(io.StringIO()) as output:
                app.custom_hypothesis_analysis(data, post_hoc=True)
            self.assertIn('Tukey' if method == 'anova' else 'Holm', output.getvalue())

    def test_no_posthoc_for_two_groups_or_nonsignificance(self):
        for data in [self.data(2), self.data().assign(score=np.tile(np.arange(10), 3))]:
            with patch('builtins.input', side_effect=['1', '1']), contextlib.redirect_stdout(io.StringIO()) as output:
                app.custom_hypothesis_analysis(data, post_hoc=True)
            self.assertIn('No post-hoc', output.getvalue())

    def test_custom_menu_and_switch_back(self):
        with patch('builtins.input', side_effect=['16', '4', '5', '17', '4', '0']), patch.object(app, 'load_custom_dataset', return_value=self.data()), patch.object(app, 'custom_hypothesis_analysis') as custom, patch.object(app, 'statistical_tests') as iris:
            app.main()
        self.assertEqual(custom.call_count, 2)
        self.assertTrue(custom.call_args.kwargs['post_hoc'])
        iris.assert_called_once()

    def test_iris_original_tests_remain_noninteractive(self):
        with patch('builtins.input', side_effect=AssertionError('unexpected prompt')), contextlib.redirect_stdout(io.StringIO()) as output:
            app.statistical_tests(app.load_dataset())
            app.post_hoc_analysis(app.load_dataset())
        self.assertIn('ONE-WAY ANOVA', output.getvalue())
        self.assertIn('Tukey', output.getvalue())


class IrisIntegrationTests(unittest.TestCase):
    def test_complete_analysis_generates_ml_regression_and_reports(self):
        original = Path.cwd()
        with TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                with patch('builtins.input', side_effect=['12', '0']), patch.object(app.plt, 'show'), contextlib.redirect_stdout(io.StringIO()):
                    app.main()
                for path in ['results/model_comparison.csv', 'results/roc_auc_results.csv',
                             'results/feature_importance.csv', 'results/vif_results.csv',
                             'reports/statmate_report.txt', 'reports/statmate_report.html',
                             'figures/regression_qq_plot.png', 'figures/confusion_matrix.png']:
                    self.assertGreater(Path(path).stat().st_size, 0)
                self.assertEqual(len(pd.read_csv('results/model_comparison.csv')), 4)
            finally:
                app.plt.close('all')
                os.chdir(original)


if __name__ == '__main__':
    unittest.main()

