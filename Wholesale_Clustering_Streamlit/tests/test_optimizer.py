"""Run at repository root: python -m unittest discover -s tests -v"""
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from optimizer import (WHOLESALE, baseline_config, recommended_config, evaluate,
                       reference_space, prep_apply, segment_profile, experiments)

ROOT = Path(__file__).resolve().parents[1]

class TestEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ws = pd.read_csv(ROOT / 'data/wholesale_customers.csv')

    def test_original_score(self):
        output, *_ = evaluate(self.ws, baseline_config(WHOLESALE))
        self.assertAlmostEqual(output['silhouette'], .29032849, places=5)

    def test_improvement(self):
        cfg = recommended_config(WHOLESALE)
        score, labels, model, X, prep = evaluate(self.ws, cfg,
                                                   reference=reference_space(self.ws, WHOLESALE),
                                                   validation=self.ws.Channel)
        self.assertAlmostEqual(score['silhouette'], .46850755, places=5)
        self.assertGreater(score['ari_vs_channel'], .62)
        self.assertGreater(score['smallest_cluster_pct'], 30)
        self.assertAlmostEqual(score['pca_variance_pct'], 78.35287, places=2)
        prof = segment_profile(self.ws, WHOLESALE, labels)
        self.assertEqual(prof['Số khách'].sum(), len(self.ws))
        pred = model.predict(prep_apply(self.ws[WHOLESALE].iloc[:3], cfg, prep))
        np.testing.assert_array_equal(pred, labels[:3])

    def test_grid_sorting(self):
        rows = experiments(self.ws, {'all': WHOLESALE}, ['sqrt'], ['MinMax'],
                           ['KMeans'], [2, 3], [0, 2], [100], 20, validation=self.ws.Channel)
        self.assertGreater(len(rows), 0)
        self.assertTrue(rows.iloc[0]['eligible'])
        self.assertGreater(rows.iloc[0]['silhouette'], .46)

if __name__ == '__main__':
    unittest.main()
