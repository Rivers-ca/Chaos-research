"""Checks for diagnostic aggregation and an end-to-end report on a tiny run."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

import DiagnoseDQN as diagnostics
import DeepQLearning as deep
from TestDeepQLearning import tiny_settings


class DiagnosticTests(unittest.TestCase):
    def test_trailing_average_ignores_warmup_nan_without_using_future(self):
        np.testing.assert_allclose(
            diagnostics.rolling([np.nan, 2, 4, 8], window=2),
            [np.nan, 2, 3, 6], equal_nan=True,
        )

    def test_binning_retains_spikes_and_excludes_nonfinite(self):
        x, stats = diagnostics.binned([1, np.nan, 3, 100], bins=2)
        np.testing.assert_allclose(x, [1.5, 3.5])
        np.testing.assert_allclose(stats[:, 3], [1, 100])
        self.assertEqual(diagnostics.binned([])[1].shape, (0, 4))

    def test_report_contains_valid_images_and_matching_trial_accounting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = deep.run_deep_q_learning(tiny_settings())
            archive = deep.save_deep_q_learning_run(run, root / deep.DEEP_RUN_FILENAME)
            result = diagnostics.create_report(archive, root / 'report', include_standard=False)
            images = list((root / 'report').glob('*.png'))
            self.assertEqual(len(images), 7)
            for path in images:
                with Image.open(path) as image:
                    image.verify()
            self.assertEqual(result['episodes'], 3)
            self.assertEqual(result['environment_steps'], sum(run['history']['episode_lengths']))
            self.assertLessEqual(result['trials_improved_over_zero_forcing'], 4)
            self.assertEqual(json.loads((root / 'report/metrics.json').read_text()), result)
            self.assertTrue((root / 'report/index.html').exists())


if __name__ == '__main__':
    unittest.main()
