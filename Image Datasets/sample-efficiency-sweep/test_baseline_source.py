import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'comparison-sweep'))
from support import conditions, prepare


class SourceTests(unittest.TestCase):
    def test_saved_subsets_are_reused_and_seed_changes_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg = dict(results_dir=temp, source_run='old', seed=123, folds=10,
                       medical_datasets=['ct'], train_percentages=[10], sampling_seeds=[123])
            source = root / 'old'
            source.mkdir()
            (source / 'sweep.resolved.json').write_text(json.dumps(cfg))
            partition = dict(train=[dict(patient_id='a', bag_id='a', label=1, images=[0])],
                             val=[dict(patient_id='b', bag_id='b', label=0, images=[1])])
            manifest = dict(conditions={k: dict(v, folds=[partition]*10) for k, v in conditions(cfg).items()})
            (source / 'manifest.json').write_text(json.dumps(manifest))
            (source / 'subset_counts.csv').write_text('condition,fold\nct_p10_s123,1\n')
            prepare(cfg, root / 'new')
            self.assertEqual(json.loads((root / 'new/manifest.json').read_text()), manifest)
            with self.assertRaisesRegex(ValueError, 'seed'):
                prepare(dict(cfg, seed=456), root / 'bad')


if __name__ == '__main__':
    unittest.main()
