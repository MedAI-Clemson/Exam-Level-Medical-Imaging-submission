import json
import tempfile
import unittest
from pathlib import Path

from bags import assign_folds, construct, digest, import_saved_designs, write_json


class ImportTests(unittest.TestCase):
    def test_exact_import_and_mismatched_folds_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            metadata = root / 'metadata.csv'
            metadata.write_text('fixture')
            patients = [dict(patient_id=f'p{i}', severity=i % 2 * 2, label=i % 2, selected=[1]) for i in range(40)]
            inventory = {p['patient_id']: {1: f"/{p['patient_id']}/img1.png", 2: f"/{p['patient_id']}/img2.png"} for p in patients}
            cfg = dict(seed=123, folds=10, bag_sizes=[5, 50], replacement_if_needed=True,
                       data_dir=temp, metadata_csv=str(metadata), image_number_regex='img', image_extensions=['.png'],
                       results_dir=temp, source_runs={'5': 'small', '50': 'large'})
            assignments = assign_folds(patients, 10, 123)
            designs, shortages = construct(patients, inventory, assignments, cfg)
            self.assertFalse(shortages)
            saved = dict(configuration=cfg, metadata_sha256=digest(metadata), patients=patients,
                         inventory=inventory, folds=assignments, designs=designs)
            for run in ('small', 'large'):
                write_json(root / run / 'design.json', saved)
            import_saved_designs(cfg, root / 'new')
            actual = json.loads((root / 'new/design.json').read_text())
            self.assertEqual(actual['designs'], designs)
            saved['folds']['p0'] = 99
            write_json(root / 'large/design.json', saved)
            with self.assertRaisesRegex(ValueError, 'different folds'):
                import_saved_designs(cfg, root / 'new')


if __name__ == '__main__':
    unittest.main()
