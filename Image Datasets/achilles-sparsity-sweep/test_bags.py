"""CPU-only checks; no training libraries, source images, or GPU required."""
import csv
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from bags import assign_folds, construct, digest, image_inventory, read_patients, write_csv, write_json, composition_rows
from metrics import evaluate, fold_summary
from run_experiment import summarize


class BagTests(unittest.TestCase):
    def setUp(self):
        self.patients = [dict(patient_id=f'patient{i}', severity=i % 4, label=int(i % 4 >= 2), selected=[0, 1, 2])
                         for i in range(40)]
        self.inventory = {p['patient_id']: {i: f"/{p['patient_id']}/img{i}.png" for i in range(60)}
                          for p in self.patients}
        self.cfg = dict(seed=123, folds=10, bag_sizes=[10, 20, 50])
        self.folds = assign_folds(self.patients, 10, 123)

    def test_metadata_mapping_missing_and_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'metadata.csv'
            with path.open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['PatiendID', 'Label: description', '1st Most Useful', '2nd Most Useful', '3rd Most Useful'])
                for i in range(4):
                    writer.writerow([f'patient{i}', i, 0, 2, ''])
                writer.writerow(['missing', 3, '', '', ''])
            patients, excluded = read_patients(path)
            self.assertEqual([p['label'] for p in patients], [0, 0, 1, 1])
            self.assertEqual(patients[0]['selected'], [0, 2])
            self.assertEqual(excluded[0]['patient_id'], 'missing')

    def test_filename_mapping_and_missing_selected(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / 'patient0'
            directory.mkdir()
            for i in [0, 1, 2]:
                (directory / f'img{i}.png').touch()
            cfg = dict(data_dir=tmp, image_number_regex=r'img(?P<image_number>\d+)', image_extensions=['.png'])
            inventory = image_inventory(cfg, [self.patients[0]])
            self.assertEqual(set(inventory['patient0']), {0, 1, 2})
            (directory / 'img2.png').unlink()
            with self.assertRaisesRegex(ValueError, 'selected image numbers not found'):
                image_inventory(cfg, [self.patients[0]])

    def test_fold_isolation_donors_uniqueness_and_nesting(self):
        designs, shortages = construct(self.patients, self.inventory, self.folds, self.cfg)
        self.assertFalse(shortages)
        self.assertEqual(designs, construct(self.patients, self.inventory, self.folds, self.cfg)[0])
        seen_validation = []
        labels = {p['patient_id']: p['label'] for p in self.patients}
        for fold in range(1, 11):
            condition = designs[str(fold)]
            seen_validation.extend(r['patient_id'] for r in condition['10']['val'])
            for split in ['train', 'val']:
                previous = {}
                for size in [10, 20, 50]:
                    for row in condition[str(size)][split]:
                        paths = {i['path'] for i in row['images']}
                        self.assertEqual(len(paths), size)
                        self.assertEqual(sum(i['role'] == 'source' for i in row['images']), 3)
                        for image in row['images']:
                            self.assertEqual(self.folds[image['patient_id']] == fold, split == 'val')
                            if image['role'] == 'distractor':
                                self.assertEqual(labels[image['patient_id']], 0)
                                self.assertNotEqual(image['patient_id'], row['patient_id'])
                        self.assertTrue(previous.get(row['patient_id'], set()) <= paths)
                        previous[row['patient_id']] = paths
        self.assertEqual(sorted(seen_validation), sorted(self.folds))

    def test_donor_shortage_is_reported(self):
        tiny = {p: {i: path for i, path in images.items() if i < 3} for p, images in self.inventory.items()}
        _, shortages = construct(self.patients, tiny, self.folds, self.cfg)
        self.assertTrue(any(r['split'] == 'val' and r['available'] < r['required'] for r in shortages))

    def test_replacement_preserves_unique_prefix_and_partition_isolation(self):
        inventory = {p: {i: path for i, path in images.items() if i < 25} for p, images in self.inventory.items()}
        old, shortages = construct(self.patients, inventory, self.folds, dict(self.cfg, bag_sizes=[10,20,25]))
        self.assertFalse(shortages)
        cfg = dict(self.cfg, bag_sizes=[10,20,25,50,100], replacement_if_needed=True)
        new, shortages = construct(self.patients, inventory, self.folds, cfg)
        self.assertFalse(shortages)
        self.assertEqual(new, construct(self.patients, inventory, self.folds, cfg)[0])
        only50, _ = construct(self.patients, inventory, self.folds, dict(cfg, bag_sizes=[50]))
        labels = {p['patient_id']: p['label'] for p in self.patients}
        for fold, sizes in new.items():
            for size in ('10','20','25'):
                self.assertEqual(old[fold][size], sizes[size])
            self.assertEqual(only50[fold]['50'], sizes['50'])
            for split in ('train','val'):
                smaller = {r['patient_id']: Counter(i['path'] for i in r['images']) for r in sizes['50'][split]}
                for row in sizes['100'][split]:
                    self.assertEqual(len(row['images']), 100)
                    self.assertEqual(sum(i['role']=='source' for i in row['images']), 3)
                    self.assertTrue(smaller[row['patient_id']] <= Counter(i['path'] for i in row['images']))
                    for image in row['images']:
                        self.assertEqual(self.folds[image['patient_id']]==int(fold), split=='val')
                        if image['role']=='distractor':
                            self.assertEqual(labels[image['patient_id']],0)
                            self.assertNotEqual(image['patient_id'],row['patient_id'])
        audit = composition_rows(new)
        self.assertTrue(any(r['repeated_distractors']>0 for r in audit))
        self.assertTrue(all(r['unique_images']+r['repeated_distractors']==r['bag_size'] for r in audit))
        self.assertTrue(all(r['repeated_distractors']==0 for r in audit if r['bag_size']<=25))

    def test_replacement_cannot_borrow_from_another_partition(self):
        assignments = dict(self.folds)
        for patient in self.patients:
            if patient['label']==0:
                assignments[patient['patient_id']] = 2
        _, shortages = construct(self.patients, self.inventory, assignments,
                                 dict(self.cfg, bag_sizes=[100], replacement_if_needed=True))
        self.assertTrue(any(r['fold']==1 and r['split']=='val' and r['available']==0 for r in shortages))

    def test_no_patient_in_both_partitions_and_both_classes_present(self):
        for fold in range(1, 11):
            val = [p for p in self.patients if self.folds[p['patient_id']] == fold]
            self.assertEqual({p['label'] for p in val}, {0, 1})
        self.assertEqual(self.folds, assign_folds(self.patients, 10, 123))


class MetricTests(unittest.TestCase):
    def test_threshold_counts_and_auc(self):
        rows = [dict(label=0, severity=0, probability=.1), dict(label=0, severity=1, probability=.5),
                dict(label=1, severity=2, probability=.5), dict(label=1, severity=3, probability=.9)]
        metrics = evaluate(rows, .5)
        self.assertEqual((metrics['tp'], metrics['tn'], metrics['fp'], metrics['fn']), (2, 1, 1, 0))
        self.assertEqual(metrics['auroc'], .875)
        self.assertEqual(metrics['sensitivity'], 1)
        self.assertEqual(metrics['specificity'], .5)
        self.assertAlmostEqual(metrics['brier'], .13)

    def test_absent_subgroup_and_ci(self):
        rows = [dict(label=0, severity=0, probability=.1), dict(label=1, severity=2, probability=.9)]
        result = evaluate(rows, .5)
        self.assertIsNone(result['class_3_sensitivity'])
        folds = [dict(fold=i, **result) for i in range(1, 11)]
        summary = fold_summary(folds)
        self.assertEqual(summary['sensitivity_ci90_low'], 1)
        with self.assertRaises(ValueError):
            fold_summary(folds[:-1])

    def test_summary_collects_completed_predictions_and_rejects_stale_design(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / 'runs/n10/avg'
            write_json(root / 'design.json', {'seed': 123})
            write_csv(run / 'predictions.csv', [dict(patient_id='p0', probability=.1)])
            write_csv(run / 'fold_metrics.csv', [dict(fold=1, auroc=1.0)])
            write_json(root / 'completed/n10/avg.json', dict(bag_size=10, pooling='avg',
                       run_dir=str(run), design_sha256=digest(root / 'design.json')))
            summarize(dict(bag_sizes=[10], pooling=['avg']), root)
            with (root / 'predictions.csv').open(newline='') as stream:
                self.assertEqual(list(csv.DictReader(stream))[0]['patient_id'], 'p0')
            self.assertTrue((root / 'summary.csv').exists())
            write_json(root / 'design.json', {'seed': 124})
            with self.assertRaisesRegex(ValueError, 'Stale result'):
                summarize(dict(bag_sizes=[10], pooling=['avg']), root)


if __name__ == '__main__':
    unittest.main()
