"""Run locally or on Palmetto: python -m unittest discover -p 'test_*.py' -v."""
import copy
import importlib.util
import json
import math
import random
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from support import BudgetBatchSampler, conditions, immutable_json, nested_subsets, prepare, resolve_config


def bags(count=31):
    return [dict(patient_id=f'p{i:03d}', bag_id=f'b{i:03d}', label=i % 3, images=[i]) for i in range(count)]


class SamplingTests(unittest.TestCase):
    def test_nested_reproducible_class_complete_and_full_coverage(self):
        source = bags()
        subsets = nested_subsets(source, [10,20,50,100], 123)
        self.assertEqual(subsets, nested_subsets(source, [100,10,50,20], 123))
        previous = set()
        for percent in [10,20,50,100]:
            current = {b['patient_id'] for b in subsets[percent]}
            self.assertTrue(previous <= current)
            self.assertEqual({b['label'] for b in subsets[percent]}, {0,1,2})
            self.assertEqual(len(current), sum(math.ceil(n*percent/100) for n in [11,10,10]))
            previous = current
        self.assertEqual(subsets[100], source)
        self.assertNotEqual(subsets[20], nested_subsets(source, [20], 456)[20])
        subsets[10][0]['images'].append(999)
        self.assertEqual(source, bags())

    def test_multiple_bags_and_labels_stay_with_patient(self):
        source = bags(20)
        source.append(dict(patient_id='p000', bag_id='second', label=2, images=[101]))
        for subset in nested_subsets(source, [10,20,50,100], 10).values():
            selected = {b['patient_id'] for b in subset}
            self.assertEqual(subset, [b for b in source if b['patient_id'] in selected])
            self.assertEqual({b['label'] for b in subset}, {0,1,2})

    def test_fixed_update_budget_including_short_last_batch(self):
        for size in [3,6,15,31]:
            sampler = BudgetBatchSampler(size, 8, 31, 123)
            first = list(sampler)
            self.assertEqual(len(first), len(sampler))
            self.assertEqual([len(b) for b in first], [8,8,8,7])
            self.assertEqual(set(i for batch in first for i in batch), set(range(size)))
            self.assertEqual(first, list(BudgetBatchSampler(size, 8, 31, 123)))
            self.assertNotEqual(first, list(sampler))
        with self.assertRaises(ValueError):
            BudgetBatchSampler(0, 8, 31, 123)

    def test_run_provenance_cannot_be_replaced(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'manifest.json'
            immutable_json(path, {'seed':123})
            immutable_json(path, {'seed':123})
            with self.assertRaisesRegex(ValueError, 'new run_name'):
                immutable_json(path, {'seed':124})
            self.assertEqual(json.loads(path.read_text()), {'seed':123})

    def test_grid_is_128_jobs_with_1280_fold_fits(self):
        cfg = dict(medical_datasets=['achilles','ct','mri','xray'], train_percentages=[10,20,50,100], sampling_seeds=[123])
        self.assertEqual(len(conditions(cfg)), 16)
        self.assertEqual(len(conditions(cfg))*8, 128)
        self.assertEqual(len(conditions(cfg))*8*10, 1280)

    def test_comparator_budget_preserves_original_single_pass(self):
        # The optional comparator budget must not change existing experiments.
        path = Path(__file__).resolve().parent.parent / 'comparison-sweep' / 'train_comparison.py'
        spec = importlib.util.spec_from_file_location('tested_comparator_trainer', path)
        module = importlib.util.module_from_spec(spec)
        fake_design = types.SimpleNamespace(digest=None, write_csv=None, write_json=None)
        with patch.dict(sys.modules, {'design':fake_design}):
            spec.loader.exec_module(module)
        random.seed(7)
        expected = list(range(17))
        random.shuffle(expected)
        random.seed(7)
        self.assertEqual(module.training_order(17), expected)
        result = module.training_order(3,17)
        self.assertEqual(len(result),17)
        self.assertEqual(set(result),{0,1,2})
        for start in range(0,15,3):
            self.assertEqual(set(result[start:start+3]),{0,1,2})
        with self.assertRaises(ValueError):
            module.training_order(0,10)

    def test_configuration_rejects_invalid_grid_and_budgets(self):
        cfg = dict(project_dir='..', results_dir='../results', methods=['avg'], medical_datasets=['achilles'],
                   folds=10, seed=123, sampling_seeds=[123], train_percentages=[10,20,50,100],
                   save_models=False, cpus=8, feature_batch_size=32, mem_mb=64000, runtime_minutes=1440,
                   num_workers=0, epochs_override=None, smoke_bags_per_epoch=None, run_name='test')
        valid = resolve_config(cfg, Path.cwd())
        self.assertTrue(Path(valid['project_dir']).is_absolute())
        for key,value in [('folds',5),('train_percentages',[10,10]),('train_percentages',[101]),
                          ('sampling_seeds',[-1]),('epochs_override',0),('save_models',True)]:
            with self.subTest(key=key,value=value), self.assertRaises(ValueError):
                resolve_config(dict(cfg,**{key:value}), Path.cwd())

    def test_preparation_preserves_validation_and_counts(self):
        source = bags(60)
        folds = []
        for number in range(10):
            val = source[number*6:(number+1)*6]
            selected = {b['patient_id'] for b in val}
            folds.append(dict(train=[b for b in source if b['patient_id'] not in selected], val=val))
        full = dict(datasets={'medical_achilles':dict(task='achilles', num_classes=3, images=[])},
                    conditions={'medical_achilles':dict(folds=folds)}, source_hashes={})
        def fake_prepare(cfg, root):
            root.mkdir(parents=True)
            (root/'manifest.json').write_text(json.dumps(full))
        def check_partition(split):
            self.assertFalse({b['patient_id'] for b in split['train']} & {b['patient_id'] for b in split['val']})
        def fake_config(path, overrides, net_overrides):
            self.assertEqual(overrides['pretraining'], 'imagenet')
            self.assertTrue(overrides['freeze_enc'])
            self.assertFalse(overrides['freeze_clf'])
            self.assertFalse(overrides['instance_load_clf'])
            return dict(overrides, net_kwargs=net_overrides)
        cfg = dict(project_dir='/unused', seed=123, medical_datasets=['achilles'],
                   train_percentages=[10,20,50,100], sampling_seeds=[123])
        fake_design = types.SimpleNamespace(prepare=fake_prepare, check_partition=check_partition,
                                           digest=lambda path:'hash', write_csv=lambda path,rows:None)
        with tempfile.TemporaryDirectory() as temp, patch.dict(sys.modules, {
                'design':fake_design, 'configuration':types.SimpleNamespace(load_config=fake_config)}):
            root=Path(temp)
            prepare(cfg, root)
            result=json.loads((root/'manifest.json').read_text())
            for i in range(10):
                previous=set()
                for p in [10,20,50,100]:
                    condition=result['conditions'][f'achilles_p{p}_s123']
                    partition=condition['folds'][i]
                    self.assertEqual(partition['val'], folds[i]['val'])
                    self.assertEqual(condition['full_train_bags'][i],54)
                    ids={b['patient_id'] for b in partition['train']}
                    self.assertTrue(previous <= ids)
                    previous=ids


if __name__ == '__main__':
    unittest.main()
