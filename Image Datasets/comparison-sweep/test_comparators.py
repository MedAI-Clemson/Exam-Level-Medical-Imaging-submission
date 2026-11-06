"""Run with the training environment: python -m unittest discover -p 'test_*.py' -v."""

import importlib.util
import math
import tempfile
import unittest
from pathlib import Path

from design import check_cv_coverage, check_partition, conditions, digest, prepare, write_json
from train_comparison import profile

HAS_TORCH = importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import torch
    from torch import nn
    from comparator_models import CLAM, DTFDAFS, LeNet, ProMIL, bernstein_logs, smooth_top1


class DesignTests(unittest.TestCase):
    def test_default_condition_count_and_no_pretraining_grid(self):
        cfg = dict(experiments=['medical', 'mnist', 'achilles_sparsity'],
                   medical_datasets=['achilles', 'ct', 'mri', 'xray'], achilles_bag_sizes=[5, 10, 25],
                   mnist_densities_by_bag_size={10: [10, 20, 50], 20: [5, 10, 25, 50], 100: [1, 5, 10, 25, 50]})
        jobs = conditions(cfg)
        self.assertEqual(len(jobs), 19)
        self.assertEqual(len(jobs) * 3, 57)
        self.assertEqual(len({j['dataset'] for j in jobs.values()}), 6)

    def test_partition_checks_source_and_donor_overlap(self):
        fold = dict(train=[dict(patient_id='a', images=[dict(path='/a.png')])],
                    val=[dict(patient_id='b', images=[dict(path='/b.png')])])
        check_partition(fold)
        fold['val'][0]['images'].append(dict(path='/a.png'))
        with self.assertRaisesRegex(ValueError, 'overlap'):
            check_partition(fold)

    def test_different_native_profiles_and_smoke_override(self):
        cfg = dict(profiles={'promil': {'mnist': {'epochs': 100, 'warmup_epochs': 0},
                                       'medical': {'epochs': 125, 'warmup_epochs': 25}}})
        self.assertEqual(profile(cfg, 'promil', True)['epochs'], 100)
        self.assertEqual(profile(cfg, 'promil', False)['warmup_epochs'], 25)
        cfg['epochs_override'] = 1
        self.assertEqual(profile(cfg, 'promil', False)['epochs'], 1)
        self.assertEqual(cfg['profiles']['promil']['medical']['epochs'], 125)

    def test_cv_must_hold_out_each_source_once(self):
        first = dict(train=[dict(bag_id='a')], val=[dict(bag_id='b')])
        second = dict(train=[dict(bag_id='b')], val=[dict(bag_id='a')])
        check_cv_coverage([first, second])
        with self.assertRaisesRegex(ValueError, 'exactly once'):
            check_cv_coverage([first, first])

    def test_exact_achilles_design_reuse_and_label_check(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = dict(project_dir=tmp, results_dir=tmp, experiments=['achilles_sparsity'],
                       achilles_bag_sizes=[5], achilles_source_run='old', folds=2)
            bags = []
            for index in range(4):
                images = []
                for image_number in range(5):
                    path = root / f'p{index}_img{image_number}.png'
                    path.touch()
                    images.append(dict(patient_id=f'p{index}', path=str(path)))
                bags.append(dict(patient_id=f'p{index}', severity=2 if index % 2 else 0,
                                 label=index % 2, source_count=3, images=images))
            source = dict(designs={'1': {'5': {'train': bags[:2], 'val': bags[2:]}},
                                   '2': {'5': {'train': bags[2:], 'val': bags[:2]}}})
            path = root / 'old/design.json'
            write_json(path, source)
            prepare(cfg, root / 'new')
            result = json.loads((root / 'new/manifest.json').read_text())
            self.assertEqual(result['source_hashes'][str(path)], digest(path))
            row = result['conditions']['achilles_n5']['folds'][0]['val'][0]
            actual = [result['datasets']['achilles_sparsity']['images'][i]['path'] for i in row['images']]
            self.assertEqual(actual, [i['path'] for i in bags[2]['images']])
            source['designs']['1']['5']['train'][0]['label'] = 1
            write_json(path, source)
            with self.assertRaisesRegex(ValueError, 'classes 0/1'):
                prepare(cfg, root / 'bad')


@unittest.skipUnless(HAS_TORCH, 'PyTorch is required; run these checks in the pytorch environment')
class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        torch.set_num_threads(1)

    def test_clam_small_bags_and_all_three_classes(self):
        for size in (1, 2, 5, 10, 25):
            model = CLAM(3, width=12, dropout=0)
            loss, probability, info = model.objective(torch.randn(size, 12), torch.tensor(2))
            self.assertEqual(info['selected_k'], min(8, size//2))
            self.assertEqual(tuple(probability.shape), (1, 3))
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
            self.assertIsNotNone(model.clf.weight.grad)
            if size > 1:
                self.assertIsNotNone(model.instance_clfs[2].weight.grad)

    def test_clam_inference_is_permutation_invariant(self):
        model, x = CLAM(2, width=12, dropout=0).eval(), torch.randn(15, 12)
        torch.testing.assert_close(model(x)[0], model(x[torch.randperm(15)])[0])

    def test_smooth_svm_reference(self):
        logits = torch.tensor([[2., 1.], [-1., 2.]])
        loss = smooth_top1(logits, torch.tensor([0, 1]))
        expected = (math.log1p(math.exp(0)) + math.log1p(math.exp(-2))) / 2
        self.assertAlmostEqual(float(loss), expected, places=6)

    def test_dtfd_afs_is_weighted_features_and_separate_updates(self):
        model = DTFDAFS(2, width=12, groups=4)
        x = torch.randn(5, 12)
        seed = 77
        pseudo, features = model.distill(x, torch.Generator().manual_seed(seed))
        order = torch.randperm(5, generator=torch.Generator().manual_seed(seed))
        h = model.project(x)
        expected = torch.cat([model.attention1(h[g])[0] for g in torch.tensor_split(order, 4)])
        torch.testing.assert_close(features, expected)
        before1, before2 = model.clf1.weight.detach().clone(), model.clf2.weight.detach().clone()
        opts = [torch.optim.Adam(p, lr=.001) for p in (model.tier1_parameters(), model.tier2_parameters())]
        loss, probability, _ = model.train_step(x, torch.tensor(1), opts)
        self.assertTrue(torch.isfinite(loss))
        self.assertFalse(torch.equal(before1, model.clf1.weight))
        self.assertFalse(torch.equal(before2, model.clf2.weight))
        model.eval()
        torch.testing.assert_close(model.predict(x, seed), model.predict(x, seed))
        self.assertEqual(tuple(model.predict(x[:1], seed).shape), (1, 2))

    def test_promil_matches_author_bernstein_equation_and_gradients(self):
        logits = torch.tensor([[-2.], [0.], [1.], [3.]], dtype=torch.float64, requires_grad=True)
        raw = torch.tensor([.2], dtype=torch.float64, requires_grad=True)
        p, not_p = bernstein_logs(logits, raw)
        alpha = raw.sigmoid()
        weights = torch.stack([math.comb(3, k)*alpha**(3-k)*(1-alpha)**k for k in range(4)]).reshape(4, 1)
        expected = (weights*logits.sigmoid().sort(dim=0).values).sum()
        torch.testing.assert_close(p.exp().squeeze(), expected)
        torch.testing.assert_close(p.exp()+not_p.exp(), torch.ones_like(p))
        (-p.sum()).backward()
        self.assertTrue(torch.isfinite(raw.grad).all())
        self.assertGreater(abs(raw.grad.item()), 0)
        self.assertTrue(torch.isfinite(logits.grad).all())
        permutation = torch.tensor([2, 0, 3, 1])
        torch.testing.assert_close(bernstein_logs(logits[permutation], raw)[0], p)

    def test_promil_extremes_singleton_and_multiclass(self):
        for value in (-1000., 1000.):
            x = torch.full((100, 1), value, requires_grad=True)
            p, not_p = bernstein_logs(x, torch.tensor([.2]))
            loss = -p.sum()-not_p.sum()
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(torch.isfinite(x.grad).all())
        x = torch.tensor([[.4]])
        torch.testing.assert_close(bernstein_logs(x, torch.tensor([.2]))[0].exp(), x.sigmoid())
        model = ProMIL(nn.Identity(), 12, num_classes=3)
        loss, probability, _ = model.objective(torch.randn(5, 12), torch.tensor(2))
        loss.backward()
        torch.testing.assert_close(probability.sum(), torch.tensor(1.))

    def test_lenet_dimensions_and_freezing(self):
        model = ProMIL(LeNet(), 500, mnist=True)
        loss, probability, _ = model.objective(torch.randn(3, 1, 28, 28), torch.tensor(1))
        loss.backward()
        self.assertEqual(tuple(probability.shape), (1, 2))
        model.set_encoder_frozen(True)
        model.train()
        self.assertFalse(model.encoder.training)
        self.assertTrue(all(not p.requires_grad for p in model.encoder.parameters()))
        model.set_encoder_frozen(False)
        self.assertTrue(all(p.requires_grad for p in model.encoder.parameters()))


if __name__ == '__main__':
    unittest.main()
