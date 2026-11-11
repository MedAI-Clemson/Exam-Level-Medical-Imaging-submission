"""Small objective checks; run with the training environment's Python."""
import unittest

try:
    import torch
    from baselines import baseline_batch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch is not installed in this environment')
class BaselineTests(unittest.TestCase):
    def model(self, logits):
        class Fixed:
            num_out = logits.shape[1]
            def __call__(self, images, lengths):
                return logits
        return Fixed()

    def test_instance_mean_probabilities_and_inherited_labels(self):
        logits = torch.tensor([[-4.], [1.], [2.]], requires_grad=True)
        labels = torch.tensor([1, 0])
        loss, probs, count = baseline_batch(self.model(logits), None, [2, 1], labels, 'instance', True)
        expected = torch.nn.functional.binary_cross_entropy_with_logits(logits[:, 0], torch.tensor([1., 1., 0.]))
        self.assertTrue(torch.allclose(loss, expected))
        self.assertEqual(count, 3)
        self.assertTrue(torch.allclose(probs[:, 0], torch.stack([logits[:2].sigmoid().mean(), logits[2, 0].sigmoid()])))
        self.assertFalse(torch.allclose(probs[0, 0], logits[:2].mean().sigmoid()))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        val, _, count = baseline_batch(self.model(logits), None, [2, 1], labels, 'instance', False)
        self.assertTrue(torch.allclose(val, torch.nn.functional.binary_cross_entropy(probs[:, 0], labels.float())))
        self.assertEqual(count, 2)

    def test_multiclass(self):
        logits = torch.tensor([[2., 0., -1.], [0., 3., 1.], [-1., 1., 2.]])
        labels = torch.tensor([2, 1])
        loss, probs, _ = baseline_batch(self.model(logits), None, [1, 2], labels, 'instance', True)
        self.assertTrue(torch.allclose(loss, torch.nn.functional.cross_entropy(logits, torch.tensor([2, 1, 1]))))
        self.assertTrue(torch.allclose(probs.sum(-1), torch.ones(2)))

    def test_noisy_or_real_model(self):
        from models import Classifier
        model = Classifier(torch.nn.Identity(), 1, num_out=1, pooling='stan')
        with torch.no_grad():
            model.clf[-1].weight.fill_(1)
            model.clf[-1].bias.zero_()
        images = torch.tensor([[-2.], [-1.], [-2.]])
        loss, probs, count = baseline_batch(model, images, [2, 1], torch.tensor([1, 0]), 'stan', True)
        expected = torch.tensor([1-(1-torch.sigmoid(images[0, 0]))*(1-torch.sigmoid(images[1, 0])), torch.sigmoid(images[2, 0])])
        self.assertTrue(torch.allclose(probs[:, 0], expected))
        self.assertEqual(count, 2)
        self.assertTrue(torch.isfinite(loss))


if __name__ == '__main__':
    unittest.main()
