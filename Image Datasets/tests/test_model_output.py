"""Model-output option tests; only the Python standard library is required."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training_options import FoldSelection, configure_model_output


class ModelOutputTests(unittest.TestCase):
    def test_defaults_are_last_epoch_and_save_models(self):
        cfg = {}
        configure_model_output(cfg)
        self.assertEqual(cfg, {"model_selection": "last", "save_models": True})

    def test_explicit_options_are_preserved(self):
        cfg = {"model_selection": "best_loss", "save_models": False}
        configure_model_output(cfg)
        self.assertEqual(cfg, {"model_selection": "best_loss", "save_models": False})

    def test_invalid_selection_is_rejected(self):
        for method in ("best", "lowest", None, False):
            with self.subTest(method=method), self.assertRaisesRegex(ValueError, "model_selection"):
                configure_model_output({"model_selection": method})

    def test_save_models_requires_a_boolean(self):
        for value in ("false", "yes", 0, 1, None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "save_models"):
                configure_model_output({"save_models": value})

    def test_last_epoch_is_selected_even_when_loss_increases(self):
        selection = FoldSelection()
        for epoch, loss in enumerate((0.7, 0.2, 0.9), 1):
            self.assertTrue(selection.update(loss, epoch))
        self.assertEqual((selection.epoch, selection.loss), (3, 0.9))

    def test_lowest_loss_keeps_the_first_tied_epoch(self):
        selection = FoldSelection("best_loss")
        updates = [selection.update(loss, epoch) for epoch, loss in enumerate((0.7, 0.2, 0.2, 0.9), 1)]
        self.assertEqual(updates, [True, True, False, False])
        self.assertEqual((selection.epoch, selection.loss), (2, 0.2))

    def test_nonfinite_losses_do_not_replace_selected_epoch(self):
        for method in ("last", "best_loss"):
            selection = FoldSelection(method)
            selection.update(0.2, 1)
            for loss in (float("nan"), float("inf"), -float("inf")):
                with self.subTest(method=method, loss=loss), self.assertRaises(ValueError):
                    selection.update(loss, 2)
                self.assertEqual(selection.epoch, 1)


if __name__ == "__main__":
    unittest.main()
