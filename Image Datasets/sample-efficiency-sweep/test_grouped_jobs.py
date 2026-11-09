"""Check restart behavior without training or importing PyTorch."""
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run


class GroupedJobTests(unittest.TestCase):
    def test_resume_and_invalidated_results(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            calls = []
            fail = {4}

            def child(command, **kwargs):
                number = int(command[-1])
                calls.append(number)
                if number in fail:
                    raise subprocess.CalledProcessError(1, command)
                directory, marker = run.locations(root, 'example', 'avg', number)
                directory.mkdir(parents=True, exist_ok=True)
                marker.parent.mkdir(parents=True, exist_ok=True)
                for filename in ('predictions.csv', 'fold_metrics.csv'):
                    (directory / filename).write_text(f'fold,value\n{number},0.8\n')
                (directory / 'split.json').write_text('{}')
                marker.write_text('{}')

            complete = root / 'completed/example/avg/complete.json'
            with patch.object(run, 'training_signature', return_value='v1') as signature, patch.object(run.subprocess, 'run', side_effect=child):
                with self.assertRaises(subprocess.CalledProcessError):
                    run.train_all({}, root, 'example', 'avg')
                self.assertFalse(complete.exists())
                self.assertTrue((root / 'completed/example/avg/fold1.json').exists())
                fail.clear()
                calls.clear()
                run.train_all({}, root, 'example', 'avg')
                self.assertEqual(calls, list(range(4, 11)))
                self.assertEqual(json.loads(complete.read_text())['folds'], 10)
                self.assertEqual(len((root / 'runs/example/avg/predictions.csv').read_text().splitlines()), 11)
                calls.clear()
                (root / 'runs/example/avg/fold2/predictions.csv').unlink()
                run.train_all({}, root, 'example', 'avg')
                self.assertEqual(calls, [2])
                signature.return_value = 'v2'
                calls.clear()
                run.train_all({}, root, 'example', 'avg')
                self.assertEqual(calls, list(range(1, 11)))


if __name__ == '__main__':
    unittest.main()
