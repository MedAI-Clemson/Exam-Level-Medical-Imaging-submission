"""Small CPU-only checks of the experimental design; no training or downloads."""
import unittest
from bags import make_bags, source_split


class DesignTests(unittest.TestCase):
    def setUp(self):
        self.labels = list(range(10)) * 100
        self.train, self.val = source_split(self.labels, .2, 123)

    def test_disjoint_stratified_sources(self):
        self.assertFalse(set(self.train) & set(self.val))
        self.assertEqual(set(self.train) | set(self.val), set(range(1000)))
        for digit in range(10):
            self.assertEqual(sum(self.labels[i] == digit for i in self.val), 20)

    def test_all_compositions_and_balance(self):
        for size, densities in {10: (10,20,50), 20: (5,10,25,50), 100: (1,5,10,25,50)}.items():
            for density in densities:
                rows = make_bags(self.labels, self.train, 100, size, density, 9, 123)
                self.assertEqual(sum(row['label'] for row in rows), 50)
                for row in rows:
                    self.assertEqual(len(row['indices']), size)
                    self.assertTrue(set(row['indices']) <= set(self.train))
                    actual = sum(self.labels[i] == 9 for i in row['indices'])
                    self.assertEqual(actual, size * density // 100 if row['label'] else 0)
                    self.assertEqual(actual, row['target_count'])

    def test_reproducibility(self):
        args = (self.labels, self.val, 20, 100, 5, 9, 123)
        self.assertEqual(make_bags(*args), make_bags(*args))
        self.assertNotEqual(make_bags(*args), make_bags(*args[:-1], 124))


if __name__ == '__main__':
    unittest.main()
