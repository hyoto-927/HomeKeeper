import unittest
from copy import deepcopy
from src.synthetic import generate, FEATURES
from src.evaluate import partition, validate_features, design, weighted_signal

class DemoTests(unittest.TestCase):
    def setUp(self):
        self.rows = generate()

    def test_deterministic(self):
        self.assertEqual(self.rows, generate())
        self.assertEqual(len(self.rows), 960)

    def test_partition(self):
        train, test = partition(self.rows)
        self.assertEqual((len(train), len(test)), (450, 80))
        self.assertFalse({r["zone"] for r in train} & {r["zone"] for r in test})
        self.assertLess(max(r["month"] + 1 for r in train), min(r["month"] for r in test))

    def test_target_never_allowed(self):
        with self.assertRaises(ValueError):
            validate_features(self.rows, FEATURES + ("future_rate",))

    def test_target_copy_rejected(self):
        copied = deepcopy(self.rows)
        for r in copied:
            r["price_pressure"] = r["future_rate"]
        with self.assertRaisesRegex(ValueError, "Target duplicate"):
            design(copied)

    def test_design_and_weight_bounds(self):
        self.assertEqual(len(design(self.rows)[0]), 4)
        self.assertTrue(all(0 <= x[0] <= 1 for x in weighted_signal(self.rows)))

if __name__ == "__main__":
    unittest.main()

