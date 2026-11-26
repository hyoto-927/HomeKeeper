import unittest
from src.score_review import WEIGHTS, final_v4, top_k_overlap
from src.contract_demo import simulate, aggregate, quantile_map


class ReviewTests(unittest.TestCase):
    def test_visible_weights(self):
        self.assertAlmostEqual(final_v4(dict.fromkeys(WEIGHTS, 1)), 0.65)
        self.assertAlmostEqual(final_v4(dict.fromkeys(WEIGHTS, 0)), 0)
        with self.assertRaises(ValueError):
            final_v4(dict.fromkeys(WEIGHTS, 1.2))

    def test_top_k(self):
        matrix = top_k_overlap({"a": {"X": 3, "Y": 2, "Z": 1},
                                "b": {"X": 2, "Y": 1, "Z": 3}}, 2)
        self.assertEqual(matrix, {"a": {"a": 2, "b": 1}, "b": {"a": 1, "b": 2}})
        with self.assertRaises(ValueError):
            top_k_overlap({"a": {"X": 1}, "b": {"Y": 2}}, 1)

    def test_mapping(self):
        self.assertEqual(quantile_map([-1, 1, 4], [0, 1, 2], [10, 20, 30]), [10, 20, 30])
        self.assertEqual(quantile_map([1], [1, 1, 1], [0, 10]), [5])

    def test_simulation_and_aggregation(self):
        rows = simulate(42, 240, "A")
        self.assertEqual(rows, simulate(42, 240, "A"))
        self.assertFalse({r["contract"] for r in rows} & {r["contract"] for r in simulate(43, 100, "B")})
        grouped = aggregate(rows)
        self.assertEqual(sum(g["contracts"] for g in grouped.values()), 240)
        self.assertEqual(sum(g["fictional_positive"] for g in grouped.values()), sum(r["fictional_label"] for r in rows))
