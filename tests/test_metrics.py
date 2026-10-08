import unittest

from cola.metrics import classification_metrics


class MetricsTests(unittest.TestCase):
    def test_paper_metrics(self) -> None:
        metrics = classification_metrics(
            ["favor", "favor", "against", "neutral"],
            ["favor", "against", "against", "neutral"],
        )

        self.assertAlmostEqual(metrics["accuracy"], 0.75)
        self.assertAlmostEqual(metrics["f_avg"], 2 / 3)
        self.assertAlmostEqual(metrics["macro_f1"], 7 / 9)

    def test_lengths_must_match(self) -> None:
        with self.assertRaises(ValueError):
            classification_metrics(["favor"], [])


if __name__ == "__main__":
    unittest.main()
