import unittest

from cola.client import DeterministicMockClient
from cola.pipeline import COLAPipeline, parse_judge_option


class PipelineTests(unittest.TestCase):
    def test_pipeline_runs_all_seven_agents(self) -> None:
        client = DeterministicMockClient()
        result = COLAPipeline(client).predict(
            text=(
                "The only way I support Hillary was if Elizabeth Warren ran or "
                "Karl Marx was running. #2016 #Clinton2016"
            ),
            target="Hillary Clinton",
        )

        self.assertEqual(result.label, "against")
        self.assertEqual(len(result.analyses), 3)
        self.assertEqual(len(result.debates), 3)
        self.assertEqual(len(client.calls), 7)

    def test_parse_judge_option_accepts_constrained_and_word_outputs(self) -> None:
        self.assertEqual(parse_judge_option("A"), "against")
        self.assertEqual(parse_judge_option("B: Favor"), "favor")
        self.assertEqual(parse_judge_option("neutral"), "neutral")

    def test_parse_judge_option_rejects_unknown_output(self) -> None:
        for raw in ("I cannot decide", "A or B", "A: Favor", "A long explanation"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_judge_option(raw)


if __name__ == "__main__":
    unittest.main()
