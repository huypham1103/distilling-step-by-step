import unittest

import pandas as pd

import build_judge_cqa_outstanding_pack as cqa_pack


class CqaBoundarySpecialistPackTest(unittest.TestCase):
    def setUp(self):
        self._originals = {
            "load_gold_records": cqa_pack.load_gold_records,
            "load_candidates": cqa_pack.load_candidates,
            "build_family": cqa_pack.build_family,
            "build_boundary_rows": cqa_pack.build_boundary_rows,
            "build_shortcut_rows": cqa_pack.build_shortcut_rows,
            "build_derived_packs": cqa_pack.build_derived_packs,
            "save_dataset": cqa_pack.save_dataset,
        }

    def tearDown(self):
        for name, value in self._originals.items():
            setattr(cqa_pack, name, value)

    def test_main_builds_cqa_boundary_specialist_with_esnli_rule(self):
        captured = {}

        def boundary_row(example_id, source, band, score):
            return {
                "premise": f"question {example_id}",
                "hypothesis": "['a', 'b', 'c', 'd', 'e']",
                "prompt": "",
                "rationale": f"{source} rationale {example_id} {band}",
                "split": "train",
                "correct_index": "",
                "LLM_answer": "a",
                "judge_source": source,
                "judge_score": score,
                "candidate_label": "a",
                "gold_label": "a",
                "paper_gold_label": "a",
                "voted_label": "a",
                "voted_label_support": 3.0,
                "voted_label_margin": 2.0,
                "label_match": True,
                "word_count": 6,
                "overlap_score": 0.5,
                "agreement_count": 4,
                "agreement_ratio": 0.8,
                "judge_view_rank": 1,
                "boundary_band": band,
            }

        boundary_rows = pd.DataFrame(
            [
                boundary_row("same", "historical", "boundary", 3.0),
                boundary_row("same", "contrastive", "bridge", 2.9),
                boundary_row("same", "if_else", "boundary", 2.8),
                boundary_row("easy", "historical", "easy", 3.5),
                boundary_row("hard", "comparative", "hard", 3.4),
                boundary_row("causal", "causal", "boundary", 3.3),
                boundary_row("neutral", "neutral", "bridge", 3.2),
                boundary_row("other", "consensus", "bridge", 3.1),
            ]
        )

        cqa_pack.load_gold_records = lambda: []
        cqa_pack.load_candidates = lambda source: {}
        cqa_pack.build_family = lambda gold_records, candidate_tables, name, profile: pd.DataFrame()
        cqa_pack.build_boundary_rows = lambda gold_records, candidate_tables: boundary_rows
        cqa_pack.build_shortcut_rows = lambda gold_records, candidate_tables: pd.DataFrame()
        cqa_pack.build_derived_packs = lambda packs: {}

        def capture_dataset(name, df):
            captured[name] = df.copy()
            return {"num_examples": int(len(df))}

        cqa_pack.save_dataset = capture_dataset

        cqa_pack.main()

        self.assertIn("judge_student_boundary_specialist_balanced", captured)
        specialist = captured["judge_student_boundary_specialist_balanced"]
        self.assertEqual({"boundary", "bridge"}, set(specialist["boundary_band"]))
        self.assertLessEqual(
            set(specialist["judge_source"]),
            {"historical", "contrastive", "comparative", "if_else", "consensus"},
        )
        same_example_count = (
            specialist[specialist["premise"] == "question same"].shape[0]
        )
        self.assertEqual(2, same_example_count)


if __name__ == "__main__":
    unittest.main()
