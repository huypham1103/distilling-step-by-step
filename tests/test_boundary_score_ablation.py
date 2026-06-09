import unittest

import pandas as pd

import build_boundary_focus_pack as boundary
import build_boundary_score_ablation as ablation
import build_judge_cqa_outstanding_pack as cqa


class BoundaryScoreComponentTest(unittest.TestCase):
    def setUp(self):
        self.candidate = {
            "source": "historical",
            "premise": "A child runs through a park.",
            "hypothesis": "A child is outdoors.",
            "rationale": (
                "The correct answer is entailment because a park is outdoors, "
                "so the answer is entailment."
            ),
        }

    def test_disabling_each_component_removes_only_its_weighted_contribution(self):
        full_score, contributions = boundary.candidate_quality(
            self.candidate,
            training_label="entailment",
            vote_margin=2.5,
            agreement_count=5,
            total_candidates=7,
            return_contributions=True,
        )

        self.assertEqual(set(boundary.SCORE_COMPONENTS), set(contributions))
        self.assertAlmostEqual(full_score, sum(contributions.values()))

        for component, contribution in contributions.items():
            with self.subTest(component=component):
                ablated_score = boundary.candidate_quality(
                    self.candidate,
                    training_label="entailment",
                    vote_margin=2.5,
                    agreement_count=5,
                    total_candidates=7,
                    disabled_components={component},
                )
                self.assertAlmostEqual(ablated_score, full_score - contribution)

    def test_unknown_component_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown score components"):
            boundary.candidate_quality(
                self.candidate,
                training_label="entailment",
                vote_margin=2.5,
                agreement_count=5,
                total_candidates=7,
                disabled_components={"unknown"},
            )


class BoundarySelectionChangeTest(unittest.TestCase):
    @staticmethod
    def row(example, rationale, label="entailment", source="causal"):
        return {
            "premise": example,
            "hypothesis": f"{example} hypothesis",
            "rationale": rationale,
            "LLM_answer": label,
            "judge_source": source,
        }

    def test_selection_change_counts_changed_rationales_and_examples(self):
        full = pd.DataFrame(
            [
                self.row("one", "rationale a"),
                self.row("two", "rationale b"),
            ]
        )
        variant = pd.DataFrame(
            [
                self.row("one", "rationale changed", source="if_else"),
                self.row("three", "rationale c"),
            ]
        )

        report = ablation.compare_selection(full, variant)

        self.assertEqual(report["full_rows"], 2)
        self.assertEqual(report["variant_rows"], 2)
        self.assertEqual(report["shared_selected_rows"], 0)
        self.assertEqual(report["changed_selected_rows"], 2)
        self.assertEqual(report["shared_examples"], 1)
        self.assertEqual(report["changed_examples"], 1)
        self.assertEqual(report["selection_change"], 1.0)
        self.assertEqual(report["example_change"], 0.5)

    def test_rank_reports_orders_largest_selection_change_first(self):
        reports = {
            "source": {"selection_change": 0.2, "changed_selected_rows": 20},
            "ground": {"selection_change": 0.5, "changed_selected_rows": 50},
            "cue": {"selection_change": 0.1, "changed_selected_rows": 10},
        }

        ranked = ablation.rank_reports(reports)

        self.assertEqual(["ground", "source", "cue"], [item["component"] for item in ranked])


class CqaBoundaryScoreComponentTest(unittest.TestCase):
    def setUp(self):
        self.candidate = {
            "source": "causal",
            "premise": "Where would a bird build a nest?",
            "hypothesis": "['tree', 'ocean', 'desk', 'road', 'car']",
            "rationale": "The correct answer is tree because birds build nests in trees.",
        }
        self.profile = {
            "ideal_word_max": 96,
            "hard_word_max": 128,
            "overlap_weight": 0.62,
            "agreement_weight": 0.18,
            "margin_weight": 0.10,
            "preferred_sources": {"causal", "if_else", "neutral", "contrastive"},
        }

    def test_cqa_disabling_each_component_removes_only_its_weighted_contribution(self):
        full_score, contributions = cqa.score_candidate(
            self.candidate,
            gold_label="tree",
            vote_margin=2.5,
            agreement_count=5,
            total_candidates=7,
            source_prior=cqa.BASE_SOURCE_PRIOR,
            profile=self.profile,
            return_contributions=True,
        )

        self.assertEqual(set(boundary.SCORE_COMPONENTS), set(contributions))
        self.assertAlmostEqual(full_score, sum(contributions.values()))

        for component, contribution in contributions.items():
            with self.subTest(component=component):
                ablated_score = cqa.score_candidate(
                    self.candidate,
                    gold_label="tree",
                    vote_margin=2.5,
                    agreement_count=5,
                    total_candidates=7,
                    source_prior=cqa.BASE_SOURCE_PRIOR,
                    profile=self.profile,
                    disabled_components={component},
                )
                self.assertAlmostEqual(ablated_score, full_score - contribution)

    def test_cqa_component_split_preserves_the_existing_full_score(self):
        full_score = cqa.score_candidate(
            self.candidate,
            gold_label="tree",
            vote_margin=2.5,
            agreement_count=5,
            total_candidates=7,
            source_prior=cqa.BASE_SOURCE_PRIOR,
            profile=self.profile,
        )
        rationale_lower = self.candidate["rationale"].lower()
        overlap = cqa.support_overlap(
            self.candidate["premise"],
            self.candidate["hypothesis"],
            self.candidate["rationale"],
        )
        legacy_cue_hits = sum(1 for cue in cqa.SHORTCUT_CUES if cue in rationale_lower)
        expected_legacy_score = (
            cqa.BASE_SOURCE_PRIOR[self.candidate["source"]]
            + self.profile["overlap_weight"] * overlap
            + self.profile["agreement_weight"] * 5
            + 0.12 * (5 / 7)
            + self.profile["margin_weight"] * 2.5
            + 0.20
            + 0.08 * legacy_cue_hits
            + 0.12
            + 0.25
        )

        self.assertAlmostEqual(full_score, expected_legacy_score)


if __name__ == "__main__":
    unittest.main()
