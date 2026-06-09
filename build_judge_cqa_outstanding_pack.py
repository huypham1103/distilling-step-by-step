import ast
import json
import math
import re
from collections import Counter
from pathlib import Path

import pandas as pd


API_DIR = Path("[API] CQA")
SOURCES = ["historical", "consensus", "contrastive", "causal", "neutral", "if_else", "comparative"]

BASE_SOURCE_PRIOR = {
    "causal": 1.00,
    "if_else": 0.99,
    "neutral": 0.98,
    "contrastive": 0.975,
    "historical": 0.94,
    "consensus": 0.84,
    "comparative": 0.80,
}

SUPERCLEAN_SOURCE_PRIOR = {
    "causal": 1.00,
    "if_else": 0.995,
    "neutral": 0.985,
    "contrastive": 0.975,
    "historical": 0.95,
    "consensus": 0.80,
    "comparative": 0.76,
}

DIVERSE_SOURCE_PRIOR = {
    "causal": 1.00,
    "historical": 0.98,
    "if_else": 0.97,
    "neutral": 0.96,
    "contrastive": 0.95,
    "consensus": 0.92,
    "comparative": 0.90,
}

EXPERT_SOURCE_PRIOR = {
    "causal": 1.00,
    "if_else": 0.995,
    "neutral": 0.985,
    "contrastive": 0.975,
    "historical": 0.94,
    "consensus": 0.82,
    "comparative": 0.79,
}

FORMAT_CUES = ("so the answer is", "the correct answer")
REASONING_CUES = ("because", "therefore", "if", "then", "means")
SHORTCUT_CUES = REASONING_CUES + FORMAT_CUES
SCORE_COMPONENTS = (
    "source",
    "ground",
    "cue",
    "explicit",
    "format",
    "brief",
    "agreement_count",
    "agreement_ratio",
    "margin",
)


def normalize_text(text):
    if pd.isna(text):
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def normalize_answer(text):
    normalized = normalize_text(text).lower()
    normalized = normalized.replace("**", "")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip(" .,:;!?\"'")


def tokenize(text):
    return set(re.findall(r"[a-z0-9]+", normalize_text(text).lower()))


def parse_choices(text):
    normalized = normalize_text(text)
    if not normalized:
        return []
    try:
        parsed = ast.literal_eval(normalized)
        if isinstance(parsed, list):
            return [normalize_text(item) for item in parsed]
    except Exception:
        pass
    return [part.strip() for part in normalized.split("',") if part.strip()]


def example_key(premise, hypothesis):
    return normalize_text(premise).lower() + "</s>" + normalize_text(hypothesis).lower()


def rationale_jaccard(left, right):
    left_tokens = tokenize(left)
    right_tokens = tokenize(right)
    if not left_tokens and not right_tokens:
        return 1.0
    union = left_tokens | right_tokens
    if not union:
        return 0.0
    return len(left_tokens & right_tokens) / len(union)


def load_gold_records():
    paper = pd.read_csv(API_DIR / "paper.csv")
    records = []
    for row in paper.to_dict("records"):
        raw_input = str(row.get("input", ""))
        if "\nAnswer Choices:\n" not in raw_input:
            continue
        question, choices_block = raw_input.split("\nAnswer Choices:\n", 1)
        choices = []
        for line in choices_block.split("\n"):
            if ") " in line:
                choices.append(normalize_text(line.split(") ", 1)[1]))
        hypothesis = str(choices)
        records.append({
            "key": example_key(question, hypothesis),
            "premise": normalize_text(question),
            "hypothesis": hypothesis,
            "gold_label": normalize_answer(row.get("label", "")),
            "paper_rationale": normalize_text(row.get("rationale", "")),
        })
    return records


def load_candidates(source_name):
    df = pd.read_csv(API_DIR / f"{source_name} - full.csv")
    table = {}
    for row in df.to_dict("records"):
        key = example_key(row["premise"], row["hypothesis"])
        table[key] = {
            "source": source_name,
            "premise": normalize_text(row["premise"]),
            "hypothesis": normalize_text(row["hypothesis"]),
            "label": normalize_answer(row["LLM_answer"]),
            "rationale": normalize_text(row["rationale"]),
            "prompt": row.get("prompt", ""),
            "split": row.get("split", ""),
            "correct_index": row.get("correct_index", ""),
        }
    return table


def weighted_vote(candidates, source_prior):
    scores = {}
    counts = Counter()
    for candidate in candidates:
        label = candidate["label"]
        if not label:
            continue
        scores[label] = scores.get(label, 0.0) + source_prior.get(candidate["source"], 0.0)
        counts[label] += 1
    if not scores:
        return "", 0.0, 0.0, counts
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    winner, winner_score = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
    return winner, winner_score, runner_up, counts


def support_overlap(premise, hypothesis, rationale):
    support_tokens = tokenize(premise) | tokenize(hypothesis)
    rationale_tokens = tokenize(rationale)
    if not rationale_tokens:
        return 0.0
    return len(support_tokens & rationale_tokens) / len(rationale_tokens)


def shortcut_profile(premise, hypothesis, answer):
    question_tokens = tokenize(premise)
    choices = parse_choices(hypothesis)
    answer_tokens = tokenize(answer)
    answer_overlap = len(answer_tokens & question_tokens) / max(1, len(answer_tokens))
    option_overlaps = []
    for choice in choices:
        choice_tokens = tokenize(choice)
        overlap = len(choice_tokens & question_tokens) / max(1, len(choice_tokens))
        option_overlaps.append((normalize_answer(choice), overlap))
    best_other = max((score for label, score in option_overlaps if label != answer), default=0.0)
    gap = answer_overlap - best_other
    spread = sum(abs(score - answer_overlap) < 0.15 for _, score in option_overlaps)

    score = 0.0
    if best_other >= answer_overlap:
        score += 0.9
    if answer_overlap == 0.0 and best_other > 0.0:
        score += 0.8
    if gap < 0.1:
        score += 0.55
    if spread >= 3:
        score += 0.45

    if score >= 1.5:
        band = "challenge"
    elif score >= 0.8:
        band = "bridge"
    else:
        band = "easy"

    return {
        "shortcut_score": round(score, 6),
        "shortcut_band": band,
        "answer_overlap": round(answer_overlap, 6),
        "best_other_overlap": round(best_other, 6),
        "overlap_gap": round(gap, 6),
    }


def score_candidate(
    candidate,
    gold_label,
    vote_margin,
    agreement_count,
    total_candidates,
    source_prior,
    profile,
    disabled_components=None,
    return_contributions=False,
):
    disabled_components = frozenset(disabled_components or ())
    unknown_components = disabled_components.difference(SCORE_COMPONENTS)
    if unknown_components:
        raise ValueError(f"Unknown score components: {sorted(unknown_components)}")

    rationale = candidate["rationale"]
    rationale_lower = rationale.lower()
    word_count = len(re.findall(r"\w+", rationale))
    overlap = support_overlap(candidate["premise"], candidate["hypothesis"], rationale)
    explicit_answer = 1.0 if gold_label and gold_label in rationale_lower else 0.0
    cue_hits = sum(1 for cue in REASONING_CUES if cue in rationale_lower)
    format_hits = sum(1 for cue in FORMAT_CUES if cue in rationale_lower)
    brevity = 0.25 if 10 <= word_count <= profile["ideal_word_max"] else (-0.18 if word_count > profile["hard_word_max"] else 0.0)
    agreement_ratio = agreement_count / max(1, total_candidates)

    contributions = {
        "source": source_prior.get(candidate["source"], 0.0) + 0.12 * (candidate["source"] in profile["preferred_sources"]),
        "ground": profile["overlap_weight"] * overlap,
        "cue": 0.08 * cue_hits,
        "explicit": 0.20 * explicit_answer,
        "format": 0.08 * format_hits,
        "brief": brevity,
        "agreement_count": profile["agreement_weight"] * agreement_count,
        "agreement_ratio": 0.12 * agreement_ratio,
        "margin": profile["margin_weight"] * vote_margin,
    }
    score = sum(
        contribution
        for component, contribution in contributions.items()
        if component not in disabled_components
    )
    if return_contributions:
        return score, contributions
    return score


def choose_secondary(primary, matching, profile):
    alternatives = []
    for candidate in matching:
        if candidate is primary:
            continue
        if candidate["source"] == primary["source"]:
            continue
        similarity = rationale_jaccard(primary["rationale"], candidate["rationale"])
        if similarity >= profile["similarity_limit"]:
            continue
        if candidate["judge_score"] < primary["judge_score"] - profile["max_secondary_gap"]:
            continue
        alternatives.append((1.0 - similarity, candidate))
    if not alternatives:
        return None
    alternatives.sort(key=lambda item: (item[1]["judge_score"], item[0]), reverse=True)
    return alternatives[0][1]


def append_row(rows, gold, candidate, voted_label, winner_score, vote_margin, label_counts, view_rank):
    rows.append({
        "premise": gold["premise"],
        "hypothesis": gold["hypothesis"],
        "prompt": candidate.get("prompt", ""),
        "rationale": candidate["rationale"],
        "split": candidate.get("split", ""),
        "correct_index": candidate.get("correct_index", ""),
        "LLM_answer": gold["gold_label"],
        "judge_source": candidate["source"],
        "judge_score": candidate["judge_score"],
        "candidate_label": candidate["label"],
        "gold_label": gold["gold_label"],
        "paper_gold_label": gold["gold_label"],
        "voted_label": voted_label,
        "voted_label_support": winner_score,
        "voted_label_margin": vote_margin,
        "label_match": candidate["label"] == gold["gold_label"],
        "word_count": len(candidate["rationale"].split()),
        "overlap_score": candidate["support_overlap"],
        "agreement_count": label_counts.get(gold["gold_label"], 0),
        "agreement_ratio": label_counts.get(gold["gold_label"], 0) / max(1, sum(label_counts.values())),
        "judge_view_rank": view_rank,
    })


def source_balanced_select(df, cap, max_per_example, source_order):
    if df.empty:
        return df.copy()
    df = df.copy()
    df["example_key"] = df.apply(lambda row: example_key(row["premise"], row["hypothesis"]), axis=1)
    df["row_key"] = (
        df["example_key"]
        + "||"
        + df["rationale"].fillna("").map(normalize_text).str.lower()
        + "||"
        + df["judge_source"].fillna("").astype(str).str.lower()
    )
    df = df.sort_values(
        by=["judge_score", "agreement_count", "voted_label_margin"],
        ascending=[False, False, False],
    ).drop_duplicates(subset=["row_key"], keep="first")

    target = min(cap, len(df))
    buckets = {source: df[df["judge_source"] == source].to_dict("records") for source in source_order}
    selected = []
    selected_keys = set()
    example_counts = Counter()

    while len(selected) < target:
        progressed = False
        for source in source_order:
            rows = buckets.get(source, [])
            while rows:
                row = rows.pop(0)
                if row["row_key"] in selected_keys:
                    continue
                if example_counts[row["example_key"]] >= max_per_example:
                    continue
                selected.append(row)
                selected_keys.add(row["row_key"])
                example_counts[row["example_key"]] += 1
                progressed = True
                break
            if len(selected) >= target:
                break
        if not progressed:
            break

    if len(selected) < target:
        remaining = df[~df["row_key"].isin(selected_keys)].to_dict("records")
        for row in remaining:
            if len(selected) >= target:
                break
            if example_counts[row["example_key"]] >= max_per_example:
                continue
            selected.append(row)
            selected_keys.add(row["row_key"])
            example_counts[row["example_key"]] += 1

    return pd.DataFrame(selected)


def build_family(gold_records, candidate_tables, name, profile):
    rows = []
    for gold in gold_records:
        key = gold["key"]
        candidates = []
        for source_name, table in candidate_tables.items():
            candidate = table.get(key)
            if candidate is not None and candidate["rationale"]:
                candidates.append(candidate.copy())
        if not candidates or not gold["gold_label"]:
            continue

        voted_label, winner_score, runner_up, label_counts = weighted_vote(candidates, profile["vote_prior"])
        vote_margin = winner_score - runner_up
        matching = []
        for candidate in candidates:
            if candidate["label"] != gold["gold_label"]:
                continue
            enriched = candidate.copy()
            enriched["support_overlap"] = support_overlap(candidate["premise"], candidate["hypothesis"], candidate["rationale"])
            enriched["judge_score"] = score_candidate(
                enriched,
                gold["gold_label"],
                vote_margin,
                label_counts.get(gold["gold_label"], 0),
                len(candidates),
                profile["score_prior"],
                profile,
            )
            matching.append(enriched)

        if not matching:
            continue

        matching.sort(key=lambda item: item["judge_score"], reverse=True)
        best = matching[0]
        if label_counts.get(gold["gold_label"], 0) < profile["min_agreement"]:
            continue
        if vote_margin < profile["min_margin"]:
            continue
        if best["judge_score"] < profile["min_score"]:
            continue

        append_row(rows, gold, best, voted_label, winner_score, vote_margin, label_counts, view_rank=1)

        if profile["allow_secondary"]:
            secondary = choose_secondary(best, matching, profile)
            if secondary is not None:
                append_row(rows, gold, secondary, voted_label, winner_score, vote_margin, label_counts, view_rank=2)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    if profile["post_quantile"] > 0.0:
        cutoff = df["judge_score"].quantile(profile["post_quantile"])
        df = df[df["judge_score"] >= cutoff].copy()

    return source_balanced_select(
        df,
        cap=profile["cap"],
        max_per_example=profile["max_per_example"],
        source_order=profile["source_order"],
    )


def build_boundary_rows(gold_records, candidate_tables, disabled_components=None):
    rows = []
    for gold in gold_records:
        key = gold["key"]
        candidates = []
        for source_name, table in candidate_tables.items():
            candidate = table.get(key)
            if candidate is not None and candidate["rationale"]:
                candidates.append(candidate.copy())
        if not candidates or not gold["gold_label"]:
            continue

        voted_label, winner_score, runner_up, label_counts = weighted_vote(candidates, BASE_SOURCE_PRIOR)
        vote_margin = winner_score - runner_up
        matching = []
        for candidate in candidates:
            if candidate["label"] != gold["gold_label"]:
                continue
            enriched = candidate.copy()
            enriched["support_overlap"] = support_overlap(candidate["premise"], candidate["hypothesis"], candidate["rationale"])
            enriched["judge_score"] = score_candidate(
                enriched,
                gold["gold_label"],
                vote_margin,
                label_counts.get(gold["gold_label"], 0),
                len(candidates),
                BASE_SOURCE_PRIOR,
                {
                    "ideal_word_max": 96,
                    "hard_word_max": 128,
                    "overlap_weight": 0.62,
                    "agreement_weight": 0.18,
                    "margin_weight": 0.10,
                    "preferred_sources": {"contrastive", "if_else", "causal", "neutral"},
                },
                disabled_components=disabled_components,
            )
            matching.append(enriched)
        if not matching:
            continue
        matching.sort(key=lambda item: item["judge_score"], reverse=True)
        best = matching[0]

        if vote_margin >= 2.4 and label_counts.get(gold["gold_label"], 0) >= 5:
            boundary_band = "easy"
        elif vote_margin >= 1.6 and label_counts.get(gold["gold_label"], 0) >= 3:
            boundary_band = "boundary"
        else:
            boundary_band = "hard"

        append_row(rows, gold, best, voted_label, winner_score, vote_margin, label_counts, view_rank=1)
        rows[-1]["boundary_band"] = boundary_band

        partner = choose_secondary(
            best,
            matching,
            {
                "similarity_limit": 0.76,
                "max_secondary_gap": 0.5,
            },
        )
        if partner is not None and boundary_band != "hard":
            append_row(rows, gold, partner, voted_label, winner_score, vote_margin, label_counts, view_rank=2)
            rows[-1]["boundary_band"] = "bridge"

    return pd.DataFrame(rows)


def build_shortcut_rows(gold_records, candidate_tables):
    rows = []
    for gold in gold_records:
        key = gold["key"]
        candidates = []
        for source_name, table in candidate_tables.items():
            candidate = table.get(key)
            if candidate is not None and candidate["rationale"]:
                candidates.append(candidate.copy())
        if not candidates or not gold["gold_label"]:
            continue

        profile = shortcut_profile(gold["premise"], gold["hypothesis"], gold["gold_label"])
        voted_label, winner_score, runner_up, label_counts = weighted_vote(candidates, DIVERSE_SOURCE_PRIOR)
        vote_margin = winner_score - runner_up

        matching = []
        for candidate in candidates:
            if candidate["label"] != gold["gold_label"]:
                continue
            enriched = candidate.copy()
            enriched["support_overlap"] = support_overlap(candidate["premise"], candidate["hypothesis"], candidate["rationale"])
            base_score = score_candidate(
                enriched,
                gold["gold_label"],
                vote_margin,
                label_counts.get(gold["gold_label"], 0),
                len(candidates),
                DIVERSE_SOURCE_PRIOR if profile["shortcut_band"] != "challenge" else EXPERT_SOURCE_PRIOR,
                {
                    "ideal_word_max": 100,
                    "hard_word_max": 136,
                    "overlap_weight": 0.60,
                    "agreement_weight": 0.16,
                    "margin_weight": 0.08,
                    "preferred_sources": {"contrastive", "if_else", "historical", "causal", "neutral"},
                },
            )
            enriched["judge_score"] = base_score + 0.12 * profile["shortcut_score"]
            matching.append(enriched)
        if not matching:
            continue

        matching.sort(key=lambda item: item["judge_score"], reverse=True)
        best = matching[0]
        append_row(rows, gold, best, voted_label, winner_score, vote_margin, label_counts, view_rank=1)
        rows[-1].update(profile)

        partner = choose_secondary(
            best,
            matching,
            {
                "similarity_limit": 0.72 if profile["shortcut_band"] == "challenge" else 0.68,
                "max_secondary_gap": 0.42,
            },
        )
        if partner is not None and (profile["shortcut_band"] != "easy" or label_counts.get(gold["gold_label"], 0) >= 5):
            append_row(rows, gold, partner, voted_label, winner_score, vote_margin, label_counts, view_rank=2)
            rows[-1].update(profile)

    return pd.DataFrame(rows)


def band_select(df, cap, max_per_example, band_column, allowed_bands=None, band_mix=None, source_order=None):
    if df.empty:
        return df.copy()
    work = df.copy()
    if allowed_bands is not None:
        work = work[work[band_column].isin(set(allowed_bands))].copy()
    if work.empty:
        return work

    work["example_key"] = work.apply(lambda row: example_key(row["premise"], row["hypothesis"]), axis=1)
    work["row_key"] = (
        work["example_key"]
        + "||"
        + work["rationale"].fillna("").map(normalize_text).str.lower()
        + "||"
        + work["judge_source"].fillna("").astype(str).str.lower()
    )
    work = work.sort_values(
        by=["judge_score", "agreement_count", "voted_label_margin"],
        ascending=[False, False, False],
    ).drop_duplicates(subset=["row_key"], keep="first")

    target = min(cap, len(work))
    chosen = []
    chosen_keys = set()
    example_counts = Counter()

    if band_mix:
        for band, ratio in band_mix:
            subset = work[work[band_column] == band]
            desired = min(len(subset), int(math.floor(target * ratio)))
            for _, row in subset.iterrows():
                if row["row_key"] in chosen_keys:
                    continue
                if example_counts[row["example_key"]] >= max_per_example:
                    continue
                chosen.append(row)
                chosen_keys.add(row["row_key"])
                example_counts[row["example_key"]] += 1
                if len([item for item in chosen if item[band_column] == band]) >= desired:
                    break

    if len(chosen) < target:
        if source_order:
            remainder = source_balanced_select(
                work[~work["row_key"].isin(chosen_keys)],
                cap=target - len(chosen),
                max_per_example=max_per_example,
                source_order=source_order,
            )
            for _, row in remainder.iterrows():
                if example_counts[row["example_key"]] >= max_per_example:
                    continue
                chosen.append(row)
                example_counts[row["example_key"]] += 1
        else:
            remaining = work[~work["row_key"].isin(chosen_keys)]
            for _, row in remaining.iterrows():
                if len(chosen) >= target:
                    break
                if example_counts[row["example_key"]] >= max_per_example:
                    continue
                chosen.append(row)
                example_counts[row["example_key"]] += 1

    return pd.DataFrame(chosen[:target])


def add_bonus(df, bonus):
    boosted = df.copy()
    boosted["quality_score"] = boosted["judge_score"] + bonus
    return boosted


def dedupe_by_quality(df):
    if df.empty:
        return df.copy()
    work = df.copy()
    if "quality_score" not in work.columns:
        work["quality_score"] = work["judge_score"]
    work["example_key"] = work.apply(lambda row: example_key(row["premise"], row["hypothesis"]), axis=1)
    work["row_key"] = (
        work["example_key"]
        + "||"
        + work["rationale"].fillna("").map(normalize_text).str.lower()
        + "||"
        + work["judge_source"].fillna("").astype(str).str.lower()
    )
    return work.sort_values(
        by=["quality_score", "judge_score", "agreement_count", "voted_label_margin"],
        ascending=[False, False, False, False],
    ).drop_duplicates(subset=["row_key"], keep="first")


def build_derived_packs(base_packs):
    packs = {}
    hybrid = base_packs["judge_student_multiview_hybrid_balanced"]
    superclean = base_packs["judge_student_singleview_superclean_balanced"]
    diverse = base_packs["judge_student_singleview_diverse_balanced"]
    multiview = base_packs["judge_student_multiview"]

    hybrid_core = hybrid.copy()
    hybrid_core["score_cutoff"] = hybrid_core["judge_score"].quantile(0.45)
    hybrid_core["margin_cutoff"] = hybrid_core["voted_label_margin"].quantile(0.35)
    hybrid_core["agree_cutoff"] = hybrid_core["agreement_count"].median()
    hybrid_core = hybrid_core[
        (hybrid_core["judge_score"] >= hybrid_core["score_cutoff"])
        & (hybrid_core["voted_label_margin"] >= hybrid_core["margin_cutoff"])
        & (hybrid_core["agreement_count"] >= hybrid_core["agree_cutoff"])
    ].drop(columns=["score_cutoff", "margin_cutoff", "agree_cutoff"])
    packs["judge_hybrid_core_balanced"] = source_balanced_select(hybrid_core, 8000, 1, list(EXPERT_SOURCE_PRIOR))

    expert_rows = pd.concat(
        [
            add_bonus(hybrid[hybrid["judge_view_rank"] == 1], 0.24),
            add_bonus(superclean, 0.32),
            add_bonus(multiview[multiview["judge_source"].isin({"contrastive", "if_else", "causal"})], 0.18),
        ],
        ignore_index=True,
    )
    packs["judge_expert_hybrid_fusion_balanced"] = source_balanced_select(dedupe_by_quality(expert_rows), 9000, 2, list(EXPERT_SOURCE_PRIOR))

    precision_diverse = pd.concat(
        [
            add_bonus(superclean, 0.36),
            add_bonus(diverse, 0.18),
            add_bonus(hybrid, 0.14),
            add_bonus(multiview, 0.08),
        ],
        ignore_index=True,
    )
    packs["judge_precision_diverse_mix_balanced"] = source_balanced_select(dedupe_by_quality(precision_diverse), 9000, 2, list(DIVERSE_SOURCE_PRIOR))

    cleanfusion = pd.concat(
        [
            add_bonus(superclean, 0.40),
            add_bonus(hybrid[hybrid["judge_view_rank"] == 1], 0.10),
            add_bonus(hybrid[hybrid["judge_view_rank"] == 2], 0.04),
        ],
        ignore_index=True,
    )
    packs["judge_student_multiview_hybrid_cleanfusion"] = source_balanced_select(dedupe_by_quality(cleanfusion), 9500, 2, list(EXPERT_SOURCE_PRIOR))

    return packs


def save_dataset(name, df):
    output_path = API_DIR / f"{name} - full.csv"
    clean_df = df.copy()
    clean_df.index.name = "Unnamed: 0"
    clean_df.to_csv(output_path)

    report = {
        "output_csv": str(output_path),
        "num_examples": int(len(clean_df)),
        "judge_source_counts": clean_df["judge_source"].value_counts().to_dict() if "judge_source" in clean_df else {},
        "view_rank_counts": clean_df["judge_view_rank"].value_counts().to_dict() if "judge_view_rank" in clean_df else {},
        "average_judge_score": round(float(clean_df["judge_score"].mean()), 6) if "judge_score" in clean_df and not clean_df.empty else None,
        "average_agreement_count": round(float(clean_df["agreement_count"].mean()), 6) if "agreement_count" in clean_df and not clean_df.empty else None,
        "average_voted_label_margin": round(float(clean_df["voted_label_margin"].mean()), 6) if "voted_label_margin" in clean_df and not clean_df.empty else None,
    }
    if "boundary_band" in clean_df:
        report["boundary_band_counts"] = clean_df["boundary_band"].value_counts().to_dict()
    if "shortcut_band" in clean_df:
        report["shortcut_band_counts"] = clean_df["shortcut_band"].value_counts().to_dict()
        report["average_shortcut_score"] = round(float(clean_df["shortcut_score"].mean()), 6)

    report_path = API_DIR / f"{name}_judge_report.json"
    with report_path.open("w") as handle:
        json.dump(report, handle, indent=2)
    return report


def main():
    gold_records = load_gold_records()
    candidate_tables = {source: load_candidates(source) for source in SOURCES}

    base_profiles = {
        "judge_student_multiview": {
            "vote_prior": BASE_SOURCE_PRIOR,
            "score_prior": BASE_SOURCE_PRIOR,
            "preferred_sources": {"contrastive", "if_else", "causal", "neutral"},
            "ideal_word_max": 110,
            "hard_word_max": 140,
            "overlap_weight": 0.58,
            "agreement_weight": 0.18,
            "margin_weight": 0.08,
            "min_agreement": 2,
            "min_margin": 1.2,
            "min_score": 1.65,
            "allow_secondary": True,
            "similarity_limit": 0.78,
            "max_secondary_gap": 0.50,
            "post_quantile": 0.12,
            "cap": 12000,
            "max_per_example": 2,
            "source_order": list(BASE_SOURCE_PRIOR),
        },
        "judge_student_multiview_hybrid_balanced": {
            "vote_prior": BASE_SOURCE_PRIOR,
            "score_prior": EXPERT_SOURCE_PRIOR,
            "preferred_sources": {"contrastive", "if_else", "causal", "neutral", "historical"},
            "ideal_word_max": 100,
            "hard_word_max": 132,
            "overlap_weight": 0.62,
            "agreement_weight": 0.20,
            "margin_weight": 0.10,
            "min_agreement": 3,
            "min_margin": 1.6,
            "min_score": 1.95,
            "allow_secondary": True,
            "similarity_limit": 0.74,
            "max_secondary_gap": 0.42,
            "post_quantile": 0.22,
            "cap": 10500,
            "max_per_example": 2,
            "source_order": list(EXPERT_SOURCE_PRIOR),
        },
        "judge_student_singleview_superclean_balanced": {
            "vote_prior": SUPERCLEAN_SOURCE_PRIOR,
            "score_prior": SUPERCLEAN_SOURCE_PRIOR,
            "preferred_sources": {"contrastive", "if_else", "causal", "neutral"},
            "ideal_word_max": 96,
            "hard_word_max": 124,
            "overlap_weight": 0.66,
            "agreement_weight": 0.22,
            "margin_weight": 0.12,
            "min_agreement": 3,
            "min_margin": 1.8,
            "min_score": 2.05,
            "allow_secondary": False,
            "similarity_limit": 0.72,
            "max_secondary_gap": 0.0,
            "post_quantile": 0.34,
            "cap": 8000,
            "max_per_example": 1,
            "source_order": list(SUPERCLEAN_SOURCE_PRIOR),
        },
        "judge_student_singleview_diverse_balanced": {
            "vote_prior": DIVERSE_SOURCE_PRIOR,
            "score_prior": DIVERSE_SOURCE_PRIOR,
            "preferred_sources": {"contrastive", "historical", "if_else", "causal", "consensus", "comparative"},
            "ideal_word_max": 104,
            "hard_word_max": 132,
            "overlap_weight": 0.60,
            "agreement_weight": 0.18,
            "margin_weight": 0.09,
            "min_agreement": 2,
            "min_margin": 1.45,
            "min_score": 1.78,
            "allow_secondary": False,
            "similarity_limit": 0.72,
            "max_secondary_gap": 0.0,
            "post_quantile": 0.24,
            "cap": 8500,
            "max_per_example": 1,
            "source_order": list(DIVERSE_SOURCE_PRIOR),
        },
    }

    packs = {}
    for name, profile in base_profiles.items():
        packs[name] = build_family(gold_records, candidate_tables, name, profile)

    boundary_rows = build_boundary_rows(gold_records, candidate_tables)
    packs["judge_student_boundary_mix_balanced"] = band_select(
        boundary_rows,
        cap=8500,
        max_per_example=1,
        band_column="boundary_band",
        allowed_bands={"easy", "boundary"},
        source_order=list(EXPERT_SOURCE_PRIOR),
    )
    packs["judge_student_boundary_bridge_balanced"] = band_select(
        boundary_rows,
        cap=10500,
        max_per_example=2,
        band_column="boundary_band",
        allowed_bands={"easy", "boundary", "bridge"},
        source_order=list(EXPERT_SOURCE_PRIOR),
    )
    packs["judge_student_boundary_specialist_balanced"] = band_select(
        boundary_rows[boundary_rows["judge_source"].isin({"historical", "contrastive", "comparative", "if_else", "consensus"})],
        cap=9000,
        max_per_example=2,
        band_column="boundary_band",
        allowed_bands={"boundary", "bridge"},
        source_order=list(EXPERT_SOURCE_PRIOR),
    )

    shortcut_rows = build_shortcut_rows(gold_records, candidate_tables)
    packs["judge_student_shortcut_aware_balanced"] = band_select(
        shortcut_rows,
        cap=9000,
        max_per_example=2,
        band_column="shortcut_band",
        allowed_bands={"challenge", "bridge", "easy"},
        band_mix=[("challenge", 0.35), ("bridge", 0.40), ("easy", 0.25)],
        source_order=list(DIVERSE_SOURCE_PRIOR),
    )

    packs.update(build_derived_packs(packs))

    for name, df in packs.items():
        report = save_dataset(name, df)
        print(f"{name}: {report['num_examples']} rows")


if __name__ == "__main__":
    main()
