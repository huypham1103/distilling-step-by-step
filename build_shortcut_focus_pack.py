import json
import math
import re
from collections import Counter

import pandas as pd

from build_judge_esnli import (
    API_DIR,
    DEFAULT_SOURCES,
    LABEL_TEACHING_CUES,
    load_candidates,
    load_gold_records,
    normalize_label,
    normalize_text,
    rationale_jaccard,
    tokenize_for_overlap,
)


LABELS = ["entailment", "neutral", "contradiction"]
NEGATION_TOKENS = {"no", "not", "never", "none", "nobody", "nothing", "without", "cannot", "can't", "dont", "don't"}
NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}

SHORTCUT_SOURCE_PRIOR = {
    "entailment": {
        "challenge": {
            "historical": 1.00,
            "consensus": 0.98,
            "causal": 0.96,
            "contrastive": 0.86,
            "neutral": 0.82,
            "if_else": 0.74,
            "comparative": 0.72,
        },
        "bridge": {
            "historical": 1.00,
            "consensus": 0.97,
            "causal": 0.93,
            "contrastive": 0.88,
            "neutral": 0.83,
            "if_else": 0.77,
            "comparative": 0.73,
        },
        "easy": {
            "consensus": 1.00,
            "historical": 0.99,
            "causal": 0.90,
            "contrastive": 0.84,
            "neutral": 0.80,
            "if_else": 0.75,
            "comparative": 0.72,
        },
    },
    "neutral": {
        "challenge": {
            "if_else": 1.00,
            "comparative": 0.98,
            "neutral": 0.95,
            "contrastive": 0.91,
            "causal": 0.84,
            "consensus": 0.80,
            "historical": 0.71,
        },
        "bridge": {
            "if_else": 1.00,
            "neutral": 0.97,
            "comparative": 0.96,
            "contrastive": 0.89,
            "causal": 0.85,
            "consensus": 0.81,
            "historical": 0.72,
        },
        "easy": {
            "neutral": 1.00,
            "if_else": 0.98,
            "comparative": 0.94,
            "contrastive": 0.88,
            "causal": 0.83,
            "consensus": 0.78,
            "historical": 0.71,
        },
    },
    "contradiction": {
        "challenge": {
            "comparative": 1.00,
            "contrastive": 0.99,
            "causal": 0.96,
            "neutral": 0.88,
            "consensus": 0.82,
            "historical": 0.74,
            "if_else": 0.70,
        },
        "bridge": {
            "contrastive": 1.00,
            "comparative": 0.98,
            "causal": 0.95,
            "neutral": 0.88,
            "consensus": 0.83,
            "historical": 0.75,
            "if_else": 0.71,
        },
        "easy": {
            "contrastive": 1.00,
            "causal": 0.96,
            "comparative": 0.95,
            "neutral": 0.87,
            "consensus": 0.81,
            "historical": 0.72,
            "if_else": 0.69,
        },
    },
}

SHORTCUT_CUES = {
    "entailment": ("means", "shows", "indicates", "describes", "therefore", "so"),
    "neutral": ("not enough", "cannot know", "does not say", "might", "could", "unclear"),
    "contradiction": ("not", "no", "never", "different", "instead", "but", "whereas"),
}


def build_example_key(premise, hypothesis):
    return normalize_text(premise).lower() + "</s>" + normalize_text(hypothesis).lower()


def extract_number_markers(text):
    markers = set()
    normalized = normalize_text(text).lower()
    for match in re.findall(r"\d+", normalized):
        markers.add(int(match))
    for token in re.findall(r"[a-z]+", normalized):
        if token in NUMBER_WORDS:
            markers.add(NUMBER_WORDS[token])
    return markers


def has_negation(text):
    tokens = tokenize_for_overlap(text)
    return any(token in NEGATION_TOKENS for token in tokens)


def pair_overlap_metrics(premise, hypothesis):
    premise_tokens = tokenize_for_overlap(premise)
    hypothesis_tokens = tokenize_for_overlap(hypothesis)
    shared = premise_tokens & hypothesis_tokens
    union = premise_tokens | hypothesis_tokens
    hypothesis_coverage = len(shared) / max(1, len(hypothesis_tokens))
    jaccard = len(shared) / max(1, len(union))
    negation_mismatch = has_negation(premise) != has_negation(hypothesis)
    number_mismatch = bool(extract_number_markers(premise) or extract_number_markers(hypothesis)) and (
        extract_number_markers(premise) != extract_number_markers(hypothesis)
    )
    return {
        "premise_tokens": premise_tokens,
        "hypothesis_tokens": hypothesis_tokens,
        "hypothesis_coverage": hypothesis_coverage,
        "jaccard": jaccard,
        "negation_mismatch": negation_mismatch,
        "number_mismatch": number_mismatch,
    }


def shortcut_profile(label, premise, hypothesis):
    metrics = pair_overlap_metrics(premise, hypothesis)
    coverage = metrics["hypothesis_coverage"]
    jaccard = metrics["jaccard"]
    negation_mismatch = metrics["negation_mismatch"]
    number_mismatch = metrics["number_mismatch"]

    score = 0.0
    if label == "contradiction":
        if coverage >= 0.62:
            score += 1.15
        elif coverage >= 0.48:
            score += 0.6
        if negation_mismatch:
            score += 0.8
        if number_mismatch:
            score += 0.75
        if 0.28 <= jaccard <= 0.75:
            score += 0.25
    elif label == "entailment":
        if coverage <= 0.42:
            score += 1.1
        elif coverage <= 0.58:
            score += 0.55
        if jaccard <= 0.25:
            score += 0.7
        elif jaccard <= 0.38:
            score += 0.3
        if not negation_mismatch and not number_mismatch:
            score += 0.2
    elif label == "neutral":
        if coverage >= 0.55:
            score += 1.0
        elif coverage >= 0.38:
            score += 0.5
        if not negation_mismatch and not number_mismatch and jaccard >= 0.22:
            score += 0.65
        elif negation_mismatch or number_mismatch:
            score += 0.15

    if score >= 1.5:
        band = "challenge"
    elif score >= 0.8:
        band = "bridge"
    else:
        band = "easy"

    metrics["shortcut_score"] = round(score, 6)
    metrics["shortcut_band"] = band
    return metrics


def vote_labels(candidates):
    scores = {label: 0.0 for label in LABELS}
    counts = {label: 0 for label in LABELS}
    for candidate in candidates:
        label = normalize_label(candidate["label"])
        if label not in LABELS:
            continue
        counts[label] += 1
        for band in ("challenge", "bridge", "easy"):
            scores[label] += SHORTCUT_SOURCE_PRIOR[label][band].get(candidate["source"], 0.0) / 3.0
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    winner, winner_score = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
    return winner, winner_score, runner_up, counts


def candidate_quality(candidate, training_label, shortcut_band, shortcut_score, agreement_count, vote_margin):
    rationale = candidate["rationale"]
    rationale_lower = rationale.lower()
    word_count = len(re.findall(r"\w+", rationale))
    support_tokens = tokenize_for_overlap(candidate["premise"]) | tokenize_for_overlap(candidate["hypothesis"])
    rationale_tokens = tokenize_for_overlap(rationale)
    support_overlap = len(rationale_tokens & support_tokens) / max(1, len(rationale_tokens))
    source_prior = SHORTCUT_SOURCE_PRIOR[training_label][shortcut_band].get(candidate["source"], 0.0)
    teaching_hits = sum(1 for cue in LABEL_TEACHING_CUES.get(training_label, ()) if cue in rationale_lower)
    shortcut_hits = sum(1 for cue in SHORTCUT_CUES.get(training_label, ()) if cue in rationale_lower)
    explicit_label = 1.0 if training_label in rationale_lower else 0.0
    format_bonus = 0.24 if "the correct answer" in rationale_lower or "so the answer is" in rationale_lower else 0.0
    brevity_bonus = 0.22 if 10 <= word_count <= 96 else (-0.18 if word_count > 132 else 0.0)

    return (
        source_prior
        + 0.62 * support_overlap
        + 0.18 * teaching_hits
        + 0.16 * shortcut_hits
        + 0.18 * explicit_label
        + format_bonus
        + brevity_bonus
        + 0.15 * agreement_count
        + 0.06 * vote_margin
        + 0.1 * shortcut_score
    )


def choose_secondary_view(primary, candidates, shortcut_band):
    alternatives = []
    for candidate in candidates:
        if candidate is primary:
            continue
        if candidate["source"] == primary["source"]:
            continue
        similarity = rationale_jaccard(primary["rationale"], candidate["rationale"])
        similarity_limit = 0.74 if shortcut_band == "challenge" else 0.7
        if similarity >= similarity_limit:
            continue
        if candidate["quality_score"] < primary["quality_score"] - (0.45 if shortcut_band == "challenge" else 0.35):
            continue
        alternatives.append((1.0 - similarity, candidate))
    if not alternatives:
        return None
    alternatives.sort(key=lambda item: (item[1]["quality_score"], item[0]), reverse=True)
    return alternatives[0][1]


def append_row(rows, gold, candidate, output_label, voted_label, winner_score, vote_margin, label_counts, profile, view_rank):
    rows.append({
        "premise": gold["premise"],
        "hypothesis": gold["hypothesis"],
        "prompt": candidate.get("prompt", ""),
        "rationale": candidate["rationale"],
        "split": candidate.get("split", ""),
        "correct_index": candidate.get("correct_index", ""),
        "LLM_answer": output_label,
        "judge_source": candidate["source"],
        "judge_score": candidate["quality_score"],
        "candidate_label": candidate["label"],
        "gold_label": output_label,
        "paper_gold_label": normalize_label(gold["gold_label"]),
        "voted_label": voted_label,
        "voted_label_support": winner_score,
        "voted_label_margin": vote_margin,
        "label_match": normalize_label(candidate["label"]) == output_label,
        "word_count": len(candidate["rationale"].split()),
        "overlap_score": candidate["support_overlap"],
        "agreement_count": label_counts.get(output_label, 0),
        "agreement_ratio": label_counts.get(output_label, 0) / max(1, sum(label_counts.values())),
        "judge_view_rank": view_rank,
        "shortcut_band": profile["shortcut_band"],
        "shortcut_score": profile["shortcut_score"],
        "pair_hypothesis_coverage": profile["hypothesis_coverage"],
        "pair_jaccard": profile["jaccard"],
        "negation_mismatch": profile["negation_mismatch"],
        "number_mismatch": profile["number_mismatch"],
    })


def collect_shortcut_examples():
    gold_records, _ = load_gold_records()
    candidate_tables = {source: load_candidates(source) for source in DEFAULT_SOURCES}
    rows = []

    for gold in gold_records:
        gold_label = normalize_label(gold["gold_label"])
        if gold_label not in LABELS:
            continue

        profile = shortcut_profile(gold_label, gold["premise"], gold["hypothesis"])
        example_candidates = []
        for source_name, table in candidate_tables.items():
            candidate = table.get(gold["key"])
            if candidate is None:
                continue
            example_candidates.append(candidate.copy())

        if not example_candidates:
            continue

        voted_label, winner_score, runner_up_score, label_counts = vote_labels(example_candidates)
        vote_margin = winner_score - runner_up_score
        matching = []
        for candidate in example_candidates:
            if normalize_label(candidate["label"]) != gold_label:
                continue
            enriched = candidate.copy()
            agreement_count = label_counts.get(gold_label, 0)
            enriched["support_overlap"] = len(
                tokenize_for_overlap(candidate["rationale"])
                & (tokenize_for_overlap(candidate["premise"]) | tokenize_for_overlap(candidate["hypothesis"]))
            ) / max(1, len(tokenize_for_overlap(candidate["rationale"])))
            enriched["quality_score"] = candidate_quality(
                enriched,
                gold_label,
                profile["shortcut_band"],
                profile["shortcut_score"],
                agreement_count,
                vote_margin,
            )
            matching.append(enriched)

        if not matching:
            continue

        matching.sort(
            key=lambda candidate: (
                candidate["quality_score"],
                label_counts.get(gold_label, 0),
                vote_margin,
            ),
            reverse=True,
        )
        best = matching[0]
        append_row(rows, gold, best, gold_label, voted_label, winner_score, vote_margin, label_counts, profile, view_rank=1)

        partner = choose_secondary_view(best, matching, profile["shortcut_band"])
        if partner is not None and (profile["shortcut_band"] != "easy" or label_counts.get(gold_label, 0) >= 5):
            append_row(rows, gold, partner, gold_label, voted_label, winner_score, vote_margin, label_counts, profile, view_rank=2)

    return pd.DataFrame(rows)


def take_rows_with_caps(subset, target, per_example_max):
    chosen = []
    example_counts = Counter()
    for _, row in subset.iterrows():
        key = row["example_key"]
        if example_counts[key] >= per_example_max:
            continue
        chosen.append(row)
        example_counts[key] += 1
        if len(chosen) >= target:
            break
    return chosen, example_counts


def balance_by_profile(df, cap_per_label, band_mix, per_example_max, allowed_bands=None):
    if df.empty:
        return df.copy()

    df = df.copy()
    if allowed_bands is not None:
        df = df[df["shortcut_band"].isin(set(allowed_bands))].copy()
        if df.empty:
            return df
    df["example_key"] = df.apply(lambda row: build_example_key(row["premise"], row["hypothesis"]), axis=1)
    df["row_key"] = (
        df["example_key"]
        + "||"
        + df["rationale"].fillna("").map(normalize_text).str.lower()
        + "||"
        + df["LLM_answer"].fillna("").astype(str).str.lower()
    )
    df = df.sort_values(
        by=["judge_score", "agreement_count", "voted_label_margin", "shortcut_score"],
        ascending=[False, False, False, False],
    ).drop_duplicates(subset=["row_key"], keep="first")

    label_counts = df["LLM_answer"].value_counts()
    if label_counts.empty:
        return df.head(0).copy()
    shared_target = min(cap_per_label, int(label_counts.min()))

    parts = []
    for label in LABELS:
        subset = df[df["LLM_answer"] == label].copy()
        if subset.empty:
            continue
        target = min(shared_target, len(subset))
        chosen = []
        example_counts = Counter()

        for band, ratio in band_mix:
            band_subset = subset[subset["shortcut_band"] == band]
            desired = min(len(band_subset), int(math.floor(target * ratio)))
            for _, row in band_subset.iterrows():
                key = row["example_key"]
                if example_counts[key] >= per_example_max:
                    continue
                chosen.append(row)
                example_counts[key] += 1
                if len([item for item in chosen if item["shortcut_band"] == band]) >= desired:
                    break

        if len(chosen) < target:
            chosen_keys = {
                row["row_key"]
                for row in chosen
            }
            remaining = subset[~subset["row_key"].isin(chosen_keys)]
            for _, row in remaining.iterrows():
                key = row["example_key"]
                if example_counts[key] >= per_example_max:
                    continue
                chosen.append(row)
                example_counts[key] += 1
                if len(chosen) >= target:
                    break

        parts.append(pd.DataFrame(chosen[:target]))

    return pd.concat(parts, ignore_index=True)


def save_dataset(name, df):
    output_csv = API_DIR / f"{name} - full.csv"
    clean_df = df.copy()
    clean_df.index.name = "Unnamed: 0"
    clean_df.to_csv(output_csv)

    report = {
        "output_csv": str(output_csv),
        "num_examples": int(len(clean_df)),
        "label_counts": clean_df["LLM_answer"].value_counts().to_dict(),
        "shortcut_band_counts": clean_df["shortcut_band"].value_counts().to_dict() if "shortcut_band" in clean_df else {},
        "view_rank_counts": clean_df["judge_view_rank"].value_counts().to_dict() if "judge_view_rank" in clean_df else {},
        "judge_source_counts": clean_df["judge_source"].value_counts().to_dict() if "judge_source" in clean_df else {},
        "average_judge_score": round(float(clean_df["judge_score"].mean()), 6) if "judge_score" in clean_df else None,
        "average_shortcut_score": round(float(clean_df["shortcut_score"].mean()), 6) if "shortcut_score" in clean_df else None,
        "average_pair_hypothesis_coverage": round(float(clean_df["pair_hypothesis_coverage"].mean()), 6) if "pair_hypothesis_coverage" in clean_df else None,
        "average_pair_jaccard": round(float(clean_df["pair_jaccard"].mean()), 6) if "pair_jaccard" in clean_df else None,
    }
    report_path = API_DIR / f"{name}_judge_report.json"
    with report_path.open("w") as handle:
        json.dump(report, handle, indent=2)
    return report


def main():
    all_rows = collect_shortcut_examples()

    configs = [
        (
            "judge_student_shortcut_aware_balanced",
            {
                "cap_per_label": 3000,
                "band_mix": [("challenge", 0.34), ("bridge", 0.38), ("easy", 0.28)],
                "per_example_max": 2,
                "allowed_bands": ["challenge", "bridge", "easy"],
            },
        ),
        (
            "judge_student_shortcut_bridge_balanced",
            {
                "cap_per_label": 3600,
                "band_mix": [("challenge", 0.42), ("bridge", 0.40), ("easy", 0.18)],
                "per_example_max": 2,
                "allowed_bands": ["challenge", "bridge", "easy"],
            },
        ),
        (
            "judge_student_shortcut_specialist_balanced",
            {
                "cap_per_label": 2200,
                "band_mix": [("challenge", 0.58), ("bridge", 0.42)],
                "per_example_max": 2,
                "allowed_bands": ["challenge", "bridge"],
            },
        ),
    ]

    for name, config in configs:
        dataset = balance_by_profile(
            all_rows,
            cap_per_label=config["cap_per_label"],
            band_mix=config["band_mix"],
            per_example_max=config["per_example_max"],
            allowed_bands=config["allowed_bands"],
        )
        report = save_dataset(name, dataset)
        print(f"{name}: {report['num_examples']} rows, score={report['average_judge_score']}, shortcut={report['average_shortcut_score']}")


if __name__ == "__main__":
    main()
