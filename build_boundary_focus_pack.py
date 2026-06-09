import json
from pathlib import Path

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


BOUNDARY_SOURCE_PRIOR = {
    "entailment": {
        "historical": 1.00,
        "consensus": 0.98,
        "contrastive": 0.95,
        "causal": 0.92,
        "neutral": 0.88,
        "if_else": 0.74,
        "comparative": 0.70,
    },
    "neutral": {
        "if_else": 1.00,
        "comparative": 0.99,
        "neutral": 0.96,
        "contrastive": 0.90,
        "causal": 0.84,
        "consensus": 0.82,
        "historical": 0.72,
    },
    "contradiction": {
        "comparative": 1.00,
        "contrastive": 0.99,
        "causal": 0.97,
        "neutral": 0.89,
        "consensus": 0.84,
        "historical": 0.74,
        "if_else": 0.70,
    },
}

LABELS = ["entailment", "neutral", "contradiction"]

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


def build_example_key(premise, hypothesis):
    return normalize_text(premise).lower() + "</s>" + normalize_text(hypothesis).lower()


def vote_labels(candidates):
    scores = {label: 0.0 for label in LABELS}
    counts = {label: 0 for label in LABELS}
    for candidate in candidates:
        label = normalize_label(candidate["label"])
        if label not in scores:
            continue
        scores[label] += BOUNDARY_SOURCE_PRIOR[label].get(candidate["source"], 0.0)
        counts[label] += 1
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    winner, winner_score = ranked[0]
    runner_up_score = ranked[1][1] if len(ranked) > 1 else 0.0
    return winner, winner_score, runner_up_score, counts


def candidate_quality(
    candidate,
    training_label,
    vote_margin,
    agreement_count,
    total_candidates,
    disabled_components=None,
    return_contributions=False,
):
    disabled_components = frozenset(disabled_components or ())
    unknown_components = disabled_components.difference(SCORE_COMPONENTS)
    if unknown_components:
        raise ValueError(f"Unknown score components: {sorted(unknown_components)}")

    rationale = candidate["rationale"]
    rationale_lower = rationale.lower()
    word_count = len(rationale.split())
    overlap_tokens = tokenize_for_overlap(candidate["premise"]) | tokenize_for_overlap(candidate["hypothesis"])
    rationale_tokens = tokenize_for_overlap(rationale)
    overlap_score = len(overlap_tokens & rationale_tokens) / max(1, len(rationale_tokens))
    source_prior = BOUNDARY_SOURCE_PRIOR.get(training_label, {}).get(candidate["source"], 0.0)
    teaching_hits = sum(1 for cue in LABEL_TEACHING_CUES.get(training_label, ()) if cue in rationale_lower)
    explicit_label = 1.0 if training_label in rationale_lower else 0.0
    format_bonus = 0.25 if "the correct answer" in rationale_lower or "so the answer is" in rationale_lower else 0.0
    brevity = 0.25 if 10 <= word_count <= 96 else (-0.15 if word_count > 128 else 0.0)
    agreement_ratio = agreement_count / max(1, total_candidates)
    contributions = {
        "source": source_prior,
        "ground": 0.55 * overlap_score,
        "cue": 0.22 * teaching_hits,
        "explicit": 0.20 * explicit_label,
        "format": format_bonus,
        "brief": brevity,
        "agreement_count": 0.12 * agreement_count,
        "agreement_ratio": 0.08 * agreement_ratio,
        "margin": 0.03 * vote_margin,
    }
    score = sum(
        contribution
        for component, contribution in contributions.items()
        if component not in disabled_components
    )
    if return_contributions:
        return score, contributions
    return score


def choose_boundary_partner(primary, matching_candidates):
    alternatives = []
    for candidate in matching_candidates:
        if candidate is primary:
            continue
        if candidate["source"] == primary["source"]:
            continue
        similarity = rationale_jaccard(primary["rationale"], candidate["rationale"])
        if similarity >= 0.76:
            continue
        alternatives.append((1.0 - similarity, candidate))
    if not alternatives:
        return None
    alternatives.sort(key=lambda item: (item[1]["quality_score"], item[0]), reverse=True)
    return alternatives[0][1]


def append_row(rows, gold, candidate, output_label, vote_margin, label_counts, winner_score, boundary_band):
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
        "paper_gold_label": gold["gold_label"],
        "voted_label": output_label,
        "voted_label_support": winner_score,
        "voted_label_margin": vote_margin,
        "label_match": True,
        "word_count": len(candidate["rationale"].split()),
        "overlap_score": candidate["overlap_score"],
        "agreement_count": label_counts.get(output_label, 0),
        "agreement_ratio": label_counts.get(output_label, 0) / max(1, sum(label_counts.values())),
        "judge_view_rank": candidate.get("judge_view_rank", 1),
        "boundary_band": boundary_band,
    })


def collect_examples(gold_records=None, candidate_tables=None, disabled_components=None):
    if gold_records is None:
        gold_records, _ = load_gold_records()
    if candidate_tables is None:
        candidate_tables = {source: load_candidates(source) for source in DEFAULT_SOURCES}
    rows = []

    for gold in gold_records:
        key = gold["key"]
        candidates = []
        for source_name, table in candidate_tables.items():
            candidate = table.get(key)
            if candidate is None:
                continue
            scored = candidate.copy()
            candidates.append(scored)

        if not candidates:
            continue

        voted_label, winner_score, runner_up_score, label_counts = vote_labels(candidates)
        vote_margin = winner_score - runner_up_score
        paper_label = normalize_label(gold["gold_label"])

        if paper_label in LABELS and (paper_label == voted_label or vote_margin < 1.55):
            training_label = paper_label
        elif voted_label in LABELS:
            training_label = voted_label
        else:
            continue

        matching = []
        for candidate in candidates:
            label = normalize_label(candidate["label"])
            if label != training_label:
                continue
            agreement_count = label_counts.get(training_label, 0)
            total_candidates = max(1, len(candidates))
            quality = candidate_quality(
                candidate,
                training_label,
                vote_margin,
                agreement_count,
                total_candidates,
                disabled_components=disabled_components,
            )
            rationale_tokens = tokenize_for_overlap(candidate["rationale"])
            support_tokens = tokenize_for_overlap(candidate["premise"]) | tokenize_for_overlap(candidate["hypothesis"])
            overlap_score = len(rationale_tokens & support_tokens) / max(1, len(rationale_tokens))
            enriched = candidate.copy()
            enriched["quality_score"] = quality
            enriched["overlap_score"] = overlap_score
            matching.append(enriched)

        if not matching:
            continue

        matching.sort(key=lambda cand: cand["quality_score"], reverse=True)
        best = matching[0]

        if vote_margin >= 2.1 and label_counts.get(training_label, 0) >= 5:
            boundary_band = "easy"
        elif vote_margin >= 1.1 and label_counts.get(training_label, 0) >= 2:
            boundary_band = "boundary"
        else:
            boundary_band = "hard"

        append_row(rows, gold, best, training_label, vote_margin, label_counts, winner_score, boundary_band)

        partner = choose_boundary_partner(best, matching)
        if partner is not None and boundary_band != "hard":
            partner = partner.copy()
            partner["judge_view_rank"] = 2
            append_row(rows, gold, partner, training_label, vote_margin, label_counts, winner_score, "bridge")

    return pd.DataFrame(rows)


def balance_dataset(df, cap_per_label, allowed_bands, max_per_example):
    if df.empty:
        return df.copy()

    df = df[df["boundary_band"].isin(allowed_bands)].copy()
    df["example_key"] = df.apply(lambda row: build_example_key(row["premise"], row["hypothesis"]), axis=1)
    df["row_key"] = (
        df["example_key"]
        + "||"
        + df["rationale"].fillna("").map(normalize_text).str.lower()
        + "||"
        + df["LLM_answer"].fillna("").astype(str).str.lower()
    )
    df = df.sort_values(
        by=["judge_score", "agreement_count", "voted_label_margin"],
        ascending=[False, False, False],
    ).drop_duplicates(subset=["row_key"], keep="first")

    counts = df["LLM_answer"].value_counts()
    target = min(cap_per_label, int(counts.min()))
    parts = []

    for label in LABELS:
        subset = df[df["LLM_answer"] == label].sort_values(
            by=["judge_score", "agreement_count", "voted_label_margin"],
            ascending=[False, False, False],
        )
        chosen = []
        example_counts = {}
        for _, row in subset.iterrows():
            key = row["example_key"]
            if example_counts.get(key, 0) >= max_per_example:
                continue
            chosen.append(row)
            example_counts[key] = example_counts.get(key, 0) + 1
            if len(chosen) >= target:
                break
        parts.append(pd.DataFrame(chosen))

    return pd.concat(parts, ignore_index=True)


def save_dataset(name, df):
    output_csv = API_DIR / f"{name} - full.csv"
    df = df.copy()
    df.index.name = "Unnamed: 0"
    df.to_csv(output_csv)

    report = {
        "output_csv": str(output_csv),
        "num_examples": int(len(df)),
        "label_counts": df["LLM_answer"].value_counts().to_dict(),
        "judge_source_counts": df["judge_source"].value_counts().to_dict(),
        "boundary_band_counts": df["boundary_band"].value_counts().to_dict(),
        "average_judge_score": float(df["judge_score"].mean()),
        "average_agreement_count": float(df["agreement_count"].mean()),
        "average_voted_label_margin": float(df["voted_label_margin"].mean()),
        "extra_view_count": int((df["judge_view_rank"] > 1).sum()),
    }
    report_path = API_DIR / f"{name}_judge_report.json"
    with report_path.open("w") as handle:
        json.dump(report, handle, indent=2)

    return output_csv, report_path, report


def main():
    rows = collect_examples()

    datasets = {
        "judge_student_boundary_mix_balanced": balance_dataset(
            rows,
            cap_per_label=3200,
            allowed_bands={"easy", "boundary"},
            max_per_example=1,
        ),
        "judge_student_boundary_bridge_balanced": balance_dataset(
            rows,
            cap_per_label=4200,
            allowed_bands={"easy", "boundary", "bridge"},
            max_per_example=2,
        ),
        "judge_student_boundary_specialist_balanced": balance_dataset(
            rows[rows["judge_source"].isin({"historical", "contrastive", "comparative", "if_else", "consensus"})],
            cap_per_label=3000,
            allowed_bands={"boundary", "bridge"},
            max_per_example=2,
        ),
    }

    for name, df in datasets.items():
        output_csv, report_path, report = save_dataset(name, df)
        print(json.dumps(report, indent=2))
        print(f"Saved judged rationale CSV to {output_csv}")
        print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
