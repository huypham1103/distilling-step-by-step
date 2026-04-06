import argparse
import json
import math
import re
from pathlib import Path

import pandas as pd


API_DIR = Path("[API] ESNLI")
DEFAULT_SOURCES = [
    "neutral",
    "contrastive",
    "historical",
    "comparative",
    "causal",
    "consensus",
    "if_else",
]

THESIS_PREFERRED_SOURCES = ["neutral", "contrastive", "historical"]

LABEL_NORMALIZATION = {
    "entailment": "entailment",
    "entailed": "entailment",
    "neutral": "neutral",
    "contradiction": "contradiction",
    "contradicted": "contradiction",
}

TYPE_PRIOR = {
    "neutral": 1.00,
    "contrastive": 0.95,
    "historical": 0.90,
    "comparative": 0.82,
    "causal": 0.78,
    "consensus": 0.74,
    "if_else": 0.70,
    "paper": 0.55,
}

THESIS_TYPE_PRIOR = {
    "neutral": 1.00,
    "contrastive": 0.98,
    "historical": 0.96,
    "comparative": 0.55,
    "causal": 0.52,
    "consensus": 0.48,
    "if_else": 0.42,
    "paper": 0.35,
}

REASONING_CUES = (
    "because",
    "therefore",
    "however",
    "but",
    "although",
    "so the answer is",
    "the correct answer",
    "implies",
    "not necessarily",
)


def normalize_text(text):
    if pd.isna(text):
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def normalize_key(premise, hypothesis):
    return normalize_text(premise).lower() + "</s>" + normalize_text(hypothesis).lower()


def normalize_label(label):
    normalized = normalize_text(label).lower()
    normalized = normalized.replace("**", "").replace(".", "")
    if normalized in LABEL_NORMALIZATION:
        return LABEL_NORMALIZATION[normalized]
    if "contrad" in normalized:
        return "contradiction"
    if "neutral" in normalized:
        return "neutral"
    if "entail" in normalized:
        return "entailment"
    return normalized


def tokenize_for_overlap(text):
    return set(re.findall(r"[a-z]+", normalize_text(text).lower()))


def load_gold_records():
    paper_path = API_DIR / "paper - full.csv"
    if not paper_path.exists():
        raise FileNotFoundError(f"Missing gold anchor file: {paper_path}")

    paper = pd.read_csv(paper_path)
    records = []
    for row in paper.to_dict("records"):
        records.append({
            "key": normalize_key(row["premise"], row["hypothesis"]),
            "premise": normalize_text(row["premise"]),
            "hypothesis": normalize_text(row["hypothesis"]),
            "gold_label": normalize_label(row["LLM_answer"]),
            "paper_rationale": normalize_text(row.get("rationale", "")),
        })
    return records


def load_candidates(source_name):
    path = API_DIR / f"{source_name} - full.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing candidate rationale file: {path}")

    dataframe = pd.read_csv(path)
    required_columns = {"premise", "hypothesis", "rationale", "LLM_answer"}
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        raise ValueError(f"{path} is missing required columns: {sorted(missing_columns)}")

    candidates = {}
    for row in dataframe.to_dict("records"):
        key = normalize_key(row["premise"], row["hypothesis"])
        rationale = normalize_text(row["rationale"])
        if not rationale:
            continue
        candidates[key] = {
            "source": source_name,
            "premise": normalize_text(row["premise"]),
            "hypothesis": normalize_text(row["hypothesis"]),
            "label": normalize_label(row["LLM_answer"]),
            "rationale": rationale,
            "prompt": row.get("prompt", ""),
            "split": row.get("split", ""),
            "correct_index": row.get("correct_index", ""),
        }
    return candidates


def score_candidate(candidate, gold_label, strategy):
    rationale = candidate["rationale"]
    premise = candidate["premise"]
    hypothesis = candidate["hypothesis"]
    predicted_label = candidate["label"]
    source = candidate["source"]

    label_match = predicted_label == gold_label
    rationale_lower = rationale.lower()
    word_count = len(re.findall(r"\w+", rationale))

    if strategy == "thesis":
        if word_count < 8:
            length_score = -0.8
        elif word_count <= 64:
            length_score = 1.15
        elif word_count <= 120:
            length_score = 0.55
        elif word_count <= 180:
            length_score = 0.05
        else:
            length_score = -0.85
    else:
        if word_count < 6:
            length_score = -1.0
        elif word_count <= 80:
            length_score = 1.0
        elif word_count <= 180:
            length_score = 0.6
        elif word_count <= 260:
            length_score = 0.2
        else:
            length_score = -0.6

    support_tokens = tokenize_for_overlap(premise) | tokenize_for_overlap(hypothesis)
    rationale_tokens = tokenize_for_overlap(rationale)
    overlap_score = 0.0
    if rationale_tokens:
        overlap_score = len(rationale_tokens & support_tokens) / max(1, len(rationale_tokens))

    cue_bonus = sum(1 for cue in REASONING_CUES if cue in rationale_lower)
    cue_bonus = min(cue_bonus * 0.12, 0.48)

    label_bonus = 0.0
    if gold_label in rationale_lower:
        label_bonus += 0.25
    if "the correct answer" in rationale_lower or "so the answer is" in rationale_lower:
        label_bonus += 0.15

    type_prior = THESIS_TYPE_PRIOR if strategy == "thesis" else TYPE_PRIOR
    source_prior = type_prior.get(source, 0.5)
    if strategy == "thesis":
        label_score = 3.2 if label_match else -2.2
        preferred_bonus = 0.45 if source in THESIS_PREFERRED_SOURCES else 0.0
        total = label_score + source_prior + preferred_bonus + 1.0 * length_score + 0.85 * overlap_score + cue_bonus + label_bonus
    else:
        label_score = 3.0 if label_match else -2.0
        total = label_score + source_prior + 0.9 * length_score + 0.9 * overlap_score + cue_bonus + label_bonus

    return {
        "judge_score": round(total, 6),
        "label_match": label_match,
        "word_count": word_count,
        "overlap_score": round(overlap_score, 6),
    }


def infer_gold_label(candidates, strategy):
    scores = {}
    type_prior = THESIS_TYPE_PRIOR if strategy == "thesis" else TYPE_PRIOR
    for candidate in candidates:
        label = normalize_label(candidate.get("label", ""))
        if label not in {"entailment", "neutral", "contradiction"}:
            continue
        scores[label] = scores.get(label, 0.0) + type_prior.get(candidate["source"], 0.5)
    if not scores:
        return None
    return max(scores.items(), key=lambda item: item[1])[0]


def choose_best_candidate(candidates, strategy):
    if strategy == "thesis":
        preferred = [candidate for candidate in candidates if candidate["source"] in THESIS_PREFERRED_SOURCES]
        if preferred:
            return max(preferred, key=lambda candidate: (candidate["judge_score"], THESIS_TYPE_PRIOR.get(candidate["source"], 0.0)))
    type_prior = THESIS_TYPE_PRIOR if strategy == "thesis" else TYPE_PRIOR
    return max(candidates, key=lambda candidate: (candidate["judge_score"], type_prior.get(candidate["source"], 0.0)))


def build_judged_dataset(source_names, output_name, strategy):
    gold_records = load_gold_records()
    candidate_tables = {source: load_candidates(source) for source in source_names}

    rows = []
    source_counts = {}
    fallback_count = 0
    inferred_gold_count = 0
    skipped_count = 0

    for gold in gold_records:
        key = gold["key"]
        candidates = []
        for source_name, table in candidate_tables.items():
            candidate = table.get(key)
            if candidate is None:
                continue
            scored = candidate.copy()
            candidates.append(scored)

        gold_label = gold["gold_label"]
        if gold_label not in {"entailment", "neutral", "contradiction"}:
            gold_label = infer_gold_label(candidates, strategy)
            if gold_label is None:
                skipped_count += 1
                continue
            inferred_gold_count += 1

        for candidate in candidates:
            candidate.update(score_candidate(candidate, gold_label, strategy))

        matching_candidates = [candidate for candidate in candidates if candidate["label_match"]]

        if matching_candidates:
            best = choose_best_candidate(matching_candidates, strategy)
        else:
            fallback_count += 1
            best = {
                "source": "paper",
                "premise": gold["premise"],
                "hypothesis": gold["hypothesis"],
                "label": gold_label,
                "rationale": gold["paper_rationale"],
                "prompt": "",
                "split": "",
                "correct_index": "",
                "judge_score": (THESIS_TYPE_PRIOR if strategy == "thesis" else TYPE_PRIOR)["paper"],
                "label_match": True,
                "word_count": len(re.findall(r"\w+", gold["paper_rationale"])),
                "overlap_score": 0.0,
            }

        source_counts[best["source"]] = source_counts.get(best["source"], 0) + 1
        rows.append({
            "premise": gold["premise"],
            "hypothesis": gold["hypothesis"],
            "prompt": best.get("prompt", ""),
            "rationale": best["rationale"],
            "split": best.get("split", ""),
            "correct_index": best.get("correct_index", ""),
            "LLM_answer": gold_label,
            "judge_source": best["source"],
            "judge_score": best["judge_score"],
            "candidate_label": best["label"],
            "gold_label": gold_label,
            "label_match": best["label_match"],
            "word_count": best["word_count"],
            "overlap_score": best["overlap_score"],
        })

    judged = pd.DataFrame(rows)
    judged.index.name = "Unnamed: 0"

    output_csv = API_DIR / f"{output_name} - full.csv"
    judged.to_csv(output_csv)

    report = {
        "output_csv": str(output_csv),
        "num_examples": int(len(judged)),
        "source_counts": source_counts,
        "fallback_to_paper_count": int(fallback_count),
        "inferred_gold_count": int(inferred_gold_count),
        "skipped_count": int(skipped_count),
        "label_match_rate": float(judged["label_match"].mean()) if not judged.empty else math.nan,
        "average_judge_score": float(judged["judge_score"].mean()) if not judged.empty else math.nan,
        "sources_considered": source_names,
        "strategy": strategy,
    }
    report_path = API_DIR / f"{output_name}_judge_report.json"
    with report_path.open("w") as handle:
        json.dump(report, handle, indent=2)

    return output_csv, report_path, report


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-name", type=str, default="judge")
    parser.add_argument("--sources", nargs="+", default=DEFAULT_SOURCES)
    parser.add_argument("--strategy", type=str, choices=["baseline", "thesis"], default="baseline")
    return parser.parse_args()


def main():
    args = parse_args()
    output_csv, report_path, report = build_judged_dataset(args.sources, args.output_name, args.strategy)
    print(json.dumps(report, indent=2))
    print(f"Saved judged rationale CSV to {output_csv}")
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
