import argparse
import json
from pathlib import Path

import pandas as pd

from build_boundary_focus_pack import (
    SCORE_COMPONENTS,
    balance_dataset,
    build_example_key,
    collect_examples,
)
from build_judge_esnli import API_DIR, DEFAULT_SOURCES, load_candidates, load_gold_records, normalize_text


DEFAULT_OUTPUT_DIR = Path("result/boundary_score_ablation")


def selected_row_key(row):
    return (
        build_example_key(row["premise"], row["hypothesis"])
        + "||"
        + normalize_text(row["rationale"]).lower()
        + "||"
        + normalize_text(row["LLM_answer"]).lower()
    )


def selection_keys(df):
    if df.empty:
        return set()
    return set(df.apply(selected_row_key, axis=1))


def example_keys(df):
    if df.empty:
        return set()
    return set(df.apply(lambda row: build_example_key(row["premise"], row["hypothesis"]), axis=1))


def compare_selection(full_df, variant_df):
    full_rows = selection_keys(full_df)
    variant_rows = selection_keys(variant_df)
    full_examples = example_keys(full_df)
    variant_examples = example_keys(variant_df)
    shared_rows = full_rows & variant_rows
    shared_examples = full_examples & variant_examples
    changed_rows = full_rows - variant_rows
    changed_examples = full_examples - variant_examples

    return {
        "full_rows": len(full_rows),
        "variant_rows": len(variant_rows),
        "shared_selected_rows": len(shared_rows),
        "changed_selected_rows": len(changed_rows),
        "selection_change": len(changed_rows) / max(1, len(full_rows)),
        "full_examples": len(full_examples),
        "variant_examples": len(variant_examples),
        "shared_examples": len(shared_examples),
        "changed_examples": len(changed_examples),
        "example_change": len(changed_examples) / max(1, len(full_examples)),
        "source_counts": variant_df["judge_source"].value_counts().to_dict(),
    }


def rank_reports(reports):
    ranked = [
        {"component": component, **report}
        for component, report in reports.items()
    ]
    return sorted(
        ranked,
        key=lambda item: (item["selection_change"], item["changed_selected_rows"]),
        reverse=True,
    )


def build_boundary_mix(gold_records, candidate_tables, disabled_components=None):
    rows = collect_examples(
        gold_records=gold_records,
        candidate_tables=candidate_tables,
        disabled_components=disabled_components,
    )
    return balance_dataset(
        rows,
        cap_per_label=3200,
        allowed_bands={"easy", "boundary"},
        max_per_example=1,
    )


def save_csv(path, df):
    output = df.copy()
    output.index.name = "Unnamed: 0"
    output.to_csv(path)


def generate_ablation_study(output_dir=DEFAULT_OUTPUT_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gold_records, _ = load_gold_records()
    candidate_tables = {source: load_candidates(source) for source in DEFAULT_SOURCES}

    full_df = build_boundary_mix(gold_records, candidate_tables)
    save_csv(output_dir / "boundary_mix_full_score.csv", full_df)

    reports = {}
    for component in SCORE_COMPONENTS:
        variant_df = build_boundary_mix(
            gold_records,
            candidate_tables,
            disabled_components={component},
        )
        save_csv(output_dir / f"boundary_mix_without_{component}.csv", variant_df)
        reports[component] = compare_selection(full_df, variant_df)

    ranked = rank_reports(reports)
    summary = {
        "reference_dataset": str(API_DIR / "judge_student_boundary_mix_balanced - full.csv"),
        "generated_full_score_rows": int(len(full_df)),
        "ranking_metric": "selection_change",
        "top_three_components": [item["component"] for item in ranked[:3]],
        "ranked_components": ranked,
    }
    with (output_dir / "ablation_selection_report.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    pd.DataFrame(ranked).drop(columns=["source_counts"]).to_csv(
        output_dir / "ablation_selection_ranking.csv",
        index=False,
    )
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    summary = generate_ablation_study(args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
