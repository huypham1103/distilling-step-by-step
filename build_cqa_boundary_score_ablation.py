import argparse
import json
from pathlib import Path

from build_boundary_score_ablation import compare_selection
from build_judge_cqa_outstanding_pack import (
    BASE_SOURCE_PRIOR,
    SCORE_COMPONENTS,
    SOURCES,
    band_select,
    build_boundary_rows,
    load_candidates,
    load_gold_records,
)


DEFAULT_COMPONENTS = ("format", "cue", "source")
DEFAULT_OUTPUT_DIR = Path("result/boundary_score_ablation_cqa")


def build_boundary_mix(gold_records, candidate_tables, disabled_components=None):
    rows = build_boundary_rows(
        gold_records,
        candidate_tables,
        disabled_components=disabled_components,
    )
    return band_select(
        rows,
        cap=8500,
        max_per_example=1,
        band_column="boundary_band",
        allowed_bands={"easy", "boundary"},
        source_order=list(BASE_SOURCE_PRIOR),
    )


def save_csv(path, df):
    output = df.copy()
    output.index.name = "Unnamed: 0"
    output.to_csv(path)


def generate_cqa_ablation_datasets(components=DEFAULT_COMPONENTS, output_dir=DEFAULT_OUTPUT_DIR):
    components = tuple(components)
    unknown_components = set(components).difference(SCORE_COMPONENTS)
    if unknown_components:
        raise ValueError(f"Unknown score components: {sorted(unknown_components)}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gold_records = load_gold_records()
    candidate_tables = {source: load_candidates(source) for source in SOURCES}

    full_df = build_boundary_mix(gold_records, candidate_tables)
    save_csv(output_dir / "boundary_mix_full_score.csv", full_df)

    reports = {}
    for component in components:
        variant_df = build_boundary_mix(
            gold_records,
            candidate_tables,
            disabled_components={component},
        )
        save_csv(output_dir / f"boundary_mix_without_{component}.csv", variant_df)
        reports[component] = compare_selection(full_df, variant_df)

    summary = {
        "components": list(components),
        "generated_full_score_rows": int(len(full_df)),
        "selection_reports": reports,
    }
    with (output_dir / "cqa_ablation_selection_report.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--components", nargs="+", default=list(DEFAULT_COMPONENTS))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    summary = generate_cqa_ablation_datasets(args.components, args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
