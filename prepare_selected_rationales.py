import argparse
import json
from pathlib import Path

from selection_utils import (
    ESNLI_RATIONALE_TYPES,
    build_canonical_esnli_dataset,
    maybe_attach_official_esnli_splits,
    select_rationales,
    validate_latest_rationale_files,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--rationale-root', type=str, default='[API] ESNLI')
    parser.add_argument('--label-source', type=str, default='[API] ESNLI/paper.csv')
    parser.add_argument('--output-dir', type=str, default='artifacts/esnli_selection')
    parser.add_argument('--selection-policy', type=str, default='heuristic')
    parser.add_argument('--num-selected-rationales', type=int, default=1)
    parser.add_argument('--judge-model-name', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-examples', type=int, default=None)
    args = parser.parse_args()

    rationale_root = Path(args.rationale_root)
    label_source = Path(args.label_source)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    validation_report, validation_errors = validate_latest_rationale_files(rationale_root, ESNLI_RATIONALE_TYPES)
    if validation_errors:
        raise SystemExit('\n'.join(validation_errors))

    canonical, canonical_report = build_canonical_esnli_dataset(
        rationale_root=rationale_root,
        label_source_path=label_source,
        rationale_types=ESNLI_RATIONALE_TYPES,
    )
    canonical, split_report = maybe_attach_official_esnli_splits(canonical)

    if args.max_examples is not None:
        canonical = canonical.head(args.max_examples).copy()

    selected, selection_report = select_rationales(
        canonical=canonical,
        rationale_types=ESNLI_RATIONALE_TYPES,
        selection_policy=args.selection_policy,
        num_selected_rationales=args.num_selected_rationales,
        judge_model_name=args.judge_model_name,
        seed=args.seed,
    )

    canonical_path = output_dir / 'canonical_esnli_rationales.csv'
    selected_path = output_dir / f'selected_{args.selection_policy}_top{args.num_selected_rationales}.csv'
    report_path = output_dir / f'selected_{args.selection_policy}_top{args.num_selected_rationales}_report.json'

    canonical.to_csv(canonical_path, index=False)
    selected.to_csv(selected_path, index=False)
    with report_path.open('w', encoding='utf-8') as file_handle:
        json.dump(
            {
                'validation_report': validation_report,
                'canonical_report': canonical_report,
                'split_report': split_report,
                'selection_report': selection_report,
            },
            file_handle,
            indent=2,
        )

    print(f'Canonical dataset saved to {canonical_path}')
    print(f'Selected dataset saved to {selected_path}')
    print(f'Report saved to {report_path}')


if __name__ == '__main__':
    main()
