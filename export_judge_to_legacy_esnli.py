import argparse
import json
import re
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a judge-selected rationale CSV into legacy [API] ESNLI/* - full.csv files."
    )
    parser.add_argument("--selected_csv", type=str, required=True)
    parser.add_argument("--output_prefix", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="[API] ESNLI")
    return parser.parse_args()


def ensure_premise_hypothesis(df: pd.DataFrame) -> pd.DataFrame:
    if "premise" in df.columns and "hypothesis" in df.columns:
        return df

    if "input" not in df.columns:
        raise ValueError("Selected CSV must contain either premise/hypothesis or input.")

    split_pairs = df["input"].astype(str).str.split("</s>", n=1, expand=True)
    if split_pairs.shape[1] != 2:
        raise ValueError("Could not split input into premise and hypothesis using </s>.")

    df = df.copy()
    df["premise"] = split_pairs[0]
    df["hypothesis"] = split_pairs[1]
    return df


def main() -> None:
    args = parse_args()
    selected_path = Path(args.selected_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(selected_path)
    df = ensure_premise_hypothesis(df)

    rationale_columns = sorted(
        column for column in df.columns if re.fullmatch(r"rationale_\d+", column)
    )
    if not rationale_columns:
        raise ValueError("Selected CSV does not contain any rationale_<n> columns.")

    exported = []
    for rationale_column in rationale_columns:
        suffix = rationale_column.split("_")[-1]
        rationale_type_column = f"rationale_type_{suffix}"
        output_name = f"{args.output_prefix}_r{suffix}"
        output_path = output_dir / f"{output_name} - full.csv"

        export_df = df[["premise", "hypothesis", rationale_column, "label"]].copy()
        export_df = export_df.rename(
            columns={
                rationale_column: "rationale",
                "label": "LLM_answer",
            }
        )

        if rationale_type_column in df.columns:
            export_df["rationale_type"] = df[rationale_type_column]

        export_df = export_df.dropna(subset=["premise", "hypothesis", "rationale", "LLM_answer"])
        export_df = export_df[export_df["rationale"].astype(str).str.strip() != ""]
        export_df.to_csv(output_path, index=False)

        exported.append(
            {
                "rationale_column": rationale_column,
                "output_name": output_name,
                "output_path": str(output_path),
                "num_rows": int(len(export_df)),
            }
        )

    report = {
        "selected_csv": str(selected_path),
        "output_dir": str(output_dir),
        "output_prefix": args.output_prefix,
        "exported_files": exported,
    }
    report_path = output_dir / f"{args.output_prefix}_legacy_export_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
