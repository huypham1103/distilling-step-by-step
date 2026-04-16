import json
from pathlib import Path

import pandas as pd


API_DIR = Path("[API] ESNLI")
HYBRID_PATH = API_DIR / "judge_student_multiview_hybrid_balanced - full.csv"
SUPERCLEAN_PATH = API_DIR / "judge_student_singleview_superclean_balanced - full.csv"

LABELS = ["entailment", "neutral", "contradiction"]


def normalize_text(text):
    if pd.isna(text):
        return ""
    return " ".join(str(text).split())


def example_key(frame):
    return (
        frame["premise"].fillna("").map(normalize_text).str.lower()
        + "</s>"
        + frame["hypothesis"].fillna("").map(normalize_text).str.lower()
    )


def row_key(frame):
    return (
        example_key(frame)
        + "||"
        + frame["rationale"].fillna("").map(normalize_text).str.lower()
        + "||"
        + frame["LLM_answer"].fillna("").astype(str).str.lower()
    )


def load_frame(path, source_pack):
    df = pd.read_csv(path)
    df["source_pack"] = source_pack
    df["example_key"] = example_key(df)
    df["row_key"] = row_key(df)
    if "judge_view_rank" not in df.columns:
        df["judge_view_rank"] = 1
    df["judge_view_rank"] = df["judge_view_rank"].fillna(1).astype(int)
    df["agreement_count"] = df["agreement_count"].fillna(0)
    df["voted_label_margin"] = df["voted_label_margin"].fillna(0.0)
    df["judge_score"] = df["judge_score"].fillna(0.0)
    df["quality_score"] = (
        df["judge_score"]
        + 0.16 * df["agreement_count"]
        + 0.10 * df["voted_label_margin"]
        + (df["judge_view_rank"] == 1).astype(float) * 0.08
    )
    return df


def dedupe(df):
    if df.empty:
        return df.copy()
    return (
        df.sort_values(
            by=["quality_score", "judge_score", "agreement_count", "voted_label_margin"],
            ascending=[False, False, False, False],
        )
        .drop_duplicates(subset=["row_key"], keep="first")
        .copy()
    )


def balance_top(df, cap_per_label):
    df = dedupe(df)
    counts = df["LLM_answer"].value_counts()
    target = min(cap_per_label, int(counts.min()))
    parts = []
    for label in LABELS:
        subset = df[df["LLM_answer"] == label].sort_values(
            by=["quality_score", "agreement_count", "judge_score", "voted_label_margin"],
            ascending=[False, False, False, False],
        ).head(target)
        parts.append(subset)
    return pd.concat(parts, ignore_index=True)


def balance_with_example_cap(df, cap_per_label, max_per_example):
    df = dedupe(df)
    counts = df["LLM_answer"].value_counts()
    target = min(cap_per_label, int(counts.min()))
    selected_parts = []

    for label in LABELS:
        subset = df[df["LLM_answer"] == label].sort_values(
            by=["quality_score", "agreement_count", "judge_score", "voted_label_margin"],
            ascending=[False, False, False, False],
        )
        selected = []
        example_counts = {}
        for _, row in subset.iterrows():
            key = row["example_key"]
            if example_counts.get(key, 0) >= max_per_example:
                continue
            selected.append(row)
            example_counts[key] = example_counts.get(key, 0) + 1
            if len(selected) >= target:
                break
        selected_parts.append(pd.DataFrame(selected))

    return pd.concat(selected_parts, ignore_index=True)


def add_bonus(df, source_pack, bonus):
    boosted = df.copy()
    boosted.loc[boosted["source_pack"] == source_pack, "quality_score"] += bonus
    return boosted


def build_firstview_strict(hybrid):
    first = hybrid[hybrid["judge_view_rank"] == 1].copy()
    first["score_cutoff"] = first.groupby("LLM_answer")["judge_score"].transform(lambda s: s.quantile(0.50))
    first["margin_cutoff"] = first.groupby("LLM_answer")["voted_label_margin"].transform(lambda s: s.quantile(0.35))
    first["agree_cutoff"] = first.groupby("LLM_answer")["agreement_count"].transform("median")
    strict = first[
        (first["judge_score"] >= first["score_cutoff"])
        & (first["voted_label_margin"] >= first["margin_cutoff"])
        & (first["agreement_count"] >= first["agree_cutoff"])
    ].drop(columns=["score_cutoff", "margin_cutoff", "agree_cutoff"])
    return balance_top(strict, 3000)


def build_pruned_secondview(hybrid):
    first = hybrid[hybrid["judge_view_rank"] == 1].copy()
    second = hybrid[hybrid["judge_view_rank"] == 2].copy()
    second["score_cutoff"] = second.groupby("LLM_answer")["judge_score"].transform(lambda s: s.quantile(0.62))
    second["margin_cutoff"] = second.groupby("LLM_answer")["voted_label_margin"].transform(lambda s: s.quantile(0.50))
    second["agree_cutoff"] = second.groupby("LLM_answer")["agreement_count"].transform("median")
    second = second[
        (second["judge_score"] >= second["score_cutoff"])
        & (second["voted_label_margin"] >= second["margin_cutoff"])
        & (second["agreement_count"] >= second["agree_cutoff"])
    ].drop(columns=["score_cutoff", "margin_cutoff", "agree_cutoff"])
    mixed = pd.concat([first, second], ignore_index=True)
    return balance_with_example_cap(mixed, 4500, max_per_example=2)


def build_cleanfusion(hybrid, superclean):
    first = hybrid[hybrid["judge_view_rank"] == 1].copy()
    second = hybrid[hybrid["judge_view_rank"] == 2].copy()
    second["score_cutoff"] = second.groupby("LLM_answer")["judge_score"].transform(lambda s: s.quantile(0.70))
    second = second[second["judge_score"] >= second["score_cutoff"]].drop(columns=["score_cutoff"])
    boosted = pd.concat(
        [
            add_bonus(superclean, "singleview_superclean", 0.42),
            add_bonus(first, "multiview_hybrid", 0.08),
            add_bonus(second, "multiview_hybrid", 0.02),
        ],
        ignore_index=True,
    )
    return balance_with_example_cap(boosted, 3600, max_per_example=2)


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
        "source_pack_counts": df["source_pack"].value_counts().to_dict(),
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
    hybrid = load_frame(HYBRID_PATH, "multiview_hybrid")
    superclean = load_frame(SUPERCLEAN_PATH, "singleview_superclean")

    datasets = {
        "judge_student_multiview_hybrid_firstview_strict": build_firstview_strict(hybrid),
        "judge_student_multiview_hybrid_pruned": build_pruned_secondview(hybrid),
        "judge_student_multiview_hybrid_cleanfusion": build_cleanfusion(hybrid, superclean),
    }

    for name, df in datasets.items():
        output_csv, report_path, report = save_dataset(name, df)
        print(json.dumps(report, indent=2))
        print(f"Saved judged rationale CSV to {output_csv}")
        print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
