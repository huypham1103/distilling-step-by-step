import json
from pathlib import Path

import pandas as pd


API_DIR = Path("[API] ESNLI")

SOURCE_FILES = {
    "signal": API_DIR / "judge_student_signal - full.csv",
    "label_expert_guarded": API_DIR / "judge_label_expert_guarded - full.csv",
    "singleview_superclean": API_DIR / "judge_student_singleview_superclean_balanced - full.csv",
    "singleview_diverse": API_DIR / "judge_student_singleview_diverse_balanced - full.csv",
    "multiview_hybrid": API_DIR / "judge_student_multiview_hybrid_balanced - full.csv",
    "multiview_raw": API_DIR / "judge_student_multiview - full.csv",
    "multiview_hardclean": API_DIR / "judge_student_multiview_hardclean_balanced - full.csv",
    "multiview_sourceblend": API_DIR / "judge_student_multiview_sourceblend_balanced - full.csv",
}

LABEL_SOURCE_PRIORITY = {
    "entailment": ["historical", "consensus", "contrastive", "causal", "neutral", "if_else", "comparative"],
    "neutral": ["if_else", "comparative", "neutral", "contrastive", "causal", "consensus", "historical"],
    "contradiction": ["comparative", "contrastive", "causal", "neutral", "consensus", "historical", "if_else"],
}

MULTIVIEW_CAP = 5500
SINGLEVIEW_CAP = 3200


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


def load_sources():
    frames = {}
    for name, path in SOURCE_FILES.items():
        df = pd.read_csv(path)
        df["source_pack"] = name
        df["example_key"] = example_key(df)
        df["row_key"] = row_key(df)
        if "judge_view_rank" in df.columns:
            df["judge_view_rank"] = df["judge_view_rank"].fillna(1).astype(int)
        else:
            df["judge_view_rank"] = 1
        if "agreement_count" in df.columns:
            df["agreement_count"] = df["agreement_count"].fillna(0)
        else:
            df["agreement_count"] = 0
        if "voted_label_margin" in df.columns:
            df["voted_label_margin"] = df["voted_label_margin"].fillna(0.0)
        else:
            df["voted_label_margin"] = 0.0
        if "judge_score" in df.columns:
            df["judge_score"] = df["judge_score"].fillna(0.0)
        else:
            df["judge_score"] = 0.0
        df["quality_score"] = (
            df["judge_score"]
            + 0.18 * df["agreement_count"]
            + 0.10 * df["voted_label_margin"]
            + (df["judge_view_rank"] == 1).astype(float) * 0.25
        )
        frames[name] = df
    return frames


def dedupe_keep_best(df):
    if df.empty:
        return df.copy()
    sort_cols = ["quality_score", "judge_score", "agreement_count", "voted_label_margin"]
    deduped = df.sort_values(by=sort_cols, ascending=[False, False, False, False])
    return deduped.drop_duplicates(subset=["row_key"], keep="first").copy()


def balance_top(df, cap_per_label):
    if df.empty:
        return df.copy()
    df = dedupe_keep_best(df)
    counts = df["LLM_answer"].value_counts()
    target = min(cap_per_label, int(counts.min()))
    parts = []
    for label in ["entailment", "neutral", "contradiction"]:
        subset = df[df["LLM_answer"] == label].sort_values(
            by=["quality_score", "agreement_count", "judge_score", "voted_label_margin"],
            ascending=[False, False, False, False],
        ).head(target)
        parts.append(subset)
    return pd.concat(parts, ignore_index=True)


def balance_diverse(df, cap_per_label, max_per_example):
    if df.empty:
        return df.copy()
    df = dedupe_keep_best(df)
    counts = df["LLM_answer"].value_counts()
    target = min(cap_per_label, int(counts.min()))
    selected_parts = []

    for label in ["entailment", "neutral", "contradiction"]:
        subset = df[df["LLM_answer"] == label].copy()
        subset["source_bucket"] = subset["source_pack"] + "::" + subset["judge_source"].fillna("")
        subset = subset.sort_values(
            by=["quality_score", "agreement_count", "judge_score", "voted_label_margin"],
            ascending=[False, False, False, False],
        )

        buckets = {}
        for source in LABEL_SOURCE_PRIORITY[label]:
            rows = subset[subset["judge_source"] == source].to_dict("records")
            if rows:
                buckets[source] = rows

        for source in subset["judge_source"].dropna().unique():
            rows = subset[subset["judge_source"] == source].to_dict("records")
            if rows and source not in buckets:
                buckets[source] = rows

        chosen = []
        chosen_row_keys = set()
        example_counts = {}

        while len(chosen) < target:
            progressed = False
            for source in LABEL_SOURCE_PRIORITY[label] + [s for s in buckets if s not in LABEL_SOURCE_PRIORITY[label]]:
                rows = buckets.get(source, [])
                while rows:
                    row = rows.pop(0)
                    if row["row_key"] in chosen_row_keys:
                        continue
                    if example_counts.get(row["example_key"], 0) >= max_per_example:
                        continue
                    chosen.append(row)
                    chosen_row_keys.add(row["row_key"])
                    example_counts[row["example_key"]] = example_counts.get(row["example_key"], 0) + 1
                    progressed = True
                    break
                if len(chosen) >= target:
                    break
            if not progressed:
                break

        if len(chosen) < target:
            remaining = subset[~subset["row_key"].isin(chosen_row_keys)].to_dict("records")
            for row in remaining:
                if len(chosen) >= target:
                    break
                if example_counts.get(row["example_key"], 0) >= max_per_example:
                    continue
                chosen.append(row)
                chosen_row_keys.add(row["row_key"])
                example_counts[row["example_key"]] = example_counts.get(row["example_key"], 0) + 1

        selected_parts.append(pd.DataFrame(chosen).head(target))

    return pd.concat(selected_parts, ignore_index=True)


def boost_pack(df, pack_name, bonus):
    boosted = df.copy()
    boosted.loc[boosted["source_pack"] == pack_name, "quality_score"] += bonus
    return boosted


def build_pack(frames):
    packs = {}

    hybrid = frames["multiview_hybrid"]
    superclean = frames["singleview_superclean"]
    diverse = frames["singleview_diverse"]
    hardclean = frames["multiview_hardclean"]
    sourceblend = frames["multiview_sourceblend"]
    label_expert = frames["label_expert_guarded"]
    signal = frames["signal"]

    packs["judge_hybrid_firstview_balanced"] = balance_top(
        hybrid[hybrid["judge_view_rank"] == 1],
        SINGLEVIEW_CAP,
    )

    hybrid_core = hybrid.copy()
    hybrid_core["label_score_cutoff"] = hybrid_core.groupby("LLM_answer")["judge_score"].transform(lambda s: s.quantile(0.42))
    hybrid_core["label_margin_cutoff"] = hybrid_core.groupby("LLM_answer")["voted_label_margin"].transform(lambda s: s.quantile(0.32))
    hybrid_core["label_agreement_cutoff"] = hybrid_core.groupby("LLM_answer")["agreement_count"].transform("median")
    hybrid_core = hybrid_core[
        (hybrid_core["judge_score"] >= hybrid_core["label_score_cutoff"])
        & (hybrid_core["voted_label_margin"] >= hybrid_core["label_margin_cutoff"])
        & (hybrid_core["agreement_count"] >= hybrid_core["label_agreement_cutoff"])
    ].drop(columns=["label_score_cutoff", "label_margin_cutoff", "label_agreement_cutoff"])
    packs["judge_hybrid_core_balanced"] = balance_top(hybrid_core, SINGLEVIEW_CAP)

    cleanboost = pd.concat(
        [
            boost_pack(hybrid[hybrid["judge_view_rank"] == 1], "multiview_hybrid", 0.0),
            boost_pack(superclean, "singleview_superclean", 0.55),
            boost_pack(superclean, "singleview_superclean", 0.55),
        ],
        ignore_index=True,
    )
    packs["judge_hybrid_cleanboost_balanced"] = balance_top(cleanboost, MULTIVIEW_CAP)

    diverseboost = pd.concat(
        [
            hybrid,
            boost_pack(diverse, "singleview_diverse", 0.28),
            boost_pack(sourceblend, "multiview_sourceblend", 0.24),
        ],
        ignore_index=True,
    )
    packs["judge_hybrid_diverseboost_balanced"] = balance_diverse(diverseboost, MULTIVIEW_CAP, max_per_example=2)

    singleview_fusion = pd.concat([superclean, diverse], ignore_index=True)
    packs["judge_singleview_fusion_balanced"] = balance_top(singleview_fusion, SINGLEVIEW_CAP)

    singleview_specialist = pd.concat(
        [
            boost_pack(superclean, "singleview_superclean", 0.35),
            boost_pack(diverse, "singleview_diverse", 0.15),
            boost_pack(label_expert, "label_expert_guarded", 0.18),
        ],
        ignore_index=True,
    )
    packs["judge_singleview_specialist_balanced"] = balance_diverse(singleview_specialist, 3000, max_per_example=2)

    sourceblend_core = sourceblend[
        ((sourceblend["judge_view_rank"] == 1) & (sourceblend["judge_score"] >= sourceblend["judge_score"].quantile(0.45)))
        | ((sourceblend["judge_view_rank"] == 2) & (sourceblend["judge_score"] >= sourceblend["judge_score"].quantile(0.7)))
    ]
    packs["judge_multiview_sourceblend_core_balanced"] = balance_diverse(sourceblend_core, MULTIVIEW_CAP, max_per_example=2)

    hardclean_hybrid = pd.concat(
        [
            boost_pack(hardclean, "multiview_hardclean", 0.22),
            boost_pack(hybrid, "multiview_hybrid", 0.18),
        ],
        ignore_index=True,
    )
    packs["judge_hardclean_hybrid_fusion_balanced"] = balance_top(hardclean_hybrid, MULTIVIEW_CAP)

    expert_hybrid = pd.concat(
        [
            boost_pack(hybrid[hybrid["judge_view_rank"] == 1], "multiview_hybrid", 0.24),
            boost_pack(label_expert, "label_expert_guarded", 0.2),
            boost_pack(superclean, "singleview_superclean", 0.3),
        ],
        ignore_index=True,
    )
    packs["judge_expert_hybrid_fusion_balanced"] = balance_diverse(expert_hybrid, 3000, max_per_example=2)

    precision_diverse = pd.concat(
        [
            boost_pack(superclean, "singleview_superclean", 0.36),
            boost_pack(diverse, "singleview_diverse", 0.18),
            boost_pack(sourceblend, "multiview_sourceblend", 0.2),
            boost_pack(signal, "signal", 0.1),
        ],
        ignore_index=True,
    )
    packs["judge_precision_diverse_mix_balanced"] = balance_diverse(precision_diverse, SINGLEVIEW_CAP, max_per_example=2)

    return packs


def save_pack(name, df):
    df = df.copy()
    if "source_pack" in df.columns:
        source_pack_counts = df["source_pack"].value_counts().to_dict()
    else:
        source_pack_counts = {}

    output_csv = API_DIR / f"{name} - full.csv"
    df.index.name = "Unnamed: 0"
    df.to_csv(output_csv)

    report = {
        "output_csv": str(output_csv),
        "num_examples": int(len(df)),
        "label_counts": df["LLM_answer"].value_counts().to_dict(),
        "judge_source_counts": df["judge_source"].value_counts().to_dict(),
        "source_pack_counts": source_pack_counts,
        "average_judge_score": float(df["judge_score"].mean()) if not df.empty else None,
        "average_agreement_count": float(df["agreement_count"].mean()) if not df.empty else None,
        "average_voted_label_margin": float(df["voted_label_margin"].mean()) if not df.empty else None,
        "extra_view_count": int((df["judge_view_rank"] > 1).sum()) if "judge_view_rank" in df.columns else 0,
    }
    report_path = API_DIR / f"{name}_judge_report.json"
    with report_path.open("w") as handle:
        json.dump(report, handle, indent=2)

    return output_csv, report_path, report


def main():
    frames = load_sources()
    packs = build_pack(frames)

    for name, df in packs.items():
        output_csv, report_path, report = save_pack(name, df)
        print(json.dumps(report, indent=2))
        print(f"Saved judged rationale CSV to {output_csv}")
        print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
