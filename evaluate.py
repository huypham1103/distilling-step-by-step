import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


LABEL_CANDIDATES = [
    "label",
    "gold_label",
    "paper_gold_label",
    "LLM_answer",
    "candidate_label",
    "voted_label",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="Path to the saved Hugging Face model.")
    parser.add_argument("--test-data", required=True, help="Path to the test file.")
    parser.add_argument(
        "--output-path",
        default=None,
        help="Where to save predictions. Defaults to <test_data_stem>_predictions.csv.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-input-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--prefix",
        default="predict: ",
        help="Prefix added before each input. Use an empty string for standard models.",
    )
    return parser.parse_args()


def load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".json":
        return pd.read_json(path)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported test data format: {path.suffix}")


def resolve_output_path(test_data_path: Path, output_path_arg: Optional[str]) -> Path:
    if output_path_arg:
        output_path = Path(output_path_arg)
    else:
        output_path = Path.cwd() / f"{test_data_path.stem}_predictions.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def build_inputs(df: pd.DataFrame, tokenizer) -> List[str]:
    if "input" in df.columns:
        return df["input"].fillna("").astype(str).tolist()

    if {"premise", "hypothesis"}.issubset(df.columns):
        separator = tokenizer.eos_token or "</s>"
        premise = df["premise"].fillna("").astype(str)
        hypothesis = df["hypothesis"].fillna("").astype(str)
        return (premise + separator + hypothesis).tolist()

    raise ValueError(
        "Test data must contain either an 'input' column or both 'premise' and 'hypothesis' columns."
    )


def find_label_column(df: pd.DataFrame) -> Optional[str]:
    for column in LABEL_CANDIDATES:
        if column in df.columns:
            return column
    return None


def generate_predictions(args, model, tokenizer, inputs: List[str], device: torch.device) -> List[str]:
    predictions = []
    total_batches = (len(inputs) + args.batch_size - 1) // args.batch_size

    for start in tqdm(
        range(0, len(inputs), args.batch_size),
        total=total_batches,
        desc="Generating",
    ):
        batch_texts = [args.prefix + text for text in inputs[start : start + args.batch_size]]
        batch = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_input_length,
        )
        batch = {key: value.to(device) for key, value in batch.items()}

        with torch.inference_mode():
            output_ids = model.generate(**batch, max_new_tokens=args.max_new_tokens)

        predictions.extend(tokenizer.batch_decode(output_ids, skip_special_tokens=True))

    return predictions


def main():
    args = parse_args()

    model_path = Path(args.model_path)
    test_data_path = Path(args.test_data)
    output_path = resolve_output_path(test_data_path, args.output_path)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    df = load_table(test_data_path)
    inputs = build_inputs(df, tokenizer)
    predictions = generate_predictions(args, model, tokenizer, inputs, device)

    result_df = df.copy()
    result_df["prediction"] = predictions

    label_column = find_label_column(result_df)
    if label_column is not None:
        labels = result_df[label_column].fillna("").astype(str).str.strip().str.lower()
        preds = result_df["prediction"].fillna("").astype(str).str.strip().str.lower()
        accuracy = (preds == labels).mean()
        print(f"Accuracy ({label_column}): {accuracy:.4f}")

    result_df.to_csv(output_path, index=False)
    print(f"Saved predictions to: {output_path}")


if __name__ == "__main__":
    main()
