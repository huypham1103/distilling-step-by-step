import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd


def build_tiny_selected_dataset(csv_path: Path) -> None:
    rows = [
        {
            "premise": "A person on a horse jumps over a broken down airplane.",
            "hypothesis": "A person is outdoors, on a horse.",
            "input": "A person on a horse jumps over a broken down airplane.</s>A person is outdoors, on a horse.",
            "label": "entailment",
            "rationale_1": "The correct answer is entailment because being on a horse jumping over an airplane means the person is outdoors and on a horse.",
            "rationale_type_1": "neutral",
        },
        {
            "premise": "A person on a horse jumps over a broken down airplane.",
            "hypothesis": "A person is at a diner, ordering an omelette.",
            "input": "A person on a horse jumps over a broken down airplane.</s>A person is at a diner, ordering an omelette.",
            "label": "contradiction",
            "rationale_1": "The correct answer is contradiction because the person cannot be on a horse outdoors and at a diner ordering food at the same time.",
            "rationale_type_1": "neutral",
        },
        {
            "premise": "A person on a horse jumps over a broken down airplane.",
            "hypothesis": "A person is training his horse for a competition.",
            "input": "A person on a horse jumps over a broken down airplane.</s>A person is training his horse for a competition.",
            "label": "neutral",
            "rationale_1": "The correct answer is neutral because the premise does not prove the person is training for a competition.",
            "rationale_type_1": "neutral",
        },
    ]

    full_rows = []
    for split in ["train", "valid", "test"]:
        for row in rows:
            copied = row.copy()
            copied["split"] = split
            full_rows.append(copied)

    pd.DataFrame(full_rows).to_csv(csv_path, index=False)


def run_command(command: list[str], cwd: Path) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--eval_batch_size", type=int, default=3)
    parser.add_argument("--max_input_length", type=int, default=128)
    parser.add_argument("--gen_max_len", type=int, default=8)
    parser.add_argument("--from_pretrained", type=str, default="t5-small")
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--selection_policy", type=str, default="debug")
    parser.add_argument("--num_selected_rationales", type=int, default=1)
    parser.add_argument("--min_accuracy", type=float, default=2 / 3)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix="selected-pipeline-") as tmp_dir:
        tmp_root = Path(tmp_dir)
        dataset_path = tmp_root / "tiny_selected_overfit.csv"
        eval_dir = tmp_root / "eval"
        build_tiny_selected_dataset(dataset_path)

        model_dir = (
            repo_root
            / ".."
            / "model_path"
            / f"selected_{args.selection_policy}_{args.num_selected_rationales}"
        ).resolve()
        if model_dir.exists():
            shutil.rmtree(model_dir)

        train_command = [
            sys.executable,
            "run.py",
            "--dataset",
            "esnli",
            "--selected_rationale_path",
            str(dataset_path),
            "--selection_policy",
            args.selection_policy,
            "--num_selected_rationales",
            str(args.num_selected_rationales),
            "--model_type",
            "task_prefix",
            "--label_type",
            "gt",
            "--batch_size",
            str(args.batch_size),
            "--eval_batch_size",
            str(args.eval_batch_size),
            "--grad_steps",
            "1",
            "--max_input_length",
            str(args.max_input_length),
            "--gen_max_len",
            str(args.gen_max_len),
            "--max_steps",
            str(args.max_steps),
            "--eval_steps",
            str(args.eval_steps or args.max_steps),
            "--from_pretrained",
            args.from_pretrained,
            "--alpha",
            str(args.alpha),
            "--no_log",
        ]
        run_command(train_command, repo_root)

        eval_command = [
            sys.executable,
            "evaluate_test.py",
            "--model_path",
            str(model_dir),
            "--test_data_path",
            str(dataset_path),
            "--output_dir",
            str(eval_dir),
            "--model_type",
            "task_prefix",
            "--batch_size",
            str(args.eval_batch_size),
            "--max_input_length",
            str(args.max_input_length),
            "--gen_max_len",
            str(args.gen_max_len),
        ]
        run_command(eval_command, repo_root)

        metrics_path = eval_dir / f"selected_{args.selection_policy}_{args.num_selected_rationales}_test_metrics.json"
        predictions_path = eval_dir / f"selected_{args.selection_policy}_{args.num_selected_rationales}_test_predictions.csv"
        metrics = json.loads(metrics_path.read_text())
        predictions = pd.read_csv(predictions_path)

        print(json.dumps(metrics, indent=2))
        print(predictions[["input", "label", "prediction", "prediction_is_empty"]].to_string())

        if args.strict:
            assert metrics["test_accuracy_exact"] == 1.0, metrics
        else:
            assert metrics["test_accuracy_exact"] >= args.min_accuracy, metrics
        assert metrics["prediction_empty_ratio"] == 0.0, metrics
        assert not predictions["prediction_is_empty"].any(), predictions

        print("Selected-rationale regression test passed.")


if __name__ == "__main__":
    main()
