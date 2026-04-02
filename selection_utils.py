import json
import math
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


ESNLI_RATIONALE_TYPES = (
    'condition',
    'contrastive',
    'neutral',
    'consensus',
    'causal',
    'comparative',
    'historical',
)

RATIONALE_TYPE_ALIASES = {
    'if_else': 'condition',
}

LABELS = {'entailment', 'neutral', 'contradiction'}


def normalize_whitespace(text: object) -> str:
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return ''
    text = str(text).replace('\u00a0', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def normalize_label(text: object) -> str:
    value = normalize_whitespace(text).lower()
    if value in LABELS:
        return value
    if value in {'entailed', 'entails'}:
        return 'entailment'
    if value in {'contradicted', 'contradicts', 'contradictory'}:
        return 'contradiction'
    return value


def build_pair_key(premise: object, hypothesis: object) -> str:
    premise_text = normalize_whitespace(premise).lower()
    hypothesis_text = normalize_whitespace(hypothesis).lower()
    return f'{premise_text}</s>{hypothesis_text}'


def resolve_rationale_type_name(name: str) -> str:
    return RATIONALE_TYPE_ALIASES.get(name, name)


def resolve_rationale_path(rationale_root: Path, rationale_type: str) -> Path:
    canonical_name = resolve_rationale_type_name(rationale_type)
    path = rationale_root / f'{canonical_name} - full.csv'
    if not path.exists():
        raise FileNotFoundError(f'Unable to find latest rationale file for "{rationale_type}" at {path}')
    return path


@dataclass
class CandidateScore:
    rationale_type: str
    rationale: str
    candidate_label: str
    keep: bool
    heuristic_score: float
    judge_score: float
    final_score: float
    metrics: Dict[str, object]


def _tokenize_words(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9']+", text.lower())


def _compute_length_score(token_count: int) -> float:
    if token_count <= 0:
        return 0.0
    if 16 <= token_count <= 96:
        return 1.0
    if token_count < 16:
        return max(0.0, token_count / 16.0)
    return max(0.0, 1.0 - ((token_count - 96) / 96.0))


def heuristic_evaluate_candidate(
    premise: str,
    hypothesis: str,
    gold_label: str,
    rationale: object,
    candidate_label: object,
) -> CandidateScore:
    rationale_text = normalize_whitespace(rationale)
    normalized_gold = normalize_label(gold_label)
    normalized_candidate = normalize_label(candidate_label)
    prompt_tokens = set(_tokenize_words(f'{premise} {hypothesis}'))
    rationale_tokens = _tokenize_words(rationale_text)
    overlap_count = sum(1 for token in rationale_tokens if token in prompt_tokens)
    overlap_ratio = overlap_count / max(len(rationale_tokens), 1)
    token_count = len(rationale_tokens)
    repeated_template_penalty = 1.0 if 'premise:' in rationale_text.lower() or 'hypothesis:' in rationale_text.lower() else 0.0
    has_label_mention = normalized_candidate in rationale_text.lower()
    label_match = normalized_candidate == normalized_gold and normalized_candidate in LABELS
    keep = label_match and 8 <= token_count <= 180 and repeated_template_penalty == 0.0

    length_score = _compute_length_score(token_count)
    overlap_score = min(1.0, overlap_ratio / 0.12) if rationale_tokens else 0.0
    clarity_base = (0.7 * length_score) + (0.3 * (1.0 - repeated_template_penalty))
    faithfulness_base = (0.35 * overlap_score) + (0.65 if label_match else 0.0)
    hallucination_base = max(0.0, min(1.0, 0.4 + (0.6 * overlap_score) - (0.3 * repeated_template_penalty)))
    usefulness_base = (0.5 * length_score) + (0.2 * overlap_score) + (0.3 if has_label_mention else 0.0)

    faithfulness = max(1, min(5, int(round(1 + (4 * faithfulness_base)))))
    non_hallucination = max(1, min(5, int(round(1 + (4 * hallucination_base)))))
    clarity = max(1, min(5, int(round(1 + (4 * clarity_base)))))
    student_usefulness = max(1, min(5, int(round(1 + (4 * usefulness_base)))))

    heuristic_score = (
        (0.40 * faithfulness) +
        (0.35 * student_usefulness) +
        (0.15 * non_hallucination) +
        (0.10 * clarity)
    )
    if not keep:
        heuristic_score -= 3.0

    metrics = {
        'label_match': int(label_match),
        'faithfulness': faithfulness,
        'non_hallucination': non_hallucination,
        'clarity': clarity,
        'student_usefulness': student_usefulness,
        'token_count': token_count,
        'overlap_ratio': round(overlap_ratio, 4),
        'repeated_template_penalty': repeated_template_penalty,
        'keep': keep,
    }

    return CandidateScore(
        rationale_type='',
        rationale=rationale_text,
        candidate_label=normalized_candidate,
        keep=keep,
        heuristic_score=heuristic_score,
        judge_score=heuristic_score,
        final_score=heuristic_score,
        metrics=metrics,
    )


class LocalJudge:
    def __init__(self, model_name: str, max_new_tokens: int = 256):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self._tokenizer = None
        self._model = None
        self._device = 'cpu'

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                'Local judge requires both transformers and torch to be installed.'
            ) from exc

        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self._model.to(self._device)
        self._model.eval()

    def score_candidate(
        self,
        premise: str,
        hypothesis: str,
        gold_label: str,
        rationale: str,
        candidate_label: str,
        rationale_type: str,
        heuristic_metrics: Dict[str, object],
    ) -> Dict[str, object]:
        self._ensure_loaded()

        prompt = f"""You are judging a rationale for textual entailment distillation.
Return one JSON object and nothing else.

Premise: {premise}
Hypothesis: {hypothesis}
Gold label: {gold_label}
Candidate label: {candidate_label}
Rationale type: {rationale_type}
Candidate rationale: {rationale}

Respond with JSON using:
{{
  "keep": true or false,
  "faithfulness": 1-5,
  "non_hallucination": 1-5,
  "clarity": 1-5,
  "student_usefulness": 1-5
}}
"""

        import torch

        inputs = self._tokenizer(prompt, return_tensors='pt').to(self._device)
        with torch.no_grad():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        decoded = self._tokenizer.decode(generated[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        match = re.search(r'\{.*\}', decoded, flags=re.DOTALL)
        if not match:
            raise ValueError(f'Failed to parse judge output as JSON: {decoded}')
        parsed = json.loads(match.group(0))
        for key in ('faithfulness', 'non_hallucination', 'clarity', 'student_usefulness'):
            parsed[key] = int(parsed[key])
        parsed['keep'] = bool(parsed['keep'])
        parsed['label_match'] = int(heuristic_metrics.get('label_match', 0))
        score = (
            (0.40 * parsed['faithfulness']) +
            (0.35 * parsed['student_usefulness']) +
            (0.15 * parsed['non_hallucination']) +
            (0.10 * parsed['clarity'])
        )
        parsed['score'] = score
        return parsed


def validate_latest_rationale_files(
    rationale_root: Path,
    rationale_types: Sequence[str],
) -> Tuple[Dict[str, Dict[str, int]], List[str]]:
    report: Dict[str, Dict[str, int]] = {}
    errors: List[str] = []

    for rationale_type in rationale_types:
        path = resolve_rationale_path(rationale_root, rationale_type)
        frame = pd.read_csv(path)
        frame['pair_key'] = frame.apply(lambda row: build_pair_key(row['premise'], row['hypothesis']), axis=1)
        duplicate_count = int(frame['pair_key'].duplicated().sum())
        collisions = int(
            frame.groupby('pair_key')[['premise', 'hypothesis']]
            .nunique(dropna=False)
            .max(axis=1)
            .gt(1)
            .sum()
        )
        label_na = int(frame['LLM_answer'].isna().sum()) if 'LLM_answer' in frame else 0
        report[rationale_type] = {
            'rows': int(len(frame)),
            'unique_pairs': int(frame['pair_key'].nunique()),
            'duplicate_pairs': duplicate_count,
            'collisions': collisions,
            'missing_labels': label_na,
        }
        if duplicate_count:
            errors.append(f'{rationale_type}: found {duplicate_count} duplicate pair keys')
        if collisions:
            errors.append(f'{rationale_type}: found {collisions} key collisions with inconsistent raw text')
    return report, errors


def build_canonical_esnli_dataset(
    rationale_root: Path,
    label_source_path: Path,
    rationale_types: Sequence[str] = ESNLI_RATIONALE_TYPES,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    label_source = pd.read_csv(label_source_path)
    required_columns = {'premise', 'hypothesis', 'label'}
    missing = required_columns.difference(label_source.columns)
    if missing:
        raise ValueError(f'Label source is missing required columns: {sorted(missing)}')

    canonical = label_source.copy()
    canonical['premise'] = canonical['premise'].map(normalize_whitespace)
    canonical['hypothesis'] = canonical['hypothesis'].map(normalize_whitespace)
    if 'input' not in canonical.columns:
        canonical['input'] = canonical['premise'] + '</s>' + canonical['hypothesis']
    canonical['label'] = canonical['label'].map(normalize_label)
    canonical['pair_key'] = canonical.apply(lambda row: build_pair_key(row['premise'], row['hypothesis']), axis=1)

    duplicate_label_keys = int(canonical['pair_key'].duplicated().sum())
    canonical = canonical.drop_duplicates(subset=['pair_key'], keep='first').reset_index(drop=True)

    report = {
        'label_source_rows': int(len(label_source)),
        'label_source_unique_pairs': int(canonical['pair_key'].nunique()),
        'label_source_duplicate_pairs': duplicate_label_keys,
        'matches_per_rationale': {},
    }

    for rationale_type in rationale_types:
        frame = pd.read_csv(resolve_rationale_path(rationale_root, rationale_type))
        frame = frame.copy()
        frame['premise'] = frame['premise'].map(normalize_whitespace)
        frame['hypothesis'] = frame['hypothesis'].map(normalize_whitespace)
        frame['pair_key'] = frame.apply(lambda row: build_pair_key(row['premise'], row['hypothesis']), axis=1)
        frame = frame.drop_duplicates(subset=['pair_key'], keep='last')

        merged = frame[['pair_key', 'rationale', 'LLM_answer']].rename(
            columns={
                'rationale': f'rationale_{rationale_type}',
                'LLM_answer': f'candidate_label_{rationale_type}',
            }
        )
        canonical = canonical.merge(merged, on='pair_key', how='left')
        canonical[f'rationale_type_{rationale_type}'] = rationale_type
        canonical[f'source_present_{rationale_type}'] = canonical[f'rationale_{rationale_type}'].notna()
        report['matches_per_rationale'][rationale_type] = int(canonical[f'source_present_{rationale_type}'].sum())

    available_counts = []
    for rationale_type in rationale_types:
        available_counts.append(canonical[f'source_present_{rationale_type}'].astype(int))
    canonical['available_rationale_count'] = sum(available_counts)
    canonical = canonical[canonical['available_rationale_count'] > 0].reset_index(drop=True)

    return canonical, report


def _stratified_split(labels: Sequence[str], seed: int) -> List[str]:
    rng = random.Random(seed)
    split_assignments = [None] * len(labels)
    grouped_indices: Dict[str, List[int]] = {}
    for index, label in enumerate(labels):
        grouped_indices.setdefault(label, []).append(index)

    for indices in grouped_indices.values():
        indices = list(indices)
        rng.shuffle(indices)
        count = len(indices)
        train_end = int(count * 0.8)
        valid_end = train_end + int(count * 0.1)
        for idx in indices[:train_end]:
            split_assignments[idx] = 'train'
        for idx in indices[train_end:valid_end]:
            split_assignments[idx] = 'valid'
        for idx in indices[valid_end:]:
            split_assignments[idx] = 'test'
    return split_assignments


def maybe_attach_official_esnli_splits(canonical: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
    repo_root = os.getcwd()
    original_path = list(sys.path)
    try:
        sys.path = [path for path in sys.path if path not in ('', repo_root)]
        from datasets import load_dataset  # type: ignore
        split_lookup: Dict[str, str] = {}
        label_lookup: Dict[str, str] = {}
        dataset = load_dataset('esnli', trust_remote_code=True)
        label_mapping = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}

        for split_name, split_key in [('train', 'train'), ('valid', 'validation'), ('test', 'test')]:
            frame = pd.DataFrame(dataset[split_key])[['premise', 'hypothesis', 'label']]
            frame['label'] = frame['label'].map(label_mapping)
            frame['pair_key'] = frame.apply(lambda row: build_pair_key(row['premise'], row['hypothesis']), axis=1)
            for record in frame[['pair_key', 'label']].drop_duplicates('pair_key').itertuples(index=False):
                split_lookup[record.pair_key] = split_name
                label_lookup[record.pair_key] = record.label

        canonical = canonical.copy()
        canonical['split'] = canonical['pair_key'].map(split_lookup)
        canonical['official_label'] = canonical['pair_key'].map(label_lookup)
        matched = canonical['split'].notna()
        label_mismatches = int((matched & (canonical['label'] != canonical['official_label'])).sum())
        unmatched = canonical['split'].isna()
        if unmatched.any():
            canonical.loc[unmatched, 'split'] = _stratified_split(canonical.loc[unmatched, 'label'].tolist(), seed=0)
        canonical = canonical.drop(columns=['official_label'])
        return canonical, {
            'split_source': 'official_esnli',
            'matched_pairs': int(matched.sum()),
            'unmatched_pairs': int(unmatched.sum()),
            'label_mismatches': label_mismatches,
        }
    except Exception as exc:
        canonical = canonical.copy()
        canonical['split'] = _stratified_split(canonical['label'].tolist(), seed=0)
        return canonical, {
            'split_source': 'stratified_fallback',
            'reason': str(exc),
        }
    finally:
        sys.path = original_path


def _collect_candidates(row: pd.Series, rationale_types: Sequence[str]) -> List[CandidateScore]:
    candidates: List[CandidateScore] = []
    for rationale_type in rationale_types:
        rationale = row.get(f'rationale_{rationale_type}')
        candidate_label = row.get(f'candidate_label_{rationale_type}')
        if pd.isna(rationale) or pd.isna(candidate_label):
            continue
        candidate = heuristic_evaluate_candidate(
            premise=row['premise'],
            hypothesis=row['hypothesis'],
            gold_label=row['label'],
            rationale=rationale,
            candidate_label=candidate_label,
        )
        candidate.rationale_type = rationale_type
        candidates.append(candidate)
    return candidates


def _score_with_optional_judge(
    candidate: CandidateScore,
    judge: Optional[LocalJudge],
    premise: str,
    hypothesis: str,
    gold_label: str,
) -> CandidateScore:
    if judge is None:
        candidate.judge_score = candidate.heuristic_score
        candidate.final_score = candidate.heuristic_score
        return candidate

    judged = judge.score_candidate(
        premise=premise,
        hypothesis=hypothesis,
        gold_label=gold_label,
        rationale=candidate.rationale,
        candidate_label=candidate.candidate_label,
        rationale_type=candidate.rationale_type,
        heuristic_metrics=candidate.metrics,
    )
    candidate.metrics.update(judged)
    candidate.keep = candidate.keep and bool(judged['keep'])
    candidate.judge_score = float(judged['score'])
    candidate.final_score = candidate.judge_score
    return candidate


def _choose_top_candidates(
    candidates: List[CandidateScore],
    top_k: int,
    selection_mode: str,
    rng: random.Random,
) -> List[CandidateScore]:
    valid_candidates = [candidate for candidate in candidates if candidate.keep]
    if not valid_candidates:
        return []

    if selection_mode == 'random':
        rng.shuffle(valid_candidates)
        valid_candidates = sorted(valid_candidates, key=lambda candidate: candidate.rationale_type)
        rng.shuffle(valid_candidates)
        return valid_candidates[:top_k]

    ranked = sorted(valid_candidates, key=lambda candidate: candidate.final_score, reverse=True)
    if top_k == 1:
        return ranked[:1]

    selected = [ranked[0]]
    for candidate in ranked[1:]:
        if candidate.rationale_type != selected[0].rationale_type:
            selected.append(candidate)
            break
    if len(selected) < top_k and len(ranked) > 1:
        selected.append(ranked[1])
    return selected[:top_k]


def select_rationales(
    canonical: pd.DataFrame,
    rationale_types: Sequence[str],
    selection_policy: str,
    num_selected_rationales: int,
    judge_model_name: Optional[str] = None,
    seed: int = 0,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    if num_selected_rationales not in (1, 2):
        raise ValueError('num_selected_rationales must be 1 or 2 in the current implementation')

    mode = selection_policy.lower()
    if mode not in {'heuristic', 'judge', 'random'}:
        raise ValueError('selection_policy must be one of: heuristic, judge, random')

    judge = LocalJudge(judge_model_name) if mode == 'judge' and judge_model_name else None
    rng = random.Random(seed)
    selected_rows = []
    dropped_rows = 0
    selection_counts = {rationale_type: 0 for rationale_type in rationale_types}

    for row in canonical.itertuples(index=False):
        series = pd.Series(row._asdict())
        candidates = _collect_candidates(series, rationale_types)
        if mode == 'judge':
            candidates = [
                _score_with_optional_judge(candidate, judge, series['premise'], series['hypothesis'], series['label'])
                for candidate in candidates
            ]
        chosen = _choose_top_candidates(candidates, num_selected_rationales, 'random' if mode == 'random' else 'ranked', rng)
        if not chosen:
            dropped_rows += 1
            continue

        row_dict = {
            'premise': series['premise'],
            'hypothesis': series['hypothesis'],
            'input': series['input'],
            'label': series['label'],
            'split': series.get('split', 'train'),
            'pair_key': series['pair_key'],
            'available_rationale_count': series['available_rationale_count'],
        }
        for index in range(1, num_selected_rationales + 1):
            if index <= len(chosen):
                candidate = chosen[index - 1]
                row_dict[f'rationale_{index}'] = candidate.rationale
                row_dict[f'rationale_type_{index}'] = candidate.rationale_type
                row_dict[f'rationale_score_{index}'] = round(candidate.final_score, 4)
                row_dict[f'rationale_keep_{index}'] = candidate.keep
                row_dict[f'rationale_selection_source_{index}'] = mode
                selection_counts[candidate.rationale_type] += 1
            else:
                fallback = chosen[0]
                row_dict[f'rationale_{index}'] = fallback.rationale
                row_dict[f'rationale_type_{index}'] = fallback.rationale_type
                row_dict[f'rationale_score_{index}'] = round(fallback.final_score, 4)
                row_dict[f'rationale_keep_{index}'] = fallback.keep
                row_dict[f'rationale_selection_source_{index}'] = f'{mode}_fallback_duplicate'
        selected_rows.append(row_dict)

    selected = pd.DataFrame(selected_rows)
    report = {
        'selection_policy': mode,
        'num_selected_rationales': num_selected_rationales,
        'rows_before_selection': int(len(canonical)),
        'rows_after_selection': int(len(selected)),
        'dropped_rows': int(dropped_rows),
        'selection_counts': selection_counts,
    }
    return selected, report
