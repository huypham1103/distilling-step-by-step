# Boundary-Aware Rationale Selection: A Family Pipeline for Step-by-Step Distillation

> Focused paper draft.  
> Main idea: one shared boundary-aware pipeline, three family variants.  
> Proposed final method: `judge_student_boundary_mix_balanced`.  
> Comparison scope: boundary family only, with baseline as reference.  
> Core downstream comparison: `boundary_mix` vs `boundary_bridge`; `boundary_specialist` is used as a selection-analysis variant.

---

## Abstract

Step-by-step distillation trains a student model with both task labels and natural-language rationales. Although rationales can provide useful supervision, automatically generated rationales may be noisy: they can predict the wrong label, explain the input weakly, or introduce unsupported reasoning. This paper proposes a **boundary-aware rationale selection family** that filters rationale candidates before student training.

All variants in the family share the same pipeline. First, rationale candidates from multiple sources are grouped by the original example. Second, their predicted labels or answers are combined through weighted voting. Third, the voting margin is used to estimate whether the example is reliable, near the decision boundary, or too unstable. Fourth, candidate rationales are scored using source compatibility, input overlap, explanation cues, answer explicitness, brevity, and multi-source agreement. Finally, different family variants select rationales using different final selection rules.

The main proposed variant is `judge_student_boundary_mix_balanced`, which keeps one rationale from examples in the `easy` and `boundary` regions and removes `hard` examples. A second evaluated variant, `judge_student_boundary_bridge_balanced`, adds a secondary diverse rationale when available. A third variant, `judge_student_boundary_specialist_balanced`, is used for selection analysis because complete downstream results are not available for both datasets. Experiments on ESNLI and CommonsenseQA show that `judge_student_boundary_mix_balanced` improves over the baseline on both datasets, from `83.93` to `84.16` on ESNLI and from `60.36` to `62.41` on CommonsenseQA. The family comparison shows that `boundary_bridge` achieves stronger ESNLI performance, while `boundary_mix` is more consistent in the current two-dataset comparison and gives the strongest CommonsenseQA result within the boundary family.

---

## 1. Introduction

Large language models can generate rationales that explain why an answer or label should be selected. These rationales are useful for step-by-step distillation because the student model can learn not only the final answer, but also an intermediate explanation. However, generated rationales are not always reliable. Some rationales are fluent but too generic. Some mention the correct answer without explaining it. Some are generated from sources that disagree with each other.

The central problem is therefore not simply:

```text
How do we generate rationales?
```

but:

```text
Which generated rationales should be used for training?
```

This paper focuses on one method family for this problem: the **boundary-aware rationale selection family**. Instead of presenting many unrelated rationale selection strategies, the paper studies one shared pipeline and compares variants that differ only in the final selection step.

The intuition is simple. If many rationale sources agree on the same answer, the example is likely to provide clean supervision. If the sources mostly agree but the example is close to a confusing boundary, it may still be useful because it teaches the student a subtle distinction. If the sources strongly disagree, the example may be too noisy for training.

The proposed family therefore divides examples into:

| Region | Meaning | Training role |
|---|---|---|
| `easy` | Strong agreement, high margin | Clean supervision |
| `boundary` | Moderate agreement, near decision boundary | Useful difficult supervision |
| `hard` | Weak agreement, unstable decision | Removed from main training set |
| `bridge` | Secondary diverse rationale for a non-hard example | Optional extra view |

The main proposed method is:

```text
judge_student_boundary_mix_balanced
```

This method keeps one rationale from `easy` and `boundary` examples and filters out `hard` examples. It is chosen as the main method because it improves performance on both ESNLI and CommonsenseQA.

The contributions of this paper are:

1. We define a shared boundary-aware rationale selection pipeline for step-by-step distillation.
2. We compare two evaluated variants from the same family, `boundary_mix` and `boundary_bridge`, and include `boundary_specialist` as an analysis variant.
3. We show that the family variants differ mainly in the final selection rule, making the method easy to explain and extend.
4. We show that `judge_student_boundary_mix_balanced` improves over the baseline on both ESNLI and CommonsenseQA.
5. We provide data-level analysis showing what each family variant selects.

---

## 2. Background

### 2.1. Step-by-Step Distillation

In step-by-step distillation, a student model is trained with examples that include both the target answer and a rationale. A training instance can be written as:

```text
input -> rationale -> answer
```

The rationale is intended to guide the student toward the reasoning process behind the answer. This is useful when the student model is smaller than the teacher model and needs additional supervision.

### 2.2. Why Rationale Selection Is Needed

If every generated rationale is used directly, the training set may contain noisy explanations. Noise can appear in several forms:

| Noise type | Example problem |
|---|---|
| Wrong label | The rationale supports a different label from the gold label |
| Weak grounding | The rationale does not refer clearly to the input |
| Generic explanation | The rationale says the answer is correct but does not explain why |
| Source disagreement | Different rationale sources produce different answers |

The boundary-aware family addresses this by using multiple rationale sources as evidence. The method selects rationales only when the example-level evidence is strong enough.

---

## 3. Boundary-Aware Family Overview

The family contains two fully evaluated variants and one selection-analysis variant:

| Variant | Shared pipeline | Variant-specific step | Main purpose |
|---|---|---|---|
| `judge_student_boundary_mix_balanced` | Yes | Keep `easy + boundary`, one rationale per example | Main proposed method |
| `judge_student_boundary_bridge_balanced` | Yes | Add a second diverse rationale when possible | Fully evaluated family variant |
| `judge_student_boundary_specialist_balanced` | Yes | Focus on selected specialist sources and non-easy examples | Selection-analysis variant |

The important point is that these are not three unrelated methods. They share the same core pipeline:

```text
collect candidates
-> group by example
-> weighted vote
-> compute margin
-> score candidates
-> assign band
-> apply variant selection rule
-> balance output
```

All variants share the same core voting, scoring, and banding pipeline. They differ in the final filtering rule and, for bridge-style variants, whether a second diverse rationale is allowed.

---

## 4. Shared Pipeline

### 4.1. Candidate Sources

For each example, rationale candidates are collected from seven source styles:

```text
neutral
contrastive
historical
comparative
causal
consensus
if_else
```

Each candidate contains:

| Field | Meaning |
|---|---|
| `premise` | Input sentence, question, or context |
| `hypothesis` | Hypothesis or answer choices |
| `rationale` | Generated explanation |
| `LLM_answer` | Predicted label or answer |
| `judge_source` | Source style of the rationale |

### 4.2. Group Candidates by Example

Candidates from different source files must be grouped if they describe the same original example. The method creates an example key:

```text
example_key = normalize(premise) + "</s>" + normalize(hypothesis)
```

`normalize` removes extra spaces, strips leading and trailing whitespace, and lowercases the text when constructing the key. This ensures that candidates from different source files can be aligned.

After grouping, one example can have multiple rationale candidates. The group is the unit used for voting and boundary assignment.

### 4.3. Source Prior

The family uses source priors in weighted voting. A source prior is a manually defined compatibility weight between a rationale style and a label or answer type.

The prior is:

- not learned from data
- not taken from the original step-by-step distillation paper
- not random
- a manually defined inductive bias

It should be understood as a relative compatibility matrix. For example, a contrastive rationale is naturally useful for contradiction-like reasoning because it emphasizes differences, while an if-else rationale is naturally useful for neutral-like reasoning because it often expresses uncertainty or missing conditions.

These values should not be interpreted as calibrated probabilities. They only define soft preferences used during voting and candidate scoring. A stronger future version should either tune these values on a validation set or compare them with an equal-vote baseline.

### 4.4. ESNLI Source Prior

For ESNLI, source prior depends on the candidate label.

| Label | historical | consensus | contrastive | causal | neutral | if_else | comparative |
|---|---:|---:|---:|---:|---:|---:|---:|
| entailment | 1.00 | 0.98 | 0.95 | 0.92 | 0.88 | 0.74 | 0.70 |
| neutral | 0.72 | 0.82 | 0.90 | 0.84 | 0.96 | 1.00 | 0.99 |
| contradiction | 0.74 | 0.84 | 0.99 | 0.97 | 0.89 | 0.70 | 1.00 |

### 4.5. CommonsenseQA Source Prior

For CommonsenseQA, source prior is defined per rationale source:

| Source | Prior |
|---|---:|
| causal | 1.000 |
| if_else | 0.990 |
| neutral | 0.980 |
| contrastive | 0.975 |
| historical | 0.940 |
| consensus | 0.840 |
| comparative | 0.800 |

### 4.6. Weighted Voting

Each candidate votes for its predicted label or answer. Instead of counting every source equally, the vote weight is the source prior.

For each label `y`:

```text
score(y) = sum source_prior(source_i, y)
           for candidates whose predicted label is y
```

The winning label is:

```text
voted_label = argmax_y score(y)
```

Example:

| Source | Predicted label | Weight |
|---|---|---:|
| historical | entailment | 1.00 |
| consensus | entailment | 0.98 |
| causal | entailment | 0.92 |
| if_else | neutral | 1.00 |
| comparative | contradiction | 1.00 |

The weighted scores are:

```text
score(entailment) = 1.00 + 0.98 + 0.92 = 2.90
score(neutral) = 1.00
score(contradiction) = 1.00
```

So the voted label is `entailment`.

### 4.7. Vote Margin

The vote margin measures how strongly the winning label beats the second-best label:

```text
voted_label_margin = winner_score - runner_up_score
```

This is important because a label that wins by a large margin is more reliable than a label that wins by a small margin.

| Margin | Interpretation |
|---|---|
| High | Sources strongly agree |
| Medium | Useful but near a decision boundary |
| Low | Sources disagree, example may be noisy |

### 4.8. Boundary Band Assignment

The shared pipeline assigns a band to each example.

For ESNLI:

```text
easy:
  voted_label_margin >= 2.1
  and label_counts[training_label] >= 5

boundary:
  voted_label_margin >= 1.1
  and label_counts[training_label] >= 2

hard:
  otherwise
```

For CommonsenseQA:

```text
easy:
  voted_label_margin >= 2.4
  and label_counts[gold_answer] >= 5

boundary:
  voted_label_margin >= 1.6
  and label_counts[gold_answer] >= 3

hard:
  otherwise
```

The `hard` band does not mean the rationale text is hard to read. It means the example group is unstable because source agreement is weak.

### 4.9. Candidate Scoring

After voting, the method chooses the best rationale candidate among candidates that match the selected training label or gold answer.

The score combines:

| Signal | Role |
|---|---|
| Source prior | Prefer source styles compatible with the label |
| Input overlap | Prefer rationales grounded in premise/hypothesis or question/options |
| Teaching cues | Prefer rationales with explanation words such as because, means, therefore |
| Explicit answer/label | Prefer rationales that state the target label or answer |
| Brevity | Prefer rationales with reasonable length |
| Agreement count | Prefer examples supported by multiple sources |
| Agreement ratio | Prefer higher fraction of source agreement |
| Vote margin | Prefer clearer voting outcomes |

This score is a rule-based judge score. It is not a learned neural judge.

### 4.10. Balance Output

After selection, the output is balanced by label where applicable. For ESNLI `boundary_mix`, the selected data contains:

| Label | Count |
|---|---:|
| entailment | 3018 |
| neutral | 3018 |
| contradiction | 3018 |

Balancing prevents the selected rationale dataset from being dominated by one label.

---

## 5. Family Variants

This section explains how each variant changes the final step of the shared pipeline.

### 5.1. Variant 1: `judge_student_boundary_mix_balanced`

This is the main proposed method.

Selection rule:

```text
allowed_bands = {"easy", "boundary"}
max_per_example = 1
```

Meaning:

- keep one best rationale per example
- keep `easy` examples for clean supervision
- keep `boundary` examples for useful near-boundary supervision
- remove `hard` examples
- balance selected data

Why this is the main method:

```text
It is simple, avoids noisy hard examples, does not add extra rationale views, and improves both ESNLI and CommonsenseQA.
```

### 5.2. Variant 2: `judge_student_boundary_bridge_balanced`

This variant uses the same shared pipeline, but adds a second rationale when a useful diverse partner exists.

Selection rule:

```text
allowed_bands = {"easy", "boundary", "bridge"}
max_per_example = 2
```

Extra step:

```text
choose a secondary rationale from a different source
if it is not too similar to the primary rationale
```

In implementation, similarity is checked with rationale token overlap. A secondary rationale is treated as a `bridge` view when it provides another explanation for the same selected label or answer.

Purpose:

```text
Test whether a second diverse rationale helps student training.
```

This variant improves ESNLI strongly in the current results, but it gives only a small improvement on CommonsenseQA.

### 5.3. Variant 3: `judge_student_boundary_specialist_balanced`

This variant also uses the same shared pipeline, but focuses on selected specialist sources:

```text
historical
contrastive
comparative
if_else
consensus
```

Selection rule:

```text
allowed_bands = {"boundary", "bridge"}
max_per_example = 2
```

Purpose:

```text
Analyze whether specialist rationale styles are useful for difficult or secondary-view examples.
```

In the current project outputs, this variant is available for ESNLI selection analysis. It is not used as the main proposed method because the thesis target is a method that works clearly on both ESNLI and CommonsenseQA.

### 5.4. Variant Comparison Summary

| Variant | Keeps easy? | Keeps boundary? | Keeps bridge? | Keeps hard? | Rationale count |
|---|---|---|---|---|---:|
| `boundary_mix` | Yes | Yes | No | No | 1 |
| `boundary_bridge` | Yes | Yes | Yes | No | Up to 2 |
| `boundary_specialist` | No | Yes | Yes | No | Up to 2 |

This table is the core family comparison. The variants are different final choices over the same shared pipeline.

---

## 6. Experimental Setup

### 6.1. Datasets

The experiments use two datasets:

| Dataset | Task | Output |
|---|---|---|
| ESNLI | Natural language inference | entailment, neutral, contradiction |
| CommonsenseQA | Commonsense question answering | multiple-choice answer |

These two datasets test different reasoning abilities. ESNLI focuses on sentence-pair logical relations. CommonsenseQA focuses on commonsense answer selection.

### 6.2. Baseline

The baseline is the original student training setup used in the project.

| Dataset | Baseline accuracy |
|---|---:|
| ESNLI | 83.93 |
| CommonsenseQA | 60.36 |

TODO/VERIFY before final submission:

- student model backbone
- training hyperparameters
- random seed
- number of training runs
- exact train/validation/test split

### 6.3. Evaluation Metric

The evaluation metric is accuracy.

This is suitable because:

- ESNLI is a classification task
- CommonsenseQA is a multiple-choice classification task

### 6.4. What We Compare

The paper discusses these boundary-family variants:

```text
baseline
boundary_mix
boundary_bridge
boundary_specialist
```

However, `boundary_specialist` is reported only for selection statistics in the current project output because its downstream performance table is not available for both datasets.

The downstream performance comparison focuses on:

```text
baseline vs boundary_mix vs boundary_bridge
```

This paper does not claim that `boundary_specialist` is a complete downstream competitor. It is included to show how the same pipeline can be restricted to specialist rationale sources, but the main empirical claim is based on `boundary_mix` and `boundary_bridge`.

---

## 7. Selection Statistics

Selection statistics are important because this paper proposes a data/rationale selection method. We need to show not only final accuracy, but also what the method selected.

### 7.1. Boundary Mix Selection

| Dataset | Selected rows | Band distribution | Average agreement | Average margin |
|---|---:|---|---:|---:|
| ESNLI | 9054 | easy: 9032, boundary: 22 | 6.85 | 5.95 |
| CommonsenseQA | 7843 | easy: 7787, boundary: 56 | 6.67 | 5.98 |

Interpretation:

`boundary_mix` mostly selects high-agreement easy examples, but it also keeps a small number of boundary examples. This makes the selected set clean while still preserving some near-boundary supervision.

Although the final selected data is dominated by `easy` examples, the method is still boundary-aware because the selection decision is made after estimating the example's region. In other words, the method does not assume all examples are equally useful. It explicitly identifies `easy`, `boundary`, and `hard` regions, keeps reliable and near-boundary examples, and removes unstable examples. The small number of selected `boundary` rows indicates that most retained examples have strong multi-source agreement, while the boundary mechanism still protects the training set from low-margin noisy cases.

### 7.2. Boundary Bridge Selection

| Dataset | Selected rows | Band distribution | Extra bridge views | Average agreement | Average margin |
|---|---:|---|---:|---:|---:|
| ESNLI | 12600 | easy: 7747, bridge: 4853 | 4853 | 6.94 | 6.08 |
| CommonsenseQA | 10500 | bridge: 6037, easy: 4462, boundary: 1 | 6037 | 6.91 | 6.37 |

Interpretation:

`boundary_bridge` selects more data because it can include a second rationale view. This increases training coverage, especially through bridge rationales.

### 7.3. Boundary Specialist Selection

| Dataset | Selected rows | Band distribution | Extra bridge views | Average agreement | Average margin |
|---|---:|---|---:|---:|---:|
| ESNLI | 3921 | bridge: 3912, boundary: 9 | 3912 | 6.84 | 5.93 |

Interpretation:

`boundary_specialist` is much smaller and focuses on specialist sources. It is useful for analysis, but less suitable as the main cross-dataset method because current outputs do not provide the same complete evaluation on both ESNLI and CommonsenseQA.

### 7.4. Source Distribution in Boundary Mix

For ESNLI `boundary_mix`, selected rationales come mainly from:

| Source | Count |
|---|---:|
| causal | 3525 |
| neutral | 2558 |
| if_else | 1648 |
| contrastive | 478 |
| comparative | 371 |
| historical | 243 |
| consensus | 231 |

For CommonsenseQA `boundary_mix`, selected rationales come mainly from:

| Source | Count |
|---|---:|
| if_else | 6112 |
| causal | 948 |
| neutral | 679 |
| contrastive | 102 |
| historical | 1 |
| consensus | 1 |

This shows that the selected source distribution is dataset-dependent. ESNLI uses a more mixed source distribution, while CommonsenseQA is dominated by `if_else`, with support from `causal` and `neutral`.

---

## 8. Performance Results

### 8.1. Boundary Family Performance

| Method | ESNLI | CommonsenseQA |
|---|---:|---:|
| baseline | 83.93 | 60.36 |
| `judge_student_boundary_mix_balanced` | 84.16 | 62.41 |
| `judge_student_boundary_bridge_balanced` | 85.06 | 60.52 |

### 8.2. Improvement over Baseline

| Dataset | Baseline | Boundary Mix | Improvement |
|---|---:|---:|---:|
| ESNLI | 83.93 | 84.16 | +0.23 |
| CommonsenseQA | 60.36 | 62.41 | +2.05 |

### 8.3. Main Interpretation

`boundary_bridge` gives the best ESNLI result:

```text
85.06
```

However, its CommonsenseQA score is:

```text
60.52
```

which is only slightly above the baseline.

`boundary_mix` gives:

```text
ESNLI: 84.16
CommonsenseQA: 62.41
```

This is why `boundary_mix` is selected as the main proposed method. It is not the highest on every single dataset, but it improves both datasets and performs especially well on CommonsenseQA.

The key conclusion from the current two-dataset comparison is:

```text
boundary_mix is the most consistent family variant across the two datasets.
```

---

## 9. Step-by-Step Family Comparison

This section explains the family comparison as a pipeline difference.

### 9.1. Shared Steps

All family variants use these steps:

| Step | Used by mix | Used by bridge | Used by specialist |
|---|---|---|---|
| Collect candidates from sources | Yes | Yes | Yes |
| Group by example | Yes | Yes | Yes |
| Weighted vote | Yes | Yes | Yes |
| Compute vote margin | Yes | Yes | Yes |
| Score candidate rationales | Yes | Yes | Yes |
| Assign `easy/boundary/hard` band | Yes | Yes | Yes |
| Remove `hard` examples | Yes | Yes | Yes |

### 9.2. Different Final Steps

| Final step | `boundary_mix` | `boundary_bridge` | `boundary_specialist` |
|---|---|---|---|
| Keep `easy` | Yes | Yes | No |
| Keep `boundary` | Yes | Yes | Yes |
| Add `bridge` rationale | No | Yes | Yes |
| Filter specialist sources | No | No | Yes |
| Max rationale per example | 1 | 2 | 2 |

This table is useful for the paper because it shows that the variants are controlled changes over the same pipeline.

### 9.3. Why Boundary Mix Is Preferred

`boundary_mix` is preferred as the main thesis method because:

1. It uses the simplest final selection rule.
2. It avoids unstable `hard` examples.
3. It avoids adding a second rationale that may introduce extra noise.
4. It improves over baseline on both datasets.
5. It achieves the strongest CommonsenseQA performance among the evaluated boundary variants.

### 9.4. Why Boundary Bridge May Help ESNLI

ESNLI is a sentence-pair reasoning task. Multiple rationales can explain the same logical relation from different angles. For example, one rationale may focus on lexical entailment, while another may focus on semantic generalization. This may explain why bridge rationales improve ESNLI in the current results.

However, the same extra-view strategy is less effective on CommonsenseQA in the current table. CommonsenseQA answer choices can be close in meaning, and extra rationales may introduce associations that are not consistently useful. This interpretation should be treated as a hypothesis until additional ablations are run.

### 9.5. Why Boundary Specialist Is Not the Main Method

`boundary_specialist` is useful for studying specialist source behavior, but it is not the best main method because:

- it is smaller
- it excludes easy examples
- current outputs are ESNLI-focused
- it is less directly comparable across both datasets

Therefore, it is better treated as a family analysis variant rather than the main thesis method.

---

## 10. Qualitative Examples

### 10.1. Easy Example

Example from ESNLI `boundary_mix`:

| Field | Value |
|---|---|
| Premise | A young man wearing a backpack and dark glasses approaches the brick building where there is a bit of graffiti on the wall. |
| Hypothesis | The young man has glasses on his face. |
| Selected label | entailment |
| Source | neutral |
| Agreement count | 7 |
| Vote margin | 6.17 |
| Band | easy |

Rationale:

```text
The correct answer is entailment. This means that the hypothesis is supported by the premise. The premise and the hypothesis are both about a young man wearing glasses.
```

This is an easy example because all seven sources agree and the vote margin is high.

### 10.2. Boundary Example

Example from ESNLI before final balancing:

| Field | Value |
|---|---|
| Premise | Two men, one standing and one seated on the ground are attempting to wrangle a bull as dust from the action is being kicked up. |
| Hypothesis | A rodeo clown tries to save a rider from a bull. |
| Selected label | neutral |
| Source | consensus |
| Agreement count | 4 |
| Vote margin | 1.11 |
| Band | boundary |

Rationale:

```text
The commonly agreed-upon answer is neutral. The hypothesis is neither entailed nor contradicted by the premise. It is possible that the two men are rodeo clowns trying to save a rider from a bull, but it is not necessarily the case.
```

This example is useful because the hypothesis is plausible but not guaranteed. That is exactly the kind of distinction a boundary example should teach.

### 10.3. Hard Example

Example from ESNLI before filtering:

| Field | Value |
|---|---|
| Premise | A group of construction workers are working. |
| Hypothesis | A group of guys outside. |
| Selected label | neutral |
| Source | contrastive |
| Agreement count | 3 |
| Vote margin | 0.01 |
| Band | hard |

Rationale:

```text
The most likely answer is neutral. This is because the hypothesis is not necessarily true or false based on the premise. A group of guys outside could be construction workers, but they could also be doing something else.
```

The rationale is not obviously bad, but the example has an extremely low vote margin. This means the sources are unstable. The boundary-aware family removes it from `boundary_mix`.

### 10.4. CommonsenseQA Boundary Example

| Field | Value |
|---|---|
| Question | Where did his wife ask him to look first when he complained of missing deodorant? |
| Choices | grocery store, nightstand, armpits, own bathroom, medicine chest |
| Selected answer | medicine chest |
| Source | causal |
| Agreement count | 4 |
| Vote margin | 1.61 |
| Band | boundary |

Rationale:

```text
The correct answer is medicine chest. His wife asked him to look first in the medicine chest because that is where he usually keeps his deodorant.
```

This example is near the boundary because several choices are plausible household-related options, but the rationale gives a clear reason for the selected answer.

---

## 11. Discussion

### 11.1. Main Finding

The main finding is that a single shared pipeline can produce several useful rationale-selection variants. This makes the method easier to understand than a collection of unrelated selection methods.

The family pipeline provides:

- example grouping
- multi-source agreement
- vote margin
- boundary bands
- rationale scoring
- controlled final selection variants

### 11.2. Why the Family View Is Better

The family view is useful for a thesis because it gives a clean story:

```text
We propose one pipeline.
Then we compare controlled variants of that pipeline.
```

This is stronger than comparing many unrelated method names because each result can be explained as the effect of a specific final selection rule.

### 11.3. Dataset Difference

The results suggest that ESNLI and CommonsenseQA may benefit from different variants.

ESNLI benefits from `boundary_bridge` in the current result, possibly because sentence-pair inference can use multiple logical explanations. CommonsenseQA benefits more from `boundary_mix` in the current result, possibly because extra rationales may add noisy commonsense associations.

This supports the decision to choose `boundary_mix` as the main method for cross-dataset consistency in this experiment.

---

## 12. Limitations

The method has several limitations.

First, source priors are manually defined. They encode a reasonable inductive bias, but they are not learned from data.

Second, band thresholds are rule-based. The thresholds may need tuning for other datasets.

Third, the judge score is also rule-based. It is interpretable, but it may not capture all aspects of rationale quality.

Fourth, the final `boundary_mix` selection is dominated by `easy` examples. This does not invalidate the boundary-aware pipeline, because the method still estimates and filters by boundary regions, but it means that the main performance improvement may come more from removing unstable examples than from adding many boundary examples.

Fifth, `boundary_specialist` does not currently have the same complete downstream evaluation on both ESNLI and CommonsenseQA.

Sixth, the current results do not report repeated runs or standard deviation. Therefore, claims about cross-dataset consistency should be interpreted as observations from the current comparison, not as statistically verified stability.

Seventh, the current draft does not include ablation experiments. Ablation is important to test whether weighted voting, source prior, boundary filtering, and bridge rationale selection each contribute to performance.

---

## 13. Future Work

Future work should evaluate:

| Future experiment | Purpose |
|---|---|
| Equal vote vs weighted vote | Test whether source priors help |
| Easy only vs easy + boundary | Test whether boundary examples help |
| With hard examples vs without hard examples | Test whether hard filtering matters |
| One rationale vs bridge rationale | Test whether the second rationale helps by dataset |
| Learned source prior | Replace manual prior with validation-learned weights |
| Threshold tuning | Learn boundary thresholds from validation data |
| Repeated runs with standard deviation | Verify whether improvements are robust |
| Validation-set threshold search | Reduce dependence on manually selected thresholds |

The most important ablation is:

```text
equal vote vs weighted vote
```

This directly tests the value of the prior matrix.

---

## 14. Conclusion

This paper presents a boundary-aware rationale selection family for step-by-step distillation. Instead of comparing many unrelated rationale selection methods, the paper focuses on one shared pipeline and three controlled variants. The shared pipeline groups rationale candidates by example, performs weighted voting, computes vote margin, assigns boundary bands, scores candidate rationales, and then applies a variant-specific selection rule.

The main proposed method, `judge_student_boundary_mix_balanced`, keeps one rationale from `easy` and `boundary` examples while removing `hard` examples. It improves over the baseline on both ESNLI and CommonsenseQA, from `83.93` to `84.16` on ESNLI and from `60.36` to `62.41` on CommonsenseQA.

The family comparison shows that `boundary_bridge` is stronger on ESNLI, but `boundary_mix` is more consistent across the two evaluated datasets and performs best on CommonsenseQA. Therefore, `boundary_mix` is selected as the main thesis method, while `boundary_bridge` and `boundary_specialist` are used to analyze how different final selection rules affect the shared boundary pipeline.

---

## References

TODO: add final citations.

Recommended citation groups:

1. Knowledge distillation.
2. Step-by-step distillation.
3. Chain-of-thought prompting.
4. ESNLI dataset.
5. CommonsenseQA dataset.
6. Rationale evaluation and faithfulness.
7. Data filtering and curriculum learning.

---

## Appendix A. Paper Story in Vietnamese

Luận điểm chính:

```text
Chúng ta không trình bày nhiều method rời rạc.
Chúng ta trình bày một family pipeline chung.
Các type trong family chỉ khác nhau ở bước chọn cuối cùng.
```

Pipeline chung:

```text
gom rationale theo example
-> weighted vote
-> tính margin
-> gán easy/boundary/hard
-> chấm điểm rationale
-> chọn theo rule của từng variant
```

Ba variant:

| Variant | Giải thích ngắn |
|---|---|
| `boundary_mix` | giữ easy + boundary, mỗi example 1 rationale |
| `boundary_bridge` | thêm rationale thứ hai khác source, ít trùng nội dung |
| `boundary_specialist` | tập trung vào source chuyên biệt và bridge/boundary |

Vì sao chọn `boundary_mix`:

```text
Nó cải thiện cả ESNLI và CQA.
Nó đơn giản hơn bridge.
Nó không thêm rationale phụ có thể gây nhiễu.
Nó đạt CQA cao nhất trong boundary family.
```
