# Paper Outline: Boundary-Aware Rationale Selection

Tài liệu này là outline để bắt đầu viết thesis paper/report dựa trên method chính:

```text
judge_student_boundary_mix_balanced
```

Ý tưởng trung tâm của paper:

```text
Không phải rationale nào sinh ra từ LLM cũng nên dùng để train student model.
Ta cần chọn rationale vừa đúng, vừa bám vào input, vừa có mức độ đồng thuận hợp lý giữa nhiều source.
```

Method đề xuất:

```text
Boundary-Aware Rationale Selection
```

Implementation chính:

```text
judge_student_boundary_mix_balanced
```

---

## 1. Working Title

Gợi ý title:

```text
Boundary-Aware Rationale Selection for Step-by-Step Knowledge Distillation
```

Title tiếng Việt để giải thích với giáo viên:

```text
Lựa chọn rationale theo vùng biên cho distillation từng bước
```

Ý nghĩa:

- `Rationale Selection`: chọn lời giải thích nào sẽ dùng để train student model.
- `Boundary-Aware`: không chỉ chọn rationale tốt, mà còn xét example đó thuộc vùng dễ, vùng biên, hay vùng quá nhiễu.
- `Step-by-Step Knowledge Distillation`: teacher/LLM tạo rationale, student học từ label + rationale.

---

## 2. Abstract

Abstract nên viết sau cùng, nhưng outline nội dung sẽ gồm:

1. Bài toán:
   - Step-by-step distillation dùng rationale từ LLM để train student model.
   - Tuy nhiên rationale sinh tự động có thể nhiễu, sai label, hoặc không bám vào input.

2. Hạn chế của cách chọn đơn giản:
   - Dùng tất cả rationale có thể đưa nhiễu vào training.
   - Chọn top-1 rationale theo score riêng lẻ có thể bỏ qua agreement giữa nhiều source.
   - Không phân biệt example dễ, example biên, và example quá khó.

3. Method đề xuất:
   - Gom nhiều rationale candidates theo cùng một example.
   - Dùng weighted voting để đo agreement giữa các source.
   - Tính margin để phân loại example thành `easy`, `boundary`, hoặc `hard`.
   - Chỉ giữ `easy + boundary`, bỏ `hard`.

4. Kết quả chính:
   - Method `judge_student_boundary_mix_balanced` cải thiện cả ESNLI và CQA so với baseline.
   - ESNLI: `83.93 -> 84.16`
   - CQA: `60.36 -> 62.41`
   - Method đạt performance ổn định nhất trên cả hai dataset, đặc biệt cao nhất trên CQA trong bảng so sánh.

5. Kết luận:
   - Lọc rationale theo độ tin cậy và vùng biên giúp student học từ các example có ích hơn.

---

## 3. Introduction

### 3.1. Motivation

Cần mở bài bằng vấn đề lớn:

```text
Large language models can produce explanations, but not all explanations are equally useful for training smaller models.
```

Diễn giải bằng tiếng Việt:

LLM có thể tạo nhiều rationale khác nhau cho cùng một example. Nhưng trong training, nếu đưa toàn bộ rationale vào student model thì student có thể học cả tín hiệu tốt lẫn tín hiệu nhiễu.

Ví dụ:

```text
Premise: A man is playing guitar on stage.
Hypothesis: A person is performing music.
Gold label: entailment
```

Một rationale tốt:

```text
Because playing guitar on stage is a form of performing music.
```

Một rationale yếu:

```text
Because the two sentences are about a man.
```

Rationale thứ hai không sai hoàn toàn, nhưng quá chung và không chỉ rõ quan hệ logic giữa premise và hypothesis.

### 3.2. Problem Statement

Paper nên định nghĩa vấn đề như sau:

```text
Given multiple rationale candidates generated for the same training example, how can we select a reliable subset that improves student model performance?
```

Trong bài này, mỗi example có:

- input: premise/hypothesis hoặc question/answer choices
- gold label hoặc gold answer
- nhiều rationale candidates từ các source khác nhau
- predicted label/answer từ từng rationale source

Mục tiêu:

```text
Chọn rationale dùng để train student model sao cho performance tốt hơn baseline.
```

### 3.3. Research Gap

Các cách chọn rationale thông thường có thể thiếu 3 yếu tố:

| Thiếu yếu tố | Vấn đề |
|---|---|
| Multi-source agreement | Không biết nhiều source có đồng ý không |
| Example difficulty | Không biết example quá dễ, vùng biên, hay quá nhiễu |
| Rationale-input grounding | Không biết rationale có bám vào premise/hypothesis không |

Paper của mình tập trung vào 3 điểm này.

### 3.4. Contribution

Nên viết contributions rõ ràng:

1. Đề xuất một pipeline chọn rationale dựa trên boundary-aware filtering.
2. Dùng weighted voting giữa nhiều rationale sources để đo độ đồng thuận label/answer.
3. Dùng `voted_label_margin` để phân loại example thành `easy`, `boundary`, và `hard`.
4. Chọn `easy + boundary` examples để train student, giúp tránh rationale quá nhiễu.
5. Thực nghiệm trên ESNLI và CQA cho thấy method cải thiện so với baseline trên cả hai dataset.

---

## 4. Related Work

Phần này nên viết ngắn, tập trung vào các nhóm công trình.

### 4.1. Knowledge Distillation

Nội dung cần viết:

- Knowledge distillation train một student model nhỏ hơn từ teacher model.
- Student học từ label, logits, hoặc intermediate supervision.
- Trong setting của mình, student học từ label + rationale.

Citation placeholder:

```text
[Hinton et al., 2015]
```

### 4.2. Step-by-Step Distillation

Nội dung cần viết:

- Step-by-step distillation dùng rationales hoặc chain-of-thought explanations làm supervision.
- Rationale giúp student học không chỉ đáp án cuối, mà cả quá trình giải thích.
- Nhưng chất lượng rationale rất quan trọng.

Citation placeholder:

```text
[Distilling Step-by-Step paper]
```

### 4.3. Rationale Quality and Selection

Nội dung cần viết:

- Rationale có thể faithful hoặc không faithful.
- Rationale có thể đúng label nhưng giải thích yếu.
- Vì vậy cần scoring/filtering rationale.

Liên hệ với method của mình:

```text
Khác với việc chỉ chọn rationale đơn lẻ, method của mình xét cả agreement giữa nhiều source và độ khó của example.
```

### 4.4. Data Filtering and Curriculum Learning

Nội dung cần viết:

- Training data không đồng đều về chất lượng.
- Một số example dễ, một số example khó, một số example nhiễu.
- Curriculum learning và data filtering đều cho rằng chọn hoặc sắp xếp data có thể ảnh hưởng performance.

Liên hệ với method:

```text
Boundary-aware selection giống một dạng data filtering: giữ example có tín hiệu học tốt, bỏ example quá nhiễu.
```

---

## 5. Method

Đây là phần quan trọng nhất của paper.

Tên method đề xuất:

```text
Boundary-Aware Rationale Selection
```

Implementation chính:

```text
judge_student_boundary_mix_balanced
```

### 5.1. Overview

Mô tả pipeline bằng lời:

1. Thu thập rationale candidates từ nhiều source.
2. Group candidates theo cùng một example.
3. Normalize label/answer.
4. Dùng weighted vote để tính label/answer được nhiều source ủng hộ nhất.
5. Tính agreement và margin.
6. Chấm điểm từng rationale candidate.
7. Gán boundary band cho example.
8. Chỉ giữ candidates thuộc `easy + boundary`.
9. Balance dataset theo label.

Figure cần dùng:

```text
assets/boundary_family_pipeline.svg
```

### 5.2. Candidate Generation

Mỗi example có nhiều rationale candidates từ 7 source:

```text
neutral
contrastive
historical
comparative
causal
consensus
if_else
```

Mỗi candidate gồm:

| Field | Ý nghĩa |
|---|---|
| `premise` | input thứ nhất |
| `hypothesis` | input thứ hai |
| `rationale` | lời giải thích |
| `LLM_answer` | label/answer do source dự đoán |
| `source` | loại rationale source |

### 5.3. Grouping by Example

Một example key được tạo từ input đã normalize:

```text
example_key = normalize(premise) + "</s>" + normalize(hypothesis)
```

Với CQA, có thể hiểu tương tự:

```text
example_key = normalize(question/context) + "</s>" + normalize(answer/options)
```

Mục tiêu:

```text
Tất cả rationale candidates nói về cùng một example phải nằm trong cùng một group.
```

### 5.4. Source Prior

Source prior là trọng số thủ công mô tả mức độ phù hợp tương đối giữa:

```text
rationale source/style
```

và:

```text
label/answer type
```

Quan trọng:

```text
Các prior này không lấy từ original paper.
Các prior này không được học tự động từ training.
Các prior này không phải random.
Chúng là heuristic thủ công, được dùng như một relative compatibility matrix.
```

Ví dụ trực giác cho ESNLI:

- `historical` phù hợp hơn với `entailment`, vì nó thường kể lại/diễn giải sự kiện theo hướng hỗ trợ.
- `if_else` phù hợp hơn với `neutral`, vì nó hay nói điều kiện chưa đủ chắc.
- `comparative` và `contrastive` phù hợp hơn với `contradiction`, vì chúng nhấn mạnh khác biệt/xung đột.

#### ESNLI Prior Values

| Label | historical | consensus | contrastive | causal | neutral | if_else | comparative |
|---|---:|---:|---:|---:|---:|---:|---:|
| entailment | 1.00 | 0.98 | 0.95 | 0.92 | 0.88 | 0.74 | 0.70 |
| neutral | 0.72 | 0.82 | 0.90 | 0.84 | 0.96 | 1.00 | 0.99 |
| contradiction | 0.74 | 0.84 | 0.99 | 0.97 | 0.89 | 0.70 | 1.00 |

#### CQA Prior Values

| Source | Prior |
|---|---:|
| causal | 1.000 |
| if_else | 0.990 |
| neutral | 0.980 |
| contrastive | 0.975 |
| historical | 0.940 |
| consensus | 0.840 |
| comparative | 0.800 |

Cách viết trong paper:

```text
We define source priors as manually specified compatibility weights between rationale styles and predicted labels. These priors are not learned parameters; instead, they encode an inductive bias about which rationale styles are more suitable for each label type. The final effectiveness of this heuristic is evaluated empirically through downstream student performance.
```

### 5.5. Weighted Voting

Mỗi source đưa ra một label/answer.

Thay vì đếm phiếu như nhau:

```text
1 source = 1 vote
```

ta dùng:

```text
vote weight = source prior
```

Ví dụ ESNLI:

| Source | Predicted label | Weight for that label |
|---|---|---:|
| historical | entailment | 1.00 |
| consensus | entailment | 0.98 |
| causal | entailment | 0.92 |
| if_else | neutral | 1.00 |
| comparative | contradiction | 1.00 |

Tổng điểm:

```text
score(entailment) = 1.00 + 0.98 + 0.92 = 2.90
score(neutral) = 1.00
score(contradiction) = 1.00
```

Voted label:

```text
entailment
```

### 5.6. Voted Label Margin

Margin đo khoảng cách giữa label thắng và label đứng thứ hai:

```text
voted_label_margin = winner_score - runner_up_score
```

Ví dụ:

```text
winner_score = 2.90
runner_up_score = 1.00
margin = 1.90
```

Ý nghĩa:

| Margin | Diễn giải |
|---|---|
| Cao | Nhiều source đồng ý, example đáng tin hơn |
| Trung bình | Có tín hiệu đúng nhưng vẫn gần vùng nhầm |
| Thấp | Source chia phiếu mạnh, example dễ nhiễu |

Lý do dùng công thức này:

```text
Không chỉ cần biết label nào thắng, mà còn cần biết nó thắng rõ hay thắng sát nút.
```

### 5.7. Boundary Band Assignment

Sau khi có margin và label counts, gán example vào band.

Với ESNLI:

```text
easy:
  voted_label_margin >= 2.1
  and label_counts[gold_label] >= 5

boundary:
  voted_label_margin >= 1.1
  and label_counts[gold_label] >= 2

hard:
  otherwise
```

Với CQA:

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

Ý nghĩa:

| Band | Có giữ không trong boundary_mix? | Lý do |
|---|---|---|
| `easy` | Có | Tín hiệu rõ, ít nhiễu |
| `boundary` | Có | Có ích để học ranh giới quyết định |
| `hard` | Không | Quá nhiễu hoặc source không đồng thuận |

### 5.8. Candidate Quality Score

Mỗi rationale candidate được chấm điểm bằng nhiều tín hiệu:

| Thành phần | Ý nghĩa |
|---|---|
| `source_prior` | Source/style này có phù hợp label không |
| `overlap_score` | Rationale có dùng từ/cụm từ liên quan tới input không |
| `teaching_hits` | Rationale có cue giải thích tốt không |
| `explicit_label` | Rationale có nói rõ label/answer không |
| `format_bonus` | Rationale có format giải thích rõ không |
| `brevity` | Rationale có độ dài vừa phải không |
| `agreement_count` | Có bao nhiêu source cùng label/answer |
| `agreement_ratio` | Tỉ lệ source đồng ý |
| `vote_margin` | Label thắng có thắng rõ không |

Điểm này dùng để chọn candidate tốt nhất trong group.

### 5.9. Final Selection

Với `judge_student_boundary_mix_balanced`:

```text
allowed_bands = {"easy", "boundary"}
max_per_example = 1
```

Tức là:

- chỉ giữ example thuộc `easy` hoặc `boundary`
- bỏ `hard`
- mỗi example chỉ chọn 1 rationale tốt nhất
- balance số lượng theo label

---

## 6. Boundary Family Variants

Paper nên giới thiệu family để chứng minh method chính không đứng một mình.

| Method | Bands | Max rationale/example | Mục tiêu |
|---|---|---:|---|
| `judge_student_boundary_mix_balanced` | easy + boundary | 1 | Method chính, ổn định nhất trên hai dataset |
| `judge_student_boundary_bridge_balanced` | easy + boundary + bridge | 2 | Thêm rationale phụ để tăng coverage |
| `judge_student_boundary_specialist_balanced` | boundary + bridge | 2 | Tập trung vào rationale source chuyên biệt, ESNLI only |

Điểm cần nhấn mạnh:

```text
boundary_mix là lựa chọn chính vì nó đơn giản hơn bridge, ít đưa thêm rationale phụ, và cho kết quả tốt nhất trên CQA đồng thời vẫn cải thiện ESNLI.
```

---

## 7. Experimental Setup

### 7.1. Datasets

Sử dụng 2 dataset:

| Dataset | Task | Output |
|---|---|---|
| ESNLI | Natural language inference | entailment / neutral / contradiction |
| CQA | Commonsense question answering | answer choice |

ESNLI kiểm tra khả năng suy luận quan hệ giữa premise và hypothesis.

CQA kiểm tra khả năng suy luận tri thức thường thức.

### 7.2. Baseline

Baseline là original student training setup.

Trong bảng performance:

| Dataset | Baseline |
|---|---:|
| ESNLI | 83.93 |
| CQA | 60.36 |

### 7.3. Compared Methods

Các method cần đưa vào bảng chính:

```text
baseline
judge_student_boundary_mix_balanced
judge_student_boundary_bridge_balanced
judge_expert_hybrid_fusion_balanced
judge_student_multiview
judge_student_singleview_diverse_balanced
judge_student_singleview_superclean_balanced
judge_precision_diverse_mix_balanced
judge_student_shortcut_aware_balanced
judge_student_multiview_hybrid_balanced
```

Nếu paper cần ngắn hơn, bảng chính chỉ nên giữ:

```text
baseline
judge_student_multiview
judge_student_singleview_diverse_balanced
judge_student_boundary_bridge_balanced
judge_student_boundary_mix_balanced
judge_expert_hybrid_fusion_balanced
```

### 7.4. Evaluation Metric

Metric chính:

```text
accuracy
```

Lý do:

- ESNLI và CQA đều là classification/multiple-choice tasks.
- Performance table hiện tại dùng accuracy.

---

## 8. Results

### 8.1. Main Result Table

Bảng chính từ performance bạn cung cấp:

| Method | ESNLI | CQA |
|---|---:|---:|
| baseline | 83.93 | 60.36 |
| `judge_student_boundary_mix_balanced` | 84.16 | 62.41 |
| `judge_expert_hybrid_fusion_balanced` | 84.01 | 61.43 |
| `judge_hybrid_core_balanced` | 83.96 | 58.48 |
| `judge_student_multiview` | 83.97 | 60.77 |
| `judge_student_boundary_bridge_balanced` | 85.06 | 60.52 |
| `judge_precision_diverse_mix_balanced` | 84.02 | 61.99 |
| `judge_student_multiview_hybrid_balanced` | 84.77 | 61.99 |
| `judge_student_multiview_hybrid_cleanfusion` | 84.35 | 57.90 |
| `judge_student_shortcut_aware_balanced` | 84.22 | 61.51 |
| `judge_student_singleview_diverse_balanced` | 84.02 | 62.33 |
| `judge_student_singleview_superclean_balanced` | 84.16 | 61.02 |

### 8.2. Main Interpretation

Key message:

```text
Không có method nào cao nhất trên cả hai dataset.
```

Nhưng:

```text
judge_student_boundary_mix_balanced là lựa chọn tốt nhất cho thesis vì nó cải thiện cả hai dataset và đạt CQA cao nhất.
```

So với baseline:

| Dataset | Baseline | Boundary Mix | Improvement |
|---|---:|---:|---:|
| ESNLI | 83.93 | 84.16 | +0.23 |
| CQA | 60.36 | 62.41 | +2.05 |

### 8.3. Boundary Family Result

| Method | ESNLI | CQA | Nhận xét |
|---|---:|---:|---|
| baseline | 83.93 | 60.36 | Original method |
| `boundary_mix` | 84.16 | 62.41 | Ổn định nhất trên cả hai dataset |
| `boundary_bridge` | 85.06 | 60.52 | ESNLI cao nhất, nhưng CQA thấp hơn boundary_mix |

Interpretation:

```text
Bridge rationales có thể giúp ESNLI vì task NLI hưởng lợi từ nhiều cách diễn giải logic.
Tuy nhiên, với CQA, thêm rationale phụ có thể đưa thêm nhiễu hoặc làm student học tín hiệu không ổn định.
```

---

## 9. Analysis

### 9.1. Why Boundary Mix Works

Boundary mix hiệu quả vì nó cân bằng giữa hai loại example:

| Example type | Vai trò |
|---|---|
| easy | Cung cấp tín hiệu sạch |
| boundary | Giúp student học trường hợp gần ranh giới quyết định |

Nó bỏ:

| Example type | Lý do bỏ |
|---|---|
| hard | Có khả năng nhiễu, source không đồng thuận, label/answer không chắc |

### 9.2. Why Not Only Easy Examples

Nếu chỉ học easy examples:

- student có thể học các pattern quá rõ
- thiếu khả năng xử lý case gần ranh giới
- generalization có thể yếu hơn

Boundary examples giúp student học:

```text
Khi nào hai câu gần giống nhưng không entailment?
Khi nào answer nghe hợp lý nhưng không phải tốt nhất?
```

### 9.3. Why Not Include Hard Examples

Hard examples không đơn giản là “khó nhưng tốt”.

Trong method này, hard có nghĩa:

```text
Các source không đồng thuận đủ mạnh, margin thấp, hoặc ít source vote đúng gold label/answer.
```

Do đó hard có thể chứa:

- rationale sai
- label prediction sai
- explanation không bám input
- nhiều source mâu thuẫn nhau

### 9.4. Why Prior Is Acceptable

Prior không nên trình bày như một con số học được.

Nên trình bày đúng:

```text
The prior is a manually defined inductive bias.
```

Tức là:

- nó là giả định có chủ đích
- dựa trên quan hệ tương đối giữa rationale style và label type
- được kiểm chứng gián tiếp bằng downstream performance
- cần ablation trong future work để kiểm tra thêm

Nên tránh nói:

```text
The prior is learned from data.
The prior is from the original paper.
The prior is statistically calibrated.
```

---

## 10. Ablation Study

Nếu có thời gian, paper nên có ablation.

### 10.1. Suggested Ablations

| Ablation | Mục tiêu |
|---|---|
| Equal vote instead of weighted vote | Kiểm tra prior có giúp không |
| Remove overlap score | Kiểm tra grounding vào input |
| Use easy only | Kiểm tra vai trò của boundary examples |
| Use boundary only | Kiểm tra boundary có đủ ổn định không |
| Include hard examples | Kiểm tra hard có gây nhiễu không |
| Remove label balance | Kiểm tra balancing có quan trọng không |

### 10.2. Most Important Ablation

Quan trọng nhất:

```text
Weighted vote vs equal vote
```

Vì giáo viên có thể hỏi:

```text
Tại sao đặt prior như vậy?
```

Ablation này sẽ giúp trả lời:

```text
Nếu equal vote thấp hơn weighted vote, prior heuristic có tác dụng.
Nếu equal vote tương đương, ta có thể nói method vẫn robust và prior chỉ là soft preference.
```

---

## 11. Discussion

### 11.1. Strengths

Điểm mạnh của method:

- đơn giản, dễ implement
- không cần train thêm judge model
- tận dụng nhiều rationale sources
- có thể áp dụng cho cả ESNLI và CQA
- cải thiện performance trên cả hai dataset

### 11.2. Limitations

Cần viết thật trung thực:

- source prior hiện tại là heuristic thủ công
- threshold cho `easy/boundary/hard` cũng là rule-based
- method phụ thuộc vào chất lượng rationale candidates ban đầu
- chưa chứng minh được rationale được chọn là faithful tuyệt đối
- chưa có ablation đầy đủ cho từng thành phần

### 11.3. Future Work

Hướng phát triển:

- học source prior tự động từ validation set
- tune threshold bằng validation performance
- thêm ablation equal-vote vs weighted-vote
- dùng learned judge model để score rationale
- kiểm tra faithfulness của rationale được chọn

---

## 12. Conclusion

Conclusion nên ngắn:

```text
This work proposes a boundary-aware rationale selection method for step-by-step distillation. Instead of using all generated rationales or selecting rationales independently, the method groups candidates by example, measures multi-source agreement through weighted voting, assigns examples into difficulty bands, and keeps only easy and boundary examples for training. Experiments on ESNLI and CQA show that judge_student_boundary_mix_balanced improves over the baseline on both datasets, suggesting that filtering rationales by useful difficulty and agreement can improve student model training.
```

Tiếng Việt:

```text
Nghiên cứu này đề xuất phương pháp chọn rationale theo vùng biên cho step-by-step distillation. Thay vì dùng toàn bộ rationale hoặc chỉ chọn rationale riêng lẻ, phương pháp gom các rationale theo từng example, đo độ đồng thuận giữa nhiều source bằng weighted voting, phân loại example thành easy/boundary/hard, và chỉ giữ easy + boundary để train student. Kết quả trên ESNLI và CQA cho thấy judge_student_boundary_mix_balanced cải thiện so với baseline trên cả hai dataset.
```

---

## 13. Figures and Tables to Prepare

### 13.1. Figures

| Figure | File/Status | Mục tiêu |
|---|---|---|
| Boundary family pipeline | `assets/boundary_family_pipeline.svg` | Explain method overview |
| Example grouping diagram | optional | Show 7 candidates grouped into one example |
| Voting example | optional | Show weighted vote and margin |

### 13.2. Tables

| Table | Nội dung |
|---|---|
| Main results | Baseline vs methods on ESNLI/CQA |
| Boundary family comparison | mix vs bridge vs specialist |
| Source prior table | ESNLI and CQA prior values |
| Band rule table | easy/boundary/hard threshold |
| Ablation table | if available |

---

## 14. Recommended Writing Order

Không nên viết từ Introduction trước. Nên viết theo thứ tự này:

1. Method
2. Experimental Setup
3. Results
4. Analysis
5. Introduction
6. Related Work
7. Abstract
8. Conclusion

Lý do:

```text
Method và Results là phần mình đã rõ nhất. Viết hai phần đó trước sẽ làm câu chuyện paper chắc hơn.
```

---

## 15. One-Sentence Paper Story

Câu chuyện ngắn nhất của paper:

```text
We improve step-by-step distillation by selecting rationales from examples that are not only correct, but also supported by multi-source agreement and located in useful difficulty regions.
```

Tiếng Việt:

```text
Chúng tôi cải thiện step-by-step distillation bằng cách chọn rationale từ những example không chỉ đúng, mà còn có sự đồng thuận giữa nhiều source và nằm ở vùng độ khó có ích cho training.
```
