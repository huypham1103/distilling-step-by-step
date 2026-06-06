# Tao `judge_student_multiview` tu dau

Tai lieu nay giai thich tung buoc de tao pack `judge_student_multiview`.

Muc tieu la tra loi 3 cau hoi:

- can nhung file nao
- builder chon `view 1` va `view 2` nhu the nao
- chay lenh nao de tao ra file cuoi

Trong repo nay, `judge_student_multiview` co 2 bien the theo dataset:

- ESNLI: tao bang `build_judge_esnli.py`
- CQA: tao bang `build_judge_cqa_outstanding_pack.py`

---

## 1. Y tuong ngan gon

Mot example co the co nhieu loi giai thich.

Vi du ESNLI:

- premise: `A child is running in a park.`
- hypothesis: `A kid is outdoors.`
- label dung: `entailment`

Bay gio 7 source cung viet rationale:

- `historical`
- `consensus`
- `contrastive`
- `causal`
- `neutral`
- `if_else`
- `comparative`

`judge_student_multiview` khong lay tat ca 7 rationale.

No lam 4 viec:

1. gom 7 rationale cua cung mot example lai
2. xem cac source doan label nao
3. cham diem tung rationale
4. giu rationale tot nhat lam `view 1`, va neu co rationale thu hai du tot thi giu lam `view 2`

Ket qua cuoi cung la file:

```text
[API] ESNLI/judge_student_multiview - full.csv
```

hoac:

```text
[API] CQA/judge_student_multiview - full.csv
```

---

## 2. Dau vao can co

### 2.1. Dau vao ESNLI

Builder ESNLI can 1 file gold anchor va 7 file source.

Gold anchor:

```text
[API] ESNLI/paper - full.csv
```

Neu co local JSON day du, script uu tien:

```text
datasets/esnli/esnli_train.json
datasets/esnli/esnli_valid.json
datasets/esnli/esnli_test.json
```

7 file source:

```text
[API] ESNLI/neutral - full.csv
[API] ESNLI/contrastive - full.csv
[API] ESNLI/historical - full.csv
[API] ESNLI/comparative - full.csv
[API] ESNLI/causal - full.csv
[API] ESNLI/consensus - full.csv
[API] ESNLI/if_else - full.csv
```

Moi file source can toi thieu cac cot:

```text
premise
hypothesis
rationale
LLM_answer
```

### 2.2. Dau vao CQA

Builder CQA can file gold anchor:

```text
[API] CQA/paper.csv
```

Va 7 file source:

```text
[API] CQA/historical - full.csv
[API] CQA/consensus - full.csv
[API] CQA/contrastive - full.csv
[API] CQA/causal - full.csv
[API] CQA/neutral - full.csv
[API] CQA/if_else - full.csv
[API] CQA/comparative - full.csv
```

Moi file source cung can cac cot chinh:

```text
premise
hypothesis
rationale
LLM_answer
```

Voi CQA:

- `premise` la cau hoi
- `hypothesis` la list cac answer choices
- `LLM_answer` la dap an source do chon

---

## 3. Vi du nho de hieu cach chon

Gia su ta co 1 example ESNLI:

```text
premise:    A child is running through a park.
hypothesis: A kid is outdoors.
gold label: entailment
```

7 source sinh ra nhu sau:

| source | predicted label | rationale ngan |
|---|---|---|
| historical | entailment | A child is a kid, and a park is outdoors, so the hypothesis follows. |
| consensus | entailment | The premise supports that a kid is outside. |
| contrastive | entailment | It is not neutral or contradiction because the hypothesis is directly supported. |
| causal | entailment | Running through a park means the child is outdoors. |
| neutral | neutral | The premise does not say why the child is running. |
| if_else | entailment | If a child runs through a park, then the child is outdoors. |
| comparative | entailment | The hypothesis is more general than the premise. |

### Buoc 1. Vote label

Builder dem xem source nao chon label nao.

Trong vi du:

```text
entailment: 6 source
neutral:    1 source
```

Nhung builder khong chi dem bang so source. No dung weighted vote.

Nghia la source co prior cao hon se co phieu nang hon.

Vi du don gian:

```text
historical chon entailment: +1.00
consensus chon entailment:  +0.99
neutral chon neutral:       +0.98
```

Sau khi cong het, label co diem cao nhat la `voted_label`.

### Buoc 2. Chon training label

Voi `student_multiview`, builder thich dung gold label neu gold label dang tin.

Neu gold label la:

```text
entailment
```

thi builder se tim cac candidate co predicted label la:

```text
entailment
```

Nhung candidate do moi duoc goi la `matching_candidates`.

Candidate doan `neutral` bi loai khoi viec chon rationale cho example nay.

### Buoc 3. Cham diem tung rationale

Builder cham `judge_score` dua tren nhieu tin hieu:

- predicted label co khop training label khong
- source nao sinh ra rationale
- co bao nhieu source cung dong y label do
- margin giua label thang va label ve nhi co lon khong
- rationale co do dai vua phai khong
- rationale co bam vao tu trong premise va hypothesis khong
- rationale co cue lap luan nhu `because`, `therefore`, `implies`, `not necessarily` khong
- rationale co nhac dung label khong

Vi du diem gia lap:

| source | label khop | agreement | overlap | judge_score |
|---|---:|---:|---:|---:|
| historical | yes | 6 | high | 19.4 |
| if_else | yes | 6 | high | 19.1 |
| causal | yes | 6 | high | 18.9 |
| contrastive | yes | 6 | medium | 18.2 |
| neutral | no | 1 | medium | bi loai |

### Buoc 4. Chon `view 1`

`view 1` la candidate tot nhat trong cac candidate khop label.

Trong vi du:

```text
view 1 = historical
```

vi co `judge_score` cao nhat.

### Buoc 5. Chon `view 2`

Builder chi them `view 2` neu rationale thu hai:

- khac source voi `view 1`
- khong trung gan nhu y het rationale cua `view 1`
- diem khong thap hon `view 1` qua xa
- co agreement du tot

Trong vi du:

```text
view 2 = if_else
```

vi no giai bang cau truc `if ... then ...`, khac wording voi `historical`, va diem van cao.

### Buoc 6. Ghi ra 2 dong

Cung mot example co the thanh 2 dong trong output:

| premise | hypothesis | LLM_answer | judge_source | judge_view_rank |
|---|---|---|---|---:|
| A child is running... | A kid is outdoors. | entailment | historical | 1 |
| A child is running... | A kid is outdoors. | entailment | if_else | 2 |

Day la ly do ten type co chu `multiview`: mot example co the co nhieu view tot.

---

## 4. Tao `judge_student_multiview` cho ESNLI

### Buoc 1. Kiem tra file dau vao

Chay:

```bash
for s in neutral contrastive historical comparative causal consensus if_else; do
  test -f "[API] ESNLI/$s - full.csv" && echo "ok $s" || echo "missing $s"
done
```

Kiem tra gold anchor:

```bash
test -f "[API] ESNLI/paper - full.csv" && echo "ok paper" || echo "missing paper"
```

### Buoc 2. Chay builder

Lenh chinh:

```bash
python build_judge_esnli.py \
  --strategy student_multiview \
  --output-name judge_student_multiview
```

Lenh nay se tao:

```text
[API] ESNLI/judge_student_multiview - full.csv
[API] ESNLI/judge_student_multiview_judge_report.json
```

### Buoc 3. Doc report

Chay:

```bash
cat "[API] ESNLI/judge_student_multiview_judge_report.json"
```

Report hien tai trong repo co dang:

```json
{
  "output_csv": "[API] ESNLI/judge_student_multiview - full.csv",
  "num_examples": 19541,
  "extra_view_count": 9769,
  "label_match_rate": 1.0,
  "strategy": "student_multiview"
}
```

Cac field quan trong:

- `num_examples`: tong so row trong output, khong phai so example goc
- `extra_view_count`: so luong view phu duoc them
- `source_counts`: moi source dong gop bao nhieu rationale
- `label_match_rate`: ty le candidate label khop label dung
- `average_judge_score`: diem trung binh cua cac rationale duoc giu

### Buoc 4. Kiem tra file output

Chay:

```bash
python - <<'PY'
import pandas as pd

path = "[API] ESNLI/judge_student_multiview - full.csv"
df = pd.read_csv(path)

print(df.shape)
print(df["judge_view_rank"].value_counts().sort_index())
print(df["judge_source"].value_counts())
print(df[[
    "premise",
    "hypothesis",
    "LLM_answer",
    "judge_source",
    "judge_score",
    "agreement_count",
    "voted_label_margin",
    "judge_view_rank",
]].head(5))
PY
```

Neu thay `judge_view_rank` co ca `1` va `2`, nghia la multiview dang hoat dong.

---

## 5. Tao `judge_student_multiview` cho CQA

### Buoc 1. Kiem tra file dau vao

Chay:

```bash
for s in historical consensus contrastive causal neutral if_else comparative; do
  test -f "[API] CQA/$s - full.csv" && echo "ok $s" || echo "missing $s"
done
```

Kiem tra gold anchor:

```bash
test -f "[API] CQA/paper.csv" && echo "ok paper" || echo "missing paper"
```

### Buoc 2. Chay builder CQA

Lenh chinh:

```bash
python build_judge_cqa_outstanding_pack.py
```

Luu y: script CQA hien khong chi tao `judge_student_multiview`. No tao nhieu pack cung luc, trong do co:

```text
[API] CQA/judge_student_multiview - full.csv
[API] CQA/judge_student_multiview_judge_report.json
```

### Buoc 3. Hieu profile CQA cua `judge_student_multiview`

Trong `build_judge_cqa_outstanding_pack.py`, profile cua `judge_student_multiview` dung cac setting chinh:

```text
allow_secondary = True
min_agreement = 2
min_margin = 1.2
min_score = 1.65
similarity_limit = 0.78
max_secondary_gap = 0.50
cap = 12000
max_per_example = 2
```

Nghia la:

- moi question can it nhat 2 source ung ho gold answer
- vote margin phai tu 1.2 tro len
- rationale tot nhat phai co score tu 1.65 tro len
- view 2 khong duoc qua giong view 1
- moi question toi da 2 rationale
- output toi da 12000 row

### Buoc 4. Doc report

Chay:

```bash
cat "[API] CQA/judge_student_multiview_judge_report.json"
```

Report hien tai trong repo co dang:

```json
{
  "output_csv": "[API] CQA/judge_student_multiview - full.csv",
  "num_examples": 12000,
  "view_rank_counts": {
    "2": 6574,
    "1": 5426
  },
  "average_judge_score": 3.831475
}
```

Neu `view_rank_counts` co `1` va `2`, pack da co nhieu view.

### Buoc 5. Kiem tra output

Chay:

```bash
python - <<'PY'
import pandas as pd

path = "[API] CQA/judge_student_multiview - full.csv"
df = pd.read_csv(path)

print(df.shape)
print(df["judge_view_rank"].value_counts().sort_index())
print(df["judge_source"].value_counts())
print(df[[
    "premise",
    "hypothesis",
    "LLM_answer",
    "judge_source",
    "judge_score",
    "agreement_count",
    "voted_label_margin",
    "judge_view_rank",
]].head(5))
PY
```

---

## 6. Cac cot quan trong trong output

Output CSV co nhieu cot. Cac cot can nhin dau tien la:

| cot | y nghia |
|---|---|
| `premise` | input phan 1, voi ESNLI la premise, voi CQA la question |
| `hypothesis` | input phan 2, voi ESNLI la hypothesis, voi CQA la answer choices |
| `rationale` | loi giai thich duoc chon |
| `LLM_answer` | label/answer dung de train |
| `judge_source` | source sinh ra rationale duoc chon |
| `judge_score` | diem chat luong cua rationale |
| `candidate_label` | label/answer ma source da doan |
| `gold_label` | label/answer dung ma builder dung |
| `voted_label` | label/answer thang theo weighted vote |
| `voted_label_support` | tong support cua voted label |
| `voted_label_margin` | khoang cach giua label thang va label ve nhi |
| `label_match` | candidate co khop label dung khong |
| `word_count` | do dai rationale |
| `overlap_score` | muc overlap voi input |
| `agreement_count` | so source ung ho label dung |
| `agreement_ratio` | agreement_count chia tong so candidate |
| `judge_view_rank` | `1` la view chinh, `2` la view phu |

---

## 7. Dung pack nay de train trong `run.py`

Sau khi co file:

```text
[API] ESNLI/judge_student_multiview - full.csv
```

co the train ESNLI bang:

```bash
python run.py \
  --dataset esnli \
  --llm palm \
  --label_type gt \
  --model_type task_prefix \
  --type_rationale judge_student_multiview
```

Voi CQA:

```bash
python run.py \
  --dataset cqa \
  --llm palm \
  --label_type gt \
  --model_type task_prefix \
  --type_rationale judge_student_multiview
```

Trong `run.py`, khi `--llm` khac `None`, code se doc:

```text
[API] ESNLI/{type_rationale} - full.csv
```

hoac:

```text
[API] CQA/{type_rationale} - full.csv
```

Vay `--type_rationale judge_student_multiview` se nap dung pack vua tao.

---

## 8. Neu muon tao tu zero that su

Neu chua co 7 file source, can lam theo thu tu nay:

1. Tao raw rationale cho tung source.
2. Moi source ghi ra mot file `source - full.csv`.
3. Dam bao moi file co `premise`, `hypothesis`, `rationale`, `LLM_answer`.
4. Dam bao cung mot example co cung text `premise` va `hypothesis` o tat ca source.
5. Tao gold anchor `paper - full.csv` cho ESNLI hoac `paper.csv` cho CQA.
6. Chay builder.
7. Doc report.
8. Kiem tra `judge_view_rank`.
9. Train bang `run.py`.

Dieu quan trong nhat la buoc 4.

Builder gom example bang key:

```text
normalize(premise) + "</s>" + normalize(hypothesis)
```

Neu cung mot example nhung text bi lech nhe, builder se nghi do la 2 example khac nhau.

---

## 9. Loi thuong gap

### 9.1. Bao missing candidate file

Nguyen nhan:

- thieu mot file `source - full.csv`
- ten file sai, vi du `neutral.csv` thay vi `neutral - full.csv`

Cach sua:

- doi dung ten file
- hoac sua list source trong builder neu co chu dich dung it source hon

### 9.2. Output it row hon mong doi

Nguyen nhan thuong gap:

- nhieu source doan sai label
- `vote_margin` thap
- `agreement_count` thap
- rationale qua ngan hoac qua dai
- premise/hypothesis khong khop key giua cac file

Cach debug nhanh:

```bash
python - <<'PY'
import pandas as pd

df = pd.read_csv("[API] ESNLI/judge_student_multiview - full.csv")
print(df["agreement_count"].describe())
print(df["voted_label_margin"].describe())
print(df["judge_score"].describe())
PY
```

### 9.3. Khong co `view 2`

Nguyen nhan:

- candidate thu hai qua giong view 1
- candidate thu hai khac source nhung score thap
- chi co mot source doan dung label

Kiem tra:

```bash
python - <<'PY'
import pandas as pd

df = pd.read_csv("[API] ESNLI/judge_student_multiview - full.csv")
print(df["judge_view_rank"].value_counts())
PY
```

### 9.4. Train khong doc dung file

Kiem tra `--type_rationale`.

Neu goi:

```bash
--type_rationale judge_student_multiview
```

thi `run.py` se tim:

```text
[API] ESNLI/judge_student_multiview - full.csv
```

hoac:

```text
[API] CQA/judge_student_multiview - full.csv
```

Khong can them ` - full.csv` vao argument.

---

## 10. Tom tat cuc ngan

De tao ESNLI:

```bash
python build_judge_esnli.py \
  --strategy student_multiview \
  --output-name judge_student_multiview
```

De tao CQA:

```bash
python build_judge_cqa_outstanding_pack.py
```

De train:

```bash
python run.py --dataset esnli --llm palm --label_type gt --type_rationale judge_student_multiview
```

Y nghia cua `judge_student_multiview`:

- lay rationale dung va manh nhat lam `view 1`
- them rationale khac source va du khac biet lam `view 2`
- giu toi da 2 rationale cho moi example
- dung cac cot `judge_score`, `agreement_count`, `voted_label_margin`, `judge_view_rank` de kiem tra chat luong
