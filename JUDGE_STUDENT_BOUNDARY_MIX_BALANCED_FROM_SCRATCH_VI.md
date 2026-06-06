# Tao `judge_student_boundary_mix_balanced` tu dau

Tai lieu nay giai thich tung buoc de tao pack `judge_student_boundary_mix_balanced`.

Neu `judge_student_multiview` hoi:

```text
Mot example co nen co 1 hay 2 rationale tot?
```

thi `judge_student_boundary_mix_balanced` hoi:

```text
Example nay dang easy, boundary, hay hard?
```

Sau do no chi giu:

- `easy`
- `boundary`

va bo:

- `hard`

No khong co muc tieu nhan doi du lieu nhu `multiview`.

Voi type nay, moi example chi giu toi da 1 rationale trong output cuoi.

---

## 1. Y tuong ngan gon

Khong phai example nao cung tot nhu nhau de train.

Co example qua de:

```text
Tat ca source deu dong y.
Margin rat cao.
Rationale kha on dinh.
```

Co example nam gan ranh gioi:

```text
Van co dap an dung kha ro.
Nhung mot vai source co the bi nham.
Day la vung model nen hoc.
```

Co example qua hard:

```text
Source cai nhau manh.
Margin thap.
Rationale khong on dinh.
Cho vao train co the them nhieu.
```

`judge_student_boundary_mix_balanced` giu mot mixture:

- easy examples de co nen tang sach
- boundary examples de model hoc vung de nham

Nhung no bo hard examples.

---

## 2. Dau vao can co

### 2.1. Dau vao ESNLI

Script ESNLI la:

```text
build_boundary_focus_pack.py
```

Script nay dung lai helper tu:

```text
build_judge_esnli.py
```

Can gold anchor:

```text
[API] ESNLI/paper - full.csv
```

Hoac neu co local JSON day du, `build_judge_esnli.py` co the uu tien:

```text
datasets/esnli/esnli_train.json
datasets/esnli/esnli_valid.json
datasets/esnli/esnli_test.json
```

Can 7 file source:

```text
[API] ESNLI/neutral - full.csv
[API] ESNLI/contrastive - full.csv
[API] ESNLI/historical - full.csv
[API] ESNLI/comparative - full.csv
[API] ESNLI/causal - full.csv
[API] ESNLI/consensus - full.csv
[API] ESNLI/if_else - full.csv
```

Moi file source can cac cot:

```text
premise
hypothesis
rationale
LLM_answer
```

### 2.2. Dau vao CQA

Script CQA la:

```text
build_judge_cqa_outstanding_pack.py
```

Can gold anchor:

```text
[API] CQA/paper.csv
```

Can 7 file source:

```text
[API] CQA/historical - full.csv
[API] CQA/consensus - full.csv
[API] CQA/contrastive - full.csv
[API] CQA/causal - full.csv
[API] CQA/neutral - full.csv
[API] CQA/if_else - full.csv
[API] CQA/comparative - full.csv
```

Moi file source can cac cot:

```text
premise
hypothesis
rationale
LLM_answer
```

Voi CQA:

- `premise` la question
- `hypothesis` la list answer choices
- `LLM_answer` la answer source do chon

---

## 3. Vi du nho de hieu boundary mix

Gia su co 1 example ESNLI:

```text
premise:    Two dogs are running through snow.
hypothesis: Animals are outside.
gold label: entailment
```

7 source doan:

| source | predicted label |
|---|---|
| historical | entailment |
| consensus | entailment |
| contrastive | entailment |
| causal | entailment |
| neutral | entailment |
| if_else | neutral |
| comparative | entailment |

### Buoc 1. Vote label

Builder cong weighted vote.

Vi du don gian:

```text
entailment = 5.72
neutral = 1.00
contradiction = 0.00
```

Khi do:

```text
voted_label = entailment
voted_label_margin = 5.72 - 1.00 = 4.72
```

Margin lon nghia la label thang kha chac.

### Buoc 2. Dem agreement

Agreement la so source ung ho label dung.

Trong vi du:

```text
agreement_count = 6
```

vi 6 source cung chon `entailment`.

### Buoc 3. Gan band

Builder nhin 2 so chinh:

```text
agreement_count
voted_label_margin
```

Neu ca hai deu cao, example la `easy`.

Neu vua du cao, example la `boundary`.

Neu thap, example la `hard`.

Trong vi du nay:

```text
agreement_count = 6
voted_label_margin = 4.72
```

Nen example duoc gan:

```text
boundary_band = easy
```

### Buoc 4. Chon rationale tot nhat

Sau khi biet example nay duoc giu, builder cham diem cac rationale co label khop.

No lay rationale co `judge_score` cao nhat lam row cuoi.

Output co:

```text
judge_view_rank = 1
boundary_band = easy
```

### Buoc 5. Neu example qua roi thi bo

Gia su source vote nhu sau:

```text
entailment = 3 source
neutral = 3 source
contradiction = 1 source
```

Margin nho, agreement khong cao.

Example co the thanh:

```text
boundary_band = hard
```

`judge_student_boundary_mix_balanced` se bo example nay.

---

## 4. Khac gi voi `judge_student_multiview`?

`judge_student_multiview`:

- co the giu 2 rationale cho cung mot example
- output co `judge_view_rank = 1` va `judge_view_rank = 2`
- muc tieu la them nhieu goc nhin

`judge_student_boundary_mix_balanced`:

- chi giu 1 rationale cho moi example
- output gan nhu chi co `judge_view_rank = 1`
- muc tieu la chon dung do kho cua example
- giu `easy` va `boundary`
- bo `hard`

Noi ngan gon:

```text
multiview = chon nhieu view tot
boundary_mix = chon example co do kho tot de train
```

---

## 5. Tao `judge_student_boundary_mix_balanced` cho ESNLI

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

### Buoc 2. Chay builder ESNLI

Lenh chinh:

```bash
python build_boundary_focus_pack.py
```

Script nay se tao 3 pack:

```text
[API] ESNLI/judge_student_boundary_mix_balanced - full.csv
[API] ESNLI/judge_student_boundary_bridge_balanced - full.csv
[API] ESNLI/judge_student_boundary_specialist_balanced - full.csv
```

File minh can la:

```text
[API] ESNLI/judge_student_boundary_mix_balanced - full.csv
```

Report la:

```text
[API] ESNLI/judge_student_boundary_mix_balanced_judge_report.json
```

### Buoc 3. Hieu source prior ESNLI

ESNLI boundary dung prior rieng theo label.

Vi du:

```text
entailment:
  historical  = 1.00
  consensus   = 0.98
  contrastive = 0.95

neutral:
  if_else     = 1.00
  comparative = 0.99
  neutral     = 0.96

contradiction:
  comparative = 1.00
  contrastive = 0.99
  causal      = 0.97
```

Ly do:

- moi source co the manh o mot loai label khac nhau
- boundary pack can vote va score tinh hon theo tung label

### Buoc 4. Training label duoc chon nhu the nao

Builder co 2 label quan trong:

```text
paper_label
voted_label
```

Voi ESNLI boundary:

- neu `paper_label` hop le va trung `voted_label`, dung `paper_label`
- neu `paper_label` hop le va vote margin chua qua manh, van dung `paper_label`
- neu vote margin manh va voted label khac paper label, co the dung `voted_label`

Trong code, nguong quan trong la:

```text
vote_margin < 1.55
```

Neu margin khong qua manh, builder khong de voted label de dang override paper label.

### Buoc 5. Cham diem rationale

Moi candidate khop training label duoc cham `judge_score`.

Tin hieu chinh:

- source prior theo label
- overlap giua rationale va premise/hypothesis
- rationale co cue dung voi label khong
- rationale co nhac label dung khong
- co format ro nhu `the correct answer` hoac `so the answer is` khong
- do dai co vua phai khong
- agreement_count cao khong
- vote_margin cao khong

Do dai tot trong ESNLI boundary:

```text
10 <= word_count <= 96
```

Qua dai:

```text
word_count > 128
```

se bi tru nhe.

### Buoc 6. Gan `boundary_band`

ESNLI dung nguong:

```text
easy:
  vote_margin >= 2.1
  agreement_count >= 5

boundary:
  vote_margin >= 1.1
  agreement_count >= 2

hard:
  con lai
```

Nghia la:

- `easy`: rat nhieu source dong y, margin cao
- `boundary`: co tin hieu du tot, nhung khong chac bang easy
- `hard`: qua mo ho, bo khoi pack mix

### Buoc 7. Chon row cho `boundary_mix`

Trong `build_boundary_focus_pack.py`, pack mix duoc tao bang:

```text
allowed_bands = {"easy", "boundary"}
max_per_example = 1
cap_per_label = 3200
```

Nghia la:

- chi giu `easy` va `boundary`
- moi example toi da 1 row
- moi label toi da 3200 row
- dataset duoc balance theo 3 label ESNLI

### Buoc 8. Doc report

Chay:

```bash
cat "[API] ESNLI/judge_student_boundary_mix_balanced_judge_report.json"
```

Report hien tai trong repo co dang:

```json
{
  "output_csv": "[API] ESNLI/judge_student_boundary_mix_balanced - full.csv",
  "num_examples": 9054,
  "label_counts": {
    "entailment": 3018,
    "neutral": 3018,
    "contradiction": 3018
  },
  "boundary_band_counts": {
    "easy": 9032,
    "boundary": 22
  },
  "extra_view_count": 0
}
```

Dieu can de y:

- `label_counts` bang nhau, vi ESNLI co 3 label va script balance theo label
- `boundary_band_counts` cho biet co bao nhieu easy/boundary
- `extra_view_count = 0`, vi mix khong giu view 2

### Buoc 9. Kiem tra output

Chay:

```bash
python - <<'PY'
import pandas as pd

path = "[API] ESNLI/judge_student_boundary_mix_balanced - full.csv"
df = pd.read_csv(path)

print(df.shape)
print(df["LLM_answer"].value_counts())
print(df["boundary_band"].value_counts())
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
    "boundary_band",
    "judge_view_rank",
]].head(5))
PY
```

Neu dung, ban se thay:

```text
boundary_band: easy/boundary
judge_view_rank: gan nhu chi co 1
```

---

## 6. Tao `judge_student_boundary_mix_balanced` cho CQA

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

Script CQA tao nhieu pack cung luc.

File minh can la:

```text
[API] CQA/judge_student_boundary_mix_balanced - full.csv
```

Report la:

```text
[API] CQA/judge_student_boundary_mix_balanced_judge_report.json
```

### Buoc 3. Hieu source prior CQA

CQA boundary dung `BASE_SOURCE_PRIOR`.

Prior hien tai:

```text
causal      = 1.00
if_else     = 0.99
neutral     = 0.98
contrastive = 0.975
historical  = 0.94
consensus   = 0.84
comparative = 0.80
```

Nghia la CQA hoi:

- neu source khac nhau cung chon mot answer, source nao nen co phieu nang hon?

`causal` duoc uu tien cao nhat trong CQA.

### Buoc 4. Vote answer

Voi moi question:

1. gom 7 candidate cua cung question
2. moi source co `LLM_answer`
3. cong diem theo prior
4. answer co tong diem cao nhat la `voted_label`
5. tinh `voted_label_margin`

Khac ESNLI, CQA co nhieu answer text khac nhau, khong chi 3 label co dinh.

### Buoc 5. Chi cham candidate khop gold answer

CQA boundary chi xet candidate co:

```text
candidate["label"] == gold_label
```

Neu source doan sai answer, candidate do khong duoc chon lam rationale cuoi.

### Buoc 6. Cham diem rationale

Tin hieu chinh:

- source prior
- overlap voi question va answer choices
- agreement_count
- agreement_ratio
- vote_margin
- rationale co nhac answer dung khong
- cue nhu `because`, `therefore`, `if`, `then`, `so the answer is`
- source co nam trong preferred sources khong
- do dai co vua phai khong

Profile CQA boundary dung:

```text
ideal_word_max = 96
hard_word_max = 128
overlap_weight = 0.62
agreement_weight = 0.18
margin_weight = 0.10
preferred_sources = {"contrastive", "if_else", "causal", "neutral"}
```

### Buoc 7. Gan `boundary_band`

CQA dung nguong:

```text
easy:
  vote_margin >= 2.4
  agreement_count >= 5

boundary:
  vote_margin >= 1.6
  agreement_count >= 3

hard:
  con lai
```

So voi ESNLI, CQA dat nguong cao hon mot chut.

### Buoc 8. Chon row cho `boundary_mix`

Trong `build_judge_cqa_outstanding_pack.py`, pack mix duoc tao bang:

```text
allowed_bands = {"easy", "boundary"}
max_per_example = 1
cap = 8500
source_order = EXPERT_SOURCE_PRIOR order
```

Nghia la:

- chi giu `easy` va `boundary`
- bo `hard`
- bo `bridge`
- moi question toi da 1 row
- output toi da 8500 row
- khi can cat bot, script uu tien source order va score

### Buoc 9. Doc report

Chay:

```bash
cat "[API] CQA/judge_student_boundary_mix_balanced_judge_report.json"
```

Report hien tai trong repo co dang:

```json
{
  "output_csv": "[API] CQA/judge_student_boundary_mix_balanced - full.csv",
  "num_examples": 7843,
  "view_rank_counts": {
    "1": 7843
  },
  "boundary_band_counts": {
    "easy": 7787,
    "boundary": 56
  }
}
```

Dieu can de y:

- `view_rank_counts` chi co `1`
- `boundary_band_counts` chi co `easy` va `boundary`
- khong co `hard`

### Buoc 10. Kiem tra output

Chay:

```bash
python - <<'PY'
import pandas as pd

path = "[API] CQA/judge_student_boundary_mix_balanced - full.csv"
df = pd.read_csv(path)

print(df.shape)
print(df["boundary_band"].value_counts())
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
    "boundary_band",
    "judge_view_rank",
]].head(5))
PY
```

---

## 7. Cac cot quan trong trong output

| cot | y nghia |
|---|---|
| `premise` | ESNLI premise, hoac CQA question |
| `hypothesis` | ESNLI hypothesis, hoac CQA answer choices |
| `rationale` | loi giai thich duoc chon |
| `LLM_answer` | label/answer dung de train |
| `judge_source` | source sinh ra rationale duoc chon |
| `judge_score` | diem chat luong cua rationale |
| `candidate_label` | label/answer ma source doan |
| `gold_label` | label/answer dung ma builder dung |
| `paper_gold_label` | label/answer tu gold anchor |
| `voted_label` | label/answer thang theo weighted vote |
| `voted_label_support` | tong weighted support cua voted label |
| `voted_label_margin` | khoang cach giua label thang va label ve nhi |
| `label_match` | candidate co khop label dung khong |
| `word_count` | do dai rationale |
| `overlap_score` | muc overlap voi input |
| `agreement_count` | so source ung ho label/answer dung |
| `agreement_ratio` | agreement_count chia tong so candidate |
| `judge_view_rank` | voi mix thuong la `1` |
| `boundary_band` | `easy`, `boundary`, hoac `hard`; output mix chi giu `easy` va `boundary` |

---

## 8. Dung pack nay de train trong `run.py`

Sau khi co file ESNLI:

```text
[API] ESNLI/judge_student_boundary_mix_balanced - full.csv
```

train bang:

```bash
python run.py \
  --dataset esnli \
  --llm palm \
  --label_type gt \
  --model_type task_prefix \
  --type_rationale judge_student_boundary_mix_balanced
```

Voi CQA:

```bash
python run.py \
  --dataset cqa \
  --llm palm \
  --label_type gt \
  --model_type task_prefix \
  --type_rationale judge_student_boundary_mix_balanced
```

Khong them ` - full.csv` vao `--type_rationale`.

Dung:

```text
--type_rationale judge_student_boundary_mix_balanced
```

Sai:

```text
--type_rationale "judge_student_boundary_mix_balanced - full.csv"
```

---

## 9. Neu muon tao tu zero that su

Thu tu lam:

1. Tao raw rationale cho 7 source.
2. Ghi moi source thanh file `source - full.csv`.
3. Dam bao moi file co `premise`, `hypothesis`, `rationale`, `LLM_answer`.
4. Dam bao cung mot example co cung text `premise` va `hypothesis` tren moi source.
5. Tao gold anchor.
6. Chay boundary builder.
7. Doc report.
8. Kiem tra `boundary_band`.
9. Dam bao output khong co `hard`.
10. Train bang `run.py`.

Dieu quan trong nhat:

```text
premise + hypothesis phai match giua cac source
```

Builder gom example bang key:

```text
normalize(premise) + "</s>" + normalize(hypothesis)
```

Neu text lech nhau, cac source se khong duoc gom vao cung mot example.

Luc do `agreement_count` se thap va nhieu row co the bi mat.

---

## 10. Loi thuong gap

### 10.1. Output co qua it row

Nguyen nhan co the:

- nhieu source doan sai gold label
- `agreement_count` thap
- `voted_label_margin` thap
- nhieu example bi gan `hard`
- premise/hypothesis khong match giua source

Debug:

```bash
python - <<'PY'
import pandas as pd

df = pd.read_csv("[API] ESNLI/judge_student_boundary_mix_balanced - full.csv")
print(df["agreement_count"].describe())
print(df["voted_label_margin"].describe())
print(df["judge_score"].describe())
print(df["boundary_band"].value_counts())
PY
```

### 10.2. Sao khong co `judge_view_rank = 2`?

Day la dung voi `boundary_mix`.

`boundary_mix` chi giu:

```text
max_per_example = 1
```

Nen output cuoi chi co rationale chinh.

Neu muon co view phu/bridge, xem type:

```text
judge_student_boundary_bridge_balanced
```

### 10.3. Sao report ESNLI co label_counts bang nhau?

Vi ESNLI co 3 label co dinh:

```text
entailment
neutral
contradiction
```

`build_boundary_focus_pack.py` balance theo label.

### 10.4. Sao CQA khong balance theo label nhu ESNLI?

CQA answer la text tu nhieu cau hoi khac nhau.

No khong co 3 label co dinh nhu ESNLI.

Vi vay CQA dung source-balanced selection thay vi label-balanced 3 class.

---

## 11. Tom tat cuc ngan

De tao ESNLI:

```bash
python build_boundary_focus_pack.py
```

De tao CQA:

```bash
python build_judge_cqa_outstanding_pack.py
```

De train:

```bash
python run.py \
  --dataset esnli \
  --llm palm \
  --label_type gt \
  --type_rationale judge_student_boundary_mix_balanced
```

Y nghia cua `judge_student_boundary_mix_balanced`:

- vote label/answer bang weighted vote
- cham diem rationale
- gan example vao `easy`, `boundary`, hoac `hard`
- giu `easy` va `boundary`
- bo `hard`
- giu toi da 1 rationale cho moi example
