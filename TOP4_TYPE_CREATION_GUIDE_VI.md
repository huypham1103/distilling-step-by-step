# Hướng dẫn cực chi tiết để tạo 4 type tốt nhất

Tài liệu này viết cho người mới hoàn toàn.

Mục tiêu của tài liệu này là:

- giải thích như đang nói với một đứa trẻ
- không dùng từ khó mà không giải thích
- đi từng bước từ đầu đến cuối
- giúp người đọc hiểu “vì sao” chứ không chỉ biết “làm thế nào”

4 type được giải thích là:

1. `judge_student_multiview`
2. `judge_student_boundary_mix_balanced`
3. `judge_student_boundary_bridge_balanced`
4. `judge_expert_hybrid_fusion_balanced`

Đây là 4 type hiện đang có ý nghĩa tốt khi nhìn đồng thời trên ESNLI và CQA.

---

## 1. Ta đang cố làm điều gì?

Hãy tưởng tượng có một câu hỏi.

Ví dụ:

- với ESNLI: “Câu B có được suy ra từ câu A không?”
- với CQA: “Trong 5 đáp án, đáp án nào đúng?”

Bây giờ tưởng tượng có 7 người khác nhau cùng giải thích đáp án cho câu hỏi đó.

Mỗi người có một phong cách giải thích riêng:

- người thích kể theo ngữ cảnh lịch sử
- người thích so sánh hai ý với nhau
- người thích nói nguyên nhân và kết quả
- người thích kiểu “nếu ... thì ...”

Trong repo này, 7 “người” đó chính là 7 source:

- `historical`
- `consensus`
- `contrastive`
- `causal`
- `neutral`
- `if_else`
- `comparative`

Vấn đề là:

- có source giải thích hay cho câu này
- có source giải thích dở cho câu khác
- có source đoán đúng label nhưng giải thích chưa tốt
- có source giải thích nghe ổn nhưng lại đoán sai label

Nên ta không thể lấy hết tất cả rồi train model.

Ta cần làm một việc giống như chọn giáo viên để dạy học:

1. nghe tất cả các lời giải thích
2. chấm xem lời nào đáng tin hơn
3. bỏ lời yếu
4. giữ lời mạnh
5. ghép chúng thành bộ dữ liệu mới để train

Mỗi `type` là một cách khác nhau để chọn bộ lời giải thích đó.

---

## 2. Giải thích mọi từ sẽ dùng trong tài liệu

Phần này rất quan trọng.

Từ đây về sau, bất cứ khi nào thấy một từ như `example`, `label`, `rationale`, `vote`, `bridge`, người đọc phải hiểu ngay nó là gì.

### 2.1. `dataset`

`Dataset` là cả một bộ dữ liệu lớn.

Ví dụ:

- ESNLI là một dataset
- CQA là một dataset

Mỗi dataset gồm rất nhiều câu hỏi.

### 2.2. `example`

`Example` là một mẫu dữ liệu nhỏ bên trong dataset.

Có thể hiểu rất đơn giản:

- một câu hỏi
- kèm thông tin cần thiết để trả lời câu hỏi đó

Ví dụ trong ESNLI, một example thường có:

- `premise`: câu gốc
- `hypothesis`: câu giả thuyết
- `label`: nhãn đúng

Ví dụ trong CQA, một example thường có:

- `question`: câu hỏi
- `choices`: 5 đáp án lựa chọn
- `label`: đáp án đúng

### 2.3. `label`

`Label` là đáp án đúng mà model phải học.

Với ESNLI, label chỉ có 3 loại:

- `entailment`
- `neutral`
- `contradiction`

Với CQA, label là một câu trả lời cụ thể trong 5 lựa chọn của câu hỏi đó.

Ví dụ:

- `kitchen`
- `market`
- `subdivision`

### 2.4. `gold label`

`Gold label` là label chuẩn.

Đây là đáp án mà ta dùng làm mốc đúng.

Nói đơn giản:

- nếu gold label nói đáp án là A
- thì A là đáp án chuẩn để so với các source

### 2.5. `source`

`Source` là nơi sinh ra một lời giải thích.

Ta có thể tưởng tượng:

- mỗi source là một người giải bài theo một phong cách riêng

Ví dụ:

- `causal` thích giải theo nguyên nhân - kết quả
- `contrastive` thích đặt hai khả năng cạnh nhau để so sánh
- `neutral` thường cho kiểu giải thích trung tính

### 2.6. `rationale`

`Rationale` là lời giải thích vì sao một label là đúng.

Ví dụ:

- câu hỏi: “Đáp án nào đúng?”
- label: `entailment`
- rationale: “Vì premise nói trực tiếp điều đó nên hypothesis được suy ra.”

Trong tài liệu này, cứ thấy từ `rationale`, hãy hiểu là:

- lời giải thích

### 2.7. `candidate`

`Candidate` là một ứng viên.

Ở đây, một `candidate` là:

- một rationale
- đi kèm với label mà source đó dự đoán

Ví dụ:

- source `causal` đoán label là `neutral`
- source `causal` viết ra một rationale

Tổ hợp đó là một candidate.

Một example có thể có nhiều candidate vì có nhiều source.

### 2.8. `prediction`

`Prediction` là dự đoán.

Tức là:

- source nghĩ đáp án nào đúng

Prediction có thể đúng hoặc sai.

### 2.9. `vote`

`Vote` là bỏ phiếu.

Hãy tưởng tượng 7 source cùng giơ tay chọn đáp án.

Ví dụ:

- 4 source chọn `entailment`
- 2 source chọn `neutral`
- 1 source chọn `contradiction`

Ta nói:

- `entailment` đang thắng phiếu

### 2.10. `agreement`

`Agreement` là mức đồng ý với nhau.

Nếu nhiều source cùng chọn một label, agreement cao.

Nếu mỗi source chọn một label khác nhau, agreement thấp.

### 2.11. `agreement_count`

`Agreement count` là con số cụ thể của agreement.

Ví dụ:

- có 7 source
- 5 source cùng chọn `entailment`

thì:

- `agreement_count = 5`

### 2.12. `weighted vote`

`Weighted vote` là bỏ phiếu có trọng số.

Không phải source nào cũng được tính giống nhau.

Ví dụ:

- source mạnh được tính 2 điểm
- source trung bình được tính 1 điểm

Khi đó:

- 1 source mạnh có thể “nặng” hơn 1 source yếu

### 2.13. `prior`

`Prior` là mức ưu tiên có sẵn.

Nói dễ hiểu:

- ta tin source nào hơn thì cho source đó prior cao hơn

Ví dụ:

- trong CQA, nếu thực nghiệm thật cho thấy `causal` mạnh nhất
- thì ta có thể cho `causal` prior cao hơn

### 2.14. `vote margin`

`Vote margin` là khoảng cách giữa người thắng và người đứng thứ hai.

Ví dụ:

- đáp án A được 6 điểm
- đáp án B được 1 điểm

thì margin lớn.

Ví dụ khác:

- đáp án A được 4 điểm
- đáp án B được 3 điểm

thì margin nhỏ.

Margin lớn thường có nghĩa:

- hệ thống khá chắc chắn

Margin nhỏ thường có nghĩa:

- hệ thống chưa chắc

### 2.15. `overlap`

`Overlap` là mức độ rationale dùng lại từ trong input.

Ví dụ:

- input có các từ `man`, `guitar`, `stage`
- rationale cũng nói `man`, `guitar`, `stage`

thì overlap cao.

Overlap cao không phải lúc nào cũng tốt tuyệt đối.

Nhưng nhiều khi nó cho thấy:

- rationale đang bám vào dữ liệu gốc

### 2.16. `score`

`Score` là điểm tổng hợp để chấm một candidate.

Một candidate có thể được cộng điểm vì:

- source của nó mạnh
- label của nó được nhiều source khác đồng ý
- vote margin lớn
- rationale dài vừa phải
- rationale bám vào input
- rationale rõ ràng

### 2.17. `filter`

`Filter` là lọc.

Ví dụ:

- giữ những candidate có score cao
- bỏ những candidate có score thấp

### 2.18. `threshold`

`Threshold` là ngưỡng.

Ví dụ:

- chỉ giữ candidate nếu score >= 10

thì số 10 là threshold.

### 2.19. `view`

`View` là một góc nhìn giải thích.

Một example có thể có:

- 1 rationale chính
- hoặc 2 rationales khác nhau

Mỗi rationale đó là một `view`.

### 2.20. `view 1`

`View 1` là rationale chính.

Nó thường là rationale tốt nhất của example đó.

### 2.21. `view 2`

`View 2` là rationale phụ.

Nó chỉ được giữ nếu:

- đủ tốt
- không quá giống `view 1`
- đến từ source khác

### 2.22. `multiview`

`Multiview` nghĩa là:

- một example có thể có nhiều hơn 1 rationale

Trong repo này, thường là:

- `view 1`
- `view 2`

### 2.23. `balanced`

`Balanced` nghĩa là không để bộ dữ liệu cuối bị lệch quá mạnh.

Với ESNLI, điều này thường có nghĩa:

- số lượng `entailment`
- số lượng `neutral`
- số lượng `contradiction`

được làm cho tương đối cân nhau.

Với CQA, nhãn không cố định như ESNLI.

Vì vậy `balanced` trong CQA thường thiên về:

- không để một nguồn chiếm hết dataset
- không để chỉ toàn câu quá dễ
- không để một kiểu example thống trị

### 2.24. `dedupe`

`Dedupe` là xóa trùng.

Ví dụ:

- hai dòng gần như là cùng một rationale cho cùng một example

thì chỉ giữ một dòng tốt hơn.

### 2.25. `diversity`

`Diversity` là sự đa dạng.

Ta không muốn tất cả rationale đều:

- cùng một kiểu viết
- cùng một source
- cùng một góc nhìn

Ta muốn bộ dữ liệu có nhiều kiểu giải thích khác nhau.

### 2.26. `fusion`

`Fusion` là ghép nhiều bộ tốt lại với nhau.

Ví dụ:

- có một pack rất sạch
- có một pack nhiều góc nhìn
- có một pack mạnh ở câu khó

Ta có thể trộn chúng lại.

Đó gọi là `fusion`.

### 2.27. `boundary`

`Boundary` nghĩa là ranh giới.

Ở đây, nó chỉ những example nằm gần ranh giới quyết định.

Ví dụ:

- có câu rất dễ
- có câu rất khó
- có câu lưng chừng, dễ nhầm

Những câu “lưng chừng, dễ nhầm” chính là câu gần boundary.

### 2.28. `bridge`

`Bridge` nghĩa là cây cầu.

Trong tài liệu này, `bridge` là một rationale phụ giúp nối thêm một góc nhìn.

Nói dễ hiểu:

- rationale chính nói một cách
- bridge rationale nói thêm một cách khác

### 2.29. `hard`, `easy`, `boundary`

Đây là 3 nhóm mức độ khó.

`easy`:

- ví dụ dễ
- nhiều source đồng ý
- margin lớn

`boundary`:

- ví dụ không quá dễ
- có chút tranh cãi
- dễ nhầm hơn easy

`hard`:

- ví dụ quá rối
- ít agreement
- margin nhỏ
- dễ đưa nhiễu vào train

---

## 3. ESNLI và CQA khác nhau như thế nào?

Ta cần hiểu điều này trước khi build type.

Nếu không hiểu, rất dễ “copy máy móc” từ dataset này sang dataset kia.

### 3.1. ESNLI

ESNLI là bài toán NLI.

Nó hỏi:

- câu giả thuyết có được suy ra từ câu gốc không?

Nó có 3 nhãn cố định:

- `entailment`
- `neutral`
- `contradiction`

Điều này giúp ta làm nhiều việc dễ hơn:

- có thể cân bằng 3 nhãn
- có thể đặt luật riêng cho từng nhãn
- có thể nói source nào mạnh hơn với nhãn nào

### 3.2. CQA

CQA là bài toán trắc nghiệm.

Nó hỏi:

- trong 5 đáp án, đâu là đáp án đúng?

Nhãn của CQA không cố định thành 3 loại.

Mỗi câu hỏi có 5 lựa chọn riêng.

Nên với CQA:

- ta không thể bê nguyên luật của ESNLI
- ta chỉ giữ được ý tưởng
- còn cách cài đặt phải đổi

### 3.3. Điều gì giữ nguyên giữa hai dataset?

Những thứ giữ nguyên:

- ta vẫn có nhiều source
- ta vẫn có nhiều candidate cho cùng một example
- ta vẫn phải vote
- ta vẫn phải chấm điểm rationale
- ta vẫn phải bỏ rationale yếu
- ta vẫn có thể giữ 1 view hoặc 2 view

### 3.4. Điều gì phải đổi?

Những thứ phải đổi:

- cách định nghĩa label
- cách so gold answer
- cách cân bằng dữ liệu
- cách chọn source nào được ưu tiên hơn

Ví dụ rất quan trọng:

- với CQA, sau khi train thật, `causal` là original type tốt nhất
- nên trong CQA builder mới, ta lấy `causal` làm anchor mạnh

---

## 4. Nguyên liệu cần chuẩn bị trước khi build

### 4.1. Với ESNLI

Ta cần 7 file raw rationale:

- `historical - full.csv`
- `consensus - full.csv`
- `contrastive - full.csv`
- `causal - full.csv`
- `neutral - full.csv`
- `if_else - full.csv`
- `comparative - full.csv`

Ta cũng cần dữ liệu gold của ESNLI để biết đáp án chuẩn.

Các script liên quan chính:

- [build_judge_esnli.py](/Users/huypham/Desktop/Other/distilling-step-by-step/build_judge_esnli.py)
- [build_boundary_focus_pack.py](/Users/huypham/Desktop/Other/distilling-step-by-step/build_boundary_focus_pack.py)
- [build_judge_esnli_method_pack.py](/Users/huypham/Desktop/Other/distilling-step-by-step/build_judge_esnli_method_pack.py)

### 4.2. Với CQA

Ta cần 7 file raw rationale:

- `historical - full.csv`
- `consensus - full.csv`
- `contrastive - full.csv`
- `causal - full.csv`
- `neutral - full.csv`
- `if_else - full.csv`
- `comparative - full.csv`

Ta cần [paper.csv](/Users/huypham/Desktop/Other/distilling-step-by-step/[API]%20CQA/paper.csv) để lấy đáp án chuẩn.

Script liên quan chính:

- [build_judge_cqa_outstanding_pack.py](/Users/huypham/Desktop/Other/distilling-step-by-step/build_judge_cqa_outstanding_pack.py)

---

## 5. Pipeline chung trước khi tách ra từng type

Phần này là bộ xương chung.

4 type khác nhau chủ yếu ở chỗ:

- giữ bao nhiêu rationale
- giữ loại example nào
- có thêm bridge hay không
- có trộn pack khác hay không

Nhưng nền móng ban đầu là giống nhau.

### Bước 1. Chọn một example

Ví dụ ESNLI:

- premise: “A man is playing guitar on stage.”
- hypothesis: “A person is performing music.”
- gold label: `entailment`

Ví dụ CQA:

- question: “He wanted a house that was gated off from other places, where should he start looking?”
- choices: A, B, C, D, E
- gold answer: một trong 5 lựa chọn đó

### Bước 2. Gom tất cả candidate của example đó

Mỗi source có thể cho:

- một predicted label
- một rationale

Ví dụ:

- `historical`: đoán `entailment`, rationale 1
- `contrastive`: đoán `neutral`, rationale 2
- `causal`: đoán `entailment`, rationale 3

Khi đó ta có một “nhóm candidate” cho example này.

### Bước 3. Chuẩn hóa để biết tất cả đang nói về cùng một example

Ta phải chắc chắn rằng:

- premise/hypothesis giống nhau thì là cùng một example
- question/choices giống nhau thì là cùng một example

Nếu không làm kỹ bước này, hệ thống sẽ lầm rằng:

- cùng một câu nhưng là hai câu khác nhau

### Bước 4. Cho các source bỏ phiếu

Ta nhìn xem source nào đang chọn label nào.

Ví dụ:

- 4 source chọn `entailment`
- 2 source chọn `neutral`
- 1 source chọn `contradiction`

Nếu dùng weighted vote, có thể kết quả không chỉ là 4-2-1.

Ví dụ:

- `causal` được 2 điểm
- `historical` được 1.5 điểm
- source yếu được 1 điểm

Khi đó ta cộng theo điểm, không chỉ đếm đầu người.

### Bước 5. Tìm label đang thắng

Sau khi vote xong, ta biết:

- label thắng
- label đứng thứ hai
- vote margin

Đây là tín hiệu rất quan trọng.

Nếu margin lớn:

- ta tự tin hơn

Nếu margin nhỏ:

- ta nên cẩn thận hơn

### Bước 6. Chấm từng rationale

Mỗi rationale được cho một score.

Score thường dựa trên:

- source mạnh hay yếu
- prediction có trùng label đang train không
- agreement cao hay thấp
- margin cao hay thấp
- overlap có hợp lý không
- rationale quá ngắn hay quá dài
- rationale có ghi rõ ý giải thích không

### Bước 7. Bỏ rationale yếu

Nếu rationale:

- điểm quá thấp
- lạc đề
- quá ngắn
- quá giống dòng khác
- đến từ prediction quá mơ hồ

thì bỏ.

### Bước 8. Chọn rationale tốt nhất

Sau khi bỏ những rationale yếu, ta thường chọn rationale mạnh nhất làm `view 1`.

### Bước 9. Nếu type cho phép, chọn rationale phụ

Rationale phụ chỉ được giữ nếu:

- đến từ source khác
- không copy gần như y nguyên rationale chính
- vẫn đủ tốt

Rationale phụ này có thể trở thành:

- `view 2`
- hoặc `bridge`

### Bước 10. Dọn dữ liệu cuối

Ta làm thêm các việc:

- xóa trùng
- giới hạn số rationale trên mỗi example
- cân bằng dữ liệu
- chọn theo source order nếu cần

Sau đó mới ghi ra file cuối.

---

## 6. Type 1: `judge_student_multiview`

## 6.1. Ý tưởng của type này là gì?

Type này dựa trên một ý rất đơn giản:

- một bài toán không nhất thiết chỉ có một cách giải thích tốt

Nếu ta chỉ giữ 1 rationale:

- model học được 1 cách nói

Nếu ta giữ 2 rationales tốt:

- model được học 2 cách hiểu cùng một ví dụ

Nói kiểu trẻ em:

- cùng một bài toán
- cô giáo 1 giải theo cách A
- cô giáo 2 giải theo cách B
- nếu cả hai cách đều đúng và rõ ràng, ta giữ cả hai

### 6.2. Ta muốn gì từ `multiview`?

Ta muốn:

- không bị nhiễu quá nhiều
- nhưng cũng không quá nghèo góc nhìn

Nói cách khác:

- không lấy bừa 2 rationale
- chỉ lấy 2 rationale nếu rationale thứ hai thật sự hữu ích

### 6.3. Cách tạo `judge_student_multiview` trên ESNLI từ đầu

#### Bước 1. Mở 7 file raw source

Mỗi file có các cột kiểu như:

- `premise`
- `hypothesis`
- `rationale`
- `LLM_answer`

Ta đọc cả 7 file vào.

#### Bước 2. Gom các dòng theo cùng một cặp `premise + hypothesis`

Việc này có nghĩa là:

- nếu 7 source đều đang nói về cùng một example
- ta gom 7 dòng của họ vào cùng một nhóm

#### Bước 3. Xem các source đang đoán label gì

Ví dụ:

- `historical` đoán `entailment`
- `causal` đoán `entailment`
- `contrastive` đoán `contradiction`
- `neutral` đoán `entailment`

Lúc này ta thấy:

- `entailment` đang được nhiều người chọn nhất

#### Bước 4. Tính weighted vote

Không phải source nào cũng được tính như nhau.

Builder sẽ có prior để nói rằng:

- source nào mạnh hơn thì được nhiều điểm hơn

Ví dụ đơn giản:

- `historical = 1.2`
- `causal = 1.4`
- `neutral = 1.0`

Nếu `historical` và `causal` cùng chọn `entailment`, label đó sẽ tăng điểm nhanh hơn.

#### Bước 5. Tìm label thắng và margin

Sau bước này, ta biết:

- label nào thắng
- label nào về nhì
- khoảng cách giữa hai label

Ví dụ:

- `entailment = 4.8`
- `neutral = 1.3`
- `contradiction = 0.9`

thì:

- `voted_label = entailment`
- margin lớn

#### Bước 6. Chọn training label

Builder không tin mù quáng vào voted label.

Nó còn nhìn vào gold label.

Các tình huống thường gặp:

- nếu `gold label` trùng `voted_label`, rất tốt
- nếu không trùng nhưng margin rất lớn, có thể cân nhắc voted label
- nếu không trùng và margin nhỏ, ví dụ này mơ hồ, nên bỏ

Nói đơn giản:

- chỉ giữ khi hệ thống có đủ lý do để tin

#### Bước 7. Chấm điểm từng rationale

Mỗi rationale được xem xét riêng.

Ta hỏi:

- rationale này có đi cùng label đúng không?
- source của nó có đáng tin không?
- nhiều source khác có cùng ý không?
- rationale có bám sát premise/hypothesis không?
- rationale có quá ngắn kiểu “because yes” không?
- rationale có quá dài, lan man không?

Từ đó builder cộng điểm.

#### Bước 8. Chọn rationale mạnh nhất làm `view 1`

Sau khi có score, builder lấy rationale tốt nhất.

Đây là rationale chính.

Ta gọi nó là:

- `view 1`

#### Bước 9. Tìm rationale thứ hai làm `view 2`

Builder nhìn các candidate còn lại.

Nhưng không phải thấy candidate nào đứng thứ hai là lấy luôn.

Candidate đó phải thỏa:

- khác source với `view 1`
- không quá giống về câu chữ
- không quá thấp điểm
- không kém `view 1` quá xa

Lý do rất đơn giản:

- nếu rationale thứ hai gần như copy rationale đầu, giữ cũng vô ích
- nếu rationale thứ hai quá yếu, giữ sẽ thêm nhiễu

#### Bước 10. Ghi ra file cuối

Mỗi example có thể có:

- chỉ `view 1`
- hoặc `view 1` và `view 2`

Đây chính là pack:

- `[API] ESNLI/judge_student_multiview - full.csv`

### 6.4. Cách tạo `judge_student_multiview` trên CQA từ đầu

Ý tưởng vẫn giữ nguyên:

- một câu hỏi có thể được học qua 2 rationales

Nhưng cách làm phải đổi theo CQA.

#### Bước 1. Đọc `paper.csv`

Ta lấy:

- câu hỏi
- 5 lựa chọn
- đáp án chuẩn

Đáp án chuẩn này là mốc để đối chiếu.

#### Bước 2. Đọc 7 file raw rationale của CQA

Mỗi source cũng cho:

- một predicted answer
- một rationale

#### Bước 3. Dùng `causal-first` prior

Đây là điểm quan trọng.

Trong CQA, kết quả train thật cho thấy:

- `causal` là original type tốt nhất

Nên builder mới sẽ ưu tiên `causal` hơn các source khác.

Điều này không có nghĩa:

- source khác vô dụng

Nó chỉ có nghĩa:

- nếu hai rationale khá ngang nhau, builder tin `causal` hơn một chút

#### Bước 4. Vote answer

Ta cộng điểm cho từng đáp án.

Source mạnh hơn thì phiếu nặng hơn.

Sau đó ta biết:

- answer thắng
- answer đứng thứ hai
- margin

#### Bước 5. Chấm rationale

Ta xem:

- source này có mạnh không
- rationale có nhắc đúng nội dung câu hỏi không
- rationale có nhắc vào đáp án đúng không
- nhiều source khác có đồng ý không
- margin có đủ lớn không

#### Bước 6. Chọn `view 1`

Rationale tốt nhất được lấy làm view chính.

#### Bước 7. Chọn `view 2`

Rationale phụ chỉ được thêm nếu:

- khác source
- khác wording
- vẫn đủ điểm

#### Bước 8. Dọn và ghi file

Ta xóa trùng và chỉ giữ tối đa 2 rationale cho mỗi question.

Kết quả là:

- `[API] CQA/judge_student_multiview - full.csv`

### 6.5. Tóm tắt thật ngắn về `multiview`

`Multiview` nghĩa là:

- cùng một example
- có thể dạy model bằng 2 lời giải thích tốt

Điểm mạnh:

- phong phú hơn single-view

Rủi ro:

- nếu rationale phụ yếu thì dễ thêm nhiễu

---

## 7. Type 2: `judge_student_boundary_mix_balanced`

## 7.1. Ý tưởng của type này là gì?

Không phải ví dụ nào cũng đáng để cho model học như nhau.

Có ví dụ:

- quá dễ

Có ví dụ:

- vừa vừa, dễ nhầm

Có ví dụ:

- quá rối

`Boundary mix` nói rằng:

- ta không nên chỉ dạy toàn bài quá dễ
- nhưng cũng không nên ôm nhiều bài quá rối

Ta nên lấy:

- bài dễ
- và bài gần ranh giới

Đó là chữ `mix`.

### 7.2. Vì sao không lấy nhiều `hard` example?

Vì hard example thường có vấn đề:

- source cãi nhau nhiều
- margin nhỏ
- rationale kém ổn định

Nếu ta cho quá nhiều hard example vào train:

- model có thể học phải nhiễu

Nói như cho trẻ em:

- bài quá rối thì chưa chắc giúp học tốt

### 7.3. Thế nào là `easy`, `boundary`, `hard`?

#### `easy`

Ví dụ easy thường có:

- nhiều source cùng chọn một đáp án
- margin lớn

Nghĩa là:

- khá chắc

#### `boundary`

Ví dụ boundary thường có:

- vẫn có đáp án trội hơn
- nhưng không quá áp đảo

Nghĩa là:

- đây là vùng model dễ nhầm
- nhưng vẫn còn đủ tín hiệu để học

#### `hard`

Ví dụ hard thường có:

- source chia phiếu mạnh
- margin nhỏ
- khó quyết định ai đúng

### 7.4. Cách tạo `judge_student_boundary_mix_balanced` trên ESNLI

#### Bước 1. Gom các candidate theo example

Giống `multiview`.

Ta gom tất cả source của cùng một cặp `premise + hypothesis`.

#### Bước 2. Dùng prior riêng cho boundary

Builder không nhất thiết dùng đúng prior như pack khác.

Với boundary family, builder dùng prior được chỉnh để:

- chú ý hơn đến source mạnh cho từng nhãn

Ví dụ:

- có source mạnh hơn cho `contradiction`
- có source khác mạnh hơn cho `neutral`

#### Bước 3. Vote label

Ta tính label thắng và margin.

#### Bước 4. Chọn training label

Builder so giữa:

- gold label
- voted label

Nếu hai bên khá khớp hoặc voted side rất mạnh:

- giữ example

Nếu quá mơ hồ:

- bỏ example

#### Bước 5. Chấm từng rationale

Điểm rationale thường có các phần như:

- source prior
- overlap
- agreement
- margin
- độ dài hợp lý
- cue giải thích rõ

#### Bước 6. Gán band cho example

Builder nhìn agreement và margin để quyết định:

- `easy`
- `boundary`
- hay `hard`

Ví dụ:

- agreement rất cao, margin rất lớn -> `easy`
- agreement vừa, margin vừa -> `boundary`
- agreement thấp, margin thấp -> `hard`

#### Bước 7. Chọn rationale mạnh nhất

Type này chỉ cần một rationale chính cho mỗi example.

Nó không cần rationale phụ.

#### Bước 8. Chỉ giữ `easy` và `boundary`

Đây là linh hồn của type này.

Ta bỏ `hard`.

#### Bước 9. Cân bằng dữ liệu

ESNLI có 3 nhãn cố định.

Nên sau khi lọc xong, builder sẽ cố làm cho:

- số dòng `entailment`
- số dòng `neutral`
- số dòng `contradiction`

không lệch nhau quá mạnh.

#### Bước 10. Ghi file

Kết quả là:

- `[API] ESNLI/judge_student_boundary_mix_balanced - full.csv`

### 7.5. Cách tạo `judge_student_boundary_mix_balanced` trên CQA

Ý tưởng giống ESNLI:

- lấy easy
- lấy boundary
- bỏ hard

Nhưng vì CQA không có 3 nhãn cố định, cách làm cần đổi.

#### Bước 1. Vote answer theo `causal-first`

Ta dùng source prior mới của CQA.

Vì kết quả train thật nói rằng:

- `causal` là baseline mạnh nhất trong các original type

#### Bước 2. Chấm rationale

Ta xem:

- rationale có bám câu hỏi và đáp án không
- source có mạnh không
- margin có đủ lớn không
- agreement có đủ không

#### Bước 3. Gán band

Ta chia example thành:

- easy
- boundary
- hard

dựa trên agreement và margin.

#### Bước 4. Chỉ giữ easy và boundary

Đây là chữ `mix`.

Ta trộn:

- ví dụ chắc chắn
- và ví dụ gần ranh giới

#### Bước 5. Chỉ giữ 1 rationale mỗi example

Type này không lấy bridge rationale.

#### Bước 6. Xóa trùng và dọn file

Ta dedupe và giữ các dòng tốt hơn theo source order.

Kết quả là:

- `[API] CQA/judge_student_boundary_mix_balanced - full.csv`

### 7.6. Tóm tắt thật ngắn về `boundary_mix`

`Boundary mix` nghĩa là:

- không chỉ học ví dụ dễ
- cũng học ví dụ gần vùng dễ nhầm
- bỏ ví dụ quá rối

Điểm mạnh:

- dạy model về vùng quan trọng hơn vùng “quá hiển nhiên”

Rủi ro:

- nếu ranh giới đặt không đúng, có thể bỏ mất ví dụ hữu ích

---

## 8. Type 3: `judge_student_boundary_bridge_balanced`

## 8.1. Ý tưởng của type này là gì?

Type này bắt đầu từ ý tưởng của `boundary_mix`.

Nó vẫn quan tâm đến:

- easy
- boundary
- bỏ bớt hard

Nhưng nó thêm một ý mới:

- với các ví dụ không quá rối, nếu có một rationale phụ tốt, ta giữ thêm rationale đó

Rationale phụ này gọi là:

- `bridge`

### 8.2. Vì sao gọi là `bridge`?

Vì nó giống cây cầu nối thêm cách hiểu.

Rationale chính nói:

- “đây là đáp án đúng vì lý do A”

Bridge rationale nói:

- “cũng có thể hiểu theo lý do B”

Nói như cho trẻ em:

- thầy giáo giải bài một cách
- cô giáo giải lại bằng cách khác
- hai cách cùng giúp hiểu sâu hơn

### 8.3. Khi nào bridge có ích?

Bridge có ích nhất ở ví dụ:

- không quá dễ
- không quá rối
- có nhiều hơn một cách giải thích hợp lý

Bridge ít có ích khi:

- rationale phụ quá yếu
- rationale phụ quá giống rationale chính
- ví dụ quá hard

### 8.4. Cách tạo `judge_student_boundary_bridge_balanced` trên ESNLI

#### Bước 1. Làm tất cả các bước đầu giống `boundary_mix`

Tức là:

- gom candidate
- vote label
- chọn training label
- chấm rationale
- gán band

#### Bước 2. Chọn rationale chính

Rationale mạnh nhất được chọn làm rationale chính.

#### Bước 3. Tìm bridge rationale

Builder nhìn các rationale còn lại.

Nó hỏi:

- rationale này có đến từ source khác không?
- rationale này có quá giống rationale chính không?
- rationale này có đủ điểm không?
- rationale này có quá yếu so với rationale chính không?

Nếu câu trả lời ổn, rationale đó có thể trở thành bridge.

#### Bước 4. Chỉ thêm bridge cho example không quá hard

Nếu example là:

- `easy`
- hoặc `boundary`

thì có thể cho bridge vào.

Nếu example là:

- `hard`

thì không thêm bridge.

Lý do:

- hard đã rối sẵn
- thêm rationale phụ vào có thể làm dataset rối hơn nữa

#### Bước 5. Giới hạn tối đa 2 rationale mỗi example

Một example tối đa có:

- rationale chính
- bridge rationale

#### Bước 6. Cân bằng theo 3 nhãn

Giống `boundary_mix`, ESNLI vẫn có 3 nhãn cố định.

Nên builder làm balanced ở bước cuối.

#### Bước 7. Ghi file

Kết quả là:

- `[API] ESNLI/judge_student_boundary_bridge_balanced - full.csv`

### 8.5. Cách tạo `judge_student_boundary_bridge_balanced` trên CQA

#### Bước 1. Làm lại pipeline đầu của CQA

Bao gồm:

- đọc `paper.csv`
- gom candidate theo question
- vote answer theo `causal-first`
- chấm rationale
- gán easy / boundary / hard

#### Bước 2. Chọn rationale chính

Ta lấy rationale mạnh nhất trước.

#### Bước 3. Tìm bridge rationale

Rationale phụ phải:

- khác source
- không quá giống wording
- không thấp điểm quá

#### Bước 4. Chỉ dùng bridge cho easy hoặc boundary

Nếu example hard:

- bỏ bridge

#### Bước 5. Dedupe và giới hạn số rationale

Mỗi example tối đa 2 rationale.

#### Bước 6. Ghi file

Kết quả là:

- `[API] CQA/judge_student_boundary_bridge_balanced - full.csv`

### 8.6. Tóm tắt thật ngắn về `boundary_bridge`

`Boundary bridge` nghĩa là:

- lấy ví dụ dễ và ví dụ gần ranh giới
- rồi, nếu hợp lý, thêm một rationale phụ để giúp hiểu sâu hơn

Điểm mạnh:

- giữ được cả độ chắc và độ giàu góc nhìn

Rủi ro:

- nếu bridge chọn không kỹ, bridge có thể trở thành nhiễu

---

## 9. Type 4: `judge_expert_hybrid_fusion_balanced`

## 9.1. Ý tưởng của type này là gì?

Ba type trước chủ yếu đi từ raw source.

Type này khác.

Nó nói:

- thay vì chọn trực tiếp từ “nguyên liệu thô”
- ta hãy chọn từ những “bộ tốt đã được tuyển trước”

Nói như cho trẻ em:

- thay vì tuyển từng học sinh từ nhiều lớp
- ta chọn từ những đội tuyển đã mạnh sẵn

### 9.2. Tách nhỏ tên type để dễ hiểu

#### `expert`

`Expert` nghĩa là:

- tin hơn vào những pack hoặc source đã chứng minh là mạnh

#### `hybrid`

`Hybrid` nghĩa là:

- trộn nhiều kiểu dữ liệu
- không chỉ một phong cách duy nhất

Ví dụ:

- một pack rất sạch
- một pack nhiều góc nhìn
- một pack mạnh ở vùng khó

#### `fusion`

`Fusion` nghĩa là:

- ghép các pack đó lại thành một pack mới

#### `balanced`

`Balanced` nghĩa là:

- sau khi ghép xong, vẫn phải dọn lại để bộ dữ liệu cuối không bị lệch và không bị một nhóm chiếm hết

### 9.3. Cách tạo `judge_expert_hybrid_fusion_balanced` trên ESNLI

#### Bước 1. Chuẩn bị các base pack mạnh

Builder ESNLI dùng các pack nền như:

- `judge_student_multiview_hybrid_balanced`
- `judge_label_expert_guarded`
- `judge_student_singleview_superclean_balanced`

Ta cần hiểu vì sao chọn chúng:

- `multiview_hybrid_balanced`: có đa góc nhìn
- `label_expert_guarded`: có tư duy ưu tiên theo nhãn
- `singleview_superclean_balanced`: rất sạch

Tức là:

- một pack cho diversity
- một pack cho expert signal
- một pack cho precision

#### Bước 2. Đọc cả 3 pack vào cùng một bảng lớn

Sau khi nạp, mỗi dòng cần biết:

- nó đến từ pack nào
- example nào
- rationale nào
- score gốc là bao nhiêu

#### Bước 3. Tạo `quality_score`

Builder không chỉ tin đúng một cột score cũ.

Nó tính một điểm chất lượng phụ bằng cách nhìn thêm:

- agreement
- margin
- đây là `view 1` hay `view 2`
- pack nguồn mạnh hay yếu

Nói đơn giản:

- builder muốn so rationale công bằng hơn

#### Bước 4. Cộng bonus theo pack

Không phải pack nào cũng đáng tin như nhau.

Ví dụ:

- `singleview_superclean` có thể được thưởng thêm vì rất sạch
- `multiview_hybrid` được thưởng theo kiểu khác vì nhiều góc nhìn

Đây là lý do trong tên có chữ `expert`.

Nó không phải “expert” theo nghĩa huyền bí.

Nó chỉ có nghĩa:

- builder đang chủ động tin nhiều hơn vào các pack đã có lý do tốt để được tin

#### Bước 5. Đổ tất cả vào cùng một “hồ lớn”

Tưởng tượng mỗi pack là một rổ trái cây ngon.

Ta đổ cả ba rổ vào một chỗ.

Nhưng chưa lấy ăn ngay.

Ta còn phải chọn tiếp.

#### Bước 6. Xóa trùng

Có thể cùng một example, cùng một rationale xuất hiện ở hơn một pack.

Ta không muốn giữ trùng.

Nên builder dedupe:

- rationale nào tốt hơn thì giữ
- rationale nào yếu hơn thì bỏ

#### Bước 7. Chọn file cuối theo chất lượng và đa dạng

Builder không chỉ lấy top score từ đầu đến cuối.

Vì nếu làm vậy:

- có thể một source chiếm quá nhiều
- có thể một example xuất hiện quá nhiều kiểu gần giống nhau

Nên builder còn kiểm tra:

- đừng để 1 source chiếm hết
- đừng để 1 example lặp quá dày
- vẫn phải giữ diversity hợp lý

#### Bước 8. Cân bằng và ghi file

Sau cùng, ta dọn lại để file cuối gọn và hợp lý.

Kết quả là:

- `[API] ESNLI/judge_expert_hybrid_fusion_balanced - full.csv`

### 9.4. Cách tạo `judge_expert_hybrid_fusion_balanced` trên CQA

#### Bước 1. Chuẩn bị các base pack mạnh của CQA

Ta cần trước các pack như:

- `judge_student_multiview_hybrid_balanced`
- `judge_student_singleview_superclean_balanced`
- `judge_student_multiview`

Đây là những pack đã được builder CQA sinh ra trước đó.

#### Bước 2. Chọn phần nào của từng pack để trộn

Ví dụ:

- lấy `hybrid` view 1 vì đây thường là rationale chính mạnh
- lấy `superclean` vì rất sạch
- lấy phần mạnh từ `multiview` để giữ thêm diversity

#### Bước 3. Tính điểm mới sau khi fusion

Builder sẽ có:

- score gốc
- cộng với bonus tùy pack

Nói đơn giản:

- pack nào được tin hơn sẽ được cộng thêm chút lợi thế

#### Bước 4. Ưu tiên theo `causal-first`

Vì trong CQA, `causal` là original type tốt nhất sau train thật, builder hiện tại dùng:

- `causal-first` source order

Điều này giúp builder phù hợp với thực nghiệm thật của CQA hơn là proxy cũ.

#### Bước 5. Dedupe

Nếu hai pack cùng mang đến rationale gần như giống nhau:

- giữ bản mạnh hơn

#### Bước 6. Giới hạn số rationale trên mỗi example

Ta không muốn một question có quá nhiều rationale.

Nên builder giữ tối đa:

- 2 rationale mỗi example

#### Bước 7. Ghi file

Kết quả là:

- `[API] CQA/judge_expert_hybrid_fusion_balanced - full.csv`

### 9.5. Tóm tắt thật ngắn về `expert_hybrid_fusion`

`Expert hybrid fusion` nghĩa là:

- không lấy trực tiếp từ source thô
- mà lấy từ các pack mạnh đã xây trước
- rồi trộn lại, thưởng pack mạnh hơn, xóa trùng, và dọn thành một pack mới

Điểm mạnh:

- tận dụng điểm hay của nhiều pack cùng lúc

Rủi ro:

- nếu trộn không khéo, pack mới có thể bị “nửa nọ nửa kia”

---

## 10. Nếu muốn dựng lại từ số 0, nên làm theo thứ tự nào?

Phần này rất thực tế.

Nếu hoàn toàn bắt đầu lại, nên đi như sau.

### 10.1. Với ESNLI

1. Chuẩn bị 7 file raw rationale `*- full.csv`.
2. Chuẩn bị nguồn gold label.
3. Viết bước gom candidate theo `premise + hypothesis`.
4. Viết bước vote label và tính margin.
5. Viết bước chấm rationale.
6. Từ pipeline đó, build `judge_student_multiview`.
7. Dùng cùng pipeline, thêm bước gán `easy / boundary / hard`.
8. Từ đó build `judge_student_boundary_mix_balanced`.
9. Thêm bước chọn rationale phụ để build `judge_student_boundary_bridge_balanced`.
10. Sau khi đã có các pack mạnh khác, build `judge_expert_hybrid_fusion_balanced`.

### 10.2. Với CQA

1. Chuẩn bị 7 file raw rationale `*- full.csv`.
2. Chuẩn bị [paper.csv](/Users/huypham/Desktop/Other/distilling-step-by-step/[API]%20CQA/paper.csv).
3. Chuẩn hóa key theo `question + choices`.
4. Dùng `causal-first` prior để vote answer.
5. Viết bước chấm rationale theo logic CQA.
6. Build `judge_student_multiview`.
7. Thêm bước chia `easy / boundary / hard` để build `judge_student_boundary_mix_balanced`.
8. Thêm bridge rationale để build `judge_student_boundary_bridge_balanced`.
9. Dùng các pack mạnh đó làm đầu vào cho `judge_expert_hybrid_fusion_balanced`.

---

## 11. So sánh 4 type bằng ngôn ngữ cực dễ hiểu

### `judge_student_multiview`

Một bài có thể có 2 lời giải thích tốt.

Ta giữ cả 2 nếu lời giải thích thứ hai thật sự có ích.

### `judge_student_boundary_mix_balanced`

Ta lấy bài dễ và bài gần vùng dễ nhầm.

Ta bỏ bài quá rối.

### `judge_student_boundary_bridge_balanced`

Giống `boundary_mix`, nhưng nếu có một lời giải thích phụ tốt thì giữ thêm.

### `judge_expert_hybrid_fusion_balanced`

Ta không chọn từ đầu vào thô nữa.

Ta lấy từ những pack đã mạnh sẵn rồi trộn lại một cách có chọn lọc.

---

## 12. Bản tóm tắt một câu cho từng type

`judge_student_multiview`

- Mỗi example có thể dạy model bằng 2 lời giải thích tốt thay vì chỉ 1.

`judge_student_boundary_mix_balanced`

- Chỉ lấy ví dụ chắc chắn và ví dụ gần ranh giới, bỏ ví dụ quá rối.

`judge_student_boundary_bridge_balanced`

- Giữ ví dụ chắc chắn và gần ranh giới, rồi thêm một rationale phụ nếu nó giúp hiểu sâu hơn.

`judge_expert_hybrid_fusion_balanced`

- Trộn những phần mạnh nhất từ nhiều pack mạnh khác nhau để tạo pack mới.

---

## 13. Điều cốt lõi nhất phải nhớ

Điều quan trọng nhất của cả 4 type không phải là:

- càng nhiều rationale càng tốt

Điều quan trọng thật sự là:

- chọn rationale nào đáng để dạy model

Nói rất ngắn:

- `multiview` chọn nhiều góc nhìn
- `boundary_mix` chọn vùng quan trọng
- `boundary_bridge` chọn vùng quan trọng và thêm cầu nối
- `expert_hybrid_fusion` chọn cái hay nhất từ nhiều bộ tốt

Nếu phải nhớ một ý duy nhất, hãy nhớ ý này:

- chúng ta không cố thu thập mọi lời giải thích
- chúng ta cố chọn những lời giải thích giúp model học tốt nhất
