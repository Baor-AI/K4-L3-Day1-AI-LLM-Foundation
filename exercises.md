# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Ở temperature 0.0, model gần như luôn trả về cùng một sự thật (ví dụ luôn nói
> về diện tích hoặc dân số) vì nó luôn chọn token có xác suất cao nhất — output
> gần như xác định (deterministic). Khi tăng dần lên 0.5 rồi 1.0, các câu trả
> lời bắt đầu đa dạng hơn về nội dung lẫn cách diễn đạt (đôi khi kể về ẩm thực,
> lịch sử, địa lý...). Ở 1.5, phản hồi có thể trở nên kém mạch lạc hơn hoặc lạc
> đề, vì model lấy mẫu cả những token có xác suất thấp mà bình thường sẽ không
> chọn.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> đặt temperature khoảng 0.2–0.3 cho chatbot hỗ trợ khách hàng. Nhóm
> use case này cần câu trả lời phải nhất quán, đúng chính sách và dễ kiểm soát 
> temperature thấp giảm rủi ro model "sáng tạo" ra thông tin sai hoặc trả lời
> khác nhau cho cùng một câu hỏi giữa các lượt, điều rất quan trọng khi liên
> quan đến chính sách hoàn tiền, bảo hành hay hướng dẫn kỹ thuật.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**

> Theo bảng giá trong `template.py`: GPT-4o = $0.0025/1K input, $0.010/1K
> output; GPT-4o-mini = $0.00015/1K input, $0.0006/1K output. Tỷ lệ giá ở cả
> input lẫn output đều giống nhau ≈ **16,7 lần** — nên bất kể
> tỷ lệ input/output ra sao, GPT-4o luôn đắt hơn GPT-4o-mini khoảng 16–17 lần
> cho cùng một workload.
> Với kịch bản 10.000 người dùng × 3 lượt/ngày × 350 token output/lượt =
> 10.500.000 token output/ngày:
> - GPT-4o: 10.500 × $0.010 ≈ **$105/ngày** (chỉ tính output)
> - GPT-4o-mini: 10.500 × $0.0006 ≈ **$6.3/ngày**
> → chênh lệch khoảng **$98,7/ngày** (~$36.000/năm) chỉ riêng phần output.
>
> **Dùng GPT-4o** khi câu trả lời cần độ chính xác cao, suy luận phức tạp hoặc
> ảnh hưởng trực tiếp đến quyết định quan trọng — ví dụ tư vấn pháp lý/y tế sơ
> bộ, phân tích hợp đồng, nơi một câu trả lời sai gây thiệt hại lớn hơn nhiều
> lần chi phí API tiết kiệm được.
> **Dùng GPT-4o-mini** khi tác vụ đơn giản, lặp lại và có thể tolerate sai số
> nhỏ — ví dụ phân loại intent, trả lời FAQ cơ bản, tóm tắt ngắn — nơi khối
> lượng gọi API lớn khiến chi phí là yếu tố quyết định hơn chất lượng biên.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Hai phản hồi khác nhau rõ rệt dù cùng một câu hỏi. Với persona "giáo viên
> tiểu học", câu trả lời thường ngắn gọn hơn, dùng từ vựng đơn giản, tránh
> thuật ngữ chuyên môn, và hay chêm ví dụ đời thường dễ hình dung (ví dụ so
> sánh blockchain như "một cuốn sổ mà ai cũng có bản sao và không ai xóa
> được"). Với persona "chuyên gia tài chính", câu trả lời dài hơn, dùng nhiều
> thuật ngữ kỹ thuật (sổ cái phân tán, đồng thuận, băm mật mã, bất biến...),
> đi sâu vào cơ chế và ứng dụng thực tế thay vì chỉ giải thích khái niệm.
> Điều này cho thấy system prompt không chỉ đổi "giọng văn" bề mặt mà thực sự
> định hình lại mức độ chi tiết, từ vựng và cách tổ chức nội dung của model —
> cùng một kiến thức nền nhưng được "lọc" qua vai trò và đối tượng người đọc
> được chỉ định.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Với đoạn văn tiếng Việt ~100 từ, số token đếm bằng `tiktoken` thường cao hơn
> ước lượng "số từ / 0.75" khoảng 30–60%, tùy vào tỷ lệ từ có dấu và độ dài từ
> trong đoạn văn. Nguyên nhân là bộ mã hóa BPE của OpenAI được huấn luyện chủ
> yếu trên dữ liệu tiếng Anh, nên nó có sẵn nhiều token "trọn từ" cho các từ
> tiếng Anh phổ biến; còn tiếng Việt có dấu thanh và nguyên âm ghép (ví dụ
> "nghiêng", "được", "những") thường không khớp với các cụm ký tự đã được học
> sẵn, nên bị tách nhỏ thành nhiều token con (từng âm tiết, thậm chí từng ký
> tự với dấu) thay vì gộp thành một token duy nhất như một từ tiếng Anh cùng
> độ dài.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng nhất khi câu trả lời dài và người dùng đang chờ trong
> một giao diện tương tác trực tiếp (chatbot, trợ lý ảo) — thấy chữ xuất hiện
> ngay giúp cảm giác "model đang suy nghĩ và phản hồi" thay vì màn hình đứng
> im rồi bung cả đoạn văn dài, giảm cảm giác chờ đợi dù tổng thời gian xử lý
> không đổi. Ngược lại, non-streaming phù hợp hơn khi ứng dụng cần xử lý toàn
> bộ output trước khi dùng (ví dụ parse JSON, chạy qua bộ kiểm duyệt nội dung,
> hoặc gọi API nền không có giao diện hiển thị theo thời gian thực) — lúc đó
> stream từng phần chỉ thêm độ phức tạp mà không mang lại lợi ích gì.
### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> Với delay cố định, nếu server đang quá tải, tất cả client sẽ đồng loạt thử
> lại sau đúng 1 khoảng thời gian như nhau — tạo ra các đợt "sóng" request
> dồn dập lặp lại đúng chu kỳ, khiến server không kịp phục hồi và có thể sập
> hẳn (gọi là "thundering herd"). Exponential backoff giãn thời gian chờ ra
> ngày càng dài sau mỗi lần thất bại (0.1s → 0.2s → 0.4s...), giúp giảm dần
> mật độ request dồn vào server đang gặp sự cố, cho nó thời gian hồi phục,
> đồng thời tránh lãng phí tài nguyên client vào những lần retry gần như chắc
> chắn thất bại ngay sau khi vừa lỗi.
---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> Mình chọn persona: "Bạn là trợ giảng thân thiện của khóa học AI, trả lời
> ngắn gọn bằng tiếng Việt, ưu tiên ví dụ thực tế thay vì lý thuyết dài dòng."
>
> Hai lựa chọn từ ngữ quan trọng:
> - **"trả lời ngắn gọn"**: vì đây là trợ lý CLI chạy trong terminal, không
>   có định dạng đẹp như giao diện web — câu trả lời dài dễ gây rối mắt và
>   tốn thêm token output (tăng chi phí + độ trễ) mà không tăng tương xứng
>   giá trị cho người học đang cần câu trả lời nhanh, dễ nắm ý chính.
> - **"bằng tiếng Việt"**: chỉ định rõ ngôn ngữ để tránh model tự chuyển sang
>   tiếng Anh khi gặp thuật ngữ kỹ thuật (một hành vi phổ biến của các model
>   đa ngôn ngữ khi câu hỏi chứa từ vay mượn như "token", "API"), đảm bảo
>   trải nghiệm nhất quán cho người học Việt Nam.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế lớn nhất hiện tại: **history chỉ giữ 3 lượt gần nhất**, nên nếu
> người dùng hỏi lại điều gì đó đã đề cập từ 4–5 lượt trước, trợ lý sẽ "quên"
> hoàn toàn ngữ cảnh đó và có thể trả lời mâu thuẫn hoặc phải hỏi lại thông
> tin người dùng đã cung cấp.
>
> Cải thiện đề xuất: thêm một bước **tóm tắt hội thoại (summarization)** khi
> history sắp bị cắt — trước khi xóa các lượt cũ nhất, gọi model tóm tắt
> chúng thành 1–2 câu ngắn và lưu vào một biến `conversation_summary` riêng,
> rồi chèn summary đó vào đầu system prompt ở các lượt tiếp theo. Cách này
> giữ được ngữ cảnh dài hạn quan trọng mà không cần gửi lại toàn bộ history
> gốc (vốn sẽ làm input token tăng vô hạn và tốn chi phí theo từng lượt).

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
