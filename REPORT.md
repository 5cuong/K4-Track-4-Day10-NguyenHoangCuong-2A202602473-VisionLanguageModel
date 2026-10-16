# Báo cáo Lab Day 10 — Qwen3-VL-4B grounding hư hại mặt đường

Họ tên: Nguyen Hoang Cuong  
MSSV: 2A202602473  
GPU huấn luyện: NVIDIA A100-SXM4-40GB  
Dataset version: d10-21e4070301

## Bảng tổng hợp kết quả đo trên test

E0 không huấn luyện; VRAM và thời gian huấn luyện không áp dụng. Các thông số E1–E4 lấy từ manifest của các lần chạy thật. F1/CI/FP mỗi ảnh và F1 lớp lấy từ evidence của scorer trên cùng 311 ảnh test.

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút train | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| E0 | Zero-shot | 0 | — | — | 0.0887 | [0.0630, 0.1156] | 0.897 | 0.1088 | 0.0321 | 0.1868 |
| E1 | LoRA LLM, rank 16 | 33,030,144 | 29.201 | 11.236 | 0.2629 | [0.2290, 0.3031] | 1.206 | 0.1827 | 0.1821 | 0.5271 |
| E2 | E1 + projector đầy đủ | 142,135,296 | 13.388 | 36.132 | 0.3482 | [0.3075, 0.3886] | 0.932 | 0.2525 | 0.2664 | 0.5836 |
| E3 | E2 + LoRA ViT, rank 16 | 148,426,752 | 10.132 | 45.520 | 0.4254 | [0.3832, 0.4680] | 0.875 | 0.2771 | 0.3480 | 0.6993 |
| E4 rank 4 | E3, rank 4 | 118,935,552 | 23.732 | 13.435 | 0.4078 | [0.3686, 0.4499] | 1.013 | 0.3180 | 0.3121 | 0.6758 |
| E4 rank 64 | E3, rank 64 | 266,391,552 | 28.316 | 13.946 | 0.4278 | [0.3884, 0.4688] | 0.961 | 0.2996 | 0.3645 | 0.6643 |

## Trả lời câu hỏi

### Câu 1 — Kiến trúc và tham số

Đếm từ các tensor trong safetensors của đúng base model dùng trong phiên chạy: language model 4,022,468,096 tham số; vision encoder 306,242,560; projector (merger và deepstack merger) 109,105,152; tổng 4,437,815,808. E1 cập nhật LoRA ở LLM, giữ phần nhìn đóng băng. E2 mở projector đầy đủ ngoài LoRA LLM. E3 thêm LoRA ở các khối attention/MLP của ViT. Số tham số thực sự trainable của từng setup được ghi trong bảng.

### Câu 2 — E0 zero-shot

E0 có parse_valid_rate 0.8328, F1 0.0887, với 40 TP, 279 FP và 543 FN. Pattern lỗi đo được rõ nhất là bỏ sót box (FN cao hơn TP nhiều) và chất lượng định vị/phân lớp crack yếu: crack có 8 TP, 191 FP, 291 FN, F1 0.0321. File raw work/e0.jsonl không còn trong runtime hoặc gói khôi phục nên không thể xác minh hai dạng lỗi văn bản phổ biến hay trích output thô; tôi không dựng ví dụ giả. Số liệu này là thống kê thật trong report/evidence/e0.json.

### Câu 3 — Thang tọa độ

Output huấn luyện/chấm chuẩn dùng bbox 0–1000. Khi dùng cùng dự đoán E3 nhưng cố ý yêu cầu scorer diễn giải chúng như pixel, F1 giảm từ 0.4254 xuống 0.0000 (311 ảnh, tất cả lớp F1 bằng 0). Scorer chuẩn hóa pixel bằng chiều rộng/cao ảnh rồi đưa về 0–1000; diễn giải tọa độ 0–1000 như pixel làm box co sai tỷ lệ nên IoU không đạt ngưỡng. Đây là phép kiểm tra đối chứng trên prediction E3, không phải lần huấn luyện mới.

### Câu 4 — Assistant-only loss

Loss chỉ tính trên câu trả lời assistant để mô hình học sinh JSON grounding đích, không học bắt chước prompt người dùng hoặc token ảnh. Nếu tính loss trên toàn chuỗi, phần lớn token prompt/ảnh không phải nội dung cần sinh sẽ chi phối gradient và làm giảm tín hiệu học cho nhãn, lớp và tọa độ. Pipeline kiểm tra target assistant kết thúc bằng im_end.

### Câu 5 — Projector và lượng tử hóa

Projector cần là tham số trainable đầy đủ; nếu bị lượng tử hóa 4-bit thì tensor không còn dạng trọng số floating-point thông thường để cập nhật đầy đủ bằng optimizer. Cấu hình chạy đã bỏ lượng tử hóa các module visual (manifest ghi vision_quantized=false); mã cast tham số trainable của projector sang float32 và lưu projector riêng cùng adapter. Khi inference, projector được nạp lại; nếu chỉ nạp LoRA thì E2/E3 sẽ mất phần cập nhật projector.

### Câu 6 — Chia split theo phiên quay

Chia ảnh ngẫu nhiên có thể đưa khung hình gần như trùng bối cảnh/đoạn đường từ cùng phiên quay vào cả train và test, làm test dễ hơn và F1 lạc quan. Dữ liệu được chia theo 18 session không giao nhau: train 1,422 ảnh (pothole 1,074; crack 1,750; manhole 691 box), validation 276 ảnh (66; 470; 103), test 311 ảnh (121; 299; 163).

### Câu 7 — So sánh E1/E2/E3

E1 đạt F1 0.2629; E2 tăng lên 0.3482 (+0.0852), F1 manhole tăng 0.5271 lên 0.5836. E3 đạt 0.4254 (+0.0773 so với E2) và có F1 cao nhất ở cả test tổng thể lẫn crack/manhole. Kết quả ủng hộ việc mở projector và thêm LoRA ViT trên split này; tuy nhiên CI95 của E2 [0.3075, 0.3886] và E3 [0.3832, 0.4680] còn chồng lấn, nên chưa đủ kết luận chênh lệch chắc chắn. Chi phí đo được lần lượt là E1 11.236 phút/29.201 GB, E2 36.132 phút/13.388 GB, E3 45.520 phút/10.132 GB. A100 40GB là GPU chạy các số liệu này.

### Câu 8 — Độ rộng khoảng tin cậy

Độ rộng CI95 là E0 0.0526, E1 0.0741, E2 0.0810, E3 0.0847. Với 311 ảnh, sai số lấy mẫu còn đáng kể; CI của E1 và E2 chỉ tách nhẹ, còn E2/E3 chồng lấn. Vì vậy nên xem thứ hạng F1 là bằng chứng định hướng trên test hiện có, không phải bảo đảm rằng E3 tốt hơn trong mọi mẫu dữ liệu.

### Câu 9 — Khác biệt giữa lớp

Ở setup tốt nhất E3, pothole là lớp yếu nhất (F1 0.2771; 32 TP, 78 FP, 89 FN), tiếp theo crack 0.3480 và manhole 0.6993. Test có 121 pothole box so với 299 crack và 163 manhole; ít ví dụ hơn có thể làm mô hình khó bao phủ biến thiên. Kích thước/độ tương phản và nhập nhằng ngữ nghĩa là giả thuyết cần kiểm tra thêm, không được xác nhận chỉ từ bảng tổng hợp. Hình overlay thực tế của sáu ảnh test nằm tại report/figures/best_validation_setup_test_examples.png.

### Câu 10 — E4 rank ablation

E4 chỉ thay rank LoRA quanh E3 rank 16: rank 4 cho F1 0.4078 [0.3686, 0.4499], rank 64 cho 0.4278 [0.3884, 0.4688], so với E3 0.4254 [0.3832, 0.4680]. Các CI chồng lấn; rank 64 chỉ hơn E3 0.0024 nhưng dùng 266.4M tham số trainable và 28.316 GB VRAM so với 148.4M và 10.132 GB của E3. Chưa có bằng chứng đáng tin rằng tăng rank cải thiện kết quả đủ bù chi phí.

## Nhìn lại

Điểm cần cải thiện là giữ lại work/e0.jsonl cùng với evidence để có thể phân loại lỗi định dạng và trích ví dụ output thô. E0 vẫn chấm được từ evidence đã lưu, nhưng không thể tái kiểm tra văn bản gốc. Trên test E3, pothole còn nhiều FN; nên thu thập thêm ví dụ/session độc lập và phân tích kích thước box trước khi kết luận nguyên nhân.
