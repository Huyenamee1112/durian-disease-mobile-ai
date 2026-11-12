# DINOv3 — thử nghiệm recipe gần DINOv2, 3 seed

Dataset field: 480 train / 64 validation / 68 test, không thay split.
224px, batch 8, fine-tune 2 block, 20 epoch tối đa, patience 5.
Cùng augmentation, normalization, optimizer/lr và plateau scheduler như code DINOv2.
Mỗi seed chọn checkpoint theo validation macro-F1; báo toàn bộ kết quả test.

| Seed | Best epoch | Validation macro-F1 | Test accuracy | Test macro-F1 | Đúng / 68 |
|---|---:|---:|---:|---:|---:|
| 42 | 14 | 77.69% | 85.29% | 85.32% | 58/68 |
| 43 | 11 | 78.69% | 83.82% | 83.87% | 57/68 |
| 44 | 15 | 76.24% | 85.29% | 84.93% | 58/68 |

Trung bình accuracy: **84.80%**, độ lệch chuẩn mẫu
**0.85 điểm phần trăm**.
Trung bình macro-F1: **84.71%**, độ lệch chuẩn mẫu
**0.75 điểm phần trăm**.

## Đối chiếu checkpoint trước

| Thí nghiệm | Test accuracy | Macro-F1 |
|---|---:|---:|
| DINOv2 fine-tune, seed42 cũ | 92,65% | 92,62% |
| DINOv3 ban đầu, 256px, seed42 | 91,18% | 91,15% |
| DINOv3 recipe gần V2, trung bình 3 seed | 84.80% | 84.71% |

Recipe gần V2 không cải thiện V3 trong thí nghiệm này. Không chọn cấu hình mới thay
bản V3 cũ hoặc V2. Chưa thể quy nguyên nhân riêng cho kích thước ảnh, augmentation,
scheduler hay seed vì nhiều yếu tố thay đổi đồng thời giữa hai recipe V3.
Đây không chứng minh V3 kém về bản chất, chỉ phản ánh các checkpoint đã thử.

Trong ba seed mới, seed43 đạt validation macro-F1 cao nhất; nếu buộc chọn một
checkpoint của thí nghiệm mới thì dùng seed43 theo quy tắc đã định, dù test thấp
hơn hai seed còn lại. Không chọn seed42/44 vì thấy test cao hơn.

V2 chưa được chạy lại 3 seed trên cùng môi trường GPU. Tập test đã được xem nhiều
lần, nên đây là so sánh thăm dò, không phải xác nhận độc lập hoặc kiểm định ưu thế
thống kê. Không tiếp tục sửa tham số theo test cũ. Muốn khảo sát thêm nên dùng
validation và chuẩn bị tập ảnh thực địa mới để xác nhận.

App không bị thay model trong thí nghiệm này.

Notebook: https://www.kaggle.com/code/huyentran999/durian-dinov3-comparison
