# DINOv3 ViT-S/16 — kết quả train Kaggle

Lần 1/3 hoàn tất; không cần chạy lại.

- Dataset: huyentran999/durian-field-balanced110.
- Train: 480 ảnh; validation: 64 ảnh; test: 68 ảnh.
- Chạy 25 epoch; chọn epoch 22 theo validation macro-F1.
- Validation accuracy: 81,25%; macro-F1: 80,83%.
- Test accuracy: 91,18% (62/68); macro-F1: 91,15%.
- Thời gian vòng train và đánh giá: 194,8 giây (không gồm khởi động/cài đặt/tải trọng số).

| Lớp | Test F1 | Đúng / tổng ảnh thật của lớp |
|---|---:|---:|
| Đốm rong | 88,89% | 16/17 |
| Cháy lá | 88,24% | 15/17 |
| Thán thư | 90,32% | 14/17 |
| Lá khỏe | 97,14% | 17/17 |

Checkpoint: dinov3_vits16_field_4class_best.pth (~109,54 MiB).
Báo cáo đầy đủ: dinov3_vits16_field_4class_report.json.

Chọn checkpoint bằng validation, chỉ đánh giá test sau khi chọn.
Tập test nhỏ (17 ảnh mỗi lớp), kết quả chưa xác nhận hiệu quả ngoài thực địa.
Kiểm tra hash không thấy ảnh trùng byte giữa các split; chưa loại trừ ảnh gần trùng
hoặc ảnh cùng cây giữa các split. Chưa thay model mặc định trong app.
