# So sánh model bệnh lá sầu riêng — 10/11/2026

## Kết luận lựa chọn

- **Chạy trực tiếp trên điện thoại, ưu tiên gọn nhẹ/offline: MobileNetV2 bản field balanced110.**
- **Web trên điện thoại gửi ảnh về server, ưu tiên độ chính xác hiện có: DINOv2 fine-tune 2 block.**
- DINOv3 chưa cho thấy lợi ích về độ chính xác trong lần train hiện tại; giữ làm đối chứng.
- Không ưu tiên hai kiến trúc ResNet + Transformer hiện tại: lớn hơn MobileNetV2 và kết quả thấp hơn.

Đây là lựa chọn triển khai tạm thời dựa trên báo cáo hiện có, không phải kết luận
mô hình nào tốt nhất trên mọi dữ liệu. Không thay model hay sửa app trong lần so sánh này.

## Cơ sở và phạm vi

Đã đọc mọi file *report.json dưới ai/models (xem all_runs.csv).
Bảng chính dùng test 68 ảnh, 17 ảnh/lớp. Đường dẫn báo cáo MobileNetV2, DINOv2,
ResNet18 field cùng trỏ tới dữ liệu field processed. Hash nội dung test DINOv3
trong manifest khớp 68 ảnh local; test của hai bộ public + field cũng khớp hash.
ResNet50 Kaggle ghi cùng nguồn dataset field và kích thước split; chưa có manifest
riêng của lần ResNet50 để chứng minh lại bằng hash. Các báo cáo cũ không có manifest
tại thời điểm train; việc kiểm tra file local hiện tại không chứng minh dữ liệu chưa
từng thay đổi. Không đánh giá lại toàn bộ checkpoint trong báo cáo này.

Hai bản public + field có lượng train/validation khác nên không phải thí nghiệm
kiểm soát chỉ thay kiến trúc. Kết quả DINOv3 dùng ảnh 256px, DINOv2 224px và recipe
train khác; so sánh checkpoint đã có, không khẳng định hơn/kém do riêng kiến trúc.

## Kết quả trên test field 68 ảnh

Dung lượng bên dưới tính MiB (1 MiB = 1.048.576 byte).
State là tổng byte tensor trong model_state_dict, không bao gồm optimizer.
Đây chưa phải kích thước file .tflite/.mlpackage và chưa phải RAM lúc chạy.

| Model / lần train | Accuracy | Macro-F1 | Đúng/68 | Checkpoint MiB | Tensor model MiB |
|---|---:|---:|---:|---:|---:|
| DINOv2 ViT-S/14 · fine-tune 2 block | 92.65% | 92.62% | 63/68 | 84.21 | 84.15 |
| DINOv3 ViT-S/16 · fine-tune 2 block | 91.18% | 91.15% | 62/68 | 109.54 | 82.36 |
| MobileNetV2 · field balanced110 | 91.18% | 91.04% | 62/68 | 8.74 | 8.63 |
| DINOv2 ViT-S/14 · đóng băng backbone | 83.82% | 83.29% | 57/68 | 84.21 | 84.15 |
| MobileNetV2 · public + field | 82.35% | 82.45% | 56/68 | 8.74 | 8.63 |
| ResNet18 + Transformer · field | 75.00% | 74.17% | 51/68 | 58.78 | 58.73 |
| ResNet50 + Transformer · field (Kaggle) | 75.00% | 73.56% | 51/68 | 346.22 | 346.10 |
| ResNet18 + Transformer · public + field | 69.12% | 69.47% | 47/68 | 58.78 | 58.73 |

DINOv3 checkpoint gồm cả optimizer nên file 109,54 MiB không có nghĩa model suy luận
lớn hơn DINOv2 theo cùng tỷ lệ. State DINOv3 khoảng 82,36 MiB; DINOv2 84,15 MiB.
MobileNetV2 khoảng 8,63 MiB state, nhỏ hơn DINOv2 khoảng 9,7 lần.

## F1 từng lớp — ba ứng viên chính

| Model | Đốm rong | Cháy lá | Thán thư | Lá khỏe |
|---|---:|---:|---:|---:|
| DINOv2 ViT-S/14 · fine-tune 2 block | 94.12% | 88.89% | 90.32% | 97.14% |
| DINOv3 ViT-S/16 · fine-tune 2 block | 88.89% | 88.24% | 90.32% | 97.14% |
| MobileNetV2 · field balanced110 | 91.89% | 87.50% | 94.44% | 90.32% |

MobileNetV2 phát hiện đúng 17/17 ảnh thán thư của test, nhưng chỉ nhận đúng 14/17
ảnh lá khỏe. DINOv2 và DINOv3 nhận đúng 17/17 ảnh khỏe, nhưng thán thư đúng 14/17.
Đó là khác biệt cụ thể giữa các checkpoint; mẫu chỉ 17 ảnh/lớp nên chưa đủ để
khẳng định khả năng ngoài thực địa.

## Gọn nhẹ, tốc độ, điện thoại

| Tiêu chí | MobileNetV2 | DINOv2 fine-tune | DINOv3 | ResNet + Transformer |
|---|---|---|---|---|
| Dung lượng state | Nhỏ nhất, 8,63 MiB | 84,15 MiB | 82,36 MiB | 58,73–346,10 MiB |
| Ưu tiên chạy offline | Lựa chọn đầu tiên | Cần kiểm tra kỹ tài nguyên | Cần kiểm tra kỹ tài nguyên | Không ưu tiên với kết quả hiện tại |
| Độ chính xác hiện có | 91,18% | 92,65% | 91,18% | 69,12–75,00% |
| Tốc độ/RAM/pin trên điện thoại | Chưa đo | Chưa đo | Chưa đo | Chưa đo |
| Chuyển đổi mobile thực tế | Chưa thực hiện | Chưa thực hiện | Chưa thực hiện | Chưa thực hiện |

MobileNetV2 có kiến trúc hướng tới thiết bị di động, nhưng chưa đo latency, RAM,
mức tiêu thụ pin hoặc độ tương thích toán tử trên điện thoại của người dùng.
Không suy ra các số đó từ dung lượng checkpoint hay thời gian train khác phần cứng.

Ước tính dung lượng tensor nếu chỉ giảm độ rộng biểu diễn: MobileNetV2 khoảng
4,3 MiB FP16 hoặc 2,2 MiB INT8. Đây KHÔNG phải file đã export, không tính runtime,
metadata, activation và overhead. INT8 cần calibration/đánh giá lại; không đảm bảo
giữ nguyên độ chính xác hay giảm latency trên mọi thiết bị.

App hiện là Flask web: điện thoại gửi ảnh, server chạy PyTorch. Model không được
cài trực tiếp lên điện thoại. Muốn offline cần bổ sung runtime và luồng nhận diện
cục bộ, cùng bước chuyển đổi sang định dạng phù hợp. Có thể khảo sát LiteRT cho
Android hoặc Core ML cho iOS; chưa đảm bảo mọi model export thành công.

Nguồn triển khai:
- https://github.com/apple/coremltools/blob/main/docs-guides/source/convert-a-torchvision-model-from-pytorch.md
- https://github.com/google-ai-edge/litert-torch/blob/main/README.md

## Độ chắc chắn của kết luận

DINOv2 chỉ hơn MobileNetV2 và DINOv3 **1 ảnh / 68**, tức 1,47 điểm phần trăm.
Không có kiểm định dự đoán ghép cặp hoặc lặp nhiều seed; không kết luận vượt trội
có ý nghĩa thống kê. Tập test đã được dùng để so nhiều phiên bản nên cần tập thực địa
mới độc lập để xác nhận lựa chọn cuối, tránh tiếp tục tối ưu theo test cũ.

## Các lần train khác — không xếp chung do test khác

| Lần train | Train | Test | Accuracy | Macro-F1 |
|---|---:|---:|---:|---:|
| mobilenetv2/combined_public_field_4class | 1395 | 30 | 63.33% | 48.25% |
| mobilenetv2/combined_public_field_4class_retrained | 1401 | 32 | 43.75% | 33.71% |
| mobilenetv2 | 114 | 26 | 53.85% | 43.61% |
| mobilenetv2/supabase_4class_20260921_mps_verified | 480 | 46 | 78.26% | 72.55% |
| mobilenetv2/supabase_4class_20260921_retrained | 480 | 46 | 78.26% | 72.55% |
| mobilenetv2/supabase_4class_20260922_retrained | 480 | 61 | 81.97% | 81.26% |
| mobilenetv2/supabase_4class_clean | 146 | 30 | 50.00% | 35.83% |
| mobilenetv2/supabase_4class_finetuned | 152 | 32 | 53.12% | 41.96% |
| mobilenetv2/supabase_4class_from_combined_finetuned | 152 | 32 | 50.00% | 42.80% |
| mobilenetv2/supabase_4class_processed_finetuned | 480 | 32 | 56.25% | 52.55% |
| mobilenetv2/supabase_4class_processed_freeze | 480 | 32 | 56.25% | 46.67% |
| mobilenetv2/supabase_4class_retrained | 152 | 32 | 50.00% | 36.11% |

Có checkpoint ViT baseline và một số checkpoint khác không kèm report tương ứng;
không gán độ chính xác hoặc xếp hạng các file đó. Bản ResNet50 field local không có
report cùng thư mục; bảng dùng bản Kaggle có report rõ ràng.

## Quyết định đề xuất và bước xác nhận

Chọn MobileNetV2 field balanced110 nếu chỉ chọn MỘT model cho mục tiêu
gọn nhẹ và tiến tới offline trên điện thoại. Giữ DINOv2 fine-tune làm model
tham chiếu/server. Với ứng dụng web hiện tại, tiếp tục dùng DINOv2 là hợp lý.

Trước khi chốt bản mobile: export MobileNetV2, kiểm tra kết quả sau export trên
cùng dữ liệu, đo p50/p95 latency và peak RAM trên điện thoại mục tiêu, đánh giá
trên ảnh thực địa mới. Sau đó mới quyết định FP32/FP16/INT8.
