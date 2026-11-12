# Huấn luyện lại MobileNetV2 — dữ liệu thu thập 21/09/2026

## Kết quả đã kiểm chứng

Đã train đủ 15 epoch; chọn checkpoint epoch 7. Đánh giá lại validation trên CPU khớp chính xác accuracy đã ghi khi train GPU.

| Chỉ số | Lần trước | Lần mới |
|---|---:|---:|
| Validation accuracy tốt nhất | 72.7% | 83.7% |
| Test accuracy | 56.2% | 78.3% |
| Test macro F1 | 52.6% | 72.5% |
| Số ảnh test | 32 | 46 |

Accuracy thay đổi +22.0 điểm phần trăm; macro F1 thay đổi +20.0 điểm phần trăm. Test khác nhau nên không thể quy toàn bộ thay đổi cho việc thu thêm ảnh.

| Lớp | Số ảnh test | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Đốm rong | 11 | 84.6% | 100.0% | 91.7% |
| Cháy lá | 6 | 66.7% | 33.3% | 44.4% |
| Thán thư | 21 | 77.3% | 81.0% | 79.1% |
| Lá khỏe | 8 | 75.0% | 75.0% | 75.0% |

Confusion matrix: `confusion_matrix.csv`, hàng là nhãn thật, cột là dự đoán. Checkpoint: `mobilenetv2_supabase_4class_best.pth`; thứ tự nhãn và kích thước đầu vào được lưu trong checkpoint.

## Dữ liệu và tiền xử lý

- Nguồn: bản export local `ai/data/supabase_dataset`, 327 ảnh; không tải lại Supabase trong lần chạy này.
- Chọn 4 lớp; bỏ 6 ảnh Phomopsis. Loại 4 bản sao trùng hoàn toàn khỏi bản sao dataset train, giữ nguyên dữ liệu nguồn.
- Không có ảnh lỗi khi giải mã. Đã kiểm tra trùng byte và trùng pixel ở dữ liệu nguồn; cả hai tìm thấy cùng 4 cặp.
- Giữ các split của bản export; khi trùng giữa train và validation, giữ bản validation.
- 317 ảnh duy nhất: train 222, validation 49, test 46.

| Lớp | Train gốc | Train sau augment | Validation | Test |
|---|---:|---:|---:|---:|
| Đốm rong | 51 | 120 | 12 | 11 |
| Cháy lá | 28 | 120 | 6 | 6 |
| Thán thư | 102 | 120 | 22 | 21 |
| Lá khỏe | 41 | 120 | 9 | 8 |
| Tổng | 222 | 480 | 49 | 46 |

Chuẩn hóa EXIF, RGB, pad vuông nền xanh nhạt, resize 512×512. Augment offline chỉ trên train với seed 42; đầu vào mô hình 224×224, chuẩn hóa ImageNet và augment online theo script có sẵn. Đã kiểm tra tất cả 575 ảnh đầu ra, nguồn của từng ảnh augment và không có bản trùng hoàn toàn giữa các split.

## Cấu hình

MobileNetV2, fine-tune toàn bộ mạng 15 epoch; batch 16; AdamW, learning rate 0.0001, weight decay 0.0001; seed 42; GPU Apple MPS; kiểm chứng metrics trên CPU với 4 threads. Khởi tạo features từ `mobilenetv2_durian_baseline.pth` (baseline 6 lớp); classifier 4 lớp khởi tạo mới. Checkpoint gốc không kèm metadata về dữ liệu đã train. Không khởi tạo từ checkpoint field cũ vì 25 ảnh test mới từng nằm trong train của bộ field cũ.

Chọn checkpoint theo validation accuracy, giữ epoch đầu nếu bằng điểm; chỉ đánh giá test sau khi train xong.

## Tái lập

```sh
.venv/bin/python ai/scripts/build_training_datasets.py --field-source ai/data/supabase_dataset --field-output ai/data/supabase_4class_20260921_clean --field-only --deduplicate
.venv/bin/python ai/scripts/build_processed_field_dataset.py --source ai/data/supabase_4class_20260921_clean --output ai/data/supabase_4class_20260921_processed --output-size 512 --target-train-per-class 120 --seed 42
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python -u ai/training/train_supabase_mobilenetv2.py --data-dir ai/data/supabase_4class_20260921_processed --output-dir ai/models/mobilenetv2/supabase_4class_20260921_mps_verified --init-checkpoint ai/models/mobilenetv2/mobilenetv2_durian_baseline.pth --epochs 15 --batch-size 16 --image-size 224 --learning-rate 0.0001 --weight-decay 0.0001 --num-workers 0 --seed 42 --device mps
```

Các lệnh tái lập ghi lại các thư mục đầu ra đã chỉ định; đổi tên đầu ra để giữ lần chạy hiện tại.

## Giới hạn đánh giá

Bộ test cũ 32 ảnh và test mới 46 ảnh chỉ có 1 ảnh cùng ID. So sánh điểm lịch sử không phải thử nghiệm có kiểm soát trên cùng test. Chưa kiểm tra ảnh gần giống, phân nhóm theo cây/đợt chụp hoặc xác nhận lại nhãn bởi chuyên gia; kiểm tra trùng hoàn toàn chưa loại trừ các dạng rò rỉ này. Test cháy lá chỉ có 6 ảnh nên kết quả theo lớp còn dễ dao động.

Thông tin nguồn và SHA-256: `run_provenance.json`. Nhật ký: `training.log`. Báo cáo tại thư mục này dùng checkpoint GPU được lưu riêng và đã đánh giá lại độc lập; không dùng báo cáo ở thư mục chạy chung CPU/GPU.

Lần train CPU ban đầu đã được dừng qua phiên terminal gốc sau khi GPU hoàn tất. Lịch sử 15 epoch trong JSON là số liệu đầy đủ từ lần train GPU.
