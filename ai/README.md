# AI

Không gian dùng chung cho dữ liệu, tiền xử lý, huấn luyện, inference và model artifact.

## Cấu trúc

- `baseline/vit/`: thực nghiệm Vision Transformer baseline.
- `baseline/mobilenetv2/`: thực nghiệm MobileNetV2 baseline cho hướng on-device.
- `foundation_model/`: thử nghiệm Vision Foundation Model theo mục tiêu khóa luận.
- `data/`: dữ liệu thô, dữ liệu đã chia tập hoặc metadata dataset.
- `preprocessing/`: script chuẩn hóa ảnh, nhãn và augmentation.
- `training/`: script huấn luyện/fine-tune.
- `inference/`: logic dự đoán dùng chung cho backend hoặc mobile.
- `models/`: model artifact đã train/export.

## Luồng đề xuất

1. Dữ liệu thô từ `apps/data_collection_app` được đưa vào `data/`.
2. Script trong `preprocessing/` chuẩn hóa ảnh, nhãn và chia tập train/validation/test.
3. `baseline/` dùng để so sánh ViT và MobileNetV2 trên cùng dataset.
4. `foundation_model/` dùng để thử Vision Foundation Model sau khi có baseline.
5. Model phù hợp được lưu trong `models/`.
6. Logic trong `inference/` được dùng bởi backend hoặc đóng gói cho mobile.

## Ghi chú thực nghiệm

- `ViT` đang có ưu thế về độ chính xác nhưng nặng hơn, phù hợp hơn với server/cloud hoặc thiết bị mạnh.
- `MobileNetV2` nhẹ và nhanh hơn, phù hợp hơn cho inference trực tiếp trên điện thoại.
- Wi-Fi/4G chỉ giúp truyền ảnh nhanh hơn khi gọi server; nó không làm model ViT nhẹ hơn nếu chạy trực tiếp trên thiết bị.
