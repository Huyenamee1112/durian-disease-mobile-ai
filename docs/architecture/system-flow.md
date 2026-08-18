# System Flow

Tài liệu này chốt luồng triển khai cho khóa luận nhận diện bệnh sầu riêng trên thiết bị di động, dựa trên phần nghiên cứu và thực nghiệm baseline hiện tại.

## Luồng tổng thể

```text
                    ┌─────────────────────┐
                    │  DATA COLLECTION APP │
                    │     App thu thập     │
                    └──────────┬──────────┘
                               │
                         Ảnh + Label
                               │
                               ▼
                    ┌─────────────────────┐
                    │       DATASET       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   AI EXPERIMENT     │
                    │ ViT / MobileNetV2   │
                    │ Vision Foundation   │
                    └──────────┬──────────┘
                               │
                         Model được chọn
                               │
                               ▼
              ┌────────────────────────────────┐
              │       RECOGNITION APP          │
              │          App chính             │
              └───────────────┬────────────────┘
                              │
                         Chụp / chọn ảnh
                              │
                              ▼
                       Preprocessing
                              │
                              ▼
                    ┌─────────────────┐
                    │  AI INFERENCE   │
                    └────────┬────────┘
                             │
                 ┌───────────┴───────────┐
                 │                       │
          On-device model          Server model
          MobileNetV2              VFM / ViT
                 │                       │
                 └───────────┬───────────┘
                             ▼
                       Kết quả bệnh
                             │
                             ▼
                         AI AGENT
                             │
                 ┌───────────┼───────────┐
                 ▼           ▼           ▼
              Bệnh        Mức độ     Khuyến nghị
                             │
                             ▼
                          Lịch sử
```

## Ý nghĩa từng giai đoạn

1. Research: review 3-5 nghiên cứu liên quan, so sánh dataset, model, quy trình xử lý, chỉ số đánh giá và kết quả.
2. Baseline experiment: chạy cùng dataset public với ViT và MobileNetV2 để tạo đường cơ sở.
3. Vision Foundation Model: thử nghiệm/fine-tune model phù hợp với mục tiêu đề tài.
4. Data Collection App: thu thập dữ liệu thực tế để mở rộng dataset riêng.
5. Recognition App: nhận ảnh, tiền xử lý, inference và hiển thị kết quả.
6. AI Agent: diễn giải kết quả, mức độ bệnh và khuyến nghị xử lý.

## Chế độ inference trong Recognition App

- On-device: dùng MobileNetV2 hoặc model nhẹ đã export, ưu tiên chạy offline, nhanh, ít hao pin và phù hợp điện thoại tầm trung.
- Server/Cloud: upload ảnh lên backend để chạy ViT hoặc Vision Foundation Model, ưu tiên độ chính xác và phân tích sâu.

## Kết luận kiến trúc

Luồng này là hợp lý vì nối được 3 phần chính của khóa luận: nghiên cứu, thực nghiệm và ứng dụng. Tuy nhiên, không nên chốt MobileNetV2 là model cuối ngay từ baseline. MobileNetV2 phù hợp cho nhánh on-device, còn ViT/Vision Foundation Model phù hợp cho nhánh server hoặc so sánh nâng cao.

Với số liệu baseline hiện tại, kết luận nên viết là:

```text
MobileNetV2 phù hợp triển khai on-device hơn nhờ kích thước nhỏ và tốc độ nhanh.
ViT có ưu thế về độ chính xác nhưng chi phí tính toán và kích thước mô hình cao hơn,
do đó phù hợp hơn với server/cloud hoặc thiết bị mạnh.
```
