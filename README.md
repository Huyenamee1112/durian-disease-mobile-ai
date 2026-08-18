# durian-disease-mobile-ai
HỆ THỐNG NHẬN DIỆN BỆNH SẦU RIÊNG TÂY NGUYÊN SỬ DỤNG VISION FOUNDATION MODEL VÀ AI AGENT TRÊN THIẾT BỊ DI ĐỘNG

## Mục tiêu

Dự án xây dựng hệ thống hỗ trợ nhận diện bệnh trên cây sầu riêng bằng mô hình thị giác máy tính, triển khai trên thiết bị di động và có AI Agent hỗ trợ phân tích, tư vấn quy trình xử lý.

## Luồng hệ thống

1. `apps/data_collection_app`: ứng dụng thu thập ảnh, nhãn bệnh và metadata ngoài vườn.
2. `ai`: xử lý dữ liệu, thực nghiệm baseline, thử Vision Foundation Model, đánh giá và đóng gói inference.
3. `backend`: API/server trung gian nếu cần đồng bộ dữ liệu, gọi model hoặc quản lý người dùng.
4. `agent`: AI Agent dùng prompt và tools để diễn giải kết quả, gợi ý xử lý và hỗ trợ truy vấn tri thức.
5. `apps/recognition_app`: ứng dụng chính cho người dùng nhận diện bệnh từ ảnh và xem khuyến nghị.
6. `docs`: tài liệu kiến trúc, flow, sơ đồ và quyết định kỹ thuật.

## Cấu trúc thư mục

```text
durian-disease-mobile-ai/
├── apps/
│   ├── data_collection_app/
│   │   ├── lib/
│   │   ├── assets/
│   │   └── README.md
│   └── recognition_app/
│       ├── lib/
│       ├── assets/
│       └── README.md
├── ai/
│   ├── baseline/
│   │   ├── vit/
│   │   └── mobilenetv2/
│   ├── foundation_model/
│   ├── data/
│   ├── preprocessing/
│   ├── training/
│   ├── inference/
│   └── models/
├── backend/
│   ├── api/
│   ├── services/
│   └── main.py
├── agent/
│   ├── prompts/
│   ├── tools/
│   └── agent.py
├── docs/
│   ├── architecture/
│   └── diagrams/
├── README.md
└── .gitignore
```

## Vai trò từng thành phần

- `data_collection_app`: phục vụ nhóm thu thập dữ liệu, chuẩn hóa ảnh, nhãn bệnh, vị trí, thời gian, giống cây và ghi chú hiện trường.
- `recognition_app`: app di động chính cho nông dân/kỹ sư nông nghiệp chụp ảnh, nhận kết quả dự đoán và xem hướng dẫn xử lý.
- `ai`: pipeline AI dùng chung, gồm baseline ViT/MobileNetV2, Vision Foundation Model, dữ liệu, tiền xử lý, training, inference và model artifact.
- `backend`: nơi đặt API, service xử lý nghiệp vụ và tích hợp với model/agent khi app cần gọi server.
- `agent`: lớp AI Agent hỗ trợ diễn giải kết quả nhận diện và kết nối các công cụ/nguồn tri thức.
- `docs`: nơi lưu kiến trúc, flow nghiệp vụ, sơ đồ hệ thống và tài liệu triển khai.

## Định hướng triển khai

Recognition App nên hỗ trợ 2 chế độ inference:

- On-device: dùng model nhẹ như `MobileNetV2` để chạy nhanh, offline và phù hợp thiết bị di động.
- Server/Cloud: dùng model lớn hơn như `ViT` hoặc Vision Foundation Model để tăng độ chính xác và phục vụ phân tích sâu qua AI Agent.

Kết quả thực nghiệm hiện tại chỉ nên xem là baseline. Không chốt model cuối trước khi so sánh thêm Vision Foundation Model theo mục tiêu khóa luận.
