# Backend

API/server trung gian cho ứng dụng mobile, model inference và AI Agent.

## Mục tiêu

- Nhận dữ liệu từ app thu thập dữ liệu.
- Cung cấp API nhận diện bệnh nếu app không chạy model trực tiếp trên thiết bị.
- Quản lý service gọi model, lưu kết quả và tích hợp AI Agent.

## Thư mục

- `api/`: định nghĩa route/controller.
- `services/`: xử lý nghiệp vụ, model service và agent service.
- `main.py`: entrypoint server.
