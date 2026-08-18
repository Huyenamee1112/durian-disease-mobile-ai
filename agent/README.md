# AI Agent

AI Agent hỗ trợ diễn giải kết quả nhận diện, gợi ý xử lý và kết nối các công cụ/nguồn tri thức.

## Mục tiêu

- Nhận kết quả dự đoán từ model hoặc backend.
- Tạo phản hồi dễ hiểu cho người dùng cuối.
- Gọi tools khi cần tra cứu tri thức, lịch sử bệnh hoặc khuyến nghị canh tác.

## Thư mục

- `prompts/`: system prompt, instruction và template hội thoại.
- `tools/`: công cụ agent có thể gọi.
- `agent.py`: entrypoint/logic chính của agent.
