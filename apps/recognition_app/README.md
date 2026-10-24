# Recognition App

Ứng dụng web mobile-first nhận diện bệnh lá sầu riêng bằng Flask và PyTorch.

## Mục tiêu

- Nhận ảnh từ camera hoặc thư viện ảnh trên điện thoại.
- Chạy DINOv2 hoặc MobileNetV2 trên Flask backend.
- Hiển thị bệnh dự đoán, độ tin cậy và khuyến nghị xử lý.
- Kết nối AI Agent để giải thích kết quả và hỗ trợ hỏi đáp.
- Đọc kết quả bằng Web Speech API và lưu lịch sử gần đây trên trình duyệt.

## Thư mục

- `app.py`: Flask routes.
- `model_service.py`: tiền xử lý và PyTorch inference.
- `agent_service.py`: gọi LLM qua backend hoặc trả lời fallback cục bộ.
- `history_service.py`: lưu lịch sử lên bảng Supabase riêng nếu được cấu hình.
- `templates/`, `static/`: giao diện mobile-first.

## Chạy local

```bash
cd apps/recognition_app
MODEL_BACKEND=dinov2 ../../.venv/bin/python app.py
```

Mở `http://127.0.0.1:5001`. Dùng `MODEL_BACKEND=mobilenetv2` để chạy mô hình nhẹ hơn.

AI Agent trực tuyến dùng ba biến môi trường `AI_AGENT_BASE_URL`, `AI_AGENT_API_KEY`,
`AI_AGENT_MODEL`. Khi chưa cấu hình, endpoint sử dụng câu trả lời fallback an toàn.

Để lưu lịch sử Supabase, chạy `schema.sql` rồi cấu hình `SUPABASE_URL`,
`SUPABASE_SERVICE_ROLE_KEY`, `SUPABASE_HISTORY_TABLE=diagnosis_history` và tạo bucket riêng
`durian-diagnosis` (hoặc đặt `SUPABASE_DIAGNOSIS_BUCKET`) trên server.
Không đưa service-role key hoặc khóa LLM xuống JavaScript.


## Điều hướng và dữ liệu trên thiết bị

- Chẩn đoán: chụp/chọn ảnh, nhận diện và nghe kết quả.
- Bệnh: tìm kiếm bốn nhóm kết quả, mở từng mục để xem hướng dẫn quan sát.
- Lịch sử: lưu tối đa 20 kết quả mới với ảnh JPEG thu nhỏ và chi tiết dự đoán;
  mở lại, xóa từng bản ghi hoặc xóa tất cả sau xác nhận. Trang chủ hiện ba lần gần nhất.
  Bản ghi cũ thiếu chi tiết vẫn hiển thị tóm tắt.
- Kiến thức: hướng dẫn chụp ảnh, hiểu độ tin cậy và theo dõi triệu chứng, kèm nguồn tham khảo.
- Cài đặt (bánh răng trên đầu trang): chọn giọng tiếng Việt do thiết bị cung cấp,
  tốc độ đọc và tự đọc sau nhận diện (bật mặc định, có thể tắt). Không cần Google Cloud TTS.

Lịch sử dùng khóa localStorage `durianHistory`, cài đặt dùng `durianSettings`.
Dữ liệu này thuộc trình duyệt/nguồn truy cập hiện tại, không tự đồng bộ giữa thiết bị.
Xóa dữ liệu trình duyệt sẽ mất lịch sử; khi hết dung lượng, ứng dụng báo không lưu được.
Giọng đọc phụ thuộc hệ điều hành; tự đọc có thể bị trình duyệt chặn cho tới khi người dùng tương tác.

Kiểm tra thủ công trên điện thoại: chuyển bốn tab và Cài đặt; tìm bệnh không có kết quả;
nhận diện rồi mở lại lịch sử; xóa/hủy xóa; tải lại trang để kiểm tra lưu cài đặt;
nghe thử và đọc kết quả; kiểm tra khi thiết bị chưa có giọng tiếng Việt.

### Bài đọc chọn lọc

Danh mục tại `static/articles.js` gồm 6 tài liệu từ khuyến nông, tạp chí VAAS và UC IPM,
được kiểm tra ngày 24/10/2026. Mỗi mục có nguồn, tóm tắt biên tập, liên kết HTTPS,
chủ đề, loại tài liệu và ngày xuất bản khi đã xác minh. Danh mục được chọn thủ công,
không tự cập nhật tin tức. Không sao chép ảnh hay toàn văn từ nguồn bên ngoài.

Mục Kiến thức hỗ trợ tìm theo tên/nội dung/nguồn và lọc chủ đề. Kết quả nhận diện
gợi ý tối đa hai tài liệu theo tên lớp; đây là bài tham khảo, không phải bằng chứng
xác nhận chẩn đoán. Khi thêm tài liệu, cập nhật số lượng trên tiêu đề thư viện và
kiểm tra lại URL, phạm vi áp dụng, ngày đăng trước khi xuất bản.


### Cài đặt trên máy mới

Cài Python 3.12, tạo môi trường ảo và cài các thư viện:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r apps/recognition_app/requirements.txt
```

Trọng số mô hình và dữ liệu huấn luyện không được đưa lên Git. Cần đặt checkpoint
tin cậy đúng đường dẫn trước khi nhận diện:

- DINOv2: `ai/models/dinov2/field_balanced110_unfreeze2/dinov2_vits14_field_4class_best.pth`.
- MobileNetV2: `ai/models/mobilenetv2/supabase_4class_20261003_balanced110/mobilenetv2_supabase_4class_best.pth`.

DINOv2 dùng `torch.hub` để nạp mã backbone từ `facebookresearch/dinov2` ở lần đầu,
cần kết nối mạng hoặc đặt `DINOV2_HUB_REPO` phù hợp với bộ nhớ đệm đã chuẩn bị.
Giao diện có thể mở khi chưa có checkpoint nhưng chức năng nhận diện cần mô hình.
