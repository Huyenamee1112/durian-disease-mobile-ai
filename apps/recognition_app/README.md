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


## Giọng Google tiếng Việt bằng gTTS

Giống cách đọc tiếng Việt trong CoffeeAgent: dùng gTTS với lang="vi", slow=False,
dọn Markdown và ký hiệu trước khi đọc. Chỉ dự án sầu riêng được sửa.
Đây là giọng Google Translate qua thư viện gTTS, không phải Google Cloud
Text-to-Speech/Neural2. Không cần API key, thẻ hoặc Billing.

### Cài và chạy

Tại thư mục gốc dự án:

```bash
.venv/bin/python -m pip install -r apps/recognition_app/requirements.txt
cd apps/recognition_app
PORT=5002 MODEL_BACKEND=dinov2 ../../.venv/bin/python app.py
```

Cài đặt trong app: Nguồn giọng đọc → Google · Tiếng Việt (gTTS), hoặc chọn
Giọng thiết bị. Lựa chọn FPT cũ tự chuyển sang gTTS; lựa chọn Giọng thiết bị và
tùy chọn tự đọc của người dùng vẫn được giữ. gTTS không có danh sách giọng
Bắc/Trung/Nam như FPT. Tốc độ được điều khiển khi phát MP3 bằng playbackRate.

### Luồng demo và bộ nhớ đệm

- POST /api/predict chỉ nhận diện; không gọi TTS. Giao diện hiện kết quả trước,
  sau đó gửi POST /api/tts độc lập và lưu lịch sử; không chờ giọng đọc mới hiện kết quả.
- POST /api/tts nhận text (3–2000 ký tự), trả audio_url cùng cached.
  Backend tạo MP3 trước khi trả URL, không có vòng polling URL bên ngoài.
- GET /api/tts/audio/<sha256>.mp3 phát MP3 từ cache cùng nguồn với ứng dụng.
- MP3 được lưu ở apps/recognition_app/instance/tts_audio/, tên hash theo nội dung
  đã dọn + phiên bản giọng. Cache tối đa 128 MP3, 7 ngày; dọn khi tạo file mới.
  Thư mục instance đã bị Git ignore. Nội dung trùng dùng lại MP3 sau cả restart
  server; khi cache hết hạn/bị dọn, nhấn Nghe sẽ tạo lại.
- Ghi file tạm rồi đổi tên nguyên tử; file tạo dở không được phục vụ.
  Trong một process, yêu cầu trùng chờ cùng kết quả; tối đa 2 lượt tạo đồng thời.
  Nhiều worker có thể vẫn tạo trùng vì chưa dùng cache/khóa phân tán.
- Timeout mỗi kết nối/đọc mạng: 4/8 giây. Backend kiểm tra ngân sách 25 giây
  giữa các chunk; một request mạng đang chờ có thể vượt ngân sách thêm tới timeout
  mạng. Frontend chờ tối đa 40 giây cho yêu cầu/khởi phát âm thanh, tải MP3 tối đa
  15 giây. Khi lỗi/hết thời gian, thông báo và dùng giọng thiết bị nếu hỗ trợ.
- Nghe/Dừng và chuyển tab hủy request/chờ phát phía trình duyệt, không phát trễ
  hoặc chồng giọng. Backend có thể tiếp tục hoàn thành và lưu cache request đã nhận.
- Khi trình duyệt chặn autoplay, giữ MP3 sẵn sàng và hiện thanh phát để bấm ▶;
  không tự chuyển sang giọng thiết bị trong trường hợp này.

### Chuẩn bị buổi báo cáo

1. Có Internet khi tạo nội dung mới; nghe thử trước trên đúng điện thoại/trình duyệt.
2. Chạy nhận diện và nghe trước các kết quả dự định demo để tạo cache.
3. Thử Nghe lại, Dừng, đổi tab trong lúc đang tạo, chế độ mất mạng và autoplay.
4. Giữ giọng thiết bị làm dự phòng. MP3 đã có trên backend vẫn phát được khi backend
   truy cập được và cache còn tồn tại; mất kết nối điện thoại–backend thì phải dùng
   giọng thiết bị.

gTTS phụ thuộc dịch vụ bên ngoài, có thể chậm hoặc giới hạn truy cập; không đảm bảo
chất lượng/dịch vụ như Google Cloud. Không cần đăng ký Billing nhưng cũng không
có cam kết hạn mức hay SLA. Xem https://gtts.readthedocs.io/en/latest/module.html.

### Kiểm tra

```bash
cd apps/recognition_app
../../.venv/bin/python -m unittest test_app.py test_tts.py
```

Chạy kiểm tra frontend từ thư mục gốc:

```bash
node --test apps/recognition_app/test_speech.cjs
```

Backend kiểm tra input, đọc tiếng Việt, cache bền vững, lỗi mạng, ghi nguyên tử,
đồng thời, dọn cache và nhận diện không phụ thuộc TTS. Các test này mock dịch vụ
bên ngoài; cần nghe thử online riêng trước demo.

## Hướng dẫn xử lý có nguồn

Catalog knowledge_base/care_guides.json và care_service.py cung cấp hướng dẫn
biên tập cho bốn lớp, dùng chung ở kết quả nhận diện và Tra cứu bệnh.
Mỗi bước có tham chiếu nguồn; phần Nguồn tham khảo ghi cơ quan, mục/trang và ngày
đối chiếu. Các bước chụp ảnh/ghi nhận diễn biến là gợi ý của app, gắn nhãn riêng.
Xem knowledge_base/SOURCES.md để kiểm tra phạm vi nguồn và nguyên tắc biên tập.

POST /api/predict bổ sung care_guide, speech_guidance và cập nhật guidance.
Độ tin cậy thấp hoặc nhãn không xác định chỉ nhận hướng dẫn kiểm tra lại.
Nhãn Cháy lá không suy ra tác nhân hoặc thuốc từ ảnh.
Hướng dẫn không có đơn thuốc, liều hay cam kết khỏi bệnh.

Lịch sử mới giữ bản hướng dẫn và nguồn tại thời điểm nhận diện. Bản ghi cũ thiếu
hướng dẫn dùng catalog hiện tại theo label; bản ghi không có label dùng hướng dẫn
xác nhận lại. gTTS chỉ đọc tóm tắt và lưu ý, không đọc danh sách URL.
Đây là catalog chọn lọc, không phải RAG/Agent tự tra tài liệu hoặc nguồn tự cập nhật.

Kiểm tra backend: chạy python -m unittest test_app.py test_tts.py test_care.py
trong apps/recognition_app bằng Python của .venv.

Trong tab Bệnh, mở từng thẻ để xem hình dạng/màu sắc, vị trí/diễn biến, điểm dễ nhầm và hướng xử lý. Tìm kiếm bao gồm mô tả hình dạng vết bệnh; các nhận xét kiểm tra lá khỏe là gợi ý quan sát của ứng dụng.

### Trợ lý Gemini

Thêm vào `.env` ở thư mục gốc dự án (đã được Git bỏ qua), hoặc `apps/recognition_app/.env`:

```dotenv
GEMINI_API_KEY=dan_khoa_cua_ban_o_day
GEMINI_MODEL=gemini-3.1-flash-lite
```

Khởi động lại backend. Mở **Trợ lý AI**, hỏi về chăm sóc cây hoặc kết quả nhận diện.
Chat giữ tối đa 6 cặp hỏi đáp trong bộ nhớ trang, không lưu hội thoại vào lịch sử trình duyệt.
**Cuộc trò chuyện mới** xóa ngữ cảnh chat và lấy kết quả nhận diện mới nhất.
Nút **Nghe trả lời** đọc tối đa 2.000 ký tự bằng hệ thống gTTS/giọng thiết bị hiện có.
Khóa chỉ dùng ở backend; không gửi khóa qua chat, không commit `.env`.
Câu hỏi, hội thoại và ngữ cảnh bệnh được gửi tới Google; không gửi ảnh hoặc thông tin cá nhân trong lịch sử nhận diện.
Nguồn dưới câu trả lời lấy từ tài liệu đã biên tập của ứng dụng khi được trích dẫn; chưa có tìm kiếm web trực tiếp.
Khi thiếu khóa, hết quota, lỗi quyền hoặc mất mạng, app báo rõ và giữ câu hỏi để gửi lại.
Không tự động chuyển sang dịch vụ trả phí. Hạn mức và chi phí phụ thuộc dự án Google được cấu hình.
API dùng [generateContent chính thức](https://ai.google.dev/api/generate-content).
