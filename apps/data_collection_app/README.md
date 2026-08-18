# Ứng dụng góp ảnh lá sầu riêng

Ứng dụng web viết hoàn toàn bằng Python để người dân chụp ảnh lá sầu riêng và chọn tên bệnh. Ứng dụng không tự dự đoán bệnh hoặc mức độ bệnh.

## Chức năng

- Chụp một ảnh trực tiếp bằng camera trong trình duyệt; không tải ảnh có sẵn.
- Kiểm tra cơ bản độ phân giải, độ sáng và độ mờ.
- Chọn tên bệnh hoặc `Không rõ bệnh`.
- Không nhập mức độ bệnh, địa điểm hoặc ghi chú.
- Giao diện tiếng Việt, tông xanh lá và tối ưu cho màn hình điện thoại.
- Lưu ảnh JPEG và metadata vào SQLite trên máy chạy ứng dụng.

## Chạy ứng dụng

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Mở địa chỉ Streamlit hiển thị trên terminal. Muốn dùng điện thoại trong cùng mạng Wi-Fi, chạy:

```powershell
python -m streamlit run app.py --server.address 0.0.0.0
```

Sau đó mở `http://IP_MAY_TINH:8501` trên điện thoại. Dữ liệu được tạo trong thư mục `data/`; thư mục này không được đưa lên Git.

## Cấu trúc

- `app.py`: giao diện và luồng nhập dữ liệu.
- `image_quality.py`: kiểm tra chất lượng ảnh.
- `storage.py`: lưu ảnh và tên bệnh vào SQLite.
- `test_app.py`: kiểm thử chức năng chính.
