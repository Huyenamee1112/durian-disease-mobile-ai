# Hướng dẫn xử lý có nguồn

Phiên bản nội dung: 2026-11-12. Đối chiếu nguồn ngày 12/11/2026.
Nội dung do người phát triển biên tập; chưa được chuyên gia BVTV thẩm định trực tiếp.
Không tuyên bố đây là phác đồ điều trị hoặc xác nhận bệnh.

Nguồn chuẩn được khai báo trong care_guides.json, không tải lại khi nhận diện.
Ứng dụng chỉ hiển thị tóm tắt và dẫn bản gốc, không lưu lại toàn bộ tài liệu.

| Nguồn | Phạm vi dùng | Vị trí |
|---|---|---|
| Sổ tay cho nông dân canh tác sầu riêng VietGAP, Cục Trồng trọt/GIZ; Khuyến nông Lâm Đồng lưu trữ | Dấu hiệu và nguyên tắc chăm sóc các nhóm bệnh | 1.11.3.2, trang 48: thán thư; 1.11.3.3–4, trang 48–49: cháy lá; 1.11.3.5, trang 49–50: đốm rong |
| Quy trình canh tác sầu riêng bền vững, bảo đảm an toàn thực phẩm, QĐ 1899/QĐ-TTTV-TTBVTV ngày 30/06/2025; Cục Trồng trọt và BVTV | Nước, dinh dưỡng, thoát nước, thán thư, quản lý dịch hại | 3.3, 3.5–6, 5.1.1a, 5.1.2a, 5.1.4 |
| Phòng chống hạn mặn cho cây sầu riêng trong mùa khô, TTKN Quốc gia, 03/01/2024 | Kiểm tra nguồn tưới, theo dõi hạn/mặn | 1–2; không dùng dự báo 2023–2024 để mô tả hiện tại |

Liên kết và cơ quan có trong catalog, được trả kèm từng kết quả.
Trang công bố của Cục BVTV được dẫn thay PDF có tên Unicode dài dễ lỗi.

Quy tắc biên tập:
- Không suy tác nhân từ nhãn Leaf_Blight. Triệu chứng chỉ để đối chiếu.
- Không sao chép tên thuốc, liều, lịch phun từ tài liệu. Chưa chứng minh thuốc
  cụ thể còn đăng ký đúng cây/bệnh tại thời điểm dùng.
- Việc ghi ngày/ảnh, diễn biến và chụp lại là gợi ý sản phẩm, đánh dấu riêng
  với các khuyến nghị có citation.
- Điều kiện liên hệ chuyên môn là tiêu chí chuyển tiếp do app biên tập,
  không phải thang mức độ bệnh đã được kiểm định.
- Độ tin cậy thấp hoặc nhãn lạ: dùng hướng dẫn xác nhận lại.
- Khi cập nhật nội dung, sửa version/checked_at sau khi đối chiếu lại; kiểm tra
  source_ids và chạy test_care.py.
