# Thu thập ảnh bệnh lá sầu riêng thực tế

Ứng dụng web Flask viết bằng Python để người dân chụp ảnh lá sầu riêng ngoài vườn, nhập nhãn nghi ngờ và metadata thực tế. Dữ liệu dùng để xây dựng bộ dữ liệu phục vụ train/fine-tune và đánh giá model nhận diện bệnh lá sầu riêng.

## Chức năng

- Chụp một ảnh trực tiếp bằng camera trong trình duyệt hoặc camera hệ thống trên điện thoại.
- Kiểm tra cơ bản độ phân giải, độ sáng, độ mờ và ảnh có giống ảnh lá.
- Chọn tên bệnh nghi ngờ hoặc `Không rõ / cần chuyên gia xác nhận`.
- Nhập tuổi cây/giai đoạn sinh trưởng và ghi chú tự do.
- Lấy vị trí GPS nếu người dùng cấp quyền.
- Giao diện tiếng Việt, tông xanh lá và tối ưu cho màn hình điện thoại.
- Lưu ảnh JPEG chất lượng cao và metadata vào Supabase khi có cấu hình env; nếu chạy local chưa cấu hình Supabase thì lưu tạm SQLite.

## Metadata lưu cho mỗi mẫu

- Ảnh lá.
- Nhãn bệnh nghi ngờ.
- Tuổi cây hoặc giai đoạn sinh trưởng.
- Ghi chú tự do.
- Vĩ độ, kinh độ và sai số GPS nếu có.
- Tên vị trí đọc được từ OpenStreetMap nếu tra được.
- Thời gian chụp từ thiết bị và thời gian server nhận mẫu.
- Trạng thái mẫu ban đầu: `submitted`.

## Chạy ứng dụng

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Mở `http://localhost:8501` trên trình duyệt. Muốn dùng điện thoại trong cùng mạng Wi-Fi, mở `http://IP_MAY_TINH:8501`.

Trên iPhone/Android, camera trực tiếp hoạt động ổn nhất khi mở app bằng HTTPS. Nếu đang chạy local bằng HTTP, app sẽ dùng camera hệ thống của điện thoại làm phương án dự phòng.

## Deploy Vercel

Khi import repository trên Vercel:

- Branch: `vuong`
- Root Directory: `apps/data_collection_app`
- Framework Preset: `Other` hoặc để Vercel tự nhận Flask
- Build Command: để trống
- Output Directory: để trống
- Install Command: `pip install -r requirements.txt`

Vercel nhận Flask qua biến `app` trong `app.py`. Khi đã cấu hình `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY` và `SUPABASE_BUCKET`, dữ liệu trên Vercel sẽ được lưu vào Supabase Storage/Postgres. Nếu thiếu các biến này, app chỉ lưu tạm local để chạy thử.

Dữ liệu local được tạo trong thư mục `data/`; thư mục này không được đưa lên Git.

## Chuẩn bị Supabase

Supabase dùng để lưu dữ liệu thật khi deploy:

- Supabase Storage: lưu ảnh gốc/chất lượng cao.
- Supabase Postgres: lưu nhãn và metadata của mỗi mẫu.

### 1. Tạo project

1. Vào Supabase và tạo project mới.
2. Mở `Project Settings` -> `API`.
3. Ghi lại:
   - `Project URL`
   - `service_role key`

Không đưa `service_role key` vào frontend hoặc GitHub. Key này chỉ dùng trong server/env của Vercel.

### 2. Tạo bucket lưu ảnh

Vào `Storage` -> `New bucket`:

- Name: `durian-submissions`
- Public bucket: tắt nếu chỉ chuyên gia/admin xem ảnh qua backend.

Đường dẫn ảnh đề xuất:

```text
raw/{submission_id}.jpg
```

### 3. Tạo bảng metadata

Vào `SQL Editor` và chạy:

```sql
create table if not exists public.submissions (
  id uuid primary key,
  disease text not null,
  image_path text not null,
  tree_stage text,
  notes text,
  latitude double precision,
  longitude double precision,
  location_accuracy double precision,
  location_name text,
  captured_at timestamptz,
  created_at timestamptz not null default now(),
  status text not null default 'submitted',
  expert_label text,
  expert_notes text,
  reviewed_at timestamptz
);

create index if not exists submissions_status_idx
  on public.submissions (status);

create index if not exists submissions_disease_idx
  on public.submissions (disease);

create index if not exists submissions_created_at_idx
  on public.submissions (created_at desc);
```

Nếu bảng `submissions` đã được tạo trước khi có cột tên vị trí, chạy thêm:

```sql
alter table public.submissions
  add column if not exists location_name text;
```

Trạng thái đề xuất:

- `submitted`: nông dân đã gửi.
- `confirmed`: chuyên gia đã xác nhận nhãn.
- `rejected`: ảnh kém chất lượng hoặc không phù hợp.

### 4. Biến môi trường trên Vercel

Cấu hình các biến sau trên Vercel:

```text
SUPABASE_URL=https://PROJECT_ID.supabase.co
SUPABASE_SERVICE_ROLE_KEY=sb_secret_...
SUPABASE_BUCKET=durian-submissions
```

Sau khi thêm hoặc sửa env, vào Vercel `Deployments` và `Redeploy` bản mới nhất để app nhận biến môi trường.

Khi lưu thành công:

- Ảnh nằm trong Supabase `Storage` -> bucket `durian-submissions` -> thư mục `raw/`.
- Metadata nằm trong Supabase `Table Editor` -> bảng `submissions`.

## Cấu trúc

- `app.py`: giao diện và luồng nhập dữ liệu.
- `image_quality.py`: kiểm tra chất lượng ảnh.
- `storage.py`: lưu ảnh, nhãn và metadata vào Supabase hoặc SQLite fallback khi chạy local.
- `test_app.py`: kiểm thử chức năng chính.
