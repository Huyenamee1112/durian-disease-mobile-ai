import os
import sqlite3
import tempfile
import unittest
from io import BytesIO
from unittest.mock import patch

from PIL import Image, ImageDraw
from PIL import ImageFilter

os.environ["DURIAN_DATA_DIR"] = tempfile.mkdtemp(prefix="durian-test-")

from image_quality import inspect_image
import storage
from storage import DATABASE_PATH, IMAGE_DIR, StorageError, save_submission
from app import app


def make_leaf_photo() -> Image.Image:
    image = Image.new("RGB", (800, 800), "#2c7a3f")
    draw = ImageDraw.Draw(image)
    draw.ellipse((80, 120, 720, 660), fill="#367f42")
    draw.line((160, 390, 660, 360), fill="#9abf67", width=14)
    for offset in range(0, 440, 70):
        draw.line((210 + offset, 382, 275 + offset, 260), fill="#80a75a", width=6)
        draw.line((210 + offset, 382, 275 + offset, 500), fill="#80a75a", width=6)
    for center in [(250, 260), (360, 310), (470, 250), (560, 390)]:
        x, y = center
        draw.ellipse((x - 22, y - 16, x + 22, y + 16), fill="#d39b28")
    return image


def make_person_like_photo() -> Image.Image:
    image = Image.new("RGB", (800, 800), "#9fc7f2")
    draw = ImageDraw.Draw(image)
    draw.ellipse((230, 120, 570, 510), fill="#c8896f")
    draw.rectangle((120, 510, 680, 800), fill="#eee6d8")
    draw.rectangle((245, 80, 555, 210), fill="#1f1b18")
    draw.ellipse((300, 280, 330, 310), fill="#1f1b18")
    draw.ellipse((470, 280, 500, 310), fill="#1f1b18")
    return image


class DataCollectionTest(unittest.TestCase):
    def test_image_and_submission_are_saved(self) -> None:
        image = Image.new("RGB", (800, 800), "green")
        submission_id = save_submission(
            image=image,
            disease="Unknown",
            tree_stage="Cây 3 năm",
            notes="Sau mưa",
            latitude="10.123",
            longitude="106.456",
            location_accuracy="12",
            location_name="Phường A, Quận B, Thành phố C",
            captured_at="2026-08-19T09:00:00.000Z",
        )

        self.assertTrue((IMAGE_DIR / f"{submission_id}.jpg").exists())
        with sqlite3.connect(DATABASE_PATH) as connection:
            row = connection.execute(
                """
                SELECT disease, tree_stage, notes, latitude, longitude, location_name, status
                FROM submissions
                WHERE id = ?
                """,
                (submission_id,),
            ).fetchone()
        self.assertEqual(row[0], "Unknown")
        self.assertEqual(row[1], "Cây 3 năm")
        self.assertEqual(row[2], "Sau mưa")
        self.assertAlmostEqual(row[3], 10.123)
        self.assertAlmostEqual(row[4], 106.456)
        self.assertEqual(row[5], "Phường A, Quận B, Thành phố C")
        self.assertEqual(row[6], "submitted")

    def test_invalid_image_is_rejected(self) -> None:
        result = inspect_image(b"not an image")
        self.assertFalse(result.acceptable)

    def test_normal_photo_is_accepted(self) -> None:
        image = make_leaf_photo()
        buffer = BytesIO()
        image.save(buffer, "JPEG")
        self.assertTrue(inspect_image(buffer.getvalue()).acceptable)

    def test_person_photo_is_rejected(self) -> None:
        image = make_person_like_photo()
        buffer = BytesIO()
        image.save(buffer, "JPEG")
        self.assertFalse(inspect_image(buffer.getvalue()).acceptable)

    def test_blurry_leaf_photo_is_rejected(self) -> None:
        image = make_leaf_photo().filter(ImageFilter.GaussianBlur(radius=12))
        buffer = BytesIO()
        image.save(buffer, "JPEG")
        self.assertFalse(inspect_image(buffer.getvalue()).acceptable)

    def test_flask_inspect_rejects_person_photo(self) -> None:
        image = make_person_like_photo()
        buffer = BytesIO()
        image.save(buffer, "JPEG")
        buffer.seek(0)

        response = app.test_client().post(
            "/inspect",
            data={"image": (buffer, "person.jpg")},
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 400)
        self.assertFalse(response.get_json()["ok"])

    def test_large_upload_returns_json_error(self) -> None:
        original_limit = app.config["MAX_CONTENT_LENGTH"]
        app.config["MAX_CONTENT_LENGTH"] = 128
        try:
            response = app.test_client().post(
                "/inspect",
                data={"image": (BytesIO(b"x" * 2048), "large.jpg")},
                content_type="multipart/form-data",
            )
        finally:
            app.config["MAX_CONTENT_LENGTH"] = original_limit

        self.assertEqual(response.status_code, 413)
        self.assertFalse(response.get_json()["ok"])
        self.assertIn("Ảnh quá lớn", response.get_json()["message"])

    def test_flask_submit_saves_image(self) -> None:
        image = make_leaf_photo()
        buffer = BytesIO()
        image.save(buffer, "JPEG")
        buffer.seek(0)

        response = app.test_client().post(
            "/submit",
            data={
                "disease": "Không rõ / cần chuyên gia xác nhận",
                "tree_stage": "Ra đọt non",
                "notes": "Lá bị đốm ở mép",
                "latitude": "10.1",
                "longitude": "106.2",
                "location_accuracy": "20",
                "location_name": "Xã Ea Kpam, Cư M'gar, Đắk Lắk",
                "captured_at": "2026-08-19T10:00:00.000Z",
                "image": (buffer, "leaf.jpg"),
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.get_json()["ok"])
        self.assertIn("Gửi dữ liệu thành công", response.get_json()["message"])

    def test_supabase_storage_is_used_when_configured(self) -> None:
        calls = []

        def fake_request_supabase(**kwargs):
            calls.append(kwargs)
            return b""

        with (
            patch.object(storage, "SUPABASE_URL", "https://example.supabase.co"),
            patch.object(storage, "SUPABASE_SERVICE_ROLE_KEY", "sb_secret_test"),
            patch.object(storage, "SUPABASE_BUCKET", "durian-submissions"),
            patch.object(storage, "_request_supabase", side_effect=fake_request_supabase),
        ):
            submission_id = save_submission(
                image=make_leaf_photo(),
                disease="Unknown",
                captured_at="2026-08-19T10:00:00.000Z",
            )

        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["method"], "POST")
        self.assertIn(f"/storage/v1/object/durian-submissions/raw/{submission_id}.jpg", calls[0]["path"])
        self.assertEqual(calls[0]["content_type"], "image/jpeg")
        self.assertEqual(calls[1]["path"], "/rest/v1/submissions")
        self.assertEqual(calls[1]["method"], "POST")

    def test_supabase_insert_retries_without_location_name_when_column_is_missing(self) -> None:
        calls = []

        def fake_request_supabase(**kwargs):
            calls.append(kwargs)
            if kwargs["path"] == "/rest/v1/submissions" and len(calls) == 2:
                raise StorageError("Could not find the 'location_name' column")
            return b""

        with (
            patch.object(storage, "SUPABASE_URL", "https://example.supabase.co"),
            patch.object(storage, "SUPABASE_SERVICE_ROLE_KEY", "sb_secret_test"),
            patch.object(storage, "SUPABASE_BUCKET", "durian-submissions"),
            patch.object(storage, "_request_supabase", side_effect=fake_request_supabase),
        ):
            save_submission(
                image=make_leaf_photo(),
                disease="Leaf_Blight",
                location_name="Phường Hạnh Thông",
                captured_at="2026-08-19T10:00:00.000Z",
            )

        self.assertEqual(len(calls), 3)
        retry_body = calls[2]["body"].decode("utf-8")
        self.assertNotIn("location_name", retry_body)


if __name__ == "__main__":
    unittest.main()
