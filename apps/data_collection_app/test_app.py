import os
import sqlite3
import tempfile
import unittest
from io import BytesIO

from PIL import Image

os.environ["DURIAN_DATA_DIR"] = tempfile.mkdtemp(prefix="durian-test-")

from image_quality import inspect_image
from storage import DATABASE_PATH, IMAGE_DIR, save_submission


class DataCollectionTest(unittest.TestCase):
    def test_image_and_submission_are_saved(self) -> None:
        image = Image.new("RGB", (800, 800), "green")
        submission_id = save_submission(image=image, disease="Unknown")

        self.assertTrue((IMAGE_DIR / f"{submission_id}.jpg").exists())
        with sqlite3.connect(DATABASE_PATH) as connection:
            disease = connection.execute(
                "SELECT disease FROM submissions WHERE id = ?", (submission_id,)
            ).fetchone()[0]
        self.assertEqual(disease, "Unknown")

    def test_invalid_image_is_rejected(self) -> None:
        result = inspect_image(b"not an image")
        self.assertFalse(result.acceptable)

    def test_normal_photo_is_accepted(self) -> None:
        image = Image.effect_noise((800, 800), 80).convert("RGB")
        buffer = BytesIO()
        image.save(buffer, "JPEG")
        self.assertTrue(inspect_image(buffer.getvalue()).acceptable)


if __name__ == "__main__":
    unittest.main()
