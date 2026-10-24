from __future__ import annotations

import unittest
from io import BytesIO

from PIL import Image

from app import create_app


class FakeRegistry:
    def available_backends(self):
        return ["mobilenetv2"]

    def predict(self, _image, backend):
        return {
            "label": "Leaf_Healthy",
            "disease_name": "Lá khỏe",
            "confidence": 0.91,
            "uncertain": False,
            "summary": "Lá khỏe",
            "guidance": "Tiếp tục theo dõi.",
            "probabilities": [],
            "model": "MobileNetV2",
            "backend": backend,
            "device": "cpu",
        }


def image_bytes():
    buffer = BytesIO()
    Image.new("RGB", (32, 24), "green").save(buffer, "JPEG")
    buffer.seek(0)
    return buffer


class RecognitionAppTest(unittest.TestCase):
    def setUp(self):
        self.client = create_app(model_registry=FakeRegistry()).test_client()

    def test_home_and_health(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertEqual(
            self.client.get("/api/health").get_json()["available_models"],
            ["mobilenetv2"],
        )

    def test_predict_requires_image(self):
        self.assertEqual(self.client.post("/api/predict").status_code, 400)

    def test_predict_image(self):
        response = self.client.post(
            "/api/predict",
            data={"image": (image_bytes(), "leaf.jpg"), "model": "mobilenetv2"},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["result"]["label"], "Leaf_Healthy")


if __name__ == "__main__":
    unittest.main()
