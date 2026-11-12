from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch, MagicMock

from app import create_app
from test_app import FakeRegistry
import tts_service as tts


class TTSTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cache = Path(self.temp.name)
        self.app = create_app(model_registry=FakeRegistry())
        self.app.config["TTS_CACHE_DIR"] = self.cache
        self.client = self.app.test_client()

    @staticmethod
    def fake_generate(text, path):
        path.write_bytes(b"ID3-demo-mp3")

    def test_input_validation(self):
        with patch.object(tts, "_generate") as generate:
            for data in [[], {}, {"text": None}, {"text": "a"}, {"text": "x"*2001}, {"text": "***"}]:
                self.assertEqual(self.client.post("/api/tts", json=data).status_code, 400)
            generate.assert_not_called()

    def test_vietnamese_cleanup(self):
        self.assertEqual(
            tts.clean_for_speech("## **Lá khỏe**\n- Độ tin cậy 92%. [Nguồn: abc]\n[Đọc thêm](https://example.com)"),
            "Lá khỏe Độ tin cậy 92 phần trăm. Đọc thêm",
        )

    def test_generate_serve_and_reuse_persistent_cache(self):
        with patch.object(tts, "_generate", side_effect=self.fake_generate) as generate:
            first = self.client.post("/api/tts", json={"text": "**Xin chào**"}).get_json()
            second = self.client.post("/api/tts", json={"text": "Xin chào"}).get_json()
            self.assertFalse(first["cached"])
            self.assertTrue(second["cached"])
            self.assertEqual(first["audio_url"], second["audio_url"])
            generate.assert_called_once()
        with self.client.get(first["audio_url"]) as response:
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.mimetype, "audio/mpeg")
            self.assertEqual(response.data, b"ID3-demo-mp3")
        # New Flask instance still reuses file cache rather than process memory.
        other = create_app(model_registry=FakeRegistry())
        other.config["TTS_CACHE_DIR"] = self.cache
        with patch.object(tts, "_generate") as generate:
            self.assertTrue(other.test_client().post("/api/tts", json={"text": "Xin chào"}).get_json()["cached"])
            generate.assert_not_called()

    def test_bad_or_missing_audio_returns_404(self):
        for token in ["bad", "a"*64, ".."]:
            self.assertEqual(self.client.get(f"/api/tts/audio/{token}.mp3").status_code, 404)

    def test_provider_failure_sanitized_and_next_request_can_retry(self):
        with patch.object(tts, "_generate", side_effect=RuntimeError("private provider details")):
            response = self.client.post("/api/tts", json={"text": "Xin chào"})
            self.assertEqual(response.status_code, 503)
            self.assertNotIn("private provider", response.get_data(as_text=True))
        self.assertEqual(list(self.cache.glob("*.mp3")), [])
        with patch.object(tts, "_generate", side_effect=self.fake_generate):
            self.assertEqual(self.client.post("/api/tts", json={"text": "Xin chào"}).status_code, 200)

    def test_same_text_concurrent_requests_generate_once(self):
        entered = threading.Event()
        release = threading.Event()
        def generate(text, path):
            entered.set()
            self.assertTrue(release.wait(2))
            self.fake_generate(text, path)
        with patch.object(tts, "_generate", side_effect=generate) as mock, ThreadPoolExecutor(2) as executor:
            first = executor.submit(tts.synthesize, "Xin chào", self.cache)
            self.assertTrue(entered.wait(2))
            second = executor.submit(tts.synthesize, "Xin chào", self.cache)
            release.set()
            self.assertEqual(first.result()[0], second.result()[0])
            self.assertEqual(mock.call_count, 1)

    def test_partial_generation_not_published(self):
        fake = MagicMock()
        def broken_stream():
            yield b"partial"
            raise TimeoutError()
        fake.return_value.stream.side_effect = broken_stream
        with patch("gtts.gTTS", fake):
            with self.assertRaises(tts.TTSUnavailable):
                tts.synthesize("Xin chào", self.cache)
        self.assertEqual(list(self.cache.iterdir()), [])
        kwargs = fake.call_args.kwargs
        self.assertEqual(kwargs["lang"], "vi")
        self.assertFalse(kwargs["slow"])
        self.assertEqual(kwargs["timeout"], (4, 8))

    def test_pruning_and_expired_files(self):
        stale = self.cache / ("a"*64 + ".mp3")
        stale.write_bytes(b"old")
        import os
        os.utime(stale, (0, 0))
        self.assertIsNone(tts.audio_path("a"*64, self.cache))
        with patch.object(tts, "CACHE_LIMIT", 2), patch.object(tts, "_generate", side_effect=self.fake_generate):
            for text in ["Nội dung một", "Nội dung hai", "Nội dung ba"]:
                tts.synthesize(text, self.cache)
        self.assertLessEqual(len(list(self.cache.glob("*.mp3"))), 2)
        self.assertFalse(stale.exists())

    def test_predict_endpoint_does_not_generate_speech(self):
        from test_app import image_bytes
        with patch.object(tts, "_generate") as generate:
            response = self.client.post("/api/predict",
                data={"image": (image_bytes(), "leaf.jpg"), "model": "mobilenetv2"},
                content_type="multipart/form-data")
            self.assertEqual(response.status_code, 200)
            generate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
