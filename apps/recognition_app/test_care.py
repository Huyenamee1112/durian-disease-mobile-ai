import unittest
from unittest.mock import patch

from app import create_app
from care_service import CATALOG, care_response
from test_app import FakeRegistry, image_bytes


class CareGuideTest(unittest.TestCase):
    def test_each_class_has_actions_and_resolvable_sources(self):
        self.assertEqual(set(CATALOG["guides"]), {"Leaf_Algal", "Leaf_Blight", "Leaf_Colletotrichum", "Leaf_Healthy"})
        for label in CATALOG["guides"]:
            result = care_response(label)
            guide = result["care_guide"]
            self.assertEqual(guide["status"], "reference")
            self.assertGreaterEqual(len(guide["steps"]), 3)
            sources = {source["id"] for source in guide["sources"]}
            self.assertTrue(sources)
            for source in guide["sources"]:
                self.assertTrue(source["url"].startswith("https://"))
                self.assertTrue(source["publisher"])
                self.assertTrue(source["section"])
                self.assertEqual(source["checked_at"], "12/11/2026")
            for item in guide["symptoms"] + guide["steps"] + guide.get("appearance", []):
                self.assertTrue(set(item["source_ids"]).issubset(sources))
            self.assertEqual(result["guidance"], guide["summary"])
            self.assertNotIn("https://", result["speech_guidance"])

    def test_uncertain_healthy_or_disease_uses_confirmation_not_treatment(self):
        for label in CATALOG["guides"]:
            result = care_response(label, uncertain=True)
            self.assertEqual(result["care_guide"]["status"], "unconfirmed")
            self.assertEqual(result["care_guide"]["name"], "Cần kiểm tra thêm")
            self.assertIn("chưa chắc chắn", result["guidance"])

    def test_unknown_label_and_no_label_have_safe_generic_guidance(self):
        for label in [None, "unknown"]:
            self.assertEqual(care_response(label)["care_guide"]["status"], "unconfirmed")

    def test_responses_do_not_mutate_catalog_or_other_results(self):
        first = care_response("Leaf_Algal")
        first["care_guide"]["steps"].clear()
        self.assertTrue(care_response("Leaf_Algal")["care_guide"]["steps"])

    def test_predict_returns_sources_and_canonical_guidance(self):
        client = create_app(model_registry=FakeRegistry()).test_client()
        with patch("app.save_history", return_value=False):
            response = client.post("/api/predict", data={
                "image": (image_bytes(), "leaf.jpg"), "model": "mobilenetv2",
            }, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 200)
        result = response.get_json()["result"]
        self.assertEqual(result["care_guide"]["name"], "Lá khỏe")
        self.assertEqual(result["guidance"], care_response("Leaf_Healthy")["guidance"])
        self.assertTrue(result["care_guide"]["sources"])

    def test_low_confidence_predict_never_returns_disease_steps(self):
        class UncertainRegistry(FakeRegistry):
            def predict(self, image, backend):
                result = super().predict(image, backend)
                result.update(label="Leaf_Colletotrichum", uncertain=True, confidence=.3)
                return result
        client = create_app(model_registry=UncertainRegistry()).test_client()
        with patch("app.save_history", return_value=False):
            response = client.post("/api/predict", data={
                "image": (image_bytes(), "leaf.jpg"),
            }, content_type="multipart/form-data")
        self.assertEqual(response.get_json()["result"]["care_guide"]["status"], "unconfirmed")

    def test_home_includes_offline_guides(self):
        client = create_app(model_registry=FakeRegistry()).test_client()
        page = client.get("/").get_data(as_text=True)
        self.assertIn("window.DURIAN_CARE", page)
        self.assertIn('id="care-details"', page)
