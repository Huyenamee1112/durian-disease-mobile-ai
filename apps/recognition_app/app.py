from __future__ import annotations

import os
from io import BytesIO
from typing import Any

from flask import Flask, jsonify, render_template, request, send_file, url_for
from PIL import Image, UnidentifiedImageError

from agent_service import ask_agent
from care_service import CATALOG as CARE_CATALOG, care_response
from history_service import save_history
from model_service import registry
from pathlib import Path

from tts_service import TTSUnavailable, available as tts_available, audio_path, synthesize


ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "HEIC", "HEIF"}


def create_app(*, model_registry: Any = registry) -> Flask:
    app = Flask(__name__)
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024
    app.config["MODEL_REGISTRY"] = model_registry
    app.config["TTS_CACHE_DIR"] = Path(app.instance_path) / "tts_audio"

    @app.get("/")
    def index():
        default_backend = os.environ.get("MODEL_BACKEND", "dinov2")
        response = app.make_response(render_template(
            "index.html",
            default_backend=default_backend,
            available_backends=model_registry.available_backends(),
            tts_config={"available": tts_available(), "provider": "gtts"},
            care_catalog=CARE_CATALOG,
        ))
        response.headers["Cache-Control"] = "no-cache"
        return response

    @app.get("/api/health")
    def health():
        return jsonify(
            ok=True,
            available_models=model_registry.available_backends(),
            default_model=os.environ.get("MODEL_BACKEND", "dinov2"),
        )

    @app.post("/api/predict")
    def predict():
        upload = request.files.get("image")
        if upload is None or not upload.filename:
            return jsonify(ok=False, message="Vui lòng chụp hoặc chọn một ảnh lá."), 400
        backend = request.form.get("model") or os.environ.get("MODEL_BACKEND", "dinov2")
        try:
            image_bytes = upload.read()
            image = Image.open(BytesIO(image_bytes))
            image.load()
            if image.format and image.format.upper() not in ALLOWED_FORMATS:
                return jsonify(ok=False, message="Định dạng ảnh chưa được hỗ trợ."), 400
            result = model_registry.predict(image, backend)
        except UnidentifiedImageError:
            return jsonify(ok=False, message="Tệp đã chọn không phải ảnh hợp lệ."), 400
        except (OSError, RuntimeError, ValueError) as error:
            app.logger.exception("Prediction failed")
            return jsonify(ok=False, message=f"Không thể nhận diện ảnh: {error}"), 500
        result.update(care_response(result.get("label"), result.get("uncertain", False)))
        result["history_saved"] = save_history(
            result,
            image_bytes=image_bytes,
            content_type=upload.mimetype or "image/jpeg",
        )
        return jsonify(ok=True, result=result)

    @app.post("/api/agent")
    def agent():
        payload = request.get_json(silent=True) or {}
        if not isinstance(payload, dict):
            return jsonify(ok=False, message="Yêu cầu không hợp lệ."), 400
        question = str(payload.get("question", "")).strip()
        diagnosis = payload.get("diagnosis") or {}
        if not isinstance(diagnosis, dict) or not isinstance(payload.get("history", []), list):
            return jsonify(ok=False, message="Ngữ cảnh không hợp lệ."), 400
        if len(question) > 2000 or len(payload.get("history", [])) > 20:
            return jsonify(ok=False, message="Hội thoại quá dài. Hãy tạo cuộc trò chuyện mới."), 400
        if not question:
            return jsonify(ok=False, message="Vui lòng nhập câu hỏi."), 400
        return jsonify(ok=True, **ask_agent(question, diagnosis, payload.get("history", [])))

    @app.post("/api/tts")
    def tts():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify(ok=False, message="Nội dung yêu cầu không hợp lệ."), 400
        try:
            token, cached = synthesize(payload.get("text"), app.config["TTS_CACHE_DIR"])
        except ValueError as error:
            return jsonify(ok=False, message=str(error)), 400
        except TTSUnavailable as error:
            return jsonify(ok=False, message=str(error)), 503
        response = jsonify(ok=True, audio_url=url_for("tts_audio", token=token), cached=cached, provider="gtts")
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/api/tts/audio/<token>.mp3")
    def tts_audio(token):
        path = audio_path(token, app.config["TTS_CACHE_DIR"])
        if path is None:
            return jsonify(ok=False, message="Âm thanh không còn trong bộ nhớ đệm. Hãy nhấn Nghe để tạo lại."), 404
        response = send_file(path, mimetype="audio/mpeg", conditional=True, max_age=86400)
        response.headers["Cache-Control"] = "private, max-age=86400"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.errorhandler(413)
    def too_large(_error):
        return jsonify(ok=False, message="Ảnh vượt quá dung lượng 12 MB."), 413

    return app


app = create_app()


if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5001")), debug=debug)
