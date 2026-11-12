"""Vietnamese gTTS audio with persistent, bounded MP3 cache."""
from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import re
import tempfile
import threading
import time
import unicodedata

MAX_TEXT_LENGTH = 2000
CACHE_TTL = 7 * 24 * 3600
CACHE_LIMIT = 128
GENERATION_BUDGET = 25
NETWORK_TIMEOUT = (4, 8)
_guard = threading.Lock()
_inflight: dict[tuple[str, str], threading.Event] = {}
_slots = threading.BoundedSemaphore(2)


class TTSUnavailable(RuntimeError):
    pass


def available() -> bool:
    return importlib.util.find_spec("gtts") is not None


def clean_for_speech(text: str) -> str:
    # Follow CoffeeAgent's Vietnamese/Markdown cleanup, preserving spoken content.
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\[Nguồn:[^\]]*\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"^\s*#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-•]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[*_`]+", "", text)
    text = re.sub(r"[📋🌤️💊🛡️⚠️🌿🔍📊✓→•]", "", text)
    text = text.replace("%", " phần trăm")
    return re.sub(r"\s+", " ", text).strip()


def audio_path(token: str, cache_dir: Path) -> Path | None:
    if not re.fullmatch(r"[0-9a-f]{64}", token):
        return None
    path = Path(cache_dir) / f"{token}.mp3"
    try:
        stat = path.stat()
        if path.is_symlink() or stat.st_size == 0 or time.time() - stat.st_mtime > CACHE_TTL:
            return None
    except OSError:
        return None
    return path


def _prune(cache_dir: Path, keep: Path):
    files = sorted(cache_dir.glob("*.mp3"), key=lambda p: p.stat().st_mtime, reverse=True)
    for index, path in enumerate(files):
        if path != keep and (index >= CACHE_LIMIT or time.time() - path.stat().st_mtime > CACHE_TTL):
            path.unlink(missing_ok=True)


def _generate(text: str, destination: Path):
    # Lazy import keeps diagnosis usable if the optional voice dependency is absent.
    from gtts import gTTS

    deadline = time.monotonic() + GENERATION_BUDGET
    temp_path = None
    stream = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".tmp", delete=False) as output:
            temp_path = Path(output.name)
            tts = gTTS(text=text, lang="vi", slow=False, lang_check=False, timeout=NETWORK_TIMEOUT)
            stream = tts.stream()
            size = 0
            for chunk in stream:
                if time.monotonic() > deadline:
                    raise TimeoutError("TTS generation exceeded its budget")
                size += len(chunk)
                if size > 8 * 1024 * 1024:
                    raise ValueError("Audio exceeded maximum size")
                output.write(chunk)
            if not size:
                raise ValueError("Empty audio response")
        os.replace(temp_path, destination)
    finally:
        if stream is not None and hasattr(stream, "close"):
            stream.close()
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def synthesize(text: str, cache_dir: Path) -> tuple[str, bool]:
    if not isinstance(text, str) or not 3 <= len(text.strip()) <= MAX_TEXT_LENGTH:
        raise ValueError(f"Nội dung đọc phải từ 3 đến {MAX_TEXT_LENGTH:,} ký tự.")
    clean_text = clean_for_speech(text)
    if len(clean_text) < 3 or not any(character.isalnum() for character in clean_text):
        raise ValueError("Không có nội dung phù hợp để đọc.")
    token = hashlib.sha256(("gtts-vi-v1\0" + clean_text).encode("utf-8")).hexdigest()
    cache_dir = Path(cache_dir).resolve()
    key = (str(cache_dir), token)
    if audio_path(token, cache_dir):
        return token, True
    with _guard:
        event = _inflight.get(key)
        owner = event is None
        if owner:
            event = _inflight[key] = threading.Event()
    if not owner:
        if event.wait(40) and audio_path(token, cache_dir):
            return token, True
        raise TTSUnavailable("Giọng đọc Google chưa sẵn sàng. Vui lòng thử lại.")
    acquired = False
    try:
        # Another request/process may have completed this audio in the meantime.
        if audio_path(token, cache_dir):
            return token, True
        acquired = _slots.acquire(blocking=False)
        if not acquired:
            raise TTSUnavailable("Giọng đọc đang bận. Vui lòng thử lại sau.")
        cache_dir.mkdir(parents=True, exist_ok=True)
        _generate(clean_text, cache_dir / f"{token}.mp3")
        with _guard:
            _prune(cache_dir, cache_dir / f"{token}.mp3")
        return token, False
    except TTSUnavailable:
        raise
    except Exception as error:
        raise TTSUnavailable("Không tạo được giọng Google. Kiểm tra kết nối Internet hoặc thử lại sau.") from error
    finally:
        if acquired:
            _slots.release()
        with _guard:
            _inflight.pop(key, None)
            event.set()
