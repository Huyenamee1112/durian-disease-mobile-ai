from __future__ import annotations

import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


SYSTEM_PROMPT = """Bạn là trợ lý hỗ trợ chăm sóc lá sầu riêng. Trả lời ngắn, dễ hiểu bằng tiếng Việt.
Chỉ sử dụng kết quả nhận diện như thông tin tham khảo, không khẳng định chẩn đoán chắc chắn.
Không tự đưa liều lượng thuốc hoặc hóa chất. Khi ảnh không rõ, độ tin cậy thấp hoặc cây có biểu hiện nặng,
hãy khuyên người dùng liên hệ cán bộ bảo vệ thực vật hoặc chuyên gia địa phương."""


def _fallback_answer(question: str, diagnosis: dict[str, object]) -> str:
    disease = diagnosis.get("disease_name", "chưa xác định")
    confidence = round(float(diagnosis.get("confidence", 0)) * 100)
    guidance = diagnosis.get("guidance", "Hãy theo dõi cây và tham khảo người có chuyên môn.")
    return (
        f"Kết quả hiện tại nghiêng về {disease} với độ tin cậy {confidence}%. "
        f"{guidance} Câu hỏi của bạn đã được ghi nhận: {question.strip()}"
    )


def ask_agent(question: str, diagnosis: dict[str, object]) -> dict[str, object]:
    api_key = os.environ.get("AI_AGENT_API_KEY", "")
    base_url = os.environ.get("AI_AGENT_BASE_URL", "").rstrip("/")
    model = os.environ.get("AI_AGENT_MODEL", "")
    if not api_key or not base_url or not model:
        return {"answer": _fallback_answer(question, diagnosis), "source": "local_fallback"}

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Kết quả mô hình: {json.dumps(diagnosis, ensure_ascii=False)}\n"
                    f"Câu hỏi: {question.strip()}"
                ),
            },
        ],
        "temperature": 0.2,
    }
    request = Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=30) as response:
            result = json.loads(response.read().decode("utf-8"))
        return {"answer": result["choices"][0]["message"]["content"], "source": "llm"}
    except (HTTPError, URLError, KeyError, ValueError) as error:
        return {
            "answer": _fallback_answer(question, diagnosis),
            "source": "local_fallback",
            "warning": f"Không gọi được trợ lý trực tuyến: {error}",
        }
