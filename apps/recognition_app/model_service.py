from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch import nn
from torchvision import models, transforms


REPO_ROOT = Path(__file__).resolve().parents[2]
MOBILENET_CHECKPOINT = REPO_ROOT / "ai/models/mobilenetv2/supabase_4class_20261003_balanced110/mobilenetv2_supabase_4class_best.pth"
DINOV2_CHECKPOINT = REPO_ROOT / "ai/models/dinov2/field_balanced110_unfreeze2/dinov2_vits14_field_4class_best.pth"
PAD_COLOR = (188, 205, 184)

LABELS = {
    "Leaf_Algal": {
        "name": "Đốm rong",
        "summary": "Lá có dấu hiệu nghi mắc bệnh đốm rong.",
        "guidance": "Theo dõi vùng đốm, giữ vườn thông thoáng và nhờ cán bộ nông nghiệp kiểm tra trước khi xử lý.",
    },
    "Leaf_Blight": {
        "name": "Cháy lá",
        "summary": "Lá có dấu hiệu nghi mắc bệnh cháy lá.",
        "guidance": "Kiểm tra độ ẩm, khả năng thoát nước và mức độ lan rộng trên các lá lân cận.",
    },
    "Leaf_Colletotrichum": {
        "name": "Thán thư",
        "summary": "Lá có dấu hiệu nghi mắc bệnh thán thư.",
        "guidance": "Cách ly lá bệnh nếu phù hợp và liên hệ người có chuyên môn để xác nhận biện pháp xử lý.",
    },
    "Leaf_Healthy": {
        "name": "Lá khỏe",
        "summary": "Mô hình chưa phát hiện dấu hiệu rõ của ba bệnh đang hỗ trợ.",
        "guidance": "Tiếp tục theo dõi cây; chụp lại nếu lá xuất hiện triệu chứng mới hoặc ảnh hiện tại chưa rõ.",
    },
}


def choose_device() -> torch.device:
    requested = os.environ.get("MODEL_DEVICE", "auto")
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pad_to_square(image: Image.Image) -> Image.Image:
    width, height = image.size
    side = max(width, height)
    canvas = Image.new("RGB", (side, side), PAD_COLOR)
    canvas.paste(image, ((side - width) // 2, (side - height) // 2))
    return canvas


def prepare_image(image: Image.Image, image_size: int) -> torch.Tensor:
    normalized = ImageOps.exif_transpose(image).convert("RGB")
    normalized = pad_to_square(normalized)
    normalized = normalized.resize((512, 512), Image.Resampling.LANCZOS)
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return transform(normalized).unsqueeze(0)


class DINOv2Classifier(nn.Module):
    def __init__(self, backbone: nn.Module, embedding_dim: int, num_classes: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_dim), nn.Dropout(0.20), nn.Linear(embedding_dim, num_classes)
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        if isinstance(features, dict):
            features = features.get("x_norm_clstoken", features.get("x_prenorm"))
        if features.ndim == 3:
            features = features[:, 0]
        return self.classifier(features)


@dataclass
class LoadedModel:
    model: nn.Module
    class_names: list[str]
    image_size: int
    display_name: str


class ModelRegistry:
    def __init__(self) -> None:
        self.device = choose_device()
        self._models: dict[str, LoadedModel] = {}
        self._lock = threading.Lock()

    def available_backends(self) -> list[str]:
        result = []
        if MOBILENET_CHECKPOINT.exists():
            result.append("mobilenetv2")
        if DINOV2_CHECKPOINT.exists():
            result.append("dinov2")
        return result

    def _load_mobilenet(self) -> LoadedModel:
        checkpoint = torch.load(MOBILENET_CHECKPOINT, map_location="cpu", weights_only=False)
        class_names = checkpoint["class_names"]
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.last_channel, len(class_names))
        model.load_state_dict(checkpoint["model_state_dict"])
        return LoadedModel(model, class_names, int(checkpoint["image_size"]), "MobileNetV2")

    def _load_dinov2(self) -> LoadedModel:
        checkpoint = torch.load(DINOV2_CHECKPOINT, map_location="cpu", weights_only=False)
        model_name = checkpoint.get("model_name", "dinov2_vits14")
        hub_repo = os.environ.get("DINOV2_HUB_REPO", "facebookresearch/dinov2")
        backbone = torch.hub.load(hub_repo, model_name, pretrained=False)
        model = DINOv2Classifier(backbone, int(backbone.embed_dim), len(checkpoint["class_names"]))
        model.load_state_dict(checkpoint["model_state_dict"])
        return LoadedModel(
            model,
            checkpoint["class_names"],
            int(checkpoint["image_size"]),
            "DINOv2 ViT-S/14",
        )

    def get(self, backend: str) -> LoadedModel:
        if backend not in {"mobilenetv2", "dinov2"}:
            raise ValueError("Mô hình không được hỗ trợ")
        if backend in self._models:
            return self._models[backend]
        with self._lock:
            if backend not in self._models:
                loaded = self._load_dinov2() if backend == "dinov2" else self._load_mobilenet()
                loaded.model.to(self.device).eval()
                self._models[backend] = loaded
        return self._models[backend]

    def predict(self, image: Image.Image, backend: str) -> dict[str, object]:
        loaded = self.get(backend)
        tensor = prepare_image(image, loaded.image_size).to(self.device)
        with torch.inference_mode():
            probabilities = torch.softmax(loaded.model(tensor), dim=1)[0].cpu()
        scores = [
            {
                "label": label,
                "name": LABELS[label]["name"],
                "probability": round(float(probabilities[index]), 6),
            }
            for index, label in enumerate(loaded.class_names)
        ]
        scores.sort(key=lambda item: item["probability"], reverse=True)
        winner = scores[0]
        details = LABELS[str(winner["label"])]
        threshold = float(os.environ.get("PREDICTION_CONFIDENCE_THRESHOLD", "0.55"))
        return {
            "label": winner["label"],
            "disease_name": winner["name"],
            "confidence": winner["probability"],
            "uncertain": float(winner["probability"]) < threshold,
            "summary": details["summary"],
            "guidance": details["guidance"],
            "probabilities": scores,
            "model": loaded.display_name,
            "backend": backend,
            "device": str(self.device),
        }


registry = ModelRegistry()
