from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "ai" / "data" / "supabase_4class_20261003_balanced110_processed"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "ai" / "models" / "hybrid_resnet18_transformer"


class ResNetTransformerClassifier(nn.Module):
    """A pretrained ResNet spatial feature extractor followed by a Transformer."""

    def __init__(self, num_classes: int, *, backbone_name: str, pretrained: bool, freeze_backbone: bool) -> None:
        super().__init__()
        backbone_options = {
            "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT, 512),
            "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT, 2048),
        }
        constructor, default_weights, embedding_dim = backbone_options[backbone_name]
        resnet = constructor(weights=default_weights if pretrained else None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.embedding_dim = embedding_dim
        nhead = 8
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=nhead,
            dim_feedforward=self.embedding_dim * 2,
            dropout=0.30,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.classifier = nn.Sequential(nn.LayerNorm(self.embedding_dim), nn.Dropout(0.40), nn.Linear(self.embedding_dim, num_classes))
        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        tokens = features.flatten(2).transpose(1, 2)
        encoded = self.transformer(tokens)
        return self.classifier(encoded.mean(dim=1))


def build_transforms(image_size: int) -> dict[str, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomResizedCrop(image_size, scale=(0.84, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return {"train": train_transform, "val": eval_transform, "test": eval_transform}


def build_datasets(data_dir: Path, image_size: int) -> dict[str, datasets.ImageFolder]:
    transforms_by_split = build_transforms(image_size)
    result: dict[str, datasets.ImageFolder] = {}
    for split in ("train", "val", "test"):
        path = data_dir / split
        if not path.exists():
            raise FileNotFoundError(f"Missing split: {path}")
        result[split] = datasets.ImageFolder(path, transform=transforms_by_split[split])
    return result


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = total_correct = total_samples = 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if optimizer:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if optimizer:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += labels.size(0)
    return {"loss": total_loss / max(total_samples, 1), "accuracy": total_correct / max(total_samples, 1)}


def evaluate(model: nn.Module, loader: DataLoader, class_names: list[str], device: torch.device) -> dict[str, Any]:
    model.eval()
    matrix = torch.zeros((len(class_names), len(class_names)), dtype=torch.int64)
    with torch.no_grad():
        for images, labels in loader:
            predictions = model(images.to(device)).argmax(dim=1).cpu()
            for actual, predicted in zip(labels, predictions):
                matrix[int(actual), int(predicted)] += 1

    per_class: dict[str, dict[str, float]] = {}
    f1_scores = []
    for index, class_name in enumerate(class_names):
        true_positive = matrix[index, index].item()
        false_positive = matrix[:, index].sum().item() - true_positive
        false_negative = matrix[index, :].sum().item() - true_positive
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        per_class[class_name] = {"precision": precision, "recall": recall, "f1": f1, "support": int(matrix[index, :].sum().item())}
        f1_scores.append(f1)
    return {
        "accuracy": matrix.diag().sum().item() / max(matrix.sum().item(), 1),
        "macro_f1": sum(f1_scores) / len(f1_scores),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "class_names": class_names,
    }


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a ResNet-Transformer hybrid on a fixed field dataset.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--backbone", choices=("resnet18", "resnet50"), default="resnet18")
    parser.add_argument("--unfreeze-backbone", action="store_true", help="Fine-tune the ResNet backbone instead of keeping it frozen.")
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    datasets_by_split = build_datasets(args.data_dir, args.image_size)
    class_names = datasets_by_split["train"].classes
    if any(dataset.classes != class_names for dataset in datasets_by_split.values()):
        raise ValueError("Class directories differ between splits.")
    loaders = {split: DataLoader(dataset, batch_size=args.batch_size, shuffle=(split == "train"), num_workers=args.num_workers) for split, dataset in datasets_by_split.items()}
    model = ResNetTransformerClassifier(len(class_names), backbone_name=args.backbone, pretrained=not args.no_pretrained, freeze_backbone=not args.unfreeze_backbone).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_id = f"hybrid_{args.backbone}_transformer"
    checkpoint_path = args.output_dir / f"{model_id}_best.pth"
    best_macro_f1, stale_epochs, history = -1.0, 0, []
    started = time.time()
    print(f"Device: {device}")
    print(f"Classes: {class_names}")
    print("Dataset sizes: " + ", ".join(f"{name}={len(dataset)}" for name, dataset in datasets_by_split.items()))
    print(f"Trainable parameters: {sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad):,}")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, loaders["train"], criterion, device, optimizer)
        val_metrics = run_epoch(model, loaders["val"], criterion, device)
        validation = evaluate(model, loaders["val"], class_names, device)
        scheduler.step(validation["macro_f1"])
        row = {"epoch": epoch, "train_loss": train_metrics["loss"], "train_accuracy": train_metrics["accuracy"], "val_loss": val_metrics["loss"], "val_accuracy": val_metrics["accuracy"], "val_macro_f1": validation["macro_f1"]}
        history.append(row)
        print(f"Epoch {epoch:02d}/{args.epochs} train_acc={row['train_accuracy']:.3f} val_acc={row['val_accuracy']:.3f} val_macro_f1={row['val_macro_f1']:.3f}")
        if validation["macro_f1"] > best_macro_f1:
            best_macro_f1, stale_epochs = validation["macro_f1"], 0
            torch.save({"model_state_dict": model.state_dict(), "class_names": class_names, "image_size": args.image_size, "history": history, "source_dataset": str(args.data_dir)}, checkpoint_path)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping after {epoch} epochs.")
                break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, loaders["test"], class_names, device)
    report = {"model": f"{args.backbone.title()} + Transformer Encoder", "configuration": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, "device": str(device), "classes": class_names, "dataset_sizes": {name: len(dataset) for name, dataset in datasets_by_split.items()}, "history": history, "best_val_macro_f1": best_macro_f1, "test_metrics": test_metrics, "model_path": str(checkpoint_path), "elapsed_seconds": round(time.time() - started, 2)}
    report_path = args.output_dir / f"{model_id}_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Best validation macro F1: {best_macro_f1:.3f}")
    print(f"Test accuracy: {test_metrics['accuracy']:.3f}")
    print(f"Test macro F1: {test_metrics['macro_f1']:.3f}")
    print(f"Saved report: {report_path}")
    return 0



if __name__ == "__main__":
    import sys
    sys.argv = [
        "train_hybrid_resnet_transformer.py",
        "--backbone", "resnet50",
        "--data-dir", "/kaggle/input/durian-field-balanced110",
        "--output-dir", "/kaggle/working/resnet50_transformer_field",
        "--epochs", "15",
        "--patience", "5",
        "--batch-size", "8",
        "--device", "cuda",
    ]
    raise SystemExit(main())
