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
DEFAULT_DATA_DIR = REPO_ROOT / "ai" / "data" / "supabase_dataset"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "ai" / "models" / "mobilenetv2"
DEFAULT_INIT_CHECKPOINT = DEFAULT_OUTPUT_DIR / "mobilenetv2_durian_baseline.pth"


def build_transforms(image_size: int) -> dict[str, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomResizedCrop(image_size, scale=(0.78, 1.0)),
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
        split_dir = data_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Missing dataset split directory: {split_dir}")
        result[split] = datasets.ImageFolder(split_dir, transform=transforms_by_split[split])
    return result


def build_loaders(
    dataset_by_split: dict[str, datasets.ImageFolder],
    *,
    batch_size: int,
    num_workers: int,
) -> dict[str, DataLoader]:
    return {
        "train": DataLoader(
            dataset_by_split["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        ),
        "val": DataLoader(
            dataset_by_split["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
        "test": DataLoader(
            dataset_by_split["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
    }


def build_model(
    num_classes: int,
    checkpoint_path: Path | None,
    *,
    freeze_features: bool,
) -> nn.Module:
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)

    if checkpoint_path and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model_state = model.state_dict()
        compatible_state = {
            key: value
            for key, value in state_dict.items()
            if key in model_state and tuple(value.shape) == tuple(model_state[key].shape)
        }
        model_state.update(compatible_state)
        model.load_state_dict(model_state)
        print(
            f"Loaded {len(compatible_state)}/{len(model_state)} compatible tensors "
            f"from {checkpoint_path}"
        )
    elif checkpoint_path:
        print(f"Checkpoint not found, training from scratch: {checkpoint_path}")

    if freeze_features:
        for parameter in model.features.parameters():
            parameter.requires_grad = False
        print("Frozen MobileNetV2 feature extractor; training classifier only.")

    return model


def compute_class_weights(dataset: datasets.ImageFolder, device: torch.device) -> torch.Tensor:
    class_counts = torch.zeros(len(dataset.classes), dtype=torch.float32)
    for _, label in dataset.samples:
        class_counts[label] += 1
    weights = class_counts.sum() / (len(dataset.classes) * class_counts.clamp_min(1))
    return weights.to(device)


def run_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            if optimizer:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)

            if optimizer:
                loss.backward()
                optimizer.step()

            predictions = logits.argmax(dim=1)
            total_loss += loss.item() * labels.size(0)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    return {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": total_correct / max(total_samples, 1),
    }


def evaluate_predictions(
    *,
    model: nn.Module,
    loader: DataLoader,
    class_names: list[str],
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    num_classes = len(class_names)
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            predictions = logits.argmax(dim=1).cpu()
            for true_label, predicted_label in zip(labels, predictions):
                confusion[int(true_label), int(predicted_label)] += 1

    per_class: dict[str, dict[str, float]] = {}
    f1_scores = []
    recalls = []
    precisions = []
    for index, class_name in enumerate(class_names):
        tp = confusion[index, index].item()
        fp = confusion[:, index].sum().item() - tp
        fn = confusion[index, :].sum().item() - tp
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        per_class[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(confusion[index, :].sum().item()),
        }
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    accuracy = confusion.diag().sum().item() / max(confusion.sum().item(), 1)
    return {
        "accuracy": accuracy,
        "macro_precision": sum(precisions) / max(len(precisions), 1),
        "macro_recall": sum(recalls) / max(len(recalls), 1),
        "macro_f1": sum(f1_scores) / max(len(f1_scores), 1),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
        "class_names": class_names,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a 4-class MobileNetV2 baseline on Supabase exported images."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--init-checkpoint", type=Path, default=DEFAULT_INIT_CHECKPOINT)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument(
        "--freeze-features",
        action="store_true",
        help="Freeze MobileNetV2 features and train only the classifier.",
    )
    return parser.parse_args()


def choose_device(requested_device: str) -> torch.device:
    if requested_device != "auto":
        return torch.device(requested_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = choose_device(args.device)
    dataset_by_split = build_datasets(args.data_dir, args.image_size)
    class_names = dataset_by_split["train"].classes
    for split, dataset in dataset_by_split.items():
        if dataset.classes != class_names:
            raise ValueError(f"Class mismatch in {split}: {dataset.classes} != {class_names}")

    loaders = build_loaders(
        dataset_by_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = build_model(
        len(class_names),
        args.init_checkpoint,
        freeze_features=args.freeze_features,
    ).to(device)
    class_weights = compute_class_weights(dataset_by_split["train"], device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = args.output_dir / "mobilenetv2_supabase_4class_best.pth"
    report_path = args.output_dir / "mobilenetv2_supabase_4class_report.json"
    best_val_accuracy = -1.0
    history: list[dict[str, float]] = []

    print(f"Device: {device}")
    print(f"Classes: {class_names}")
    print(
        "Dataset sizes: "
        + ", ".join(f"{split}={len(dataset)}" for split, dataset in dataset_by_split.items())
    )

    started_at = time.time()
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=loaders["train"],
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        val_metrics = run_epoch(
            model=model,
            loader=loaders["val"],
            criterion=criterion,
            device=device,
        )
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"train_loss={row['train_loss']:.4f} train_acc={row['train_accuracy']:.3f} "
            f"val_loss={row['val_loss']:.4f} val_acc={row['val_accuracy']:.3f}"
        )

        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "image_size": args.image_size,
                    "history": history,
                    "source_dataset": str(args.data_dir),
                },
                best_model_path,
            )

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate_predictions(
        model=model,
        loader=loaders["test"],
        class_names=class_names,
        device=device,
    )
    report = {
        "configuration": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "device": str(device),
        "torch_version": str(torch.__version__),
        "classes": class_names,
        "dataset_sizes": {
            split: len(dataset) for split, dataset in dataset_by_split.items()
        },
        "history": history,
        "best_val_accuracy": best_val_accuracy,
        "test_metrics": test_metrics,
        "model_path": str(best_model_path),
        "elapsed_seconds": round(time.time() - started_at, 2),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Best validation accuracy: {best_val_accuracy:.3f}")
    print(f"Test accuracy: {test_metrics['accuracy']:.3f}")
    print(f"Test macro F1: {test_metrics['macro_f1']:.3f}")
    print(f"Saved model: {best_model_path}")
    print(f"Saved report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
