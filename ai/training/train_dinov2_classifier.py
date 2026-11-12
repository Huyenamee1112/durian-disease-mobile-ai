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
from torchvision import datasets, transforms


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "ai" / "data" / "supabase_4class_20261003_balanced110_processed"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "ai" / "models" / "dinov2" / "field_balanced110"


class DINOv2Classifier(nn.Module):
    """DINOv2 backbone with a small classification head."""

    def __init__(
        self,
        num_classes: int,
        *,
        model_name: str,
        pretrained: bool,
        unfreeze_last_blocks: int,
        hub_repo: str,
    ) -> None:
        super().__init__()
        self.backbone = torch.hub.load(hub_repo, model_name, pretrained=pretrained)
        embedding_dim = getattr(self.backbone, "embed_dim", None)
        if embedding_dim is None:
            raise ValueError(f"Could not determine embedding dimension for {model_name}")
        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Dropout(0.20),
            nn.Linear(embedding_dim, num_classes),
        )
        self.backbone.requires_grad_(False)
        blocks = getattr(self.backbone, "blocks", None)
        if unfreeze_last_blocks < 0:
            raise ValueError("unfreeze_last_blocks must be zero or greater")
        if unfreeze_last_blocks:
            if blocks is None:
                raise ValueError("DINOv2 backbone does not expose transformer blocks")
            if unfreeze_last_blocks > len(blocks):
                raise ValueError(
                    f"Cannot unfreeze {unfreeze_last_blocks} blocks; backbone has {len(blocks)}"
                )
            for block in blocks[-unfreeze_last_blocks:]:
                block.requires_grad_(True)
            final_norm = getattr(self.backbone, "norm", None)
            if final_norm is not None:
                final_norm.requires_grad_(True)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        if isinstance(features, dict):
            features = features.get("x_norm_clstoken", features.get("x_prenorm"))
        if features is None:
            raise RuntimeError("DINOv2 backbone did not return a supported feature tensor")
        if features.ndim == 3:
            features = features[:, 0]
        return self.classifier(features)


def build_transforms(image_size: int) -> dict[str, transforms.Compose]:
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return {
        "train": transforms.Compose(
            [
                transforms.Resize((image_size + 32, image_size + 32)),
                transforms.RandomResizedCrop(image_size, scale=(0.84, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
                transforms.ToTensor(),
                normalize,
            ]
        ),
        "val": transforms.Compose(
            [transforms.Resize((image_size, image_size)), transforms.ToTensor(), normalize]
        ),
        "test": transforms.Compose(
            [transforms.Resize((image_size, image_size)), transforms.ToTensor(), normalize]
        ),
    }


def build_datasets(data_dir: Path, image_size: int) -> dict[str, datasets.ImageFolder]:
    split_transforms = build_transforms(image_size)
    result: dict[str, datasets.ImageFolder] = {}
    for split in ("train", "val", "test"):
        split_dir = data_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Missing dataset split: {split_dir}")
        result[split] = datasets.ImageFolder(split_dir, transform=split_transforms[split])
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
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if optimizer is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += labels.size(0)
    return {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": total_correct / max(total_samples, 1),
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    class_names: list[str],
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    matrix = torch.zeros((len(class_names), len(class_names)), dtype=torch.int64)
    with torch.no_grad():
        for images, labels in loader:
            predictions = model(images.to(device)).argmax(dim=1).cpu()
            for actual, predicted in zip(labels, predictions):
                matrix[int(actual), int(predicted)] += 1

    per_class: dict[str, dict[str, float | int]] = {}
    precisions: list[float] = []
    recalls: list[float] = []
    f1_scores: list[float] = []
    for index, class_name in enumerate(class_names):
        true_positive = matrix[index, index].item()
        false_positive = matrix[:, index].sum().item() - true_positive
        false_negative = matrix[index, :].sum().item() - true_positive
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        per_class[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(matrix[index, :].sum().item()),
        }
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    return {
        "accuracy": matrix.diag().sum().item() / max(matrix.sum().item(), 1),
        "macro_precision": sum(precisions) / len(precisions),
        "macro_recall": sum(recalls) / len(recalls),
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
    parser = argparse.ArgumentParser(
        description="Train a four-class DINOv2 classifier on the fixed field dataset."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--model-name",
        choices=("dinov2_vits14", "dinov2_vitb14"),
        default="dinov2_vits14",
    )
    parser.add_argument(
        "--hub-repo",
        default="facebookresearch/dinov2",
        help="GitHub repository name or local DINOv2 repository path for torch.hub.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--backbone-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument(
        "--unfreeze-last-blocks",
        type=int,
        default=0,
        help="Fine-tune the last N transformer blocks and final norm; default 0 is linear probing.",
    )
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.image_size % 14 != 0:
        raise ValueError("DINOv2 ViT/14 requires --image-size to be divisible by 14")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = choose_device(args.device)
    datasets_by_split = build_datasets(args.data_dir, args.image_size)
    class_names = datasets_by_split["train"].classes
    if any(dataset.classes != class_names for dataset in datasets_by_split.values()):
        raise ValueError("Class directories differ between train, val, and test splits")
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=split == "train",
            num_workers=args.num_workers,
        )
        for split, dataset in datasets_by_split.items()
    }

    model = DINOv2Classifier(
        len(class_names),
        model_name=args.model_name,
        pretrained=not args.no_pretrained,
        unfreeze_last_blocks=args.unfreeze_last_blocks,
        hub_repo=args.hub_repo,
    ).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    backbone_parameters = [
        parameter for parameter in model.backbone.parameters() if parameter.requires_grad
    ]
    head_parameters = list(model.classifier.parameters())
    trainable_parameters = backbone_parameters + head_parameters
    parameter_groups: list[dict[str, Any]] = [
        {"params": head_parameters, "lr": args.learning_rate}
    ]
    if backbone_parameters:
        parameter_groups.append(
            {"params": backbone_parameters, "lr": args.backbone_learning_rate}
        )
    optimizer = torch.optim.AdamW(
        parameter_groups,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / f"{args.model_name}_field_4class_best.pth"
    report_path = args.output_dir / f"{args.model_name}_field_4class_report.json"
    best_val_macro_f1 = -1.0
    stale_epochs = 0
    history: list[dict[str, float | int]] = []
    started = time.time()

    print(f"Device: {device}")
    print(f"Model: {args.model_name}")
    print(f"Classes: {class_names}")
    print(
        "Dataset sizes: "
        + ", ".join(f"{split}={len(dataset)}" for split, dataset in datasets_by_split.items())
    )
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_parameters):,}")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, loaders["train"], criterion, device, optimizer)
        val_metrics = run_epoch(model, loaders["val"], criterion, device)
        validation = evaluate(model, loaders["val"], class_names, device)
        scheduler.step(validation["macro_f1"])
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": validation["macro_f1"],
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"train_acc={row['train_accuracy']:.3f} "
            f"val_acc={row['val_accuracy']:.3f} "
            f"val_macro_f1={row['val_macro_f1']:.3f}"
        )
        if validation["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = validation["macro_f1"]
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": args.model_name,
                    "class_names": class_names,
                    "image_size": args.image_size,
                    "normalization_mean": [0.485, 0.456, 0.406],
                    "normalization_std": [0.229, 0.224, 0.225],
                    "source_dataset": str(args.data_dir),
                    "unfrozen_last_blocks": args.unfreeze_last_blocks,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping after {epoch} epochs.")
                break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, loaders["test"], class_names, device)
    training_mode = (
        f"selective fine-tuning ({args.unfreeze_last_blocks} final blocks)"
        if args.unfreeze_last_blocks
        else "linear probing"
    )
    report = {
        "model": f"DINOv2 {args.model_name.replace('dinov2_', '').upper()} + classifier",
        "training_mode": training_mode,
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "device": str(device),
        "torch_version": str(torch.__version__),
        "classes": class_names,
        "dataset_sizes": {
            split: len(dataset) for split, dataset in datasets_by_split.items()
        },
        "history": history,
        "best_val_macro_f1": best_val_macro_f1,
        "test_metrics": test_metrics,
        "model_path": str(checkpoint_path),
        "elapsed_seconds": round(time.time() - started, 2),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Best validation macro F1: {best_val_macro_f1:.3f}")
    print(f"Test accuracy: {test_metrics['accuracy']:.3f}")
    print(f"Test macro F1: {test_metrics['macro_f1']:.3f}")
    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Saved report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
