"""Self-contained Kaggle DINOv3 training; test split used only after selection."""
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

OUT = Path("/kaggle/working")
MODEL = "vit_small_patch16_dinov3.lvd1689m"


def write_json(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2))


def main():
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "timm==1.0.22"], check=True)
    import timm
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    from PIL import ImageOps

    seed = int(os.environ.get("DURIAN_SEED", "42"))
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is unavailable; refusing an unintended CPU training run.")
    device = torch.device("cuda")
    matches = [p.parent for p in Path("/kaggle/input").rglob("train")
               if p.is_dir() and (p.parent / "val").is_dir() and (p.parent / "test").is_dir()]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one train/val/test dataset root; found {matches}")
    root = matches[0]
    model = timm.create_model(MODEL, pretrained=True, num_classes=0)
    config = timm.data.resolve_model_data_config(model)

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    size = 224
    train_tf = transforms.Compose([
        transforms.Resize((size + 32, size + 32)),
        transforms.RandomResizedCrop(size, scale=(0.84, 1.0)),
        transforms.RandomHorizontalFlip(), transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=.15, contrast=.15, saturation=.10),
        transforms.ToTensor(), transforms.Normalize(mean, std)])
    eval_tf = transforms.Compose([transforms.Resize((size, size)),
        transforms.ToTensor(), transforms.Normalize(mean, std)])
    sets = {s: datasets.ImageFolder(root / s, transform=train_tf if s == "train" else eval_tf)
            for s in ("train", "val", "test")}
    names = sets["train"].classes
    expected = {"Leaf_Algal", "Leaf_Blight", "Leaf_Colletotrichum", "Leaf_Healthy"}
    if set(names) != expected or any(d.classes != names for d in sets.values()):
        raise RuntimeError("Dataset class names/mapping do not match the intended four classes.")
    seen = {}
    manifest = []
    for split, dataset in sets.items():
        for filename, label in dataset.samples:
            digest = hashlib.sha256(Path(filename).read_bytes()).hexdigest()
            if digest in seen and seen[digest] != split:
                raise RuntimeError(f"Identical image content crosses {seen[digest]} and {split}.")
            seen[digest] = split
            manifest.append({"path": str(Path(filename).relative_to(root)), "split": split,
                             "label": names[label], "sha256": digest})
    write_json("split_manifest.json", manifest)
    loaders = {s: DataLoader(d, batch_size=8, shuffle=s == "train", num_workers=0,
                             pin_memory=True) for s, d in sets.items()}

    for param in model.parameters():
        param.requires_grad = False
    for block in model.blocks[-2:]:
        for param in block.parameters():
            param.requires_grad = True
    for param in model.norm.parameters():
        param.requires_grad = True
    head = nn.Sequential(nn.LayerNorm(model.num_features), nn.Dropout(.2),
                         nn.Linear(model.num_features, len(names)))
    network = nn.Sequential(model, head).to(device)
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if p.requires_grad], "lr": 1e-5},
        {"params": head.parameters(), "lr": 3e-4}], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=.5, patience=2)
    criterion = nn.CrossEntropyLoss(label_smoothing=.05)

    def evaluate(loader):
        network.eval()
        matrix = torch.zeros((len(names), len(names)), dtype=torch.int64)
        loss_sum = 0.
        with torch.no_grad():
            for images, labels in loader:
                logits = network(images.to(device))
                loss_sum += criterion(logits, labels.to(device)).item() * len(labels)
                preds = logits.argmax(1).cpu()
                matrix += torch.bincount(labels * len(names) + preds,
                                         minlength=len(names)**2).reshape(len(names), len(names))
        metrics = {}
        for i, name in enumerate(names):
            tp = matrix[i, i].item()
            precision = tp / max(matrix[:, i].sum().item(), 1)
            recall = tp / max(matrix[i].sum().item(), 1)
            metrics[name] = {"precision": precision, "recall": recall,
                "f1": 2*precision*recall/max(precision+recall, 1e-12),
                "support": matrix[i].sum().item()}
        return {"accuracy": matrix.diag().sum().item()/matrix.sum().item(),
                "macro_f1": sum(v["f1"] for v in metrics.values())/len(names),
                "loss": loss_sum/len(loader.dataset), "per_class": metrics,
                "confusion_matrix": matrix.tolist()}

    started = time.time()
    best = -1.
    stale = 0
    history = []
    run_config = {"model": MODEL, "pretrained": True, "seed": seed, "epochs": 20,
        "patience": 5, "batch_size": 8, "image_size": size, "unfreeze_blocks": 2,
        "backbone_lr": 1e-5, "head_lr": 3e-4, "mean": mean, "std": std,
        "preprocessing": "Match DINOv2 training transforms; direct resize 224, ImageNet normalization",
        "class_names": names, "dataset_sizes": {s: len(d) for s, d in sets.items()},
        "torch": torch.__version__, "timm": timm.__version__,
        "gpu": torch.cuda.get_device_name(0), "selection": "validation macro_f1"}
    write_json("run_config.json", run_config)
    print(json.dumps(run_config), flush=True)
    checkpoint = OUT / "dinov3_vits16_field_4class_best.pth"
    for epoch in range(1, 21):
        network.train()
        # Match DINOv2 training/evaluation mode.
        model.train()
        loss_sum = 0.
        for images, labels in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(network(images.to(device)), labels.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 1.)
            optimizer.step()
            loss_sum += loss.item()*len(labels)
        val = evaluate(loaders["val"])
        scheduler.step(val["macro_f1"])
        history.append({"epoch": epoch, "train_loss": loss_sum/len(sets["train"]),
                        "val_accuracy": val["accuracy"], "val_macro_f1": val["macro_f1"],
                        "val_loss": val["loss"]})
        print(json.dumps(history[-1]), flush=True)
        write_json("training_history.json", history)
        if val["macro_f1"] > best:
            best, stale = val["macro_f1"], 0
            torch.save({"model_state_dict": network.state_dict(), "config": run_config,
                        "epoch": epoch, "validation": val,
                        "optimizer_state_dict": optimizer.state_dict()}, checkpoint)
        else:
            stale += 1
        if stale >= 5:
            break
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    network.load_state_dict(saved["model_state_dict"])
    test = evaluate(loaders["test"])
    report = {"config": run_config, "best_epoch": saved["epoch"],
              "validation": saved["validation"], "test": test,
              "history": history, "elapsed_seconds": time.time()-started,
              "note": "Test evaluated once after validation-based checkpoint selection."}
    write_json("dinov3_vits16_field_4class_report.json", report)
    with (OUT / "confusion_matrix.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["actual/predicted", *names])
        writer.writerows([name, *row] for name, row in zip(names, test["confusion_matrix"]))
    print("COMPLETE " + json.dumps(test), flush=True)


if __name__ == "__main__":
    base = OUT
    reports = []
    try:
        for seed in [42, 43, 44]:
            os.environ["DURIAN_SEED"] = str(seed)
            OUT = base / f"seed_{seed}"
            OUT.mkdir(parents=True, exist_ok=True)
            main()
            report = json.loads((OUT / "dinov3_vits16_field_4class_report.json").read_text())
            reports.append({"seed": seed, "best_epoch": report["best_epoch"],
                            "validation": report["validation"], "test": report["test"]})
        import statistics
        OUT = base
        write_json("comparison_summary.json", {
            "runs": reports,
            "test_accuracy_mean": statistics.mean(r["test"]["accuracy"] for r in reports),
            "test_accuracy_sample_std": statistics.stdev(r["test"]["accuracy"] for r in reports),
            "test_macro_f1_mean": statistics.mean(r["test"]["macro_f1"] for r in reports),
            "test_macro_f1_sample_std": statistics.stdev(r["test"]["macro_f1"] for r in reports),
            "selection_rule": "Each seed selects checkpoint using validation macro-F1; report all seeds.",
            "note": "Compare with existing DINOv2 seed42 only; not a paired multi-seed architecture study."
        })
    except Exception as error:
        write_json("failure.json", {"type": type(error).__name__, "message": str(error),
                                   "traceback": traceback.format_exc()})
        raise
