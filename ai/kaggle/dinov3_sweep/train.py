"""Self-contained Kaggle DINOv3 training; test split used only after selection."""
import csv
import math
import shutil
import statistics
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


def main(seed=42, blocks=2, backbone_lr=1e-5):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "timm==1.0.22"], check=True)
    import timm
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    from PIL import ImageOps

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

    class Square:
        def __call__(self, image):
            image = ImageOps.exif_transpose(image).convert("RGB")
            size = max(image.size)
            return ImageOps.pad(image, (size, size), color=(188, 205, 184))

    mean, std = config["mean"], config["std"]
    size = 256
    train_tf = transforms.Compose([Square(), transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(), transforms.RandomRotation(12),
        transforms.ColorJitter(.12, .12, .08), transforms.ToTensor(),
        transforms.Normalize(mean, std)])
    eval_tf = transforms.Compose([Square(), transforms.Resize((size, size)),
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
    loaders = {s: DataLoader(d, batch_size=16, shuffle=s == "train", num_workers=0,
                             pin_memory=True) for s, d in sets.items()}

    for param in model.parameters():
        param.requires_grad = False
    for block in model.blocks[-blocks:]:
        for param in block.parameters():
            param.requires_grad = False
    for param in model.norm.parameters():
        param.requires_grad = False
    head = nn.Sequential(nn.LayerNorm(model.num_features), nn.Dropout(.2),
                         nn.Linear(model.num_features, len(names)))
    network = nn.Sequential(model, head).to(device)
    optimizer = torch.optim.AdamW([
        {"params": [p for b in model.blocks[-blocks:] for p in b.parameters()] + list(model.norm.parameters()), "lr": backbone_lr},
        {"params": head.parameters(), "lr": 1e-4}], weight_decay=1e-4)
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
    run_config = {"model": MODEL, "pretrained": True, "seed": seed, "epochs": 45, "head_epochs": 5, "fine_tune_epochs": 40, "warmup_epochs": 3,
        "patience": 8, "batch_size": 16, "image_size": size, "unfreeze_blocks": blocks,
        "backbone_lr": backbone_lr, "head_lr": 1e-4, "mean": mean, "std": std,
        "preprocessing": "EXIF transpose, RGB, square pad (188,205,184), resize 256",
        "class_names": names, "dataset_sizes": {s: len(d) for s, d in sets.items()},
        "torch": torch.__version__, "timm": timm.__version__,
        "gpu": torch.cuda.get_device_name(0), "selection": "validation macro_f1"}
    write_json("run_config.json", run_config)
    print(json.dumps(run_config), flush=True)
    checkpoint = OUT / "dinov3_vits16_field_4class_best.pth"
    for epoch in range(1, 46):
        fine_epoch = epoch - 5
        if fine_epoch == 1:
            best, stale = -1., 0
            for block in model.blocks[-blocks:]:
                block.requires_grad_(True)
            model.norm.requires_grad_(True)
        if fine_epoch > 0:
            factor = fine_epoch / 3 if fine_epoch <= 3 else 0.5 * (1 + math.cos(math.pi * (fine_epoch-3) / 37))
            optimizer.param_groups[0]["lr"] = backbone_lr * factor
            optimizer.param_groups[1]["lr"] = 1e-4 * factor
        network.train()
        # Keep frozen blocks deterministic while fine-tuning the final two.
        model.eval()
        if fine_epoch > 0:
            for block in model.blocks[-blocks:]:
                block.train()
        loss_sum = 0.
        for images, labels in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(network(images.to(device)), labels.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 1.)
            optimizer.step()
            loss_sum += loss.item()*len(labels)
        val = evaluate(loaders["val"])
        history.append({"epoch": epoch, "train_loss": loss_sum/len(sets["train"]),
                        "val_accuracy": val["accuracy"], "val_macro_f1": val["macro_f1"],
                        "val_loss": val["loss"]})
        print(json.dumps(history[-1]), flush=True)
        write_json("training_history.json", history)
        if val["macro_f1"] > best:
            best, stale = val["macro_f1"], 0
            torch.save({"model_state_dict": network.state_dict(), "config": run_config,
                        "epoch": epoch, "validation": val,
                        "stage": "fine_tune" if fine_epoch > 0 else "head"}, checkpoint)
        else:
            stale += 1
        if fine_epoch > 0 and stale >= 8:
            break
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    network.load_state_dict(saved["model_state_dict"])
    report = {"config": run_config, "best_epoch": saved["epoch"],
              "validation": saved["validation"], "history": history,
              "elapsed_seconds": time.time()-started,
              "note": "Validation only. No test metrics used in configuration selection."}
    write_json("validation_report.json", report)
    return report


def test_selected(folder):
    import timm
    import torch
    from torch import nn
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    from PIL import ImageOps
    saved = torch.load(folder / "dinov3_vits16_field_4class_best.pth", map_location="cpu", weights_only=False)
    config = saved["config"]
    backbone = timm.create_model(MODEL, pretrained=False, num_classes=0)
    network = nn.Sequential(backbone, nn.Sequential(nn.LayerNorm(backbone.num_features),
                            nn.Dropout(.2), nn.Linear(backbone.num_features, 4))).cuda().eval()
    network.load_state_dict(saved["model_state_dict"])
    class Square:
        def __call__(self, image):
            image = ImageOps.exif_transpose(image).convert("RGB")
            return ImageOps.pad(image, (max(image.size), max(image.size)), color=(188,205,184))
    tf = transforms.Compose([Square(), transforms.Resize((256,256)), transforms.ToTensor(),
                             transforms.Normalize(config["mean"],config["std"])])
    roots = [p.parent for p in Path("/kaggle/input").rglob("train") if (p.parent/"test").is_dir() and (p.parent/"val").is_dir()]
    dataset = datasets.ImageFolder(roots[0]/"test",transform=tf)
    assert dataset.classes == config["class_names"]
    matrix = torch.zeros((4,4),dtype=torch.int64)
    with torch.no_grad():
        for images, labels in DataLoader(dataset,batch_size=16,num_workers=0):
            predictions=network(images.cuda()).argmax(1).cpu()
            matrix += torch.bincount(labels*4+predictions,minlength=16).reshape(4,4)
    per_class={}
    for i,name in enumerate(dataset.classes):
        tp=matrix[i,i].item()
        precision=tp/max(matrix[:,i].sum().item(),1)
        recall=tp/max(matrix[i].sum().item(),1)
        per_class[name]={"precision":precision,"recall":recall,
                         "f1":2*precision*recall/max(precision+recall,1e-12),
                         "support":matrix[i].sum().item()}
    metrics={"accuracy":matrix.diag().sum().item()/matrix.sum().item(),
             "macro_f1":sum(v["f1"] for v in per_class.values())/4,
             "per_class":per_class,"confusion_matrix":matrix.tolist()}
    report=json.loads((folder/"validation_report.json").read_text())
    report["test"]=metrics
    (folder/"final_report.json").write_text(json.dumps(report,indent=2))
    return report

if __name__ == "__main__":
    base=OUT
    base.mkdir(parents=True,exist_ok=True)
    trials=[]
    try:
        for blocks in [2,4]:
            for lr in [3e-6,1e-5]:
                label=f"blocks{blocks}_lr{lr:g}"
                OUT=base/"search"/label
                OUT.mkdir(parents=True,exist_ok=True)
                report=main(42,blocks,lr)
                trials.append({"label":label,"blocks":blocks,"backbone_lr":lr,
                               "validation_macro_f1":report["validation"]["macro_f1"],
                               "best_epoch":report["best_epoch"]})
                (base/"search_results.json").write_text(json.dumps(trials,indent=2))
        # Stable tie handling: retain the earlier trial; never consult test.
        winner=max(trials,key=lambda x:x["validation_macro_f1"])
        (base/"selected_configuration.json").write_text(json.dumps(winner,indent=2))
        selected=base/"selected"
        shutil.copytree(base/"search"/winner["label"],selected/"seed_42")
        for seed in [43,44]:
            OUT=selected/f"seed_{seed}"
            OUT.mkdir(parents=True,exist_ok=True)
            main(seed,winner["blocks"],winner["backbone_lr"])
        # All configuration choices are frozen before any test evaluation.
        reports=[]
        for seed in [42,43,44]:
            reports.append(test_selected(selected/f"seed_{seed}"))
        summary={"selected_configuration":winner,
                 "runs":[{"seed":r["config"]["seed"],"best_epoch":r["best_epoch"],
                          "validation":r["validation"],"test":r["test"]} for r in reports]}
        for metric in ["accuracy","macro_f1"]:
            values=[r["test"][metric] for r in reports]
            summary["test_"+metric+"_mean"]=statistics.mean(values)
            summary["test_"+metric+"_sample_std"]=statistics.stdev(values)
        (base/"sweep_summary.json").write_text(json.dumps(summary,indent=2))
        # Only selected checkpoints are needed; keep all search reports/history.
        for trial in trials:
            (base/"search"/trial["label"]/"dinov3_vits16_field_4class_best.pth").unlink()
        print("SWEEP COMPLETE "+json.dumps(summary),flush=True)
    except Exception as error:
        (base/"failure.json").write_text(json.dumps({"type":type(error).__name__,
            "message":str(error),"traceback":traceback.format_exc()},indent=2))
        raise
