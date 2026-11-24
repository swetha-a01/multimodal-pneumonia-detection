from pathlib import Path
import argparse
import copy
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


def main():
    parser = argparse.ArgumentParser(description="Train an image-only NORMAL/PNEUMONIA baseline.")
    parser.add_argument("--data-dir", default="data/labelled",
                        help="Expected train/NORMAL, train/PNEUMONIA, val/NORMAL, val/PNEUMONIA")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()

    root = Path(args.data_dir)
    train_dir, val_dir = root / "train", root / "val"
    for folder in (train_dir, val_dir):
        if not folder.exists():
            raise FileNotFoundError(
                f"Missing {folder}. Arrange labelled images into train/ and val/ folders, "
                "each containing NORMAL/ and PNEUMONIA/ subfolders. See README.md."
            )

    train_tf = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    val_ds = datasets.ImageFolder(val_dir, transform=val_tf)
    expected = {"NORMAL", "PNEUMONIA"}
    if set(train_ds.classes) != expected or set(val_ds.classes) != expected:
        raise ValueError(f"Expected NORMAL and PNEUMONIA folders. Found train={train_ds.classes}, val={val_ds.classes}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    try:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    except Exception:
        print("Pretrained weights unavailable; initializing ResNet-18 without pretrained weights.")
        model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    best_acc, best_state = -1, None

    for epoch in range(args.epochs):
        model.train()
        running_loss, correct, count = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            count += y.size(0)
        train_acc = correct / max(count, 1)

        model.eval()
        val_correct, val_count, val_loss = 0, 0, 0.0
        with torch.inference_mode():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_loss += criterion(logits, y).item() * y.size(0)
                val_correct += (logits.argmax(1) == y).sum().item()
                val_count += y.size(0)
        val_acc = val_correct / max(val_count, 1)
        print(f"Epoch {epoch+1}/{args.epochs} | train_loss={running_loss/max(count,1):.4f} "
              f"| train_acc={train_acc:.3f} | val_loss={val_loss/max(val_count,1):.4f} "
              f"| val_acc={val_acc:.3f}")
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    out = Path("models")
    out.mkdir(exist_ok=True)
    checkpoint_path = out / "pneumonia_resnet18.pt"
    torch.save({
        "model_state": best_state,
        "classes": train_ds.classes,
        "image_size": args.image_size,
        "validation_accuracy": best_acc,
        "note": "Educational image-only baseline; not for clinical use."
    }, checkpoint_path)
    print(f"Saved best validation checkpoint to {checkpoint_path}")
    print("Reminder: a validation score on a small dataset is not evidence of clinical readiness.")


if __name__ == "__main__":
    main()
