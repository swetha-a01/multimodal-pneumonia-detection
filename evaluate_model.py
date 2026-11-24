from pathlib import Path
import torch
from torch import nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

try:
    from sklearn.metrics import (
        accuracy_score, classification_report, confusion_matrix,
        precision_recall_fscore_support, roc_auc_score, ConfusionMatrixDisplay
    )
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit(
        "Install evaluation packages first: python -m pip install scikit-learn matplotlib\n"
        f"Details: {exc}"
    )

PROJECT_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = PROJECT_DIR / "models" / "pneumonia_resnet18.pt"
DATASET_DIR = Path(
    r"D:\courseProject\deep learning\pneumonia_xray_500_complete"
    r"\pneumonia_xray_500\data\raw\chest_xray"
)
IMAGE_SIZE = 224
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def count_images(folder):
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if not folder.exists():
        return {}
    return {
        class_folder.name: sum(
            1 for f in class_folder.rglob("*")
            if f.is_file() and f.suffix.lower() in extensions
        )
        for class_folder in sorted(folder.iterdir()) if class_folder.is_dir()
    }


def build_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(nn.Dropout(p=0.25), nn.Linear(model.fc.in_features, 2))
    return model


def main():
    print("=" * 64)
    print("PULMOAI — DATASET AND MODEL EVALUATION")
    print("=" * 64)
    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Dataset: {DATASET_DIR}")

    if not CHECKPOINT_PATH.exists():
        raise SystemExit(f"Checkpoint not found: {CHECKPOINT_PATH}")
    if not DATASET_DIR.exists():
        raise SystemExit(
            f"Dataset folder not found: {DATASET_DIR}\n"
            "Edit DATASET_DIR near the top of evaluate_model.py to your dataset path."
        )

    split_counts = {}
    print("\nDATASET SIZE")
    for split in ("train", "val", "valid", "validation", "test"):
        split_dir = DATASET_DIR / split
        if split_dir.exists():
            counts = count_images(split_dir)
            split_counts[split] = counts
            print(f"{split}: {sum(counts.values())} images — " +
                  ", ".join(f"{name}: {num}" for name, num in counts.items()))

    test_dir = DATASET_DIR / "test"
    if not test_dir.exists():
        raise SystemExit(f"Test folder not found: {test_dir}")

    test_data = datasets.ImageFolder(str(test_dir), transform=transform)
    if len(test_data) == 0:
        raise SystemExit(f"No test images found in {test_dir}")
    if len(test_data.classes) != 2:
        raise SystemExit(f"Expected 2 class folders, found {test_data.classes}")

    model = build_model()
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state = checkpoint["model_state"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    else:
        state = checkpoint
    if state and all(k.startswith("module.") for k in state):
        state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(DEVICE).eval()

    loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    actual, predicted, pneumonia_prob = [], [], []
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(DEVICE))
            probabilities = torch.softmax(logits, dim=1)
            preds = probabilities.argmax(dim=1).cpu().tolist()
            actual.extend(labels.tolist())
            predicted.extend(preds)
            pneumonia_prob.extend(probabilities[:, 1].cpu().tolist())

    names = test_data.classes
    accuracy = accuracy_score(actual, predicted)
    precision, recall, f1, _ = precision_recall_fscore_support(
        actual, predicted, average="binary", pos_label=1, zero_division=0
    )
    matrix = confusion_matrix(actual, predicted, labels=list(range(len(names))))
    try:
        auc = roc_auc_score(actual, pneumonia_prob)
    except ValueError:
        auc = None

    print("\nHELD-OUT TEST RESULTS")
    print("-" * 64)
    print(f"Images evaluated: {len(test_data)}")
    print(f"Correct predictions: {sum(a == p for a, p in zip(actual, predicted))}")
    print(f"Accuracy:  {accuracy * 100:.2f}%")
    print(f"Precision (class index 1: {names[1]}): {precision * 100:.2f}%")
    print(f"Recall (class index 1: {names[1]}):    {recall * 100:.2f}%")
    print(f"F1-score (class index 1: {names[1]}):  {f1 * 100:.2f}%")
    print(f"ROC-AUC: {auc:.4f}" if auc is not None else "ROC-AUC: unavailable")
    print("\nClassification report:")
    report = classification_report(
        actual, predicted, labels=list(range(len(names))),
        target_names=names, digits=4, zero_division=0
    )
    print(report)
    print("Confusion matrix (rows=actual, columns=predicted):")
    print(f"Class order: {names}")
    print(matrix)

    out_dir = PROJECT_DIR / "evaluation_results"
    out_dir.mkdir(exist_ok=True)
    report_path = out_dir / "evaluation_report.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write(f"Checkpoint: {CHECKPOINT_PATH}\nDataset: {DATASET_DIR}\nDevice: {DEVICE}\n\n")
        f.write("Dataset counts:\n")
        for split, counts in split_counts.items():
            f.write(f"{split}: {sum(counts.values())} images\n")
            for class_name, count in counts.items():
                f.write(f"  {class_name}: {count}\n")
        f.write(f"\nTest images: {len(test_data)}\nAccuracy: {accuracy:.6f}\n")
        f.write(f"Precision ({names[1]}): {precision:.6f}\nRecall ({names[1]}): {recall:.6f}\n")
        f.write(f"F1 ({names[1]}): {f1:.6f}\nROC-AUC: {auc if auc is not None else 'unavailable'}\n\n")
        f.write(report + "\nConfusion matrix (rows=actual, columns=predicted):\n")
        f.write(f"Class order: {names}\n{matrix}\n")

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=names).plot(
        ax=ax, values_format="d", colorbar=False
    )
    ax.set_title("PulmoAI — Test Set Confusion Matrix")
    fig.tight_layout()
    image_path = out_dir / "confusion_matrix.png"
    fig.savefig(image_path, dpi=160)
    plt.close(fig)

    print("\nSaved files:")
    print(report_path)
    print(image_path)
    print("\nEducational prototype only — not for clinical diagnosis.")


if __name__ == "__main__":
    main()
