
from pathlib import Path
from io import BytesIO
import json

from datasets import load_dataset
from remotezip import RemoteZip
from PIL import Image

MAX_SAMPLES = 20

DATASET = "dz-osamu/IU-Xray"
ZIP_URL = (
    "https://huggingface.co/datasets/dz-osamu/"
    "IU-Xray/resolve/main/image.zip"
)

OUT = Path("data/paired_reports")
IMG_DIR = OUT / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

print("Loading report metadata...")
dataset = load_dataset(
    DATASET,
    split="train",
    streaming=True
)

saved = []
seen_studies = set()

try:
    with RemoteZip(ZIP_URL) as archive:
        members = archive.namelist()

        # Build a lookup using the relative image path.
        # ZIP prefix is different from the metadata path.
        lookup = {}
        for member in members:
            if not member.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            marker = "/iu_xray/iu_xray/images/"
            if marker in member:
                relative = member.split(marker, 1)[1]
                lookup[relative] = member

        print(f"Indexed {len(lookup)} image files.")

        for row in dataset:
            if len(saved) >= MAX_SAMPLES:
                break

            report = row.get("response", "")
            image_paths = row.get("images", [])

            if not isinstance(report, str) or not report.strip():
                continue

            if isinstance(image_paths, str):
                image_paths = [image_paths]

            # Choose the first view for this study.
            chosen_member = None
            study_id = None

            for path in image_paths:
                if not isinstance(path, str):
                    continue

                path = path.replace("\\", "/").lstrip("/")
                parts = path.split("/")

                if len(parts) < 3:
                    continue

                study = parts[-2]
                filename = parts[-1]
                relative = f"{study}/{filename}"

                if relative in lookup:
                    chosen_member = lookup[relative]
                    study_id = study
                    break

            if not chosen_member or study_id in seen_studies:
                continue

            try:
                image_bytes = archive.read(chosen_member)
                image = Image.open(
                    BytesIO(image_bytes)
                ).convert("RGB")

                number = len(saved) + 1
                filename = f"sample_{number:03d}.jpg"
                image.save(IMG_DIR / filename, quality=95)

                saved.append({
                    "image": f"images/{filename}",
                    "study_id": study_id,
                    "source_image": chosen_member,
                    "report": report.strip()
                })
                seen_studies.add(study_id)

                print(f"Saved {number}/{MAX_SAMPLES}: {study_id}")

            except Exception as exc:
                print(f"Could not retrieve {chosen_member}: {exc}")

except Exception as exc:
    print(f"\nDownload failed: {type(exc).__name__}: {exc}")

report_file = OUT / "reports.json"
report_file.write_text(
    json.dumps(saved, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print("\n========== RESULT ==========")
print(f"Paired samples saved: {len(saved)}")
print(f"Images: {IMG_DIR.resolve()}")
print(f"Reports: {report_file.resolve()}")

if not saved:
    print(
        "No pairs were saved. Check whether the remote server "
        "supports HTTP range requests."
    )
