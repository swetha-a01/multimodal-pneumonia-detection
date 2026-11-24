
from datasets import load_dataset
from remotezip import RemoteZip

DATASET = "dz-osamu/IU-Xray"
ZIP_URL = (
    "https://huggingface.co/datasets/dz-osamu/"
    "IU-Xray/resolve/main/image.zip"
)

print("1. Inspecting report metadata...")
ds = load_dataset(DATASET, split="train", streaming=True)

for i, row in enumerate(ds):
    print(f"\nMetadata row {i + 1}")
    print("Available fields:", list(row.keys()))

    for key, value in row.items():
        if key == "images":
            print("Images field:", repr(value)[:1500])
        elif key == "response":
            print("Report preview:", repr(value)[:300])
        else:
            print(f"{key}: {repr(value)[:300]}")

    if i == 1:
        break

print("\n2. Inspecting ZIP internal filenames...")
with RemoteZip(ZIP_URL) as archive:
    names = archive.namelist()

    for name in names[:30]:
        print(repr(name))

    print("\nTotal archive entries:", len(names))
