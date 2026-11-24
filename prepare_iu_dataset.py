from pathlib import Path
import argparse
import csv
import tarfile
import xml.etree.ElementTree as ET

BASE = Path(__file__).resolve().parent
DOWNLOADS = BASE / "downloads"
OUT = BASE / "data" / "iu"
IMAGES_DIR = OUT / "images"
REPORTS_DIR = OUT / "reports_xml"


def safe_extract(archive, dest):
    if not archive.is_file():
        raise FileNotFoundError(f"Missing archive: {archive}")
    dest.mkdir(parents=True, exist_ok=True)
    resolved = dest.resolve()
    with tarfile.open(archive, "r:gz") as tf:
        members = []
        for member in tf.getmembers():
            target = (dest / member.name).resolve()
            if target == resolved or resolved in target.parents:
                members.append(member)
        tf.extractall(dest, members=members)


def lname(tag):
    return tag.rsplit("}", 1)[-1].lower()


def txt(node):
    return " ".join(" ".join(node.itertext()).split()) if node is not None else ""


def parse_report(xml):
    root = ET.parse(xml).getroot()
    ids = []
    for node in root.iter():
        if lname(node.tag) == "parentimage":
            image_id = (node.attrib.get("id") or node.attrib.get("ID") or "").strip()
            if image_id and image_id not in ids:
                ids.append(image_id)
    sections = {k: "" for k in ("indication", "comparison", "findings", "impression")}
    for node in root.iter():
        tag = lname(node.tag)
        label = (node.attrib.get("Label") or node.attrib.get("label") or "").strip().lower()
        key = label if label in sections else (tag if tag in sections else "")
        if key and not sections[key]:
            val = txt(node)
            if val:
                sections[key] = val
    return ids, sections


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-pairs", type=int, default=100,
                   help="maximum linked image rows to write in CSV (default 100)")
    p.add_argument("--images-dir", default=None,
                   help="optional directory containing already extracted PNGs")
    p.add_argument("--reports-dir", default=None,
                   help="optional directory containing extracted XML reports")
    args = p.parse_args()
    if args.max_pairs < 1:
        p.error("--max-pairs must be >= 1")

    png_archive = DOWNLOADS / "NLMCXR_png.tgz"
    report_archive = DOWNLOADS / "NLMCXR_reports.tgz"
    if args.images_dir:
        images_root = Path(args.images_dir).resolve()
    else:
        if not png_archive.exists():
            raise FileNotFoundError(
                f"Missing {png_archive}. Download the official image archive (about 1.36 GB) "
                "or pass --images-dir with already extracted images."
            )
        safe_extract(png_archive, IMAGES_DIR)
        images_root = IMAGES_DIR

    if args.reports_dir:
        reports_root = Path(args.reports_dir).resolve()
    else:
        if not report_archive.exists():
            raise FileNotFoundError(
                f"Missing {report_archive}. Download the official XML report archive first."
            )
        safe_extract(report_archive, REPORTS_DIR)
        reports_root = REPORTS_DIR

    img_index = {p.stem.lower(): p for p in images_root.rglob("*")
                 if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}}
    xmls = sorted(reports_root.rglob("*.xml"))
    if not img_index:
        raise RuntimeError(f"No image files found under {images_root}")
    if not xmls:
        raise RuntimeError(f"No XML reports found under {reports_root}")

    rows, missing = [], []
    for xml in xmls:
        ids, sec = parse_report(xml)
        report_text = "\n".join(f"{k.upper()}: {sec[k]}" for k in sec if sec[k])
        for image_id in ids:
            img = img_index.get(Path(image_id).stem.lower())
            if img is None:
                missing.append(f"{xml.name}: {image_id}")
                continue
            rows.append({
                "image_path": str(img.resolve()),
                "image_id": image_id,
                "report_id": xml.stem,
                "report_xml": str(xml.resolve()),
                **sec,
                "report_text": report_text,
            })
            if len(rows) >= args.max_pairs:
                break
        if len(rows) >= args.max_pairs:
            break

    OUT.mkdir(parents=True, exist_ok=True)
    out_csv = OUT / "iu_xray_pairs.csv"
    fields = ["image_path", "image_id", "report_id", "report_xml",
              "indication", "comparison", "findings", "impression", "report_text"]
    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"Image files indexed: {len(img_index)}")
    print(f"XML reports scanned: {len(xmls)}")
    print(f"Linked rows written: {len(rows)}")
    print(f"CSV: {out_csv}")
    if missing:
        print(f"Unmatched IDs encountered before limit: {len(missing)}")
    print("Note: this CSV is a pairing manifest; report text is not a verified binary pneumonia label.")


if __name__ == "__main__":
    main()
