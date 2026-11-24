from pathlib import Path
import argparse
import csv
import tarfile
import urllib.request
import xml.etree.ElementTree as ET

BASE = Path(__file__).resolve().parent
ARCHIVE_DIR = BASE / "downloads"
REPORT_URL = "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz"
REPORT_ARCHIVE = ARCHIVE_DIR / "NLMCXR_reports.tgz"
OUT = BASE / "data" / "iu_limited"
REPORTS_DIR = OUT / "reports_xml"


def download(url, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size:
        print(f"Using existing archive: {target}")
        return
    print("Downloading the small XML report archive...")
    request = urllib.request.Request(url, headers={"User-Agent": "IU-Xray-Research-Prep/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response, target.open("wb") as f:
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            f.write(block)
    print(f"Saved: {target} ({target.stat().st_size:,} bytes)")


def safe_extract(archive, dest):
    dest.mkdir(parents=True, exist_ok=True)
    resolved = dest.resolve()
    with tarfile.open(archive, "r:gz") as tf:
        members = []
        for member in tf.getmembers():
            target = (dest / member.name).resolve()
            if target == resolved or resolved in target.parents:
                members.append(member)
        tf.extractall(dest, members=members)


def tag_name(tag):
    return tag.rsplit("}", 1)[-1].lower()


def node_text(node):
    return " ".join(" ".join(node.itertext()).split()) if node is not None else ""


def parse_xml(path):
    root = ET.parse(path).getroot()
    ids = []
    for node in root.iter():
        if tag_name(node.tag) == "parentimage":
            value = (node.attrib.get("id") or node.attrib.get("ID") or "").strip()
            if value and value not in ids:
                ids.append(value)
    sec = {k: "" for k in ("indication", "comparison", "findings", "impression")}
    for node in root.iter():
        tag = tag_name(node.tag)
        label = (node.attrib.get("Label") or node.attrib.get("label") or "").strip().lower()
        key = label if label in sec else (tag if tag in sec else "")
        if key and not sec[key]:
            value = node_text(node)
            if value:
                sec[key] = value
    return ids, sec


def main():
    parser = argparse.ArgumentParser(description="Prepare a limited IU X-Ray report manifest.")
    parser.add_argument("--max-reports", type=int, default=25)
    args = parser.parse_args()
    if args.max_reports < 1:
        parser.error("--max-reports must be >= 1")

    download(REPORT_URL, REPORT_ARCHIVE)
    safe_extract(REPORT_ARCHIVE, REPORTS_DIR)
    xmls = sorted(REPORTS_DIR.rglob("*.xml"))
    if not xmls:
        raise RuntimeError(f"No XML reports found in {REPORTS_DIR}")

    rows, all_ids = [], []
    for xml in xmls[:args.max_reports]:
        ids, sec = parse_xml(xml)
        report_text = "\n".join(f"{k.upper()}: {sec[k]}" for k in sec if sec[k])
        for image_id in ids:
            all_ids.append(image_id)
            rows.append({
                "image_path": f"images/{Path(image_id).stem}.png",
                "image_id": image_id,
                "report_id": xml.stem,
                "report_xml": str(xml.relative_to(OUT)).replace("\\", "/"),
                **sec,
                "report_text": report_text,
                "image_downloaded": "false",
            })

    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "selected_pairs.csv"
    fields = ["image_path", "image_id", "report_id", "report_xml", "indication",
              "comparison", "findings", "impression", "report_text", "image_downloaded"]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "needed_image_ids.txt").write_text("\n".join(dict.fromkeys(all_ids)), encoding="utf-8")
    print(f"Reports selected: {min(args.max_reports, len(xmls))}")
    print(f"Image-report rows in manifest: {len(rows)}")
    print(f"Unique image IDs needed: {len(set(all_ids))}")
    print(f"CSV: {csv_path}")
    print("No large image archive was downloaded. The manifest is not a complete image dataset yet.")


if __name__ == "__main__":
    main()
