# Pneumonia + IU X-Ray paired report starter

This ZIP contains the application, training code, and dataset preparation/download scripts.

## Important: dataset is not bundled inside the ZIP
The original IU/Open-i image archive is about 1.36 GB. The NLM asks users not to redistribute the downloaded collection outside their research group/organization. Therefore this ZIP includes scripts to fetch/prepare it, not a copy of the full image dataset.

## What works
- Streamlit web app: upload a chest X-ray and upload a report photo/scan.
- OCR extracts report text (requires Tesseract OCR installed on Windows).
- Optional ResNet-18 inference if you have trained/supplied `models/pneumonia_resnet18.pt`.
- Training script for a labelled NORMAL/PNEUMONIA folder dataset.
- IU report downloader for a limited number of XML reports and CSV manifest.
- Pairing script to create image-to-report CSV once the matching image files are present.

## 1. Create environment
Use Python 3.10 or 3.11 recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PyTorch installation fails, use the official selector at https://pytorch.org/get-started/locally/ and then install the other packages.

## 2. Get a limited report subset first (small download)
```powershell
python download_limited_iu.py --max-reports 25
```
This downloads the XML report archive and writes `data/iu_limited/selected_pairs.csv` plus the list of required image IDs. It does NOT download the large PNG archive.

## 3. Obtain image files
The official Open-i IU PNG images are distributed as one compressed archive, not a supported per-image download. To have the paired image files, download `NLMCXR_png.tgz` from:
https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz

Download reports from:
https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz

Put the two archives in `downloads/` and run:
```powershell
python prepare_iu_dataset.py --max-pairs 100
```
This extracts the archives and writes `data/iu/iu_xray_pairs.csv`, limited to at most 100 linked image rows. The PNG archive still needs to be downloaded in full (roughly 1.36 GB); the script only limits which image/report rows are written to the CSV.

## 4. Train the pneumonia image model
Provide labelled images in this structure (your existing 500-image dataset can be copied here):
```
data/labelled/
  train/
    NORMAL/
    PNEUMONIA/
  val/
    NORMAL/
    PNEUMONIA/
```
Then run:
```powershell
python train.py --epochs 3 --batch-size 8
```
The checkpoint will be saved as `models/pneumonia_resnet18.pt`.

If your dataset is already arranged as `data/processed/train/NORMAL`, `data/processed/train/PNEUMONIA`, etc., use:
```powershell
python train.py --data-dir data/processed --epochs 3
```

## 5. Run the app
```powershell
streamlit run app.py
```

For report-photo OCR on Windows:
1. Install Tesseract OCR from https://github.com/UB-Mannheim/tesseract/wiki
2. If Tesseract is not found, set the path in the app's sidebar, for example:
   `C:\Program Files\Tesseract-OCR\tesseract.exe`

## Output CSV columns
`image_path`, `image_id`, `report_id`, `report_xml`, `indication`, `comparison`, `findings`, `impression`, `report_text`.

## Medical / data limitations
- IU report-to-image links are real dataset associations, but they are not automatically binary pneumonia labels.
- The app's report-photo OCR and report text are shown separately from the X-ray classifier.
- Do not claim a trained/fused multimodal diagnosis. The included training script trains an image-only classifier from folders labelled NORMAL/PNEUMONIA.
- This is an educational project, not a clinical tool. Model scores are not calibrated probabilities.
- Review the NLM Open-i reuse terms and individual image licensing metadata before redistributing data.
