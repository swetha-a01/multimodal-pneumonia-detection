# Pneumonia Detection with Chest X-Ray and IU X-Ray Reports

An educational deep learning project that combines a **chest X-ray pneumonia classification application** with an **IU X-Ray medical report processing and image-report pairing workflow**.

The project provides an interactive Streamlit application for uploading chest X-ray images and medical report scans, extracting report text using OCR, and running image classification when a trained model is available.

> **Note:** The X-ray classifier and medical report processing are separate components. This project does not perform trained multimodal fusion or provide a combined medical diagnosis.

---

## Project Overview

This project explores two aspects of medical image and report processing:

1. **Pneumonia Image Classification** – Uses a ResNet-18 deep learning model to classify chest X-ray images into NORMAL and PNEUMONIA categories.
2. **IU X-Ray Report Processing** – Downloads and processes IU X-Ray XML reports, extracts relevant medical information, and prepares image-to-report pairings using dataset identifiers.

The Streamlit application provides a simple interface to upload an X-ray image and a medical report scan.

## Features

- Interactive Streamlit web application.
- Chest X-ray image upload and preview.
- Medical report image upload.
- OCR-based text extraction using Tesseract.
- Optional ResNet-18 model inference.
- Pneumonia classification using NORMAL and PNEUMONIA labels.
- Training script for custom labelled chest X-ray datasets.
- IU X-Ray XML report downloader.
- Image-to-report pairing and CSV generation.
- Evaluation results and confusion matrix.
- Dataset preparation and inspection scripts.

## Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning | PyTorch |
| Model Architecture | ResNet-18 |
| Web Application | Streamlit |
| Image Processing | Pillow, OpenCV (if used by the installed application) |
| OCR | Tesseract OCR, Pytesseract |
| Data Processing | Pandas |
| Medical Dataset | IU X-Ray / Open-i |
| Evaluation | Scikit-learn, Matplotlib (if used by evaluation scripts) |

## Project Workflow

### 1. Chest X-Ray Classification

- Provide a chest X-ray image through the Streamlit application.
- The image is preprocessed for model inference.
- The trained ResNet-18 model predicts NORMAL or PNEUMONIA.
- The prediction is displayed in the application.

### 2. Medical Report OCR

- Upload a medical report photo or scanned document.
- Tesseract OCR extracts readable text.
- The extracted text is displayed separately from the image classification result.

### 3. IU X-Ray Dataset Preparation

- Download the IU X-Ray XML report collection.
- Extract report information such as indication, comparison, findings, and impression.
- Obtain the corresponding image archive.
- Match report and image identifiers.
- Generate CSV files containing image-to-report associations.

## Project Structure

```text
Pneumonia_IU_Complete_Starter/
│
├── app.py
├── app_redesigned.py
│
├── train.py
├── evaluate_model.py
│
├── download_limited_iu.py
├── download_paired_report.py
├── prepare_iu_dataset.py
├── inspect_pairs.py
│
├── requirements.txt
├── README.md
├── .gitignore
│
├── evaluation_results/
│   ├── confusion_matrix.png
│   └── evaluation_report.txt
│
├── test/
│
├── models/                  # Model checkpoints (not committed)
├── data/                    # Datasets (not committed)
├── downloads/               # Downloaded archives (not committed)
└── .venv/                   # Virtual environment (not committed)
```

*The structure above is representative. Some folders may be created only after running the scripts.*

## Installation and Setup

### Prerequisites

- Python 3.10 or 3.11 recommended.
- Git.
- Tesseract OCR for report text extraction.
- Internet connection for downloading the IU X-Ray dataset.
- A trained model checkpoint for pneumonia inference.

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

Replace the repository URL with your GitHub repository URL.

### 2. Create a Virtual Environment

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PyTorch installation encounters compatibility issues, install a compatible version using the official PyTorch installation guide:

https://pytorch.org/get-started/locally/

## Dataset Preparation

### IU X-Ray Dataset

The original IU X-Ray/Open-i collection is not included in this repository.

The NLM distributes the image and report archives separately. Download and review the applicable reuse terms and image licensing metadata before using or redistributing the data.

**Official dataset resources:**

- [Open-i IU X-Ray Images](https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz)
- [Open-i IU X-Ray Reports](https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz)

### Download a Limited Report Subset

```powershell
python download_limited_iu.py --max-reports 25
```

This downloads a limited XML report subset and generates the corresponding selection information.

### Prepare Image-Report Pairs

Place the downloaded archives inside the `downloads/` directory and run:

```powershell
python prepare_iu_dataset.py --max-pairs 100
```

The script extracts the archives and generates image-to-report pairing information.

**Important:** Limiting the number of generated CSV rows does not reduce the size of the original image archive download.

## Training the Pneumonia Classifier

The training script expects a labelled dataset organized into NORMAL and PNEUMONIA folders.

Expected structure:

```text
data/
└── labelled/
    ├── train/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    │
    └── val/
        ├── NORMAL/
        └── PNEUMONIA/
```

Start training:

```powershell
python train.py --epochs 3 --batch-size 8
```

If your dataset is already stored under `data/processed/`, use:

```powershell
python train.py --data-dir data/processed --epochs 3
```

The trained checkpoint is saved as:

```text
models/pneumonia_resnet18.pt
```

The trained checkpoint is required for image classification inference in the application.

## Run the Application

Start the Streamlit application:

```powershell
streamlit run app.py
```

Open the local URL displayed in the terminal.

### Enable Medical Report OCR

1. Install Tesseract OCR on Windows.
2. Download it from the [UB Mannheim Tesseract installer page](https://github.com/UB-Mannheim/tesseract/wiki).
3. If the application cannot detect Tesseract automatically, configure its executable path:

```text
C:\Program Files\Tesseract-OCR\tesseract.exe
```

## Evaluation

The repository includes evaluation artifacts generated by the model evaluation workflow.

- Confusion matrix.
- Classification evaluation report.
- Model performance results.

See the [`evaluation_results/`](evaluation_results/) directory for the available results.

**Note:** Evaluation results depend on the dataset, training configuration, and checkpoint used. Model scores should not be interpreted as calibrated probabilities.

## Important Limitations

- The pneumonia classifier is trained using labelled NORMAL/PNEUMONIA image folders.
- IU X-Ray report-to-image associations do not automatically provide binary pneumonia labels.
- OCR output may contain errors depending on image quality and report formatting.
- Report text extraction and X-ray classification are separate processes.
- The project does not implement trained multimodal image-and-report fusion.
- A trained model checkpoint is required for pneumonia inference.
- The IU X-Ray dataset must be downloaded separately.

## Medical Disclaimer

This project is developed for **educational and research purposes only**.

It is not a clinically validated diagnostic system and must not be used for medical decision-making. Always consult qualified healthcare professionals for medical interpretation and diagnosis.

## Future Improvements

- Improve model performance through further training and validation.
- Enhance OCR accuracy and medical terminology extraction.
- Explore properly labelled paired image-report datasets.
- Develop and evaluate a genuine multimodal image-and-text model.
- Improve application usability and result visualization.

## Author

**Swetha**

BCA Student | Deep Learning | Python | Computer Vision

---

If you find this project useful, consider giving the repository a ⭐ on GitHub.
