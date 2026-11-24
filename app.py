from pathlib import Path
import re

import streamlit as st
import torch
from PIL import Image
from torchvision import models, transforms
from torch import nn

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "models" / "pneumonia_resnet18.pt"

st.set_page_config(
    page_title="PulmoAI | X-ray Lab",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Styling
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --ink: #14243a;
        --muted: #64748b;
        --line: #e5edf4;
        --blue: #2563eb;
        --teal: #0f9f9a;
        --surface: #ffffff;
        --soft: #f4f8fc;
    }
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    .stApp {
        background: #f4f7fb;
        color: var(--ink);
    }
    [data-testid="stHeader"] {
        background: rgba(244,247,251,0.88);
    }
    .block-container {
        max-width: 1440px;
        padding-top: 1.35rem;
        padding-bottom: 3rem;
    }
    [data-testid="stSidebar"] {
        background: #10223a;
        border-right: 1px solid #1d3653;
    }
    [data-testid="stSidebar"] * {
        color: #edf5ff;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #b8c9dc;
    }
    [data-testid="stSidebar"] hr {
        border-color: #29415e;
    }
    .brand {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 8px 0 24px 0;
    }
    .brand-icon {
        width: 46px;
        height: 46px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 14px;
        background: linear-gradient(135deg, #2f80ff, #11b6ad);
        font-size: 24px;
        box-shadow: 0 8px 24px rgba(25, 118, 210, .22);
    }
    .brand-name {
        font-family: 'Manrope', sans-serif;
        color: #ffffff;
        font-size: 20px;
        font-weight: 800;
        line-height: 1.1;
    }
    .brand-sub {
        color: #9fb4ca;
        font-size: 11px;
        letter-spacing: 1.5px;
        margin-top: 5px;
        text-transform: uppercase;
    }
    .hero {
        background: linear-gradient(120deg, #102b4b 0%, #174b73 56%, #087f8c 100%);
        color: white;
        padding: 28px 32px;
        border-radius: 22px;
        margin-bottom: 22px;
        box-shadow: 0 14px 36px rgba(17, 52, 83, .12);
    }
    .hero-kicker {
        color: #b7f3f0;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.7px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .hero h1 {
        font-family: 'Manrope', sans-serif;
        font-size: clamp(25px, 3vw, 36px);
        line-height: 1.2;
        margin: 0 0 10px 0;
        color: #ffffff;
        font-weight: 800;
    }
    .hero p {
        color: #d5e7f5;
        font-size: 14px;
        max-width: 760px;
        margin: 0;
        line-height: 1.65;
    }
    .section-title {
        font-family: 'Manrope', sans-serif;
        font-size: 19px;
        font-weight: 800;
        color: #172b45;
        margin: 8px 0 3px 0;
    }
    .section-subtitle {
        color: #718096;
        font-size: 13px;
        margin-bottom: 14px;
    }
    .panel {
        background: #ffffff;
        border: 1px solid #e4ebf3;
        border-radius: 18px;
        padding: 20px;
        box-shadow: 0 5px 20px rgba(31, 61, 91, .035);
        margin-bottom: 16px;
    }
    .panel-heading {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 5px;
        color: #172b45;
        font-family: 'Manrope', sans-serif;
        font-size: 16px;
        font-weight: 800;
    }
    .step {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 30px;
        height: 30px;
        border-radius: 10px;
        background: #e9f2ff;
        color: #2563eb;
        font-size: 13px;
        font-weight: 800;
    }
    .small-muted {
        font-size: 12px;
        line-height: 1.55;
        color: #718096;
    }
    .status-pill {
        display: inline-block;
        border-radius: 999px;
        padding: 6px 10px;
        background: #e2f8ef;
        color: #087a55;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: .3px;
    }
    .status-pill.offline {
        background: #fff2df;
        color: #9a5700;
    }
    .result-label {
        font-family: 'Manrope', sans-serif;
        font-size: 27px;
        font-weight: 800;
        color: #14243a;
        margin: 2px 0 4px 0;
    }
    .result-caption {
        color: #64748b;
        font-size: 12px;
        margin-bottom: 10px;
    }
    .stButton > button {
        border-radius: 11px;
        min-height: 44px;
        font-weight: 700;
        border: 1px solid #dbe6f2;
        transition: transform .15s ease, box-shadow .15s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 5px 14px rgba(37, 99, 235, .12);
        border-color: #a9c9f8;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(100deg, #2563eb, #0f9f9a);
        border: 0;
        color: white;
    }
    [data-testid="stFileUploader"] {
        border-radius: 12px;
    }
    [data-testid="stMetric"] {
        background: #f7faff;
        border: 1px solid #e5edf7;
        border-radius: 14px;
        padding: 14px 16px;
    }
    [data-testid="stMetricLabel"] {
        color: #64748b;
    }
    [data-testid="stMetricValue"] {
        color: #14243a;
        font-family: 'Manrope', sans-serif;
        font-weight: 800;
    }
    .footer-note {
        color: #7b8ba0;
        font-size: 11px;
        line-height: 1.6;
        text-align: center;
        padding-top: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div class="brand-icon">🫁</div>
            <div>
                <div class="brand-name">PulmoAI</div>
                <div class="brand-sub">X-ray testing lab</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("### Workspace")
    page = st.radio(
        "Choose a section",
        ["Test workspace", "About this prototype"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("### System status")
    if CHECKPOINT.is_file():
        st.markdown('<span class="status-pill">● MODEL FILE FOUND</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-pill offline">● MODEL FILE MISSING</span>', unsafe_allow_html=True)
    st.caption("Local model · ResNet-18")
    st.caption(f"Device: {'CUDA GPU' if torch.cuda.is_available() else 'CPU'}")
    st.divider()
    st.markdown("### Report OCR")
    tesseract_path = st.text_input(
        "Tesseract executable",
        value=r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        help="Only needed to extract text from an uploaded report image. Install Tesseract separately.",
    )
    st.caption("Your uploaded files are used for this test session. No dataset browser is shown.")

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">Medical imaging · Student research prototype</div>
        <h1>Chest X-ray testing workspace</h1>
        <p>Upload a chest X-ray to run the trained image classifier. Optionally add a radiology report image or text to review report keywords separately.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if page == "About this prototype":
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("### About PulmoAI")
    st.write(
        "PulmoAI is an educational prototype that demonstrates chest X-ray image classification "
        "with a trained ResNet-18 model and a separate, basic radiology-report text review."
    )
    st.write(
        "The report review uses OCR and keyword matching. The report is not fused with the X-ray "
        "by a trained multimodal model, and keyword matching cannot reliably interpret clinical context."
    )
    st.warning("This prototype is not a medical device and must not be used for diagnosis or treatment decisions.")
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# ---------------------------------------------------------
# Model loading and inference
# ---------------------------------------------------------
@st.cache_resource
def load_model(path_string, device_name, checkpoint_mtime):
    del checkpoint_mtime
    checkpoint = torch.load(path_string, map_location=device_name, weights_only=False)

    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.25),
        nn.Linear(model.fc.in_features, 2),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval().to(device_name)

    image_size = int(checkpoint.get("image_size", 224))
    raw_labels = checkpoint.get("labels", ["NORMAL", "PNEUMONIA"])

    # Accept labels saved as a list, integer-keyed dict, or string-keyed dict.
    if isinstance(raw_labels, dict):
        try:
            classes = [str(raw_labels[i]) if i in raw_labels else str(raw_labels[str(i)]) for i in range(2)]
        except KeyError:
            classes = ["NORMAL", "PNEUMONIA"]
    elif isinstance(raw_labels, (list, tuple)) and len(raw_labels) >= 2:
        classes = [str(raw_labels[0]), str(raw_labels[1])]
    else:
        classes = ["NORMAL", "PNEUMONIA"]

    return model, image_size, classes


def predict_image(image):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, image_size, classes = load_model(
        str(CHECKPOINT),
        device,
        CHECKPOINT.stat().st_mtime,
    )
    preprocess = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            ),
        ]
    )
    tensor = preprocess(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = model(tensor)
        scores = torch.softmax(logits, dim=1)[0]
    index = int(scores.argmax().item())
    return classes[index], {classes[i]: float(scores[i].item()) for i in range(len(classes))}


# ---------------------------------------------------------
# Report OCR and keyword review
# ---------------------------------------------------------
def ocr_report(image, executable_path):
    try:
        import pytesseract

        if executable_path and Path(executable_path).is_file():
            pytesseract.pytesseract.tesseract_cmd = executable_path
        return pytesseract.image_to_string(image)
    except Exception as exc:
        raise RuntimeError(
            "Could not run OCR. Install Tesseract OCR and the Python package "
            "'pytesseract', then verify the executable path in the sidebar. "
            f"Details: {exc}"
        ) from exc


TERMS = {
    "pneumonia": ["pneumonia"],
    "opacity": ["opacity", "opacities"],
    "consolidation": ["consolidation"],
    "infiltrate": ["infiltrate", "infiltrates"],
    "pleural effusion": ["pleural effusion", "effusions"],
    "atelectasis": ["atelectasis"],
    "edema": ["pulmonary edema", "oedema", "edema"],
    "clear/normal wording": [
        "lungs are clear",
        "clear lungs",
        "no acute cardiopulmonary",
        "no acute disease",
    ],
}
NEGATION = re.compile(r"\b(no|not|without|negative for|absent|free of|neither)\b", re.IGNORECASE)
UNCERTAIN = re.compile(
    r"\b(possible|possibly|probable|probably|may represent|cannot exclude|could represent|questionable|suspected)\b",
    re.IGNORECASE,
)


def inspect_report(text):
    findings = []
    for sentence in re.split(r"(?<=[.!?;])\s+|\n+", text):
        sentence = sentence.strip()
        if not sentence:
            continue
        lowered = sentence.lower()
        for category, terms in TERMS.items():
            matched_term = next((term for term in terms if term in lowered), None)
            if not matched_term:
                continue
            position = lowered.find(matched_term)
            context = lowered[max(0, position - 60):position]
            if UNCERTAIN.search(context) or UNCERTAIN.search(sentence):
                status = "uncertain wording"
            elif NEGATION.search(context):
                status = "possibly negated wording"
            else:
                status = "term mentioned"
            findings.append((category, status, sentence))
            break
    return list(dict.fromkeys(findings))


# ---------------------------------------------------------
# Test workspace
# ---------------------------------------------------------
st.markdown('<div class="section-title">Run a new test</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">Step 1 · Add your test files. You can test an X-ray alone or review a report separately.</div>',
    unsafe_allow_html=True,
)

xray_col, report_col = st.columns(2, gap="large")
xray = None
report_image = None

with xray_col:
    st.markdown(
        """
        <div class="panel">
            <div class="panel-heading"><span class="step">01</span> Chest X-ray</div>
            <div class="small-muted">PNG or JPG · Frontal chest radiograph recommended</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    xray_file = st.file_uploader(
        "Choose X-ray image",
        type=["png", "jpg", "jpeg"],
        key="test_xray",
        label_visibility="collapsed",
    )
    if xray_file is not None:
        try:
            xray = Image.open(xray_file).convert("RGB")
            st.image(xray, caption="X-ray selected for this test", use_container_width=True)
        except Exception as exc:
            st.error(f"Could not open the X-ray image: {exc}")

with report_col:
    st.markdown(
        """
        <div class="panel">
            <div class="panel-heading"><span class="step">02</span> Radiology report</div>
            <div class="small-muted">Optional · Upload a report scan or enter report text below</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    report_file = st.file_uploader(
        "Choose report image",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        key="test_report",
        label_visibility="collapsed",
    )
    if report_file is not None:
        try:
            report_image = Image.open(report_file).convert("RGB")
            st.image(report_image, caption="Report image selected", use_container_width=True)
        except Exception as exc:
            st.error(f"Could not open the report image: {exc}")

typed_report = st.text_area(
    "Or paste radiology report text",
    height=130,
    placeholder="Paste the report's Findings or Impression section here…",
    key="typed_report",
)

st.markdown(
    '<div class="small-muted">Privacy reminder: use only data you are authorized to test. This prototype does not provide a medical diagnosis.</div>',
    unsafe_allow_html=True,
)
st.write("")

def clear_test_inputs():
    for key in ("test_xray", "test_report", "typed_report"):
        st.session_state.pop(key, None)

button_col, reset_col, _ = st.columns([1.3, 1, 3])
with button_col:
    analyse = st.button("▶  Analyse test inputs", type="primary", use_container_width=True)
with reset_col:
    st.button("↻  Clear inputs", use_container_width=True, on_click=clear_test_inputs)

if analyse:
    if xray is None and report_image is None and not typed_report.strip():
        st.error("Add a chest X-ray, report image, or report text before starting.")
        st.stop()

    st.markdown("---")
    st.markdown('<div class="section-title">Test results</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Image classification and report keyword review are displayed separately.</div>',
        unsafe_allow_html=True,
    )

    if xray is not None:
        st.markdown('<div class="panel-heading">🩻 X-ray model output</div>', unsafe_allow_html=True)
        if not CHECKPOINT.is_file():
            st.error("The trained model checkpoint is missing. Place it at models/pneumonia_resnet18.pt.")
        else:
            try:
                with st.spinner("Analysing X-ray with the trained model…"):
                    label, scores = predict_image(xray)
                left_metric, right_metric = st.columns(2)
                left_metric.metric("Predicted image class", label)
                pneumonia_score = next((v for k, v in scores.items() if "PNEUMONIA" in k.upper()), None)
                if pneumonia_score is not None:
                    right_metric.metric("Pneumonia model score", f"{pneumonia_score:.1%}")
                else:
                    right_metric.metric("Number of output classes", len(scores))

                st.markdown("#### Class score breakdown")
                for class_name, score in scores.items():
                    st.write(f"**{class_name}**")
                    st.progress(max(0.0, min(1.0, score)), text=f"{score:.1%}")

                st.caption(
                    "These are model scores, not calibrated medical probabilities. "
                    "A prediction can be wrong and must not guide medical decisions."
                )
            except Exception as exc:
                st.error(
                    "X-ray prediction failed. Check that the checkpoint matches the model architecture. "
                    f"Details: {type(exc).__name__}: {exc}"
                )

    report_text = typed_report.strip()
    if report_image is not None:
        try:
            with st.spinner("Reading report image with OCR…"):
                extracted_text = ocr_report(report_image, tesseract_path).strip()
            if extracted_text:
                report_text = f"{report_text}\n{extracted_text}".strip() if report_text else extracted_text
            else:
                st.warning("OCR returned no text. Try a clearer scan or paste the text manually.")
        except Exception as exc:
            st.error(str(exc))

    if report_text:
        st.markdown("---")
        st.markdown('<div class="panel-heading">📄 Report text review</div>', unsafe_allow_html=True)
        with st.expander("View extracted / supplied report text", expanded=True):
            st.text_area(
                "Report text",
                value=report_text,
                height=200,
                disabled=True,
                key="display_report_text",
            )
        findings = inspect_report(report_text)
        if findings:
            st.markdown("#### Matched terms")
            for term, status, sentence in findings:
                with st.container(border=True):
                    st.markdown(f"**{term.title()}** · {status}")
                    st.write(sentence)
        else:
            st.info("No configured keywords were found. This does not mean the report is normal.")
        st.caption(
            "This is basic keyword matching, not clinical interpretation. It may miss negation, "
            "context, and medical meaning. Review the original report."
        )

st.markdown(
    """
    <div class="footer-note">
        PULMOAI · EDUCATIONAL PROTOTYPE<br>
        Not for diagnosis, treatment decisions, or emergency use. Always consult a qualified healthcare professional.
    </div>
    """,
    unsafe_allow_html=True,
)
