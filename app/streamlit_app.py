"""
Cephalometric Landmark Detection — Streamlit Demo Application

A polished, demo-ready interface for showcasing AI-powered cephalometric
landmark detection to a non-medical supervisor audience.
"""

import streamlit as st
import numpy as np
from pathlib import Path
from PIL import Image, UnidentifiedImageError
import io
import math

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
SAMPLE_DIR = PROJECT_ROOT / "sample_images"

# ---------------------------------------------------------------------------
# Attempt to import project modules — gracefully degrade if missing
# ---------------------------------------------------------------------------
_SRC_AVAILABLE = False
try:
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))

    from src.models.unet import create_model
    from src.inference.predict import heatmap_to_coordinates, compute_confidence
    from src.inference.angles import compute_sna, compute_snb, compute_anb, interpret_anb
    from src.viz.overlay import draw_landmarks, draw_angle_lines

    _SRC_AVAILABLE = True
except ImportError:
    _SRC_AVAILABLE = False

# ---------------------------------------------------------------------------
# Landmark metadata
# ---------------------------------------------------------------------------
LANDMARK_NAMES = ["Sella (S)", "Nasion (N)", "A-point (A)", "B-point (B)",
                  "Pogonion (Pog)", "Menton (Me)"]
LANDMARK_SHORT = ["S", "N", "A", "B", "Pog", "Me"]
LANDMARK_COLORS = [
    (255, 80, 80),    # S  — red
    (80, 180, 255),   # N  — blue
    (80, 255, 80),    # A  — green
    (255, 200, 60),   # B  — yellow
    (200, 80, 255),   # Pog — purple
    (255, 150, 80),   # Me — orange
]
GT_COLOR = (0, 255, 200)  # cyan-ish for ground truth

MODEL_INPUT_SIZE = (512, 512)  # H, W expected by the model


# ============================================================================
# Helper functions (fallbacks when src/ is unavailable)
# ============================================================================

def _discover_checkpoints() -> list[str]:
    """Return a sorted list of checkpoint file paths found in CHECKPOINT_DIR."""
    if not CHECKPOINT_DIR.exists():
        return []
    extensions = {".pt", ".pth", ".ckpt", ".bin"}
    found = sorted(
        str(p) for p in CHECKPOINT_DIR.rglob("*") if p.suffix in extensions
    )
    return found


def _discover_samples() -> list[Path]:
    """Return a sorted list of sample image paths."""
    if not SAMPLE_DIR.exists():
        return []
    extensions = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"}
    found = sorted(
        p for p in SAMPLE_DIR.iterdir()
        if p.suffix.lower() in extensions
    )
    return found


def _validate_image(img: Image.Image) -> tuple[bool, str]:
    """Basic validation: readable, reasonable dimensions."""
    w, h = img.size
    if w < 64 or h < 64:
        return False, f"Image is too small ({w}×{h}). Please use at least 64×64 px."
    if w > 10000 or h > 10000:
        return False, f"Image is very large ({w}×{h}). Please use an image under 10 000 px."
    return True, ""


def _load_image(source) -> Image.Image | None:
    """Try to load an image from a file path (Path) or UploadedFile."""
    try:
        if isinstance(source, Path):
            return Image.open(source).convert("RGB")
        else:
            return Image.open(source).convert("RGB")
    except (UnidentifiedImageError, Exception):
        return None


@st.cache_resource(show_spinner="Loading model…")
def _load_model(checkpoint_path: str):
    """Load a model checkpoint; returns (model, device) or None on failure."""
    if not _SRC_AVAILABLE:
        return None

    import torch
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    try:
        model = create_model(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=1,
            num_classes=6,
        )
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        # Handle both raw state_dict and wrapped checkpoint dicts
        if isinstance(state, dict) and "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        elif isinstance(state, dict) and "state_dict" in state:
            model.load_state_dict(state["state_dict"])
        else:
            model.load_state_dict(state)
        model.to(device).eval()
        return model, device
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None


def _preprocess(img: Image.Image):
    """Resize & normalise image for model input. Returns (tensor, scale_factors)."""

    # Fallback: basic preprocessing
    import torch
    original_size = img.size  # (W, H)
    resized = img.resize((MODEL_INPUT_SIZE[1], MODEL_INPUT_SIZE[0]), Image.BILINEAR)
    arr = np.array(resized.convert("L"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    scale_x = original_size[0] / MODEL_INPUT_SIZE[1]
    scale_y = original_size[1] / MODEL_INPUT_SIZE[0]
    return tensor, (scale_x, scale_y)


def _run_inference(model, device, tensor):
    """Run model inference. Returns raw heatmaps [1,6,H,W]."""
    import torch
    with torch.no_grad():
        heatmaps = model(tensor.to(device))
    return heatmaps


def _extract_coords(heatmaps, scale):
    """Extract landmark coordinates from heatmaps."""
    if _SRC_AVAILABLE:
        coords = heatmap_to_coordinates(heatmaps.squeeze(0)).cpu()  # (C, 2)
        # Scale back to original image space
        scaled = [(float(c[0]) * scale[0], float(c[1]) * scale[1]) for c in coords]
        return scaled

    # Fallback
    hm = heatmaps.squeeze(0).cpu().numpy()  # [6, H, W]
    coords = []
    for c in range(hm.shape[0]):
        channel = hm[c]
        idx = np.unravel_index(np.argmax(channel), channel.shape)
        y, x = idx
        coords.append((x * scale[0], y * scale[1]))
    return coords


def _extract_confidence(heatmaps):
    """Get per-landmark confidence (0–100)."""
    if _SRC_AVAILABLE:
        conf = compute_confidence(heatmaps.squeeze(0)).cpu()  # (C,)
        return [float(c) for c in conf]

    hm = heatmaps.squeeze(0).cpu().numpy()
    confidences = []
    for c in range(hm.shape[0]):
        peak = float(hm[c].max())
        confidences.append(min(peak * 100, 100.0))
    return confidences


def _compute_angle_at_vertex(p1, vertex, p2):
    """Compute angle at *vertex* formed by rays vertex→p1 and vertex→p2 (degrees)."""
    v1 = (p1[0] - vertex[0], p1[1] - vertex[1])
    v2 = (p2[0] - vertex[0], p2[1] - vertex[1])
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    m1 = math.hypot(*v1)
    m2 = math.hypot(*v2)
    if m1 == 0 or m2 == 0:
        return 0.0
    cos_a = max(-1.0, min(1.0, dot / (m1 * m2)))
    return math.degrees(math.acos(cos_a))


def _compute_angles(coords):
    """Compute SNA, SNB, ANB from coordinates.

    coords order: S(0), N(1), A(2), B(3), Pog(4), Me(5)
    """
    if _SRC_AVAILABLE:
        sna = compute_sna(coords)
        snb = compute_snb(coords)
        anb = compute_anb(coords)
        return sna, snb, anb

    S, N, A, B = coords[0], coords[1], coords[2], coords[3]
    sna = _compute_angle_at_vertex(S, N, A)
    snb = _compute_angle_at_vertex(S, N, B)
    anb = sna - snb
    return sna, snb, anb


def _interpret_anb_value(anb: float) -> str:
    """Plain-English interpretation of the ANB angle."""
    if _SRC_AVAILABLE:
        return interpret_anb(anb)

    if 0 <= anb <= 4:
        classification = "normal alignment (skeletal Class I)"
        suggestion = "The upper and lower jaws appear well-aligned relative to each other."
    elif anb > 4:
        classification = "upper jaw positioned forward (skeletal Class II tendency)"
        suggestion = ("The upper jaw sits noticeably ahead of the lower jaw, which an "
                      "orthodontist might consider when planning treatment.")
    else:
        classification = "lower jaw positioned forward (skeletal Class III tendency)"
        suggestion = ("The lower jaw sits ahead of the upper jaw, which can be relevant "
                      "for orthodontic or surgical treatment planning.")

    return (
        f"**ANB measures how the upper and lower jaw line up relative to each other.** "
        f"A value around 2° is typical.\n\n"
        f"This patient's ANB of **{anb:.1f}°** suggests **{classification}**. "
        f"{suggestion}"
    )


def _draw_results_on_image(img: Image.Image, coords, gt_coords=None):
    """Draw landmarks and angle lines on the image. Returns a PIL Image."""
    if _SRC_AVAILABLE:
        result = draw_landmarks(img.copy(), coords, LANDMARK_SHORT, LANDMARK_COLORS)
        if gt_coords:
            result = draw_landmarks(result, gt_coords, LANDMARK_SHORT,
                                    [GT_COLOR] * len(LANDMARK_SHORT), marker="x")
        result = draw_angle_lines(result, coords)
        return result

    # Fallback: use PIL drawing
    from PIL import ImageDraw, ImageFont
    result = img.copy()
    draw = ImageDraw.Draw(result)
    r = max(4, min(img.size) // 100)

    # Draw GT first (behind predictions)
    if gt_coords:
        for i, (x, y) in enumerate(gt_coords):
            x, y = int(x), int(y)
            draw.ellipse([x - r, y - r, x + r, y + r], outline=GT_COLOR, width=2)
            draw.line([x - r, y - r, x + r, y + r], fill=GT_COLOR, width=2)
            draw.line([x + r, y - r, x - r, y + r], fill=GT_COLOR, width=2)

    # Draw predicted landmarks
    for i, (x, y) in enumerate(coords):
        x, y = int(x), int(y)
        color = LANDMARK_COLORS[i]
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color, outline="white", width=1)
        # Label
        label = LANDMARK_SHORT[i]
        draw.text((x + r + 4, y - r), label, fill=color)

    # Draw angle lines: S-N, N-A, N-B
    S, N, A, B = [coords[i] for i in range(4)]
    line_width = max(2, min(img.size) // 200)
    draw.line([int(S[0]), int(S[1]), int(N[0]), int(N[1])],
              fill=(255, 255, 255), width=line_width)
    draw.line([int(N[0]), int(N[1]), int(A[0]), int(A[1])],
              fill=(80, 255, 80), width=line_width)
    draw.line([int(N[0]), int(N[1]), int(B[0]), int(B[1])],
              fill=(255, 200, 60), width=line_width)

    return result


# ============================================================================
# Streamlit Page Configuration
# ============================================================================
st.set_page_config(
    page_title="Cephalometric Landmark Detection",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# Custom CSS for polish
# ============================================================================
st.markdown("""
<style>
    /* Confidence bar styling */
    .confidence-bar-bg {
        background-color: rgba(128, 128, 128, 0.2);
        border-radius: 6px;
        overflow: hidden;
        height: 22px;
        margin-bottom: 6px;
    }
    .confidence-bar-fill {
        height: 100%;
        border-radius: 6px;
        display: flex;
        align-items: center;
        padding-left: 8px;
        font-size: 0.8rem;
        font-weight: 600;
        color: white;
        transition: width 0.4s ease;
    }
    .metric-card {
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Sidebar
# ============================================================================
with st.sidebar:
    st.image("https://em-content.zobj.net/source/twitter/408/tooth_1f9b7.png", width=48)
    st.title("🦷 CephAI")
    st.caption("Cephalometric Landmark Detection")
    st.divider()

    # --- Model Selection ---
    st.subheader("Model Selection")
    checkpoints = _discover_checkpoints()
    if checkpoints:
        selected_checkpoint = st.selectbox(
            "Choose a checkpoint",
            options=checkpoints,
            format_func=lambda p: Path(p).name,
            help="Select a trained model checkpoint to use for inference.",
        )
    else:
        selected_checkpoint = None
        st.info(
            "No trained models found in `checkpoints/`. "
            "Run training first:\n\n"
            "```bash\npython train.py\n```",
            icon="📦",
        )

    st.divider()

    # --- About ---
    st.subheader("About")
    st.markdown(
        "This application demonstrates **AI-powered detection of anatomical "
        "landmarks** on lateral skull X-rays (cephalograms).\n\n"
        "It locates 6 key points used in orthodontic analysis and computes "
        "the **ANB angle** — a standard measurement of jaw alignment.\n\n"
        "Built as a research demo, **not** a clinical diagnostic tool."
    )
    st.divider()
    st.caption("© 2026 Cephalometric AI Demo")


# ============================================================================
# Main Content — Tabs
# ============================================================================
tab_demo, tab_how = st.tabs(["🔬 Demo", "📖 How It Works"])


# ============================================================================
# Tab 1: Demo
# ============================================================================
with tab_demo:
    st.header("Landmark Detection Demo")
    st.markdown(
        "Select a **sample case** or **upload your own** lateral cephalogram "
        "to see the AI detect anatomical landmarks and compute jaw alignment angles."
    )

    # ------------------------------------------------------------------
    # Input selection
    # ------------------------------------------------------------------
    col_sample, col_upload = st.columns(2)

    input_image: Image.Image | None = None
    input_source: str = ""
    sample_paths = _discover_samples()

    with col_sample:
        st.subheader("📂 Sample Gallery")
        if sample_paths:
            sample_labels = {
                f"Sample Case {i+1}": p for i, p in enumerate(sample_paths[:5])
            }
            chosen_label = st.radio(
                "Pick a sample case",
                options=list(sample_labels.keys()),
                index=0,
                key="sample_radio",
            )
            if st.button("Load Sample", use_container_width=True, type="primary"):
                st.session_state["input_path"] = sample_labels[chosen_label]
                st.session_state["input_mode"] = "sample"
        else:
            st.info(
                "No sample images found. Add lateral cephalogram images to "
                "`sample_images/` to enable one-click demos.",
                icon="🖼️",
            )

    with col_upload:
        st.subheader("📤 Upload Your Own")
        uploaded_file = st.file_uploader(
            "Choose a cephalogram image",
            type=["png", "jpg", "jpeg", "tiff", "tif", "bmp"],
            help="Upload a lateral (side-profile) skull X-ray image.",
        )
        st.info(
            "**Expected input:** a lateral (side-profile) skull X-ray. "
            "Regular photos or front-facing scans won't produce meaningful results.",
            icon="ℹ️",
        )
        if uploaded_file is not None:
            st.session_state["uploaded_file"] = uploaded_file
            st.session_state["input_mode"] = "upload"

    # Determine active image
    input_mode = st.session_state.get("input_mode", None)
    if input_mode == "upload" and "uploaded_file" in st.session_state:
        input_image = _load_image(st.session_state["uploaded_file"])
        input_source = st.session_state["uploaded_file"].name
    elif input_mode == "sample" and "input_path" in st.session_state:
        input_image = _load_image(st.session_state["input_path"])
        input_source = st.session_state["input_path"].name

    # ------------------------------------------------------------------
    # Validation & Inference
    # ------------------------------------------------------------------
    if input_image is not None:
        valid, msg = _validate_image(input_image)
        if not valid:
            st.error(msg, icon="❌")
        else:
            st.divider()
            st.subheader(f"Results — {input_source}")

            # Check if we can actually run inference
            can_infer = _SRC_AVAILABLE and selected_checkpoint is not None

            if not can_infer:
                # -------------------------------------------------------
                # Fallback: show the image with a message about training
                # -------------------------------------------------------
                if not _SRC_AVAILABLE and selected_checkpoint is None:
                    st.warning(
                        "**The model source code and trained weights are not yet available.** "
                        "The full inference pipeline requires:\n"
                        "1. The `src/` package (model, inference, and visualization modules)\n"
                        "2. A trained checkpoint in `checkpoints/`\n\n"
                        "Run `python train.py` to train a model first, then restart the app.",
                        icon="⚠️",
                    )
                elif not _SRC_AVAILABLE:
                    st.warning(
                        "**The `src/` package is not available.** "
                        "Ensure the project source code is installed and importable.",
                        icon="⚠️",
                    )
                else:
                    st.warning(
                        "**No model checkpoint selected.** "
                        "Choose a checkpoint from the sidebar, or train a model first.",
                        icon="⚠️",
                    )

                # Show the input image anyway
                st.image(input_image, caption="Input Image (no inference available)",
                         use_container_width=True)

            else:
                # -------------------------------------------------------
                # Full inference pipeline
                # -------------------------------------------------------
                with st.spinner("Running inference…"):
                    try:
                        model_result = _load_model(selected_checkpoint)
                        if model_result is None:
                            st.error("Could not load the selected model checkpoint.",
                                     icon="❌")
                            st.stop()

                        model, device = model_result
                        tensor, scale = _preprocess(input_image)
                        heatmaps = _run_inference(model, device, tensor)
                        coords = _extract_coords(heatmaps, scale)
                        confidences = _extract_confidence(heatmaps)
                        sna, snb, anb = _compute_angles(coords)

                        # Try to load ground truth for sample images
                        gt_coords = None  # TODO: load from annotation files if available

                        result_img = _draw_results_on_image(
                            input_image, coords, gt_coords
                        )

                    except Exception as e:
                        st.error(
                            f"An error occurred during inference: {e}\n\n"
                            "Please check that the model checkpoint is compatible "
                            "with the current code.",
                            icon="❌",
                        )
                        st.stop()

                # -------------------------------------------------------
                # Display results
                # -------------------------------------------------------
                col_img, col_info = st.columns([3, 2])

                with col_img:
                    st.image(
                        result_img,
                        caption="X-ray with predicted landmarks and angle lines",
                        use_container_width=True,
                    )
                    # Legend
                    legend_parts = []
                    for i, name in enumerate(LANDMARK_SHORT):
                        r, g, b = LANDMARK_COLORS[i]
                        legend_parts.append(
                            f'<span style="color:rgb({r},{g},{b}); '
                            f'font-weight:bold;">● {name}</span>'
                        )
                    if gt_coords:
                        r, g, b = GT_COLOR
                        legend_parts.append(
                            f'<span style="color:rgb({r},{g},{b}); '
                            f'font-weight:bold;">✕ Ground Truth</span>'
                        )
                    st.markdown(
                        "&nbsp;&nbsp;".join(legend_parts),
                        unsafe_allow_html=True,
                    )

                with col_info:
                    # --- Angle Measurements ---
                    st.markdown("#### 📐 Angle Measurements")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("SNA", f"{sna:.1f}°")
                    m2.metric("SNB", f"{snb:.1f}°")
                    m3.metric("ANB", f"{anb:.1f}°",
                              delta=f"{anb - 2:.1f}° from typical" if abs(anb - 2) > 0.5 else "typical range",
                              delta_color="off")

                    st.divider()

                    # --- Clinical Interpretation ---
                    st.markdown("#### 🩺 Clinical Interpretation")
                    st.markdown(_interpret_anb_value(anb))

                    st.divider()

                    # --- Confidence Scores ---
                    st.markdown("#### 🎯 Landmark Confidence Scores")
                    st.caption("Sorted by confidence. ⚠️ indicates confidence below 70%.")

                    # Sort by confidence descending
                    paired = list(zip(LANDMARK_NAMES, confidences))
                    paired.sort(key=lambda x: x[1], reverse=True)

                    THRESHOLD = 70.0
                    for name, conf in paired:
                        warning = " ⚠️" if conf < THRESHOLD else ""
                        if conf >= 80:
                            color = "#4CAF50"  # green
                        elif conf >= THRESHOLD:
                            color = "#FFC107"  # amber
                        else:
                            color = "#F44336"  # red

                        st.markdown(
                            f'<div style="display:flex;align-items:center;gap:8px;">'
                            f'<span style="min-width:110px;font-size:0.85rem;">'
                            f'{name}{warning}</span>'
                            f'<div class="confidence-bar-bg" style="flex:1;">'
                            f'<div class="confidence-bar-fill" '
                            f'style="width:{conf:.0f}%;background:{color};">'
                            f'{conf:.0f}%</div></div></div>',
                            unsafe_allow_html=True,
                        )

    elif input_mode is not None:
        st.error(
            "Could not read the selected image. Please try a different file.",
            icon="❌",
        )


# ============================================================================
# Tab 2: How It Works
# ============================================================================
with tab_how:
    st.header("How It Works")

    # --- What is Cephalometry? ---
    st.subheader("🦴 What is Cephalometry?")
    st.markdown(
        """
        Cephalometry is a technique used by orthodontists and surgeons to study the
        relationships between the bones, teeth, and soft tissues of the face and skull.
        It works by taking a standardized **lateral (side-view) X-ray** — called a
        *cephalogram* — and identifying specific anatomical points, known as
        **landmarks**, on the image. By measuring the distances and angles between
        these landmarks, clinicians can assess jaw alignment, plan treatments like
        braces or surgery, and track changes over time.

        Traditionally, locating these landmarks is done **manually** by trained
        professionals, which is time-consuming and subject to variability between
        observers. Automating this process with AI can save time, improve consistency,
        and make cephalometric analysis accessible to more practitioners — which is
        exactly what this demo showcases.
        """
    )

    st.divider()

    # --- How the AI Works ---
    st.subheader("🤖 How the AI Works")
    st.markdown(
        """
        The model was **trained on X-ray images where doctors had marked these
        points by hand**, and learned to recognize the same anatomical features.
        For each X-ray, the AI produces a **confidence map** (called a *heatmap*)
        for each of the 6 landmarks — think of it as a heat-sensitive overlay where
        brighter areas indicate "the landmark is more likely here." The model then
        picks the **brightest point** on each heatmap as its prediction.

        This approach — called **heatmap regression** — is widely used in medical
        imaging because it's more robust than trying to predict exact pixel
        coordinates directly. The model can express uncertainty: a sharp, focused
        heatmap means high confidence, while a diffuse glow means the model is
        less sure.
        """
    )

    st.divider()

    # --- The 6 Landmarks ---
    st.subheader("📍 The 6 Landmarks")
    st.markdown(
        """
        This demo focuses on **6 key cephalometric landmarks**:

        | # | Landmark | Abbreviation | Description |
        |---|----------|:---:|-------------|
        | 1 | Sella | **S** | Center of the pituitary fossa (a bony pocket in the base of the skull) |
        | 2 | Nasion | **N** | The junction where the frontal bone meets the nasal bones |
        | 3 | A-point | **A** | The deepest concavity on the front of the upper jaw (maxilla) |
        | 4 | B-point | **B** | The deepest concavity on the front of the lower jaw (mandible) |
        | 5 | Pogonion | **Pog** | The most anterior (forward) point of the chin |
        | 6 | Menton | **Me** | The lowest point of the chin outline |

        From **S, N, A, and B**, the app computes three angles:
        - **SNA** — relates the upper jaw to the skull base
        - **SNB** — relates the lower jaw to the skull base
        - **ANB = SNA − SNB** — the key measurement of how the jaws relate to *each other*
        """
    )

    st.divider()

    # --- Technical Details (expandable) ---
    st.subheader("⚙️ Technical Details")
    with st.expander("Architecture & Training", expanded=False):
        st.markdown(
            """
            **Model Architecture**
            - **Backbone:** U-Net with a ResNet-34 encoder (pretrained on ImageNet)
            - **Library:** `segmentation_models_pytorch`
            - **Input:** Single-channel grayscale image, resized to 512×512
            - **Output:** 6-channel heatmap (one per landmark), same spatial resolution

            **Training Configuration**
            - **Dataset:** ISBI 2015 Cephalometric Challenge (400 images, 150 train / 250 test)
            - **Loss:** Mean Squared Error (MSE) on predicted vs. ground-truth heatmaps
            - **Optimizer:** Adam with differential learning rates
              - Encoder: ~1×10⁻⁴ (fine-tuning pretrained features)
              - Decoder: ~1×10⁻³ (training from scratch)
            - **Augmentation:** Random rotation (±15°), brightness/contrast jitter, Gaussian noise
            - **Scheduling:** ReduceLROnPlateau (patience = 5, monitoring validation loss)

            **Evaluation Metrics**
            - **Mean Radial Error (MRE):** Average Euclidean distance (mm) between predicted and true landmarks
            - **Successful Detection Rate (SDR):** % of predictions within 2mm, 2.5mm, 3mm, and 4mm thresholds
            """
        )

    with st.expander("Confidence Score Methodology", expanded=False):
        st.markdown(
            """
            Each landmark's **confidence score** is derived from the peak activation
            value in its predicted heatmap channel. A sharp, well-defined peak
            indicates the model is highly confident about the landmark's location,
            while a lower or more diffuse peak suggests uncertainty.

            Scores are normalized to a 0–100% scale. Landmarks scoring below
            **70%** are flagged with a warning icon (⚠️) to indicate the prediction
            may be less reliable.
            """
        )

    st.divider()

    # --- Limitations ---
    st.subheader("⚠️ Limitations")
    st.warning(
        """
        **This is a research demonstration, not a clinical tool.**

        - Trained on a single public dataset (ISBI 2015, 400 images) with annotations
          from two clinicians — not validated across diverse populations or equipment.
        - Detects only 6 of the standard 19+ cephalometric landmarks.
        - Performance degrades on non-standard X-rays (poor contrast, unusual
          positioning, pediatric patients, or images from different equipment).
        - No regulatory review or clinical validation has been performed.
        - Should **never** be used as the sole basis for clinical decision-making.
        """,
        icon="🚨",
    )

    st.info(
        "For questions about this project or the underlying methodology, "
        "please refer to the project README or reach out to the development team.",
        icon="💡",
    )
