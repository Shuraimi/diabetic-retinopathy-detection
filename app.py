"""
APTOS Diabetic Retinopathy Grading — DenseNet121 Streamlit App (wizard flow)

Upload a retina image and step through:
    1. The exact preprocessing pipeline (Ben Graham style)
    2. How the representation evolves through DenseNet121's blocks
    3. Grad-CAM interpretability overlay + final prediction

Run with:
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import torch
from pathlib import Path
from PIL import Image

from gradcam_utils import (
    CLASS_NAMES,
    ActivationExtractor,
    GradCAM,
    activation_to_heatmap,
    get_last_conv_layer,
    load_ben_color,
    overlay_heatmap,
    to_model_tensor,
)

# ==========================================================
#                     FIXED CONFIG
# ==========================================================

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "densenet_121_11epochs_qwk0.8846.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EXAMPLES_DIR = APP_DIR / "examples"
EXAMPLES_MANIFEST = EXAMPLES_DIR / "manifest.csv"

st.set_page_config(
    page_title="APTOS DR Grading — DenseNet121",
    page_icon="🩺",
    layout="wide",
)


# ==========================================================
#                     MODEL LOADING
# ==========================================================

@st.cache_resource(show_spinner="Loading model checkpoint...")
def load_model(checkpoint_path):
    """Loads the FULL saved model (torch.save(model), not just state_dict) from a fixed path."""
    model = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.to(DEVICE)
    model.eval()
    return model


@st.cache_data(show_spinner=False)
def load_examples_manifest():
    """
    Reads examples/manifest.csv (columns: filename, true_label).
    Only rows whose image file actually exists in examples/ are kept.
    Returns (dataframe, diagnostics_dict) so the UI can explain exactly what went wrong.
    """
    diagnostics = {
        "app_dir": str(APP_DIR),
        "examples_dir": str(EXAMPLES_DIR),
        "examples_dir_exists": EXAMPLES_DIR.exists(),
        "manifest_path": str(EXAMPLES_MANIFEST),
        "manifest_exists": EXAMPLES_MANIFEST.exists(),
        "files_found_in_examples_dir": [],
        "missing_files": [],
    }

    if EXAMPLES_DIR.exists():
        diagnostics["files_found_in_examples_dir"] = sorted(
            p.name for p in EXAMPLES_DIR.iterdir() if p.is_file()
        )

    if not EXAMPLES_MANIFEST.exists():
        return pd.DataFrame(columns=["filename", "true_label"]), diagnostics

    df = pd.read_csv(EXAMPLES_MANIFEST)
    exists_mask = df["filename"].apply(lambda f: (EXAMPLES_DIR / f).exists())
    diagnostics["missing_files"] = df.loc[~exists_mask, "filename"].tolist()
    df = df[exists_mask]
    return df.reset_index(drop=True), diagnostics


# ==========================================================
#                        SIDEBAR
# ==========================================================

st.sidebar.title("⚙️ Settings")

st.sidebar.subheader("1. Preprocessing")
img_size = st.sidebar.slider("Image size (px)", 128, 384, 224, step=16)
sigma_x = st.sidebar.slider("Ben Graham blur sigma (sigmaX)", 1, 30, 10)

st.sidebar.subheader("2. Grad-CAM")
gradcam_alpha = st.sidebar.slider("Heatmap overlay opacity", 0.1, 0.9, 0.4)

st.sidebar.caption(f"Model: `{MODEL_PATH}`")
st.sidebar.caption(f"Running on: **{DEVICE.upper()}**")


# ==========================================================
#                  SESSION STATE (WIZARD)
# ==========================================================

if "wizard_step" not in st.session_state:
    st.session_state.wizard_step = 0

STEP_TITLES = ["1. Preprocessing", "2. Inside DenseNet121", "3. Grad-CAM & Prediction"]


# ==========================================================
#                        MAIN PAGE
# ==========================================================

st.title("🩺 Diabetic Retinopathy Grading")
st.caption(
    "DenseNet121 trained on APTOS 2019 · QWK 0.88 (test) · "
    "Upload a fundus image to see preprocessing, internal representations, "
    "Grad-CAM, and the final grading."
)

model = load_model(MODEL_PATH)

st.subheader("Choose an image")
input_mode = st.radio(
    "Image source", ["Upload your own", "Try an example"], horizontal=True, label_visibility="collapsed"
)

true_label_idx = None
image_key = None

if input_mode == "Upload your own":
    uploaded_image = st.file_uploader(
        "Upload a retina / fundus image", type=["jpg", "jpeg", "png"]
    )
    if uploaded_image is None:
        st.info("👆 Upload a retina image to run the pipeline.")
        st.stop()
    pil_image = Image.open(uploaded_image).convert("RGB")
    image_key = uploaded_image.name

else:
    examples_df, diag = load_examples_manifest()
    if examples_df.empty:
        st.warning(
            "No usable example images found. Here's what the app is actually seeing "
            "on disk right now — use this to spot the mismatch:"
        )
        st.json(diag)
        st.caption(
            "Common causes: (1) `examples/` isn't sitting next to `app.py` in the deployed repo, "
            "(2) filenames in `manifest.csv` don't exactly match the files on disk (case-sensitive), "
            "(3) the images are tracked with **Git LFS** and only pointer files got pulled — check the "
            "file sizes on GitHub; real images should be tens/hundreds of KB, LFS pointers are ~130 bytes."
        )
        st.stop()

    labeled_options = [
        f"{row.filename}  —  true: {CLASS_NAMES[int(row.true_label)]} ({int(row.true_label)})"
        for row in examples_df.itertuples()
    ]
    choice = st.selectbox("Pick an example image", labeled_options)
    chosen_row = examples_df.iloc[labeled_options.index(choice)]

    pil_image = Image.open(EXAMPLES_DIR / chosen_row["filename"]).convert("RGB")
    true_label_idx = int(chosen_row["true_label"])
    image_key = chosen_row["filename"]

image_rgb = np.array(pil_image)

# Reset the wizard whenever the selected image changes
if st.session_state.get("last_file") != image_key:
    st.session_state.wizard_step = 0
    st.session_state.last_file = image_key
    st.session_state.has_run = False

run_button = st.button("🔬 Run analysis", type="primary")

if not run_button and not st.session_state.get("has_run", False):
    st.image(pil_image, caption="Selected image (click 'Run analysis')", width=350)
    st.stop()

if run_button:
    st.session_state.has_run = True

# ----------------------------------------------------------
# Preprocessing — computed unconditionally so every wizard
# step has access to it, regardless of which step is showing
# ----------------------------------------------------------
steps = load_ben_color(image_rgb, img_size=img_size, sigmaX=sigma_x)
final_image_for_model = steps["4. Ben Graham color processed"]
input_tensor = to_model_tensor(final_image_for_model).to(DEVICE)

# ----------------------------------------------------------
# Run model once (no grad) to grab block activations —
# also computed unconditionally for the same reason
# ----------------------------------------------------------
extractor = ActivationExtractor(model)
with torch.no_grad():
    _ = model(input_tensor)
activations = dict(extractor.activations)
extractor.remove()

# ----------------------------------------------------------
# Step indicator
# ----------------------------------------------------------
st.subheader(STEP_TITLES[st.session_state.wizard_step])

# ----------------------------------------------------------
# STEP 0 — Preprocessing pipeline
# ----------------------------------------------------------
if st.session_state.wizard_step == 0:
    cols = st.columns(len(steps))
    for col, (name, img) in zip(cols, steps.items()):
        with col:
            st.image(img, caption=name, use_container_width=True)

    st.markdown(
        f"""
        **What's happening at each step:**
        - **Cropped** — dark/black borders around the fundus are removed.
        - **Resized** — image standardized to `{img_size}x{img_size}` px.
        - **Ben Graham color processed** — local contrast is boosted by subtracting a
          Gaussian-blurred version of the image (`sigmaX={sigma_x}`), making lesions
          like hemorrhages and exudates easier to see.
        - Finally, the image is converted to a tensor and normalized with
          ImageNet mean/std before being fed to DenseNet121.
        """
    )

# ----------------------------------------------------------
# STEP 1 — Feature maps through the network
# ----------------------------------------------------------
elif st.session_state.wizard_step == 1:
    st.caption(
        "Each panel shows the *mean activation* across all channels at that stage "
        "(brighter = more strongly activated), resized for display. "
        "Notice how early layers respond to edges/vessels, while deeper layers "
        "activate on more abstract, lesion-like patterns."
    )

    layer_names = extractor.layer_names
    cols = st.columns(3)
    for i, name in enumerate(layer_names):
        heatmap = activation_to_heatmap(activations[name], display_size=img_size)
        with cols[i % 3]:
            st.image(heatmap, caption=name, use_container_width=True, clamp=True)
            shape = tuple(activations[name].shape)
            st.caption(f"Feature map shape: {shape}")

# ----------------------------------------------------------
# STEP 2 — Grad-CAM + final prediction
# ----------------------------------------------------------
elif st.session_state.wizard_step == 2:
    target_layer = get_last_conv_layer(model)
    gradcam = GradCAM(model, target_layer)
    heatmap, pred_idx, probs = gradcam.compute_heatmap(input_tensor)
    gradcam.remove_hooks()

    overlay, colored_heatmap = overlay_heatmap(
        heatmap, final_image_for_model, alpha=gradcam_alpha
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(final_image_for_model, caption="Preprocessed input", use_container_width=True)
    with c2:
        st.image(colored_heatmap, caption="Grad-CAM heatmap", use_container_width=True)
    with c3:
        st.image(overlay, caption="Overlay", use_container_width=True)

    st.divider()
    st.subheader("Final prediction")

    predicted_class = CLASS_NAMES[pred_idx]
    confidence = probs[pred_idx] * 100

    col_pred, col_chart = st.columns([1, 2])

    with col_pred:
        st.metric("Predicted grade", f"{pred_idx} — {predicted_class}")
        st.metric("Confidence", f"{confidence:.1f}%")

        severity_map = {0: "🟢", 1: "🟡", 2: "🟠", 3: "🔴", 4: "🔴🔴"}
        st.markdown(f"### {severity_map[pred_idx]} {predicted_class}")

        if true_label_idx is not None:
            true_label_name = CLASS_NAMES[true_label_idx]
            distance = abs(true_label_idx - pred_idx)
            st.divider()
            st.metric("True grade", f"{true_label_idx} — {true_label_name}")
            if distance == 0:
                st.success("✅ Exact match")
            elif distance == 1:
                st.warning(f"🟡 Off by {distance} grade — adjacent-class error")
            else:
                st.error(f"❌ Off by {distance} grades — significant error")

    with col_chart:
        prob_df = pd.DataFrame(
            {"Class": [f"{i} - {n}" for i, n in enumerate(CLASS_NAMES)], "Probability": probs}
        ).set_index("Class")
        st.bar_chart(prob_df, use_container_width=True)

    st.caption(
        "Grad-CAM highlights the regions the model relied on most for this prediction. "
        "This is a research/educational tool and is not a substitute for clinical diagnosis."
    )

# ----------------------------------------------------------
# Wizard navigation
# ----------------------------------------------------------
st.divider()
col_back, col_spacer, col_next = st.columns([1, 4, 1])
with col_back:
    if st.button("⬅ Back", disabled=st.session_state.wizard_step == 0):
        st.session_state.wizard_step -= 1
        st.rerun()
with col_next:
    if st.button("Next ➡", disabled=st.session_state.wizard_step == 2):
        st.session_state.wizard_step += 1
        st.rerun()
