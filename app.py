"""
APTOS Diabetic Retinopathy Grading — DenseNet121 Streamlit App

Upload a retina image and see:
    1. The exact preprocessing pipeline (Ben Graham style)
    2. How the representation evolves through DenseNet121's blocks
    3. Grad-CAM interpretability overlay
    4. Final prediction with per-class confidence

Run with:
    streamlit run app.py
"""

import io
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import torch
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

st.set_page_config(
    page_title="APTOS DR Grading — DenseNet121",
    page_icon="🩺",
    layout="wide",
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==========================================================
#                     MODEL LOADING
# ==========================================================

@st.cache_resource(show_spinner="Loading model checkpoint...")
def load_model(checkpoint_bytes: Optional[bytes], checkpoint_path: Optional[str]):
    """
    Loads the FULL saved model (torch.save(model), not just state_dict).
    Accepts either uploaded bytes or a local path.
    """
    if checkpoint_bytes is not None:
        buffer = io.BytesIO(checkpoint_bytes)
        model = torch.load(buffer, map_location=DEVICE, weights_only=False)
    elif checkpoint_path:
        model = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    else:
        return None

    model.to(DEVICE)
    model.eval()
    return model


# ==========================================================
#                        SIDEBAR
# ==========================================================

st.sidebar.title("⚙️ Settings")

st.sidebar.subheader("1. Model checkpoint")
ckpt_source = st.sidebar.radio(
    "Load model from",
    ["Upload file", "Local path on server"],
    index=0,
)

checkpoint_bytes = None
checkpoint_path = None

if ckpt_source == "Upload file":
    uploaded_ckpt = st.sidebar.file_uploader(
        "DenseNet121 checkpoint (.pt / .pth, full model)", type=["pt", "pth"]
    )
    if uploaded_ckpt is not None:
        checkpoint_bytes = uploaded_ckpt.getvalue()
else:
    checkpoint_path = st.sidebar.text_input(
        "Path to checkpoint on the server", value="densenet121_aptos.pt"
    )

st.sidebar.subheader("2. Preprocessing")
img_size = st.sidebar.slider("Image size (px)", 128, 384, 224, step=16)
sigma_x = st.sidebar.slider("Ben Graham blur sigma (sigmaX)", 1, 30, 10)

st.sidebar.subheader("3. Grad-CAM")
gradcam_alpha = st.sidebar.slider("Heatmap overlay opacity", 0.1, 0.9, 0.4)

st.sidebar.caption(f"Running on: **{DEVICE.upper()}**")


# ==========================================================
#                        MAIN PAGE
# ==========================================================

st.title("🩺 Diabetic Retinopathy Grading")
st.caption(
    "DenseNet121 trained on APTOS 2019 · QWK 0.88 (test) · "
    "Upload a fundus image to see preprocessing, internal representations, "
    "Grad-CAM, and the final grading."
)

model = load_model(checkpoint_bytes, checkpoint_path)

if model is None:
    st.info("👈 Upload your trained checkpoint (or set a server path) in the sidebar to get started.")
    st.stop()

uploaded_image = st.file_uploader(
    "Upload a retina / fundus image", type=["jpg", "jpeg", "png"]
)

if uploaded_image is None:
    st.info("👆 Upload a retina image to run the pipeline.")
    st.stop()

# Load as RGB numpy array
pil_image = Image.open(uploaded_image).convert("RGB")
image_rgb = np.array(pil_image)

run_button = st.button("🔬 Run analysis", type="primary")

if not run_button:
    st.image(pil_image, caption="Uploaded image (click 'Run analysis')", width=350)
    st.stop()

tab_preprocess, tab_layers, tab_interpret = st.tabs(
    ["1️⃣ Preprocessing", "2️⃣ Inside DenseNet121", "3️⃣ Grad-CAM & Prediction"]
)

# ----------------------------------------------------------
# TAB 1 — Preprocessing pipeline
# ----------------------------------------------------------
with tab_preprocess:
    st.subheader("Ben Graham style preprocessing")
    steps = load_ben_color(image_rgb, img_size=img_size, sigmaX=sigma_x)

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

    final_image_for_model = steps["4. Ben Graham color processed"]
    input_tensor = to_model_tensor(final_image_for_model).to(DEVICE)

# ----------------------------------------------------------
# Run model once (no grad) to grab block activations
# ----------------------------------------------------------
extractor = ActivationExtractor(model)
with torch.no_grad():
    _ = model(input_tensor)
activations = dict(extractor.activations)
extractor.remove()

# ----------------------------------------------------------
# TAB 2 — Feature maps through the network
# ----------------------------------------------------------
with tab_layers:
    st.subheader("How the representation evolves through DenseNet121")
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
# TAB 3 — Grad-CAM + final prediction
# ----------------------------------------------------------
with tab_interpret:
    st.subheader("Grad-CAM interpretability")

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

    with col_chart:
        prob_df = pd.DataFrame(
            {"Class": [f"{i} - {n}" for i, n in enumerate(CLASS_NAMES)], "Probability": probs}
        ).set_index("Class")
        st.bar_chart(prob_df, use_container_width=True)

    st.caption(
        "Grad-CAM highlights the regions the model relied on most for this prediction. "
        "This is a research/educational tool and is not a substitute for clinical diagnosis."
    )
