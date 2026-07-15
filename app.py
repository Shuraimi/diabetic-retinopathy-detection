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

MODEL_PATH = "densenet_121_11epochs_qwk0.8846.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

st.set_page_config(
    page_title="APTOS DR Grading — DenseNet121",
    page_icon="🩺",
    layout="wide",
)


# ==========================================================
#                     MODEL LOADING
# ==========================================================

@st.cache_resource(show_spinner="Loading model checkpoint...")
def load_model(checkpoint_path: str):
    """Loads the FULL saved model (torch.save(model), not just state_dict) from a fixed path."""
    model = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.to(DEVICE)
    model.eval()
    return model


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

from pathlib import Path

st.markdown("## Try an Example")

example_files = {
    "Grade 0 - No DR": "examples/grade0_normal.png",
    "Grade 1 - Mild DR": "examples/grade1_mild.png",
    "Grade 2 - Moderate DR": "examples/grade2_moderate.png",
    "Grade 3 - Severe DR": "examples/grade3_severe.png",
    "Grade 4 - Proliferative DR": "examples/grade4_proliferative.png",
}

choice = st.selectbox(
    "Choose an example image",
    ["None"] + list(example_files.keys())
)

from pathlib import Path

st.subheader("📷 Choose an image")

example_dir = Path("examples")

# filename : true label
example_images = {
    "No DR (Grade 0)": example_dir / "grade0.png",
    "Mild DR (Grade 1)": example_dir / "grade1.png",
    "Moderate DR (Grade 2)": example_dir / "grade2.png",
    "Severe DR (Grade 3)": example_dir / "grade3.png",
    "Proliferative DR (Grade 4)": example_dir / "grade4.png",
}

st.markdown("### Try one of these example images")

cols = st.columns(len(example_images))

selected_example = None

for col, (label, path) in zip(cols, example_images.items()):

    with col:

        if path.exists():

            st.image(str(path), use_container_width=True)

            st.caption(label)

            if st.button(f"Use", key=label):

                selected_example = path

                st.session_state.selected_example = path

                st.session_state.true_label = label

if "selected_example" in st.session_state:

    selected_example = st.session_state.selected_example

uploaded_image = st.file_uploader(
    "Or upload your own image",
    type=["png", "jpg", "jpeg"],
)

true_label = None
current_image_id = None

if uploaded_image is not None:

    pil_image = Image.open(uploaded_image).convert("RGB")

    current_image_id = uploaded_image.name

elif selected_example is not None:

    pil_image = Image.open(selected_example).convert("RGB")

    current_image_id = str(selected_example)

    true_label = st.session_state.get("true_label")

else:

    st.info("Select an example image or upload your own image.")

    st.stop()

image_rgb = np.array(pil_image)

if st.session_state.get("last_file") != current_image_id:

    st.session_state.wizard_step = 0

    st.session_state.last_file = current_image_id

    st.session_state.has_run = False

run_button = st.button("🔬 Run analysis", type="primary")

if not run_button and "has_run" not in st.session_state:
    st.image(pil_image, caption="Uploaded image (click 'Run analysis')", width=350)
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
