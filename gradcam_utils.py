"""
Utilities for the APTOS DenseNet121 Streamlit app.

Contains:
    - Ben Graham style preprocessing (with intermediate steps kept for display)
    - Feature-map ("activation") extraction across DenseNet121 blocks
    - Grad-CAM (adapted from user-supplied implementation)
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ==========================================================
#            BEN GRAHAM PREPROCESSING (from user code)
# ==========================================================

def crop_image_from_gray(img, tol=7):
    """Crop out dark borders from retinal fundus images."""
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]

    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol

        if mask.any():
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)

        return img

    return img


def load_ben_color(image_rgb, img_size=224, sigmaX=10):
    """
    Runs the full Ben Graham pipeline on an already-loaded RGB uint8 image
    and returns every intermediate step, so the app can display each stage.
    """
    steps = {"1. Original": image_rgb}

    cropped = crop_image_from_gray(image_rgb)
    steps["2. Cropped (dark border removed)"] = cropped

    resized = cv2.resize(cropped, (img_size, img_size))
    steps["3. Resized"] = resized

    ben = cv2.addWeighted(
        resized, 4,
        cv2.GaussianBlur(resized, (0, 0), sigmaX),
        -4, 128
    )
    steps["4. Ben Graham color processed"] = ben

    return steps


def to_model_tensor(ben_color_image):
    """Turns the final Ben-Graham processed RGB uint8 image into a normalized model input tensor."""
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    pil_img = Image.fromarray(ben_color_image)
    tensor = tfm(pil_img).unsqueeze(0)
    return tensor


# ==========================================================
#            DENSENET121 BLOCK ACTIVATION EXTRACTOR
# ==========================================================

class ActivationExtractor:
    """
    Registers forward hooks on the main DenseNet121 stages so we can
    visualize how the representation evolves through the network.
    """

    def __init__(self, model):
        self.activations = {}
        self.hooks = []

        features = model.features
        layer_map = {
            "Stem (conv0 + pool0)": features.pool0,
            "Dense Block 1": features.denseblock1,
            "Dense Block 2": features.denseblock2,
            "Dense Block 3": features.denseblock3,
            "Dense Block 4": features.denseblock4,
            "Final BN (norm5)": features.norm5,
        }
        self.layer_names = list(layer_map.keys())

        for name, layer in layer_map.items():
            h = layer.register_forward_hook(self._make_hook(name))
            self.hooks.append(h)

    def _make_hook(self, name):
        def hook(module, inp, out):
            self.activations[name] = out.detach()
        return hook

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


def activation_to_heatmap(activation_tensor, display_size=224):
    """
    Collapses a (1, C, H, W) activation tensor into a single-channel
    mean-activation heatmap, resized for display. Returns a float array in [0, 1].
    """
    fmap = activation_tensor[0].mean(dim=0).cpu().numpy()
    fmap = fmap - fmap.min()
    if fmap.max() > 0:
        fmap = fmap / fmap.max()
    fmap = cv2.resize(fmap, (display_size, display_size))
    return fmap


# ==========================================================
#                       GRAD-CAM
# ==========================================================
# (Adapted from the user-supplied implementation, unchanged logic.)

class GradCAM:
    """
    Implements Grad-CAM using PyTorch hooks.

    Forward Hook:  saves the feature maps of the target (last conv) layer.
    Backward Hook: saves the gradients flowing back through that layer.

    Heatmap Formula:
        alpha_k = mean(Gradient_k)
        Heatmap = ReLU( sum_k alpha_k * FeatureMap_k )
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.feature_maps = None
        self.gradients = None

        self.forward_hook = target_layer.register_forward_hook(self.save_feature_maps)
        self.backward_hook = target_layer.register_full_backward_hook(self.save_gradients)

    def save_feature_maps(self, module, input, output):
        self.feature_maps = output.detach()

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def compute_heatmap(self, image_tensor, class_index=None):
        self.model.eval()
        image_tensor = image_tensor.clone().requires_grad_(True)

        logits = self.model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)

        if class_index is None:
            class_index = logits.argmax(dim=1).item()

        self.model.zero_grad()
        score = logits[0, class_index]
        score.backward()

        weights = self.gradients[0].mean(dim=(1, 2))
        feature_maps = self.feature_maps[0]

        heatmap = (weights[:, None, None] * feature_maps).sum(dim=0)
        heatmap = torch.clamp(heatmap, min=0)
        heatmap = heatmap.cpu().numpy()

        if heatmap.max() != 0:
            heatmap /= heatmap.max()

        return heatmap, class_index, probabilities[0].detach().cpu().numpy()

    def remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()


def get_last_conv_layer(model):
    """Finds the last nn.Conv2d layer in the model (used as the Grad-CAM target)."""
    last_conv = None
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            last_conv = layer
    return last_conv


def overlay_heatmap(heatmap, original_image, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """Resizes heatmap to the original image size and overlays it as a color map."""
    H, W = original_image.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (W, H))

    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    colored_heatmap_bgr = cv2.applyColorMap(heatmap_uint8, colormap)
    colored_heatmap = cv2.cvtColor(colored_heatmap_bgr, cv2.COLOR_BGR2RGB)

    overlay = (1 - alpha) * original_image.astype(np.float32) + alpha * colored_heatmap.astype(np.float32)
    overlay = overlay.clip(0, 255).astype(np.uint8)

    return overlay, colored_heatmap
