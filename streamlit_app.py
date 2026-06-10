"""
Brain Tumor MRI — Slice-level vs Patient-level Streamlit demo (PRAI 2026)
========================================================================
Companion app for the PRAI 2026 paper on data leakage in brain tumor MRI
classification. For a chosen architecture and seed, it runs BOTH trained
checkpoints on the same image and shows their predictions side by side:

  • SEGMENT split (slice-level, leaky)   — trained with patient leakage
  • SUBJECT split (patient-level)        — trained leakage-free

This makes the leakage effect tangible: the same model+seed, differing only in
how train/test were split, often disagrees on previously-unseen images.

Checkpoints are read from results/checkpoints/ and are named
    {model}_{segment|subject}_train80_seed{seed}.pth
(produced by the _FULL training notebook). The 80–20 ratio is fixed, so only the
model and seed are user-selectable.
"""

import io
import re
import os
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEFAULT_CLASS_NAMES = ["meningioma", "glioma", "pituitary"]
RAW_LABEL_TO_NAME = {1: "meningioma", 2: "glioma", 3: "pituitary"}  # for .mat ground truth

SPLITS = {
    "segment": {"label": "Segment split", "sub": "slice-level · leaky", "color": "#E2603B", "emoji": "🔶"},
    "subject": {"label": "Subject split", "sub": "patient-level · leakage-free", "color": "#0E7C86", "emoji": "🟢"},
}

CKPT_DIR = Path("results/checkpoints")
CKPT_RE = re.compile(r"^(?P<model>.+)_(?P<split>segment|subject)_train(?P<frac>\d+)_seed(?P<seed>\d+)\.pth$")
PRETTY_MODEL = {"resnet50": "ResNet50", "densenet121": "DenseNet121", "efficientnet_b0": "EfficientNet-B0"}


# ----------------------------------------------------------------------------
# Model + Grad-CAM (mirrors the training pipeline)
# ----------------------------------------------------------------------------
def build_classifier_head(in_features: int, num_classes: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes),
    )


def get_model(name: str, num_classes: int) -> nn.Module:
    name = name.lower().strip()
    if name == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = build_classifier_head(model.fc.in_features, num_classes)
        return model
    if name == "densenet121":
        model = models.densenet121(weights=None)
        model.classifier = build_classifier_head(model.classifier.in_features, num_classes)
        return model
    if name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier = build_classifier_head(model.classifier[-1].in_features, num_classes)
        return model
    raise ValueError(f"Unknown model: {name}")


def get_target_layer(model_name: str, model: nn.Module) -> nn.Module:
    model_name = model_name.lower()
    if model_name == "resnet50":
        return model.layer4[-1]
    if model_name == "densenet121":
        return model.features.denseblock4
    if model_name == "efficientnet_b0":
        return model.features[-1]
    raise ValueError(f"No target layer mapping for {model_name}")


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.h1 = target_layer.register_forward_hook(lambda m, i, o: setattr(self, "activations", o))
        self.h2 = target_layer.register_full_backward_hook(lambda m, gi, go: setattr(self, "gradients", go[0]))

    def close(self):
        self.h1.remove()
        self.h2.remove()

    def generate(self, x: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.zero_grad()
        logits = self.model(x)
        logits[:, class_idx].sum().backward(retain_graph=True)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.relu(torch.sum(weights * self.activations, dim=1)).detach().cpu().numpy()[0]
        cam = cam - cam.min()
        return cam / (cam.max() + 1e-8)


def overlay_cam(img_pil: Image.Image, cam: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    img = np.array(img_pil).astype(np.float32) / 255.0
    H, W = img.shape[:2]
    cam_t = torch.tensor(cam, dtype=torch.float32)[None, None, :, :]
    cam_up = F.interpolate(cam_t, size=(H, W), mode="bilinear", align_corners=False)[0, 0].numpy()
    heat = plt.get_cmap("jet")(cam_up)[:, :, :3]
    return np.clip((1 - alpha) * img + alpha * heat, 0, 1)


# ----------------------------------------------------------------------------
# Image loading (supports normal images and figshare .mat slices)
# ----------------------------------------------------------------------------
def _to_uint8(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    mn, mx = float(img.min()), float(img.max())
    img = (img - mn) / (mx - mn) if mx > mn else np.zeros_like(img)
    return (img * 255.0).astype(np.uint8)


def load_mat_image(data: bytes):
    """Return (PIL RGB image, ground_truth_class|None) from a cjdata .mat file."""
    try:
        import h5py
        with h5py.File(io.BytesIO(data), "r") as f:
            cj = f["cjdata"]
            raw_label = int(np.array(cj["label"]).flatten()[0])
            image = np.array(cj["image"]).astype(np.float32).T
    except Exception:
        import scipy.io as sio
        m = sio.loadmat(io.BytesIO(data))
        cj = m["cjdata"][0, 0]
        raw_label = int(np.array(cj["label"]).flatten()[0])
        image = np.array(cj["image"]).astype(np.float32)
    pil = Image.fromarray(_to_uint8(image), mode="L").convert("RGB")
    return pil, RAW_LABEL_TO_NAME.get(raw_label)


def load_any(name: str, data: bytes):
    """Dispatch on extension. Returns (PIL RGB, ground_truth|None)."""
    if name.lower().endswith(".mat"):
        return load_mat_image(data)
    return Image.open(io.BytesIO(data)).convert("RGB"), None


# ----------------------------------------------------------------------------
# Checkpoint discovery + loading
# ----------------------------------------------------------------------------
def discover_checkpoints(ckpt_dir: Path):
    """Return {model: set(seeds)} for (model, seed) pairs that have BOTH splits."""
    have = {}  # (model, seed) -> set(splits)
    for p in ckpt_dir.glob("*.pth"):
        m = CKPT_RE.match(p.name)
        if not m:
            continue
        key = (m.group("model"), int(m.group("seed")))
        have.setdefault(key, set()).add(m.group("split"))
    combos = {}
    for (model, seed), splits in have.items():
        if {"segment", "subject"} <= splits:
            combos.setdefault(model, set()).add(seed)
    return combos


@st.cache_resource(show_spinner=False)
def load_prediction_model(ckpt_path: str):
    device = torch.device("cpu")
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:  # older torch without weights_only kwarg
        ckpt = torch.load(ckpt_path, map_location=device)
    model_name = ckpt.get("model_name", "densenet121")
    class_names = ckpt.get("class_names", DEFAULT_CLASS_NAMES)
    img_size = ckpt.get("img_size", 224)
    model = get_model(model_name, len(class_names))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, class_names, img_size, model_name, device


def run_inference(ckpt_path: Path, image: Image.Image, alpha: float):
    """Run one checkpoint on one image; return dict with pred/conf/probs/overlay."""
    model, class_names, img_size, model_name, device = load_prediction_model(str(ckpt_path))
    tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    x = tfms(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
    pred_idx = int(probs.argmax())
    cammer = GradCAM(model, get_target_layer(model_name, model))
    cam = cammer.generate(x, pred_idx)
    cammer.close()
    overlay = overlay_cam(image.resize((img_size, img_size)), cam, alpha=alpha)
    return {
        "class_names": class_names,
        "probs": probs,
        "pred_idx": pred_idx,
        "pred_class": class_names[pred_idx],
        "conf": float(probs[pred_idx]),
        "overlay": overlay,
    }


def render_split_column(col, split_key: str, ckpt_path: Path, image: Image.Image,
                        alpha: float, true_label):
    meta = SPLITS[split_key]
    with col:
        st.markdown(
            f"<div style='border-left:6px solid {meta['color']};padding-left:8px'>"
            f"<b>{meta['emoji']} {meta['label']}</b><br>"
            f"<span style='color:gray;font-size:0.85em'>{meta['sub']}</span></div>",
            unsafe_allow_html=True,
        )
        if not ckpt_path.exists():
            st.error(f"Checkpoint not found:\n{ckpt_path.name}")
            return None
        try:
            r = run_inference(ckpt_path, image, alpha)
        except Exception as e:
            st.error(f"Error: {e}")
            return None

        correct = (true_label is not None) and (r["pred_class"] == true_label)
        cap = f"Pred: {r['pred_class'].upper()}"
        if true_label is not None:
            cap += "  ✓" if correct else "  ✗"
        st.image(r["overlay"], caption=cap, use_container_width=True)
        st.markdown(f"**Confidence: {r['conf']*100:.2f}%**")
        st.markdown("All probabilities:")
        order = np.argsort(-r["probs"])
        for i in order:
            cn, cp = r["class_names"][i], r["probs"][i]
            if i == r["pred_idx"]:
                st.markdown(f"**• {cn}: {cp*100:.1f}%**")
            else:
                st.markdown(f"• {cn}: {cp*100:.1f}%")
        return r


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Brain Tumor AI — Leakage Demo", layout="wide")
st.title("Brain Tumor MRI — Slice-level vs Patient-level")
st.markdown(
    "For a chosen **model** and **seed**, this app runs the two checkpoints trained under "
    "different data splits and shows their predictions side by side. The **segment** model was "
    "trained with slice-level (leaky) splitting; the **subject** model was trained with "
    "patient-level (leakage-free) splitting. Same architecture, same seed — only the split differs."
)

if not CKPT_DIR.exists():
    st.error(f"Checkpoint directory not found at `{CKPT_DIR}`. "
             "Unzip the training results so that `results/checkpoints/*.pth` exists next to this app.")
    st.stop()

combos = discover_checkpoints(CKPT_DIR)
if not combos:
    st.error("No paired segment/subject checkpoints found in `results/checkpoints/`. "
             "Expected files like `densenet121_segment_train80_seed0.pth` and "
             "`densenet121_subject_train80_seed0.pth`.")
    st.stop()

# ---- Sidebar: model + seed only (80-20 split is fixed) ----
st.sidebar.header("Configuration")
st.sidebar.caption("Split ratio is fixed at 80–20.")
model_ids = sorted(combos.keys())
model_choice = st.sidebar.selectbox(
    "Model", model_ids,
    index=model_ids.index("densenet121") if "densenet121" in model_ids else 0,
    format_func=lambda m: PRETTY_MODEL.get(m, m),
)
seed_choices = sorted(combos[model_choice])
seed_choice = st.sidebar.selectbox("Seed", seed_choices, index=0)
alpha = st.sidebar.slider("Grad-CAM overlay opacity", 0.0, 1.0, 0.5, 0.05)

seg_ckpt = CKPT_DIR / f"{model_choice}_segment_train80_seed{seed_choice}.pth"
subj_ckpt = CKPT_DIR / f"{model_choice}_subject_train80_seed{seed_choice}.pth"
st.sidebar.markdown("**Active checkpoints**")
st.sidebar.markdown(f"🔶 `{seg_ckpt.name}`")
st.sidebar.markdown(f"🟢 `{subj_ckpt.name}`")

with st.expander("What am I comparing?  (paper context)"):
    st.markdown(
        "- **Segment split (slice-level, leaky):** train/test split over images, ignoring patient ID. "
        "Near-identical slices from one patient can appear in both train and test, inflating accuracy.\n"
        "- **Subject split (patient-level):** train/test split over patients, so every slice of a patient "
        "stays on one side. This reflects real, patient-independent performance.\n\n"
        "In the paper, removing leakage lowers accuracy by ~5–7 points and degrades calibration 3–5×, "
        "with meningioma and pituitary tumors affected most."
    )

# ---- Input source ----
input_source = st.radio("Select image source:", ["Upload Image", "Sample Directory"], horizontal=True)
process_queue = []  # (name, PIL image, true_label|None)

if input_source == "Upload Image":
    uploaded = st.file_uploader(
        "Upload MRI images (.jpg, .png, .tif, or figshare .mat slices)",
        type=["jpg", "jpeg", "png", "tif", "tiff", "mat"],
        accept_multiple_files=True,
    )
    for uf in uploaded or []:
        try:
            img, true_label = load_any(uf.name, uf.getvalue())
            process_queue.append((uf.name, img, true_label))
        except Exception as e:
            st.error(f"Could not read {uf.name}: {e}")
else:
    sample_dir = Path(st.text_input("Sample image folder", value="sample_images"))
    if not sample_dir.exists():
        st.info(f"Folder `{sample_dir}` not found. Create it and drop in `.mat`/`.png`/`.jpg` slices, "
                "or switch to Upload Image.")
    else:
        valid = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".mat"}
        files = sorted(str(f.relative_to(sample_dir)) for f in sample_dir.rglob("*")
                       if f.suffix.lower() in valid)
        if not files:
            st.warning("No images found in this folder or its subfolders.")
        else:
            for rel in st.multiselect("Select sample images", files):
                fp = sample_dir / rel
                try:
                    img, true_label = load_any(fp.name, fp.read_bytes())
                    process_queue.append((rel, img, true_label))
                except Exception as e:
                    st.error(f"Error loading {rel}: {e}")

# ---- Process ----
if not process_queue:
    st.info("Upload or select one or more MRI images to begin.")
else:
    st.caption(f"Model: **{PRETTY_MODEL.get(model_choice, model_choice)}**  ·  Seed: **{seed_choice}**  ·  Split: 80–20")
    for filename, image, true_label in process_queue:
        st.divider()
        st.subheader(f"📄 {filename}")
        if true_label is not None:
            st.markdown(f"Ground truth (from .mat): **{true_label}**")

        c_orig, c_seg, c_subj = st.columns([1, 1, 1])
        with c_orig:
            st.markdown("**Original**")
            st.image(image, use_container_width=True)
        r_seg = render_split_column(c_seg, "segment", seg_ckpt, image, alpha, true_label)
        r_subj = render_split_column(c_subj, "subject", subj_ckpt, image, alpha, true_label)

        # ---- Comparison note ----
        if r_seg and r_subj:
            if r_seg["pred_class"] == r_subj["pred_class"]:
                dconf = (r_seg["conf"] - r_subj["conf"]) * 100
                st.success(
                    f"Both splits predict **{r_seg['pred_class'].upper()}**. "
                    f"Segment confidence is {dconf:+.1f} points vs. the patient-level model."
                )
            else:
                st.warning(
                    f"⚠️ Predictions differ — segment (leaky) → **{r_seg['pred_class'].upper()}** "
                    f"({r_seg['conf']*100:.1f}%), subject (patient-level) → "
                    f"**{r_subj['pred_class'].upper()}** ({r_subj['conf']*100:.1f}%). "
                    "Disagreements like this are exactly what slice-level leakage hides."
                )

st.divider()
st.caption(
    "Research and educational demonstration only — not for clinical use. "
    "Models are trained on the Cheng et al. brain tumor dataset (3 classes: meningioma, glioma, pituitary)."
)
