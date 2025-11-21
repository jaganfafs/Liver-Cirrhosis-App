# app.py
# Streamlit cinematic two-screen Liver MRI app (ViT features + RandomForest)
# Put RandomForest_Cirrhosis.pkl in the same repo root as this file.
import os, io, time, base64, joblib, re
import numpy as np
from PIL import Image
import streamlit as st
import nibabel as nib
from skimage.transform import resize
import cv2
import torch, timm
from torchvision import transforms
import matplotlib.pyplot as plt

# --------------------
# Config
# --------------------
MODEL_PATH = "RandomForest_Cirrhosis.pkl"
LOWER_THRESHOLD = 0.455
UPPER_THRESHOLD = 0.475
SLICE_INFO_THRESHOLD = 0.465
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

st.set_page_config(page_title="Liver MRI Cinematic", layout="wide")

# --------------------
# Tiny inline SVGs (doctor + nurse) — copy from previous Gradio version if you want more detail
# --------------------
DOCTOR_SVG = """<svg width="140" height="140" viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg"> ... </svg>"""
NURSE_SVG  = """<svg width="120" height="120" viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg"> ... </svg>"""

# Replace ... with the same SVGs used in your Gradio app (keeps the look consistent).
# --------------------
# Load ViT backbone (feature extractor)
# --------------------
@st.cache_resource(show_spinner=False)
def load_vit():
    model = timm.create_model("vit_base_patch16_224", pretrained=True)
    if hasattr(model, "head"):
        model.head = torch.nn.Identity()
    elif hasattr(model, "fc"):
        model.fc = torch.nn.Identity()
    model.to(DEVICE)
    model.eval()
    return model

vit_model = load_vit()
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
transform_3ch = transforms.Compose([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

# --------------------
# Safe RF loader (joblib + fallback)
# --------------------
@st.cache_resource
def load_rf_model(path=MODEL_PATH):
    if not os.path.exists(path):
        return None, f"Model file not found at path: {path}"
    try:
        m = joblib.load(path)
        return m, None
    except ModuleNotFoundError as mnf:
        return None, f"ModuleNotFoundError: {mnf}. Add scikit-learn with matching version to requirements.txt"
    except Exception as e:
        # try cloudpickle fallback
        try:
            import cloudpickle
            with open(path, "rb") as fh:
                m = cloudpickle.load(fh)
            return m, None
        except Exception as e2:
            return None, f"Failed to load model. joblib error: {e}; cloudpickle error: {e2}"

rf_model, rf_load_err = load_rf_model()

# --------------------
# Helper functions
# --------------------
def extract_patient_id(filename: str) -> str:
    base = os.path.basename(filename or "").lower()
    base = re.sub(r"\.nii(\.gz)?$", "", base)
    tokens = re.split(r"[_\-\.\s]+", base)
    filtered = [t for t in tokens if t not in ["t1","t2","t1w","t2w"] and t != ""]
    return "_".join(filtered) if filtered else base or "unknown"

def validate_modalities_and_patient(t1_name, t2_name):
    if not t1_name or not t2_name:
        return False, "Please provide both files."
    if ("t2" in t1_name.lower()) and ("t1" not in t1_name.lower()):
        return False, f"T1 slot seems to contain a T2 file: {t1_name}"
    if ("t1" in t2_name.lower()) and ("t2" not in t2_name.lower()):
        return False, f"T2 slot seems to contain a T1 file: {t2_name}"
    if extract_patient_id(t1_name) != extract_patient_id(t2_name):
        return False, "Uploaded files appear to be from different patients (filename heuristic)."
    return True, None

def nlm_denoise(slice_img):
    img = np.clip((slice_img * 255.0),0,255).astype(np.uint8)
    den = cv2.fastNlMeansDenoising(img, None, h=10, templateWindowSize=7, searchWindowSize=21)
    return den.astype(np.float32) / 255.0

def preprocess_slice(sl):
    if np.nanmax(sl) - np.nanmin(sl) < 1e-6:
        sln = np.zeros_like(sl, dtype=np.float32)
    else:
        sl = np.nan_to_num(sl)
        sln = (sl - sl.min()) / (sl.max() - sl.min() + 1e-8)
    sln = nlm_denoise(sln)
    sln = resize(sln, (224,224), preserve_range=True).astype(np.float32)
    return sln

def vit_extract_batch(slices):
    if len(slices) == 0:
        feat_dim = getattr(vit_model, "embed_dim", 768)
        return np.zeros((0, feat_dim), dtype=np.float32)
    batch = []
    for s in slices:
        img = np.clip((s * 255.0), 0, 255).astype(np.uint8)
        s_rgb = np.stack([img]*3, axis=-1)
        pil = Image.fromarray(s_rgb)
        batch.append(transform_3ch(pil))
    xb = torch.stack(batch).to(DEVICE)
    with torch.no_grad():
        feats = vit_model(xb)
    return feats.cpu().numpy()

def fuse_features(t1_feats, t2_feats):
    L = min(len(t1_feats), len(t2_feats))
    if L==0:
        d1 = t1_feats.shape[1] if len(t1_feats)>0 else 0
        d2 = t2_feats.shape[1] if len(t2_feats)>0 else 0
        return np.zeros((0, d1+d2), dtype=np.float32)
    return np.concatenate([t1_feats[:L], t2_feats[:L]], axis=1)

# --------------------
# Visualization helpers
# --------------------
def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return "data:image/png;base64," + b64

def gauge_png(prob):
    fig, ax = plt.subplots(figsize=(3,3))
    ax.pie([prob, 1-prob], startangle=90, wedgeprops={'width':0.35})
    ax.text(0,0,f"{prob*100:.1f}%", ha='center', va='center', fontsize=16, weight='bold')
    ax.set(aspect='equal')
    return fig_to_b64(fig)

def bar_png(cirr, healthy):
    fig, ax = plt.subplots(figsize=(4,2))
    ax.bar(['Cirrhosis-leaning','Healthy-leaning'], [cirr, healthy], color=['#ff7b7b','#7bcff2'])
    return fig_to_b64(fig)

# --------------------
# UI layout
# --------------------
def show_intro():
    st.markdown(f"""
    <div style="background:linear-gradient(90deg,#071237,#08203a); padding:20px; border-radius:12px; color:#e6f0ff;">
      <div style="display:flex; gap:18px; align-items:center;">
        <div style="width:140px;">{DOCTOR_SVG}</div>
        <div>
          <h2 style="margin:0">Liver MRI Cinematic — Cirrhosis Assistant</h2>
          <p style="margin:6px 0 0 0;color:#cfe8ff">Upload paired T1/T2 liver MRI. This is a research tool, not diagnostic.</p>
        </div>
      </div>
      <div style="margin-top:10px;">
        <button id="start-btn" style="background:#7c3aed;color:white;border-radius:8px;padding:10px 14px;border:none;font-weight:600;">▶ Start</button>
      </div>
    </div>
    """, unsafe_allow_html=True)

# Two-step UI controlled by session_state
if "screen" not in st.session_state:
    st.session_state["screen"] = "intro"

if st.session_state["screen"] == "intro":
    show_intro()
    # JavaScript to switch to upload (Streamlit needs a small trick)
    if st.button("Start"):
        st.session_state["screen"] = "upload"

if st.session_state["screen"] == "upload":
    # left: upload, right: status & output
    col1, col2 = st.columns([1,1])
    with col1:
        st.subheader("Upload MRI Volumes")
        t1 = st.file_uploader("Upload T1 (.nii / .nii.gz)", type=["nii","gz","nii.gz"])
        t2 = st.file_uploader("Upload T2 (.nii / .nii.gz)", type=["nii","gz","nii.gz"])
        run = st.button("Start AI Analysis", key="run")
        st.markdown("**Privacy note:** keep PHI out of public demos.")
    with col2:
        status = st.empty()
        prog = st.progress(0)
        md_out = st.empty()
        gauge_slot = st.empty()
        bar_slot = st.empty()
        thumbs_slot = st.empty()

    if run:
        # show RF load error if present
        if rf_model is None:
            status.error(rf_load_err or f"RandomForest not loaded. Upload {MODEL_PATH} to repo root.")
        else:
            # validation
            ok, err = validate_modalities_and_patient(getattr(t1, "name", ""), getattr(t2, "name", ""))
            if not ok:
                status.error(err)
            else:
                status.info("🔎 Loading volumes...")
                prog.progress(5)
                try:
                    # write uploaded files to disk for nibabel to read
                    t1_path = "/tmp/t1.nii"
                    t2_path = "/tmp/t2.nii"
                    with open(t1_path,"wb") as f: f.write(t1.read())
                    with open(t2_path,"wb") as f: f.write(t2.read())
                    vol1 = nib.load(t1_path).get_fdata().astype(np.float32)
                    vol2 = nib.load(t2_path).get_fdata().astype(np.float32)
                except Exception as e:
                    status.error("Error loading NIfTI: " + str(e))
                    prog.progress(0)
                    raise

                n = min(vol1.shape[2], vol2.shape[2])
                if n <= 0:
                    status.error("Volumes appear empty.")
                    prog.progress(0)
                else:
                    status.info(f"🧹 Preprocessing {n} slices...")
                    prog.progress(10)
                    t1_slices, t2_slices = [], []
                    for i in range(n):
                        t1_slices.append(preprocess_slice(vol1[:,:,i]))
                        t2_slices.append(preprocess_slice(vol2[:,:,i]))
                        prog.progress(int(10 + 30*(i+1)/n))
                    status.info("⚙️ Extracting features (ViT)...")
                    # chunked extraction
                    feats1_chunks, feats2_chunks = [], []
                    chunk = max(1, n//6)
                    for start in range(0,n,chunk):
                        end = min(n, start+chunk)
                        feats1_chunks.append(vit_extract_batch(t1_slices[start:end]))
                        feats2_chunks.append(vit_extract_batch(t2_slices[start:end]))
                        prog.progress(int(40 + 20*end/n))
                    feats1 = np.concatenate(feats1_chunks, axis=0) if feats1_chunks else np.zeros((0, getattr(vit_model,"embed_dim",768)))
                    feats2 = np.concatenate(feats2_chunks, axis=0) if feats2_chunks else np.zeros((0, getattr(vit_model,"embed_dim",768)))
                    prog.progress(65)
                    status.info("🔗 Fusing features and classifying...")
                    fused = fuse_features(feats1, feats2)
                    if fused.shape[0] == 0:
                        status.error("No fused features produced.")
                        prog.progress(0)
                    else:
                        try:
                            probs = rf_model.predict_proba(fused)[:,1]
                        except Exception as e:
                            status.error("RF prediction failed: " + str(e))
                            prog.progress(0)
                            raise
                        final_prob = float(np.mean(probs))
                        slices_cirr = int((probs>=SLICE_INFO_THRESHOLD).sum())
                        slices_healthy = len(probs) - slices_cirr
                        prog.progress(88)
                        status.success("✅ Results ready")
                        prog.progress(100)
                        # visuals
                        md_report = f"**Diagnosis:** {'Cirrhosis' if final_prob>UPPER_THRESHOLD else ('Healthy' if final_prob<LOWER_THRESHOLD else 'Borderline / Inconclusive')}\\n**Mean prob:** {final_prob*100:.2f}%\\n**Slices:** {len(probs)}\\n**Cirrhosis-leaning:** {slices_cirr}\\n"
                        md_out.markdown(md_report)
                        gauge_slot.image(gauge_png(final_prob))
                        bar_slot.image(bar_png(slices_cirr, slices_healthy))
                        # thumbnails
                        thumbs = []
                        for i in range(min(6,n)):
                            img = (t1_slices[i]*255).astype(np.uint8)
                            thumbs.append(Image.fromarray(img).resize((180,180)))
                        thumbs_slot.image(thumbs, width=180)
                        # save report for download
                        rp = "/tmp/liver_report.md"
                        with open(rp,"w") as fh: fh.write(md_report)
                        st.success("Report saved as /tmp/liver_report.md; download via Streamlit UI or via GitHub after run.")
