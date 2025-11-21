# app.py
"""
Cinematic Streamlit app for Liver Cirrhosis Detector
- Full-screen intro with background image and Start button
- Smooth CSS transition to upload screen
- Robust NIfTI loading (preserve original filename + gzip detection)
- ViT feature extraction + RandomForest inference (safe loader with cloudpickle fallback)
- Large progress bar with step messages
- Separate result screens for Healthy / Borderline / Cirrhosis
"""
import os
import io
import re
import time
import base64
import gzip
import shutil
import joblib
import cloudpickle
import numpy as np
from PIL import Image
import streamlit as st
import nibabel as nib
from skimage.transform import resize
import cv2
import torch
import timm
from torchvision import transforms
import matplotlib.pyplot as plt

# -------------------------
# CONFIG
# -------------------------
st.set_page_config(page_title="Liver MRI Cinematic", layout="wide")
MODEL_PATH = "RandomForest_Cirrhosis.pkl"   # make sure uploaded model filename matches this
DEMO_BG_PATH = "/mnt/data/f09a1a95-f614-4982-a8e8-8e38ea70ccf7.png"  # your demo image path (embedded)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LOWER_THRESHOLD = 0.455
UPPER_THRESHOLD = 0.475
SLICE_INFO_THRESHOLD = 0.465

# -------------------------
# UTILS
# -------------------------
def read_file_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def svg_doctor():
    return """
    <svg width="130" height="130" viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">
      <circle cx="60" cy="60" r="58" fill="#e6f0ff"/>
      <g transform="translate(16,14)">
        <rect x="20" y="40" rx="6" ry="6" width="60" height="40" fill="#ffffff" stroke="#c7ddff"/>
        <circle cx="40" cy="24" r="14" fill="#ffe8d6" stroke="#f0c9a8"/>
        <rect x="34" y="10" width="12" height="6" rx="2" fill="#7da7ff"/>
      </g>
    </svg>
    """

def svg_nurse():
    return """
    <svg width="120" height="120" viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">
      <circle cx="60" cy="60" r="58" fill="#fff6e6"/>
      <g transform="translate(16,14)">
        <rect x="20" y="38" rx="6" ry="6" width="60" height="42" fill="#fff" stroke="#fee8b8"/>
        <circle cx="40" cy="24" r="14" fill="#ffdede" stroke="#f0baba"/>
        <rect x="36" y="8" width="18" height="6" rx="2" fill="#ffb4b4"/>
      </g>
    </svg>
    """

# -------------------------
# Model: ViT backbone (feature extractor)
# cached resource so it's not reloaded on each run
# -------------------------
@st.cache_resource(show_spinner=False)
def load_vit():
    model = timm.create_model("vit_base_patch16_224", pretrained=True)
    # replace classifier head with identity to get embeddings
    if hasattr(model, "head"):
        model.head = torch.nn.Identity()
    elif hasattr(model, "fc"):
        model.fc = torch.nn.Identity()
    model.to(DEVICE)
    model.eval()
    return model

vit_model = load_vit()
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
transform_3ch = transforms.Compose([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

# -------------------------
# RandomForest safe loader (joblib then cloudpickle fallback)
# -------------------------
@st.cache_resource(show_spinner=False)
def load_rf(path=MODEL_PATH):
    if not os.path.exists(path):
        return None, f"Model file not found at path: {path}"
    try:
        m = joblib.load(path)
        return m, None
    except ModuleNotFoundError as mnf:
        return None, f"ModuleNotFoundError while loading RF: {mnf}. Hint: pin scikit-learn in requirements.txt"
    except Exception as e:
        # try cloudpickle fallback
        try:
            with open(path, "rb") as fh:
                m = cloudpickle.load(fh)
            return m, None
        except Exception as e2:
            return None, f"Failed to load model. joblib error: {e}; cloudpickle error: {e2}"

rf_model, rf_err = load_rf()

# -------------------------
# Helper: safe NIfTI save & load
# -------------------------
def save_uploaded_preserve_name(uploaded_file, target_dir="/tmp"):
    """Save uploaded Streamlit file-like (UploadedFile) to disk preserving name and content."""
    orig_name = getattr(uploaded_file, "name", None) or "uploaded.nii"
    safe_name = orig_name.replace(" ", "_")
    out_path = os.path.join(target_dir, safe_name)
    with open(out_path, "wb") as fh:
        fh.write(uploaded_file.getbuffer())
    return out_path

def ensure_correct_and_load(path):
    """Detect gzip magic, correct extension if needed, try to make nibabel happy, and return nibabel image."""
    # check file magic
    with open(path, "rb") as fh:
        head = fh.read(2)
    is_gz = head == b'\x1f\x8b'
    if is_gz and not path.endswith(".gz"):
        newp = path + ".gz"
        os.rename(path, newp)
        path = newp
    if (not is_gz) and path.endswith(".gz"):
        # try to decompress into .nii
        try:
            decompressed = path[:-3]
            with gzip.open(path, "rb") as gz, open(decompressed, "wb") as out:
                shutil.copyfileobj(gz, out)
            path = decompressed
        except Exception:
            pass
    # now load via nibabel (will raise if unreadable)
    img = nib.load(path)
    return img

# -------------------------
# Feature extraction helpers
# -------------------------
def preprocess_slice(sl):
    # Handle constant slices
    if np.nanmax(sl) - np.nanmin(sl) < 1e-6:
        sln = np.zeros_like(sl, dtype=np.float32)
    else:
        sl = np.nan_to_num(sl)
        sln = (sl - sl.min()) / (sl.max() - sl.min() + 1e-8)
    sln = (nlm_denoise(sln) * 255).astype(np.uint8) / 255.0
    sln = resize(sln, (224, 224), preserve_range=True).astype(np.float32)
    return sln

def nlm_denoise(slice_img):
    img = np.clip((slice_img * 255.0), 0, 255).astype(np.uint8)
    den = cv2.fastNlMeansDenoising(img, None, h=10, templateWindowSize=7, searchWindowSize=21)
    return den.astype(np.float32) / 255.0

def vit_extract_batch(slices):
    if len(slices) == 0:
        feat_dim = getattr(vit_model, "embed_dim", 768)
        return np.zeros((0, feat_dim), dtype=np.float32)
    batch = []
    for s in slices:
        img = np.clip((s * 255.0), 0, 255).astype(np.uint8)
        rgb = np.stack([img] * 3, axis=-1)
        pil = Image.fromarray(rgb)
        batch.append(transform_3ch(pil))
    xb = torch.stack(batch).to(DEVICE)
    with torch.no_grad():
        feats = vit_model(xb)
    return feats.cpu().numpy()

def fuse_features(t1_feats, t2_feats):
    L = min(len(t1_feats), len(t2_feats))
    if L == 0:
        d1 = t1_feats.shape[1] if len(t1_feats) > 0 else 0
        d2 = t2_feats.shape[1] if len(t2_feats) > 0 else 0
        return np.zeros((0, d1 + d2), dtype=np.float32)
    return np.concatenate([t1_feats[:L], t2_feats[:L]], axis=1)

# -------------------------
# Visualization helpers
# -------------------------
def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return "data:image/png;base64," + b64

def gauge_png(prob):
    fig, ax = plt.subplots(figsize=(3,3))
    ax.pie([prob, 1-prob], startangle=90, wedgeprops={'width':0.38})
    ax.text(0, 0, f"{prob*100:.1f}%", ha='center', va='center', fontsize=18, weight='bold')
    ax.set(aspect='equal')
    return fig_to_b64(fig)

def bar_png(cirr, healthy):
    fig, ax = plt.subplots(figsize=(4,2))
    ax.bar(['Cirrhosis-leaning','Healthy-leaning'], [cirr, healthy], color=['#ff7b7b','#7bcff2'])
    ax.set_ylabel("Number of slices")
    return fig_to_b64(fig)

# -------------------------
# UI sections: intro, upload, results
# -------------------------
# CSS & animations (fade/scale for simulated morph)
INTRO_BG_B64 = read_file_b64(DEMO_BG_PATH) if os.path.exists(DEMO_BG_PATH) else None

st.markdown(
    """
    <style>
    .full-hero {
      height: 85vh;
      display:flex;
      align-items:center;
      justify-content:center;
      color: white;
      border-radius:12px;
      background-size: cover;
      background-position: center;
      position: relative;
      overflow: hidden;
    }
    .overlay {
      position:absolute; inset:0; background: linear-gradient(90deg, rgba(3,10,24,0.65), rgba(7,20,40,0.65));
      z-index:1;
    }
    .hero-content { z-index:2; text-align:center; }
    .start-button {
      background: linear-gradient(90deg,#7c3aed,#00c2ff);
      padding:14px 28px; font-size:20px; color:white; border-radius:12px; border:none; cursor:pointer;
      box-shadow: 0 6px 18px rgba(0,0,0,0.25);
    }
    .small-desc { opacity:0.9; margin-top:10px; color:#e6eef8 }
    .fade-in { animation: fadeScale 0.7s ease forwards; opacity:0; transform: scale(0.96); }
    @keyframes fadeScale {
      to { opacity:1; transform: scale(1); }
    }
    .result-healthy { background: linear-gradient(90deg,#e8fff2,#d7fff0); padding:24px; border-radius:12px;}
    .result-cirr { background: linear-gradient(90deg,#fff0f0,#ffe6e6); padding:24px; border-radius:12px;}
    .result-border { background: linear-gradient(90deg,#fffaf0,#fff5e6); padding:24px; border-radius:12px;}
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize screens
if "screen" not in st.session_state:
    st.session_state.screen = "intro"
if "progress" not in st.session_state:
    st.session_state.progress = 0

def show_intro():
    bg_style = f"background-image: url(data:image/png;base64,{INTRO_BG_B64});" if INTRO_BG_B64 else "background: linear-gradient(90deg,#071237,#08203a);"
    st.markdown(f"""
    <div class="full-hero" style="{bg_style}">
      <div class="overlay"></div>
      <div class="hero-content fade-in">
        <h1 style="font-size:44px; margin:0 0 8px 0;">Liver MRI Cinematic</h1>
        <p class="small-desc">AI-assisted liver cirrhosis screening — research tool only. Click start to continue.</p>
        <div style="margin-top:18px;">
          <button class="start-button" onclick="window.streamlitStartClicked = true;">▶ Start</button>
        </div>
        <div style="margin-top:16px; display:flex; gap:20px; justify-content:center; align-items:center;">
          <div style="width:120px;">{svg_doctor()}</div>
          <div style="width:120px;">{svg_nurse()}</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Detect the JS flag using a small trick: a Streamlit button underneath labeled "Start" (hidden)
    # User will click the visible button; instruct them fallback to click below if JS doesn't work.
    placeholder = st.empty()
    if placeholder.button("Start (If visible button didn't work)"):
        st.session_state.screen = "upload"

    # We also try to read a JS global if it was set (works in many browsers); polling via a hidden iframe isn't straightforward in Streamlit,
    # so we keep the button fallback. Users will click either the hero button or the hidden button.

def show_upload():
    st.markdown("<div class='fade-in' style='padding:18px;'><h2>Upload MRI Volumes</h2><p>Upload paired <b>T1</b> and <b>T2</b> MRI volumes (.nii / .nii.gz) from the same patient.</p></div>", unsafe_allow_html=True)
    col1, col2 = st.columns([1,1])
    with col1:
        t1 = st.file_uploader("Upload T1 (.nii / .nii.gz)", type=["nii","gz","nii.gz"])
        t2 = st.file_uploader("Upload T2 (.nii / .nii.gz)", type=["nii","gz","nii.gz"])
        use_demo = st.button("Use Demo Data (sample)", key="demo_btn")
        start = st.button("Start AI Analysis", key="analyze_btn")
    with col2:
        st.markdown("<div style='padding:12px; border-radius:10px; background:#f6fbff'><h4>Status</h4><div id='status'>Waiting for files.</div></div>", unsafe_allow_html=True)
        st.markdown("<div style='margin-top:12px'><i>Keep PHI out of public demos.</i></div>", unsafe_allow_html=True)
        # output placeholders
        status_placeholder = st.empty()
        progress_placeholder = st.empty()
        viz_col = st.empty()

    # Demo handling: generate synthetic or use demo image (we'll simulate with tiny synthetic NIfTI created from zeros)
    if use_demo:
        # create tiny synthetic volumes (64x64x16) and write as NIfTI
        demo_t1 = np.zeros((64,64,16), dtype=np.float32)
        demo_t2 = np.zeros((64,64,16), dtype=np.float32)
        # add some synthetic pattern so features not constant
        demo_t1[16:48,16:48,6:10] = 0.6
        demo_t2[18:46,18:46,6:10] = 0.4
        t1_path = "/tmp/demo_t1.nii"
        t2_path = "/tmp/demo_t2.nii"
        nib.Nifti1Image(demo_t1, affine=np.eye(4)).to_filename(t1_path)
        nib.Nifti1Image(demo_t2, affine=np.eye(4)).to_filename(t2_path)
        st.success("Demo volumes prepared. Click Start AI Analysis to run.")
        # store paths in session for processing
        st.session_state.demo_t1 = t1_path
        st.session_state.demo_t2 = t2_path
        t1 = None
        t2 = None

    # Start processing when Start AI Analysis clicked
    if start:
        # allow demo pipeline if demo_t1 exists
        if hasattr(st.session_state, "demo_t1") and hasattr(st.session_state, "demo_t2"):
            t1_path = st.session_state.demo_t1
            t2_path = st.session_state.demo_t2
            try:
                run_pipeline_from_paths(t1_path, t2_path, status_placeholder, progress_placeholder)
            except Exception as e:
                st.error("Error during pipeline: " + str(e))
        else:
            # require uploads present
            if t1 is None or t2 is None:
                st.error("Please upload both T1 and T2 volumes (or use Demo Data).")
            else:
                try:
                    p1 = save_uploaded_preserve_name(t1)
                    p2 = save_uploaded_preserve_name(t2)
                    run_pipeline_from_paths(p1, p2, status_placeholder, progress_placeholder)
                except Exception as e:
                    st.error("Error saving or loading uploaded files: " + str(e))

def run_pipeline_from_paths(t1_path, t2_path, status_placeholder, progress_placeholder):
    """
    Main processing pipeline using filesystem paths. Updates UI placeholders during the run.
    At the end, writes md report path to session and sets screen to result.
    """
    # 1) ensure nibabel can read
    try:
        status_placeholder.info("Validating volumes...")
        progress_placeholder.progress(5)
        img1 = ensure_correct_and_load(t1_path)
        img2 = ensure_correct_and_load(t2_path)
    except Exception as e:
        status_placeholder.error("Error loading NIfTI: " + str(e))
        return

    vol1 = img1.get_fdata().astype(np.float32)
    vol2 = img2.get_fdata().astype(np.float32)
    n = min(vol1.shape[2], vol2.shape[2])
    if n <= 0:
        status_placeholder.error("Empty volumes or no axial slices.")
        return

    # Preprocessing
    status_placeholder.info(f"Preprocessing {n} slices...")
    progress = 8
    progress_placeholder.progress(progress)
    t1_slices, t2_slices = [], []
    for i in range(n):
        t1_slices.append(preprocess_slice(vol1[:,:,i]))
        t2_slices.append(preprocess_slice(vol2[:,:,i]))
        progress = int(8 + 32*(i+1)/n)
        progress_placeholder.progress(progress)

    # Feature extraction (in chunks)
    status_placeholder.info("Extracting ViT features...")
    progress_placeholder.progress(42)
    chunk = max(1, n//6)
    feats1_chunks, feats2_chunks = [], []
    for start in range(0, n, chunk):
        end = min(n, start+chunk)
        feats1_chunks.append(vit_extract_batch(t1_slices[start:end]))
        feats2_chunks.append(vit_extract_batch(t2_slices[start:end]))
        progress = int(42 + 20*(end)/n)
        progress_placeholder.progress(progress)
    feats1 = np.concatenate(feats1_chunks, axis=0) if feats1_chunks else np.zeros((0, getattr(vit_model,"embed_dim",768)))
    feats2 = np.concatenate(feats2_chunks, axis=0) if feats2_chunks else np.zeros((0, getattr(vit_model,"embed_dim",768)))

    # Fuse + classify
    status_placeholder.info("Fusing features and classifying...")
    progress_placeholder.progress(68)
    fused = fuse_features(feats1, feats2)
    if fused.shape[0] == 0:
        status_placeholder.error("No valid fused features produced.")
        return

    if rf_model is None:
        status_placeholder.error("RandomForest not loaded. " + (rf_err or "Upload model file."))
        return

    try:
        status_placeholder.info("Running classifier...")
        progress_placeholder.progress(80)
        probs = rf_model.predict_proba(fused)[:,1]
    except Exception as e:
        status_placeholder.error("Classifier error: " + str(e))
        return

    final_prob = float(np.mean(probs))
    slices_cirr = int((probs >= SLICE_INFO_THRESHOLD).sum())
    slices_healthy = len(probs) - slices_cirr
    progress_placeholder.progress(92)
    status_placeholder.success("Finalizing results...")
    time.sleep(0.4)
    progress_placeholder.progress(100)

    # prepare visuals and report
    md_report = f"""# AI Liver MRI Report

**Diagnosis (AI):** { 'Cirrhosis' if final_prob>UPPER_THRESHOLD else ('Healthy' if final_prob<LOWER_THRESHOLD else 'Borderline / Inconclusive') }

**Mean probability:** {final_prob*100:.2f}%
**Slices analysed:** {len(probs)}
- Cirrhosis-leaning: {slices_cirr}
- Healthy-leaning: {slices_healthy}

This tool is for research/educational use only.
"""
    # save report
    rpt_path = "/tmp/liver_ai_report.md"
    with open(rpt_path, "w") as fh:
        fh.write(md_report)
    st.session_state["last_report"] = rpt_path
    st.session_state["last_prob"] = final_prob
    st.session_state["last_slices_cirr"] = slices_cirr
    st.session_state["last_slices_healthy"] = slices_healthy

    # choose result screen
    if final_prob < LOWER_THRESHOLD:
        st.session_state.screen = "result_healthy"
    elif final_prob > UPPER_THRESHOLD:
        st.session_state.screen = "result_cirrhosis"
    else:
        st.session_state.screen = "result_borderline"

# -------------------------
# Result screens (distinct styles)
# -------------------------
def show_result_healthy():
    st.markdown("<div class='result-healthy'><h2>✅ Result: Healthy</h2><p>The AI predicts a low cirrhosis probability.</p></div>", unsafe_allow_html=True)
    show_final_common()

def show_result_cirrhosis():
    st.markdown("<div class='result-cirr'><h2>⚠️ Result: Cirrhosis (AI)</h2><p>The AI predicts an elevated cirrhosis probability. Correlate clinically.</p></div>", unsafe_allow_html=True)
    show_final_common()

def show_result_borderline():
    st.markdown("<div class='result-border'><h2>🔶 Result: Borderline / Inconclusive</h2><p>AI result lies in a borderline zone — clinical review recommended.</p></div>", unsafe_allow_html=True)
    show_final_common()

def show_final_common():
    prob = st.session_state.get("last_prob", None)
    cirr = st.session_state.get("last_slices_cirr", 0)
    healthy = st.session_state.get("last_slices_healthy", 0)
    if prob is not None:
        st.markdown(f"**Mean probability:** {prob*100:.2f}%")
    st.markdown(f"- Slices cirrhosis-leaning: **{cirr}**  \n- Slices healthy-leaning: **{healthy}**")
    # visuals
    col1, col2 = st.columns([1,1])
    with col1:
        if prob is not None:
            st.image(gauge_png(prob))
    with col2:
        st.image(bar_png(cirr, healthy))
    st.markdown("### Download report")
    if "last_report" in st.session_state:
        with open(st.session_state["last_report"], "rb") as fh:
            btn = st.download_button("Download Markdown Report", data=fh, file_name="liver_report.md", mime="text/markdown")
    st.markdown("### Actions")
    colA, colB = st.columns([1,1])
    with colA:
        if st.button("Analyze another study"):
            st.session_state.screen = "upload"
    with colB:
        if st.button("Back to Home"):
            st.session_state.screen = "intro"

# -------------------------
# Main app flow
# -------------------------
def main():
    if st.session_state.screen == "intro":
        show_intro()
    elif st.session_state.screen == "upload":
        show_upload()
    elif st.session_state.screen == "result_healthy":
        show_result_healthy()
    elif st.session_state.screen == "result_cirrhosis":
        show_result_cirrhosis()
    elif st.session_state.screen == "result_borderline":
        show_result_borderline()
    else:
        st.info("Unknown app state. Resetting to home.")
        st.session_state.screen = "intro"

# If hero Start clicked via the big visible button, the hidden fallback clickable button will set screen to upload.
# We rely on user clicking either; Streamlit cannot reliably capture custom JS global flags across reruns without components.
main()

