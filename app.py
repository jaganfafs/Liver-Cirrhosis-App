# app.py
"""
Cinematic Streamlit Liver MRI app (corrected & robust)
- Full-screen hero with centered Start button (SVG + CSS)
- Upload T1/T2 (.nii or .nii.gz) and run pipeline
- Robust NIfTI load (preserve filename + gzip detect)
- ViT-based feature extraction (timm) and RandomForest classification (joblib/cloudpickle)
- If RF fails (unpickle or predict error), show helpful guidance and offer a simulated-demo result
- Separate result screens for Healthy / Borderline / Cirrhosis
- Progress bar and step messages
- Debug log written to /tmp/pipeline_debug.log (visible in UI)
"""
import os
import io
import time
import gzip
import shutil
import base64
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

# -----------------------
# Configuration
# -----------------------
st.set_page_config(page_title="Liver MRI Cinematic", layout="wide")
MODEL_PATH = "RandomForest_Cirrhosis.pkl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOWER_THRESHOLD = 0.455
UPPER_THRESHOLD = 0.475
SLICE_INFO_THRESHOLD = 0.465

# -----------------------
# Utility helpers
# -----------------------
def write_log(s):
    p = "/tmp/pipeline_debug.log"
    with open(p, "a") as fh:
        fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {s}\n")

def read_log_tail(n_lines=200):
    p = "/tmp/pipeline_debug.log"
    if not os.path.exists(p):
        return ""
    with open(p, "r") as fh:
        lines = fh.readlines()
    return "".join(lines[-n_lines:])

# -----------------------
# CSS / hero visuals (SVG + CSS)
# -----------------------
st.markdown("""
<style>
body { font-family: "Inter", sans-serif; }
.full-hero {
  height: 75vh;
  display:flex; align-items:center; justify-content:center;
  position:relative; border-radius:12px;
  background: linear-gradient(135deg,#071230,#082a3b);
  color: #fff; overflow:hidden;
}
.hero-content { text-align:center; z-index:2; }
.hero-title { font-size:44px; margin: 0; font-weight:700; letter-spacing: -0.02em; }
.hero-sub { opacity:0.9; margin-top:8px; color:#dbeefd; }
.big-start {
  margin-top:18px; padding:14px 28px; font-size:18px; color:#fff; border-radius:12px; border:none;
  background: linear-gradient(90deg,#7c3aed,#00c2ff); box-shadow: 0 8px 28px rgba(0,0,0,0.35); cursor:pointer;
}
.fade-in { animation: fadeScale 0.6s ease both; opacity:0; transform:scale(0.98); }
@keyframes fadeScale { to { opacity:1; transform:scale(1); } }
.section-card { padding:18px; border-radius:10px; background:#fff; box-shadow:0 8px 30px rgba(2,6,23,0.04); }
.progress-big { height:16px; border-radius:10px; background:#e6f0ff; overflow:hidden; }
.progress-fill { height:100%; background:linear-gradient(90deg,#7c3aed,#00c2ff); width:0%; transition: width 0.5s ease; }
.result-healthy { background: linear-gradient(90deg,#e8fff2,#d7fff0); padding:16px; border-radius:10px; }
.result-cirr { background: linear-gradient(90deg,#fff0f0,#ffe6e6); padding:16px; border-radius:10px; }
.result-border { background: linear-gradient(90deg,#fffaf0,#fff5e6); padding:16px; border-radius:10px; }
</style>
""", unsafe_allow_html=True)

LIVER_SVG = """
<svg width="180" height="120" viewBox="0 0 200 120" xmlns="http://www.w3.org/2000/svg">
  <path d="M20,70 C30,20 160,20 180,60 C188,80 160,110 110,110 C70,110 40,90 20,70 Z" fill="#ffb4a2" opacity="0.95"/>
</svg>
"""

DOCTOR_SVG = """
<svg width="80" height="80" viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">
  <circle cx="60" cy="60" r="58" fill="#e6f0ff"/>
  <g transform="translate(16,14)"><rect x="20" y="40" rx="6" ry="6" width="60" height="40" fill="#fff" stroke="#c7ddff"/><circle cx="40" cy="24" r="14" fill="#ffe8d6"/></g>
</svg>
"""

# -----------------------
# Load ViT (cached resource)
# -----------------------
@st.cache_resource(show_spinner=False)
def get_vit():
    model = timm.create_model("vit_base_patch16_224", pretrained=True)
    if hasattr(model, "head"):
        model.head = torch.nn.Identity()
    elif hasattr(model, "fc"):
        model.fc = torch.nn.Identity()
    model.to(DEVICE)
    model.eval()
    return model

vit_model = get_vit()
transform_3ch = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

# -----------------------
# Safe RF loader (joblib, cloudpickle fallback). Cached.
# -----------------------
@st.cache_resource(show_spinner=False)
def load_rf_safe(path=MODEL_PATH):
    if not os.path.exists(path):
        return None, f"Model not found at path: {path}"
    try:
        m = joblib.load(path)
        return m, None
    except ModuleNotFoundError as mnf:
        return None, f"ModuleNotFoundError while loading RF: {mnf}. Hint: pin scikit-learn."
    except Exception as e:
        # fallback try cloudpickle
        try:
            with open(path,"rb") as fh:
                m = cloudpickle.load(fh)
            return m, None
        except Exception as e2:
            return None, f"Failed to load model. joblib error: {e}; cloudpickle error: {e2}"

rf_model, rf_error = load_rf_safe()

# -----------------------
# NIfTI helpers
# -----------------------
def save_uploaded_preserve_name(uploaded_file, target_dir="/tmp"):
    name = getattr(uploaded_file, "name", None) or "uploaded.nii"
    safe = name.replace(" ","_")
    out = os.path.join(target_dir, safe)
    with open(out, "wb") as fh:
        fh.write(uploaded_file.getbuffer())
    write_log(f"Saved uploaded file to {out}")
    return out

def ensure_correct_and_load(path):
    # detect gzip magic
    with open(path,"rb") as fh:
        head = fh.read(2)
    is_gz = head == b'\x1f\x8b'
    if is_gz and not path.endswith(".gz"):
        newp = path + ".gz"
        os.rename(path, newp)
        path = newp
        write_log(f"Renamed to gz: {path}")
    if (not is_gz) and path.endswith(".gz"):
        try:
            outp = path[:-3]
            with gzip.open(path,"rb") as gz, open(outp,"wb") as o:
                shutil.copyfileobj(gz,o)
            path = outp
            write_log(f"Decompressed gz file to: {path}")
        except Exception as e:
            write_log(f"Failed to decompress {path}: {e}")
    img = nib.load(path)
    return img

# -----------------------
# Preprocess & feature helpers
# -----------------------
def nlm_denoise(slice_img):
    img = np.clip(slice_img*255, 0,255).astype(np.uint8)
    den = cv2.fastNlMeansDenoising(img, None, h=10, templateWindowSize=7, searchWindowSize=21)
    return den.astype(np.float32)/255.0

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
    if len(slices)==0:
        return np.zeros((0, getattr(vit_model,"embed_dim",768)), dtype=np.float32)
    batch = []
    for s in slices:
        img = np.clip(s*255,0,255).astype(np.uint8)
        rgb = np.stack([img]*3,axis=-1)
        pil = Image.fromarray(rgb)
        batch.append(transform_3ch(pil))
    xb = torch.stack(batch).to(DEVICE)
    with torch.no_grad():
        feats = vit_model(xb)
    return feats.cpu().numpy()

def fuse_features(t1f, t2f):
    L = min(len(t1f), len(t2f))
    if L==0:
        d1 = t1f.shape[1] if len(t1f)>0 else 0
        d2 = t2f.shape[1] if len(t2f)>0 else 0
        return np.zeros((0, d1+d2), dtype=np.float32)
    return np.concatenate([t1f[:L], t2f[:L]], axis=1)

# -----------------------
# Visualization helpers
# -----------------------
def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    s = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return "data:image/png;base64," + s

def gauge_png(prob):
    fig, ax = plt.subplots(figsize=(3,3))
    ax.pie([prob, 1-prob], startangle=90, wedgeprops={'width':0.38})
    ax.text(0,0,f"{prob*100:.1f}%", ha='center', va='center', fontsize=16, weight='bold')
    ax.set(aspect='equal')
    return fig_to_b64(fig)

def bar_png(cirr, healthy):
    fig, ax = plt.subplots(figsize=(4,2.2))
    ax.bar(['Cirrhosis','Healthy'], [cirr, healthy], color=['#ff7b7b','#7bcff2'])
    ax.set_ylabel('Slices')
    return fig_to_b64(fig)

# -----------------------
# Pipeline with robust error handling and simulated fallback
# -----------------------
def run_pipeline_paths(t1_path, t2_path, status_slot, prog_slot):
    write_log("Pipeline started")
    # 1. load niftis
    try:
        status_slot.info("Validating and loading volumes...")
        prog_slot.progress(5)
        img1 = ensure_correct_and_load(t1_path)
        img2 = ensure_correct_and_load(t2_path)
    except Exception as e:
        write_log(f"Nibabel load error: {e}")
        status_slot.error("Error loading NIfTI: " + str(e))
        return None, f"NIfTI load error: {e}"

    vol1 = img1.get_fdata().astype(np.float32)
    vol2 = img2.get_fdata().astype(np.float32)
    write_log(f"Volume shapes: {vol1.shape} / {vol2.shape}")
    n = min(vol1.shape[2], vol2.shape[2])
    if n<=0:
        status_slot.error("Empty volumes / no axial slices.")
        return None, "Empty volumes"

    # preprocess
    status_slot.info(f"Preprocessing {n} slices...")
    prog_slot.progress(8)
    t1_s, t2_s = [], []
    for i in range(n):
        try:
            t1s = preprocess_slice(vol1[:,:,i])
            t2s = preprocess_slice(vol2[:,:,i])
        except Exception as e:
            write_log(f"Preprocess slice {i} error: {e}")
            status_slot.error(f"Preprocessing failed at slice {i}: {e}")
            return None, f"Preprocess error: {e}"
        t1_s.append(t1s); t2_s.append(t2s)
        prog_slot.progress(int(8 + 32*(i+1)/n))

    # ViT features
    status_slot.info("Extracting ViT features...")
    prog_slot.progress(42)
    chunk = max(1, n//6)
    f1_chunks, f2_chunks = [], []
    for start in range(0, n, chunk):
        end = min(n, start+chunk)
        try:
            f1 = vit_extract_batch(t1_s[start:end])
            f2 = vit_extract_batch(t2_s[start:end])
        except Exception as e:
            write_log(f"ViT extraction error {start}:{end} -> {e}")
            status_slot.error("Feature extraction error: " + str(e))
            return None, f"ViT error: {e}"
        f1_chunks.append(f1); f2_chunks.append(f2)
        prog_slot.progress(int(42 + 20*(end)/n))
    feats1 = np.concatenate(f1_chunks, axis=0) if f1_chunks else np.zeros((0,getattr(vit_model,"embed_dim",768)))
    feats2 = np.concatenate(f2_chunks, axis=0) if f2_chunks else np.zeros((0,getattr(vit_model,"embed_dim",768)))
    write_log(f"Feats shapes: {feats1.shape}, {feats2.shape}")

    # fuse and classify
    status_slot.info("Fusing features and running classifier...")
    prog_slot.progress(68)
    fused = fuse_features(feats1, feats2)
    write_log(f"Fused shape: {fused.shape}")

    if fused.shape[0]==0:
        status_slot.error("No fused features produced.")
        return None, "No fused features"

    if rf_model is None:
        # explicit helpful message
        msg = rf_error or "RandomForest model not loaded in app."
        write_log("RF model missing or error: " + str(msg))
        status_slot.error("RandomForest unavailable: " + str(msg))
        # Offer to produce simulated output for UI demo
        prob = simulate_prob_by_image_stats(vol1, vol2)
        return {"simulated": True, "prob": prob, "n_slices": len(fused), "slices_cirr": int((prob>=SLICE_INFO_THRESHOLD)*len(fused)),
                "slices_healthy": int((1-(prob>=SLICE_INFO_THRESHOLD))*len(fused))}, None

    # Try predict_proba with clear exception handling
    try:
        probs = rf_model.predict_proba(fused)[:,1]
    except Exception as e:
        write_log(f"predict_proba error: {e}. fused.shape={fused.shape}")
        # Provide actionable guidance + simulated fallback
        status_slot.error("Model prediction failed: " + str(e))
        guidance = ("The RandomForest prediction failed. This often happens when the saved model "
                    "expects a different feature-length than the fused features produced here (feature mismatch), "
                    "or a scikit-learn version mismatch when unpickling. See debug logs.")
        write_log("Providing simulated output due to model predict error.")
        prob = simulate_prob_by_image_stats(vol1, vol2)
        return {"simulated": True, "prob": prob, "n_slices": len(fused), "slices_cirr": int((prob>=SLICE_INFO_THRESHOLD)*len(fused)),
                "slices_healthy": int((1-(prob>=SLICE_INFO_THRESHOLD))*len(fused))}, guidance

    final_prob = float(np.mean(probs))
    slices_cirr = int((probs >= SLICE_INFO_THRESHOLD).sum())
    slices_healthy = len(probs) - slices_cirr
    prog_slot.progress(92)
    status_slot.success("Finalizing results...")
    time.sleep(0.3)
    prog_slot.progress(100)

    return {"simulated": False, "prob": final_prob, "n_slices": len(probs), "slices_cirr": slices_cirr, "slices_healthy": slices_healthy}, None

# simple simulation function (clear label in UI)
def simulate_prob_by_image_stats(vol1, vol2):
    # simple heuristic: ratio of mean intensities -> scaled into [0.1,0.9]
    m1 = float(np.nanmean(vol1))
    m2 = float(np.nanmean(vol2))
    if (m1 + m2) == 0:
        return 0.25
    r = m2 / (m1 + 1e-8)
    p = 0.3 + (0.4 * (r / (1 + r)))  # map ratio into 0.3..0.7 roughly
    return float(min(max(p, 0.05), 0.95))

# -----------------------
# UI screens
# -----------------------
if "screen" not in st.session_state:
    st.session_state.screen = "intro"

def show_intro():
    st.markdown(f"<div class='full-hero'><div class='hero-content fade-in'>{LIVER_SVG if LIVER_SVG else ''}<h1 class='hero-title'>Liver MRI Cinematic</h1><p class='hero-sub'>AI-assisted research tool — not a diagnostic device. Click start to continue.</p><div><button class='big-start' onclick=''>▶ Start</button></div></div></div>", unsafe_allow_html=True)
    # fallback button (Streamlit handles interactions)
    if st.button("Start", key="start_fallback"):
        st.session_state.screen = "upload"

def show_upload():
    st.markdown("<div class='section-card fade-in'><h3>Upload paired T1 and T2 MRI volumes (.nii / .nii.gz)</h3></div>", unsafe_allow_html=True)
    col1, col2 = st.columns([1,1])
    with col1:
        t1 = st.file_uploader("Upload T1 (.nii / .nii.gz)", type=["nii","nii.gz","gz"])
        t2 = st.file_uploader("Upload T2 (.nii / .nii.gz)", type=["nii","nii.gz","gz"])
        demo_btn = st.button("Use Synthetic Demo (UI demo)")
        start_btn = st.button("Start AI Analysis")
    with col2:
        st.markdown("<div class='section-card'><h4>Status</h4><div id='status'>Waiting...</div></div>", unsafe_allow_html=True)
        status_slot = st.empty()
        progress_slot = st.empty()
        # show debug toggle
        if st.checkbox("Show debug log (last 200 lines)"):
            st.text_area("Debug log", read_log_tail(200), height=200)

    # use synthetic demo data
    if demo_btn:
        demo_t1 = np.zeros((64,64,16), dtype=np.float32)
        demo_t2 = np.zeros((64,64,16), dtype=np.float32)
        demo_t1[16:48,16:48,6:10] = 0.6
        demo_t2[18:46,18:46,6:10] = 0.4
        p1 = "/tmp/demo_t1.nii"; p2 = "/tmp/demo_t2.nii"
        nib.Nifti1Image(demo_t1, affine=np.eye(4)).to_filename(p1)
        nib.Nifti1Image(demo_t2, affine=np.eye(4)).to_filename(p2)
        st.success("Demo volumes created. Click Start AI Analysis.")
        st.session_state._demo_t1 = p1; st.session_state._demo_t2 = p2

    if start_btn:
        # which source?
        if hasattr(st.session_state, "_demo_t1") and hasattr(st.session_state, "_demo_t2"):
            p1 = st.session_state._demo_t1; p2 = st.session_state._demo_t2
        else:
            if t1 is None or t2 is None:
                st.error("Please upload both T1 and T2 files or pick Demo.")
                return
            p1 = save_uploaded_preserve_name(t1); p2 = save_uploaded_preserve_name(t2)
        status_slot.info("Starting pipeline...")
        progress_slot.progress(2)
        result, guidance = run_pipeline_paths(p1, p2, status_slot, progress_slot)
        write_log(f"Pipeline returned result={result}, guidance={guidance}")
        if result is None:
            status_slot.error("Pipeline failed. See debug log for details.")
            if guidance:
                st.error(str(guidance))
            return
        # save result in session_state
        st.session_state._last_result = result
        st.session_state._last_guidance = guidance
        # set result screen based on prob
        prob = result["prob"]
        if result.get("simulated", False):
            st.session_state._simulated = True
        else:
            st.session_state._simulated = False
        if prob < LOWER_THRESHOLD:
            st.session_state.screen = "result_healthy"
        elif prob > UPPER_THRESHOLD:
            st.session_state.screen = "result_cirrhosis"
        else:
            st.session_state.screen = "result_borderline"

def show_result_common():
    res = st.session_state.get("_last_result", {})
    sim = st.session_state.get("_simulated", False)
    prob = res.get("prob", None)
    cirr = res.get("slices_cirr", 0)
    healthy = res.get("slices_healthy", 0)
    if sim:
        st.warning("Note: This result is SIMULATED for UI demo because the RandomForest model was unavailable or failed. Upload a working pickle to get real predictions.")
    if prob is not None:
        st.markdown(f"**Mean estimated cirrhosis probability:** {prob*100:.2f}%")
    st.markdown(f"- Slices cirrhosis-leaning: **{cirr}**    \n- Slices healthy-leaning: **{healthy}**")
    col1, col2 = st.columns([1,1])
    with col1:
        if prob is not None:
            st.image(gauge_png(prob))
    with col2:
        st.image(bar_png(cirr, healthy))
    # download report
    md = f"# AI Liver MRI Report\n\nDiagnosis: {'Cirrhosis' if prob>UPPER_THRESHOLD else ('Healthy' if prob<LOWER_THRESHOLD else 'Borderline / Inconclusive')}\nMean prob: {prob*100:.2f}%\nSlices cirrhosis: {cirr}\nSlices healthy: {healthy}\n\n(This is a research tool.)"
    st.download_button("Download report (MD)", md.encode("utf-8"), "liver_report.md", mime="text/markdown")
    if st.button("Analyze another study"):
        st.session_state.screen = "upload"
    if st.button("Back to Home"):
        st.session_state.screen = "intro"

def show_result_healthy():
    st.markdown("<div class='result-healthy'><h2>✅ Healthy (AI)</h2><p>Low model-estimated cirrhosis probability.</p></div>", unsafe_allow_html=True)
    show_result_common()

def show_result_cirrhosis():
    st.markdown("<div class='result-cirr'><h2>⚠️ Cirrhosis (AI)</h2><p>Elevated model-estimated cirrhosis probability. Correlate clinically.</p></div>", unsafe_allow_html=True)
    show_result_common()

def show_result_borderline():
    st.markdown("<div class='result-border'><h2>🔶 Borderline / Inconclusive</h2><p>Model output falls within the borderline range; specialist review advised.</p></div>", unsafe_allow_html=True)
    show_result_common()

# -----------------------
# Router
# -----------------------
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
    st.session_state.screen = "intro"
    show_intro()

# helpful footer & diagnostics
st.markdown("---")
st.markdown("**Developer tips:** If the app used a simulated result, re-save your RandomForest pickle in the same environment you trained (matching scikit-learn), then upload `RandomForest_Cirrhosis.pkl` to the repo root. Example Colab snippet to re-save:\n\n```python\nimport joblib\nmodel = joblib.load('/path/to/old.pkl')\njoblib.dump(model, 'RandomForest_Cirrhosis.pkl', compress=3)\n```")
if st.checkbox("Show raw debug log"):
    st.text_area("Pipeline debug log", read_log_tail(400), height=300)

