# app.py
"""
Enhanced Cinematic Streamlit Liver MRI App
- Functional centered Start button (Streamlit-native)
- Uses uploaded MRI image from /mnt/data/... as hero background
- Vibrant upload page with better visuals
- Robust NIfTI loading and ViT + RF pipeline (with simulated fallback)
- Separate results screens with clinical + technical interpretation in report
"""
import os, io, time, gzip, shutil, base64, joblib, cloudpickle
import numpy as np
from PIL import Image
import streamlit as st
import nibabel as nib
from skimage.transform import resize
import cv2
import torch, timm
from torchvision import transforms
import matplotlib.pyplot as plt

# ----------------------
# CONFIG
# ----------------------
st.set_page_config(page_title="Liver MRI Cinematic", layout="wide")
MODEL_PATH = "RandomForest_Cirrhosis.pkl"   # upload this to repo root for real predictions
MRI_HERO_PATH = "/mnt/data/f09a1a95-f614-4982-a8e8-8e38ea70ccf7.png"  # your uploaded image path
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LOWER_THRESHOLD = 0.455
UPPER_THRESHOLD = 0.475
SLICE_INFO_THRESHOLD = 0.465

# ----------------------
# Utilities & logging
# ----------------------
LOG_PATH = "/tmp/pipeline_debug.log"
def write_log(msg):
    with open(LOG_PATH, "a") as fh:
        fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {msg}\n")

def read_log(n=200):
    if not os.path.exists(LOG_PATH):
        return ""
    with open(LOG_PATH, "r") as fh:
        lines = fh.readlines()
    return "".join(lines[-n:])

# Clear log at start (helps during dev)
open(LOG_PATH, "w").close()

# ----------------------
# CSS + Hero visuals
# ----------------------
st.markdown("""
<style>
:root { --card-bg: #ffffff; --accent1: #7c3aed; --accent2: #00c2ff; --muted: #6b7280; }
.hero {
  height: 70vh; border-radius: 16px; overflow:hidden; position:relative;
  display:flex; align-items:center; justify-content:center; color:white;
  background-size: cover; background-position:center; box-shadow: 0 20px 50px rgba(2,6,23,0.2);
}
.hero-overlay { position:absolute; inset:0; background:linear-gradient(180deg, rgba(2,6,23,0.55), rgba(2,6,23,0.65)); }
.hero-inner { z-index:2; text-align:center; padding:32px; }
.hero-title { font-size:48px; font-weight:800; margin:0; }
.hero-sub { color: #dbeafd; margin-top:10px; }
.center-btn { margin-top:22px; padding:14px 28px; font-size:18px; border-radius:12px; border:none; color:white;
  background: linear-gradient(90deg,var(--accent1), var(--accent2)); box-shadow: 0 12px 30px rgba(124,58,237,0.2); cursor:pointer; }
.upload-card { padding:20px; border-radius:12px; background:var(--card-bg); box-shadow: 0 12px 30px rgba(2,6,23,0.04); }
.vibrant { background: linear-gradient(90deg, rgba(124,58,237,0.08), rgba(0,194,255,0.06)); border-radius:12px; padding:14px; }
.progress-big { height: 16px; background:#e6f0ff; border-radius:10px; overflow:hidden; margin-top:12px; }
.progress-fill { height:100%; background: linear-gradient(90deg, var(--accent1), var(--accent2)); width:0%; transition: width 0.5s ease; }
.small-muted { color: var(--muted); font-size:14px; }
.section-title { font-size:20px; font-weight:700; margin-bottom:6px; }
.result-note { padding:16px; border-radius:12px; margin-bottom:14px; }
.result-healthy{ background:linear-gradient(90deg,#e8fff2,#d7fff0); }
.result-border{ background:linear-gradient(90deg,#fffaf0,#fff5e6); }
.result-cirr{ background:linear-gradient(90deg,#fff0f0,#ffe6e6); }
</style>
""", unsafe_allow_html=True)

# ----------------------
# Small SVGs & icons
# ----------------------
LIVER_SVG = """
<svg width="160" height="96" viewBox="0 0 200 120" xmlns="http://www.w3.org/2000/svg">
  <path d="M20,70 C30,20 160,20 180,60 C188,80 160,110 110,110 C70,110 40,90 20,70 Z" fill="#ffb4a2" opacity="0.95"/>
</svg>
"""

DOCTOR_SVG = """
<svg width="88" height="88" viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">
  <circle cx="60" cy="60" r="58" fill="#e6f0ff"/>
  <g transform="translate(16,14)"><rect x="20" y="40" rx="6" width="60" height="40" fill="#fff" stroke="#c7ddff"/><circle cx="40" cy="24" r="14" fill="#ffe8d6"/></g>
</svg>
"""

NURSE_SVG = """
<svg width="88" height="88" viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">
  <circle cx="60" cy="60" r="58" fill="#fff6e6"/>
  <g transform="translate(16,14)"><rect x="20" y="38" rx="6" width="60" height="42" fill="#fff" stroke="#fee8b8"/><circle cx="40" cy="24" r="14" fill="#ffdede"/></g>
</svg>
"""

# ----------------------
# Model resources
# ----------------------
@st.cache_resource(show_spinner=False)
def get_vit():
    model = timm.create_model("vit_base_patch16_224", pretrained=True)
    # replace classifier with identity to get embeddings
    if hasattr(model, "head"):
        model.head = torch.nn.Identity()
    elif hasattr(model, "fc"):
        model.fc = torch.nn.Identity()
    model.to(DEVICE)
    model.eval()
    return model

vit_model = get_vit()
transform_3ch = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

@st.cache_resource(show_spinner=False)
def load_rf_model(path=MODEL_PATH):
    if not os.path.exists(path):
        return None, f"Model file {path} not found."
    try:
        m = joblib.load(path)
        return m, None
    except ModuleNotFoundError as mnf:
        return None, f"ModuleNotFoundError: {mnf}"
    except Exception as e:
        # try cloudpickle fallback
        try:
            with open(path,"rb") as fh:
                m = cloudpickle.load(fh)
            return m, None
        except Exception as e2:
            return None, f"Failed to load model. joblib error: {e}; cloudpickle error: {e2}"

rf_model, rf_error = load_rf_model()

# ----------------------
# NIfTI helpers
# ----------------------
def save_uploaded_preserve_name(uploaded_file, target_dir="/tmp"):
    name = getattr(uploaded_file, "name", None) or "uploaded.nii"
    safe = name.replace(" ", "_")
    out = os.path.join(target_dir, safe)
    with open(out, "wb") as fh:
        fh.write(uploaded_file.getbuffer())
    write_log(f"Saved upload to {out}")
    return out

def ensure_correct_and_load(path):
    with open(path, "rb") as fh:
        head = fh.read(2)
    is_gz = head == b'\x1f\x8b'
    if is_gz and not path.endswith(".gz"):
        newp = path + ".gz"
        os.rename(path, newp)
        path = newp
        write_log(f"renamed to {newp}")
    if (not is_gz) and path.endswith(".gz"):
        try:
            outp = path[:-3]
            with gzip.open(path, "rb") as g, open(outp, "wb") as o:
                shutil.copyfileobj(g,o)
            path = outp
            write_log(f"decompressed to {outp}")
        except Exception as e:
            write_log(f"decompress failed: {e}")
    img = nib.load(path)
    return img

# ----------------------
# Preprocess & features
# ----------------------
def nlm_denoise(slice_img):
    img = np.clip(slice_img*255,0,255).astype(np.uint8)
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
    if len(slices) == 0:
        return np.zeros((0, getattr(vit_model,"embed_dim",768)), dtype=np.float32)
    batch = []
    for s in slices:
        img = np.clip(s*255,0,255).astype(np.uint8)
        rgb = np.stack([img]*3, axis=-1)
        pil = Image.fromarray(rgb)
        batch.append(transform_3ch(pil))
    xb = torch.stack(batch).to(DEVICE)
    with torch.no_grad():
        feats = vit_model(xb)
    return feats.cpu().numpy()

def fuse_features(t1f, t2f):
    L = min(len(t1f), len(t2f))
    if L == 0:
        d1 = t1f.shape[1] if len(t1f)>0 else 0
        d2 = t2f.shape[1] if len(t2f)>0 else 0
        return np.zeros((0, d1+d2), dtype=np.float32)
    return np.concatenate([t1f[:L], t2f[:L]], axis=1)

# ----------------------
# Visual helpers
# ----------------------
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
    fig, ax = plt.subplots(figsize=(4,2))
    ax.bar(['Cirrhosis','Healthy'], [cirr, healthy], color=['#ff7b7b','#7bcff2'])
    ax.set_ylabel("Slices")
    return fig_to_b64(fig)

# ----------------------
# Simulation fallback (clearly labeled)
# ----------------------
def simulate_prob_by_stats(vol1, vol2):
    m1 = float(np.nanmean(vol1))
    m2 = float(np.nanmean(vol2))
    if (m1 + m2) == 0:
        return 0.25
    r = m2 / (m1 + 1e-8)
    p = 0.35 + (0.4 * (r / (1 + r)))
    return float(min(max(p, 0.05), 0.95))

# ----------------------
# Pipeline
# ----------------------
def run_pipeline_from_paths(t1_path, t2_path, status_slot, prog_slot):
    write_log("pipeline start")
    try:
        status_slot.info("Validating volumes...")
        prog_slot.progress(5)
        img1 = ensure_correct_and_load(t1_path)
        img2 = ensure_correct_and_load(t2_path)
    except Exception as e:
        write_log(f"nib load error: {e}")
        status_slot.error("Error loading NIfTI: " + str(e))
        return None, f"NIfTI error: {e}"

    vol1 = img1.get_fdata().astype(np.float32)
    vol2 = img2.get_fdata().astype(np.float32)
    write_log(f"vol shapes {vol1.shape} / {vol2.shape}")
    n = min(vol1.shape[2], vol2.shape[2])
    if n <= 0:
        status_slot.error("Empty volumes or no axial slices.")
        return None, "Empty volumes"

    # preprocessing
    status_slot.info(f"Preprocessing {n} slices...")
    prog_slot.progress(8)
    t1_s, t2_s = [], []
    for i in range(n):
        try:
            t1s = preprocess_slice(vol1[:,:,i])
            t2s = preprocess_slice(vol2[:,:,i])
        except Exception as e:
            write_log(f"preprocess slice {i} error: {e}")
            status_slot.error(f"Preprocessing failed on slice {i}: {e}")
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
            write_log(f"vit extract error {start}:{end} -> {e}")
            status_slot.error("Feature extraction error: " + str(e))
            return None, f"ViT error: {e}"
        f1_chunks.append(f1); f2_chunks.append(f2)
        prog_slot.progress(int(42 + 20*(end)/n))

    feats1 = np.concatenate(f1_chunks, axis=0) if f1_chunks else np.zeros((0, getattr(vit_model,"embed_dim",768)))
    feats2 = np.concatenate(f2_chunks, axis=0) if f2_chunks else np.zeros((0, getattr(vit_model,"embed_dim",768)))
    write_log(f"feats shapes {feats1.shape} / {feats2.shape}")

    status_slot.info("Fusing features and classifying...")
    prog_slot.progress(68)
    fused = fuse_features(feats1, feats2)
    write_log(f"fused {fused.shape}")

    if fused.shape[0] == 0:
        status_slot.error("No fused features produced.")
        return None, "No fused features"

    if rf_model is None:
        # provide simulated fallback with clear label
        write_log("rf missing: providing simulated output")
        prob = simulate_prob_by_stats(vol1, vol2)
        ret = {"simulated": True, "prob": prob, "n_slices": len(fused),
               "slices_cirr": int((prob >= SLICE_INFO_THRESHOLD) * len(fused)),
               "slices_healthy": int((1 - (prob >= SLICE_INFO_THRESHOLD)) * len(fused))}
        prog_slot.progress(100)
        status_slot.success("Simulation complete (model missing).")
        return ret, "Model missing; simulated result returned."

    # try predict_proba
    try:
        probs = rf_model.predict_proba(fused)[:,1]
    except Exception as e:
        write_log(f"predict_proba error: {e} fused.shape={fused.shape}")
        # simulated fallback
        prob = simulate_prob_by_stats(vol1, vol2)
        ret = {"simulated": True, "prob": prob, "n_slices": len(fused),
               "slices_cirr": int((prob >= SLICE_INFO_THRESHOLD) * len(fused)),
               "slices_healthy": int((1 - (prob >= SLICE_INFO_THRESHOLD)) * len(fused))}
        prog_slot.progress(100)
        status_slot.error("Model prediction failed; simulated fallback used.")
        return ret, f"predict_proba error: {e}"

    final_prob = float(np.mean(probs))
    slices_cirr = int((probs >= SLICE_INFO_THRESHOLD).sum())
    slices_healthy = len(probs) - slices_cirr
    prog_slot.progress(92)
    status_slot.success("Finalizing results...")
    time.sleep(0.3)
    prog_slot.progress(100)

    return {"simulated": False, "prob": final_prob, "n_slices": len(probs), "slices_cirr": slices_cirr, "slices_healthy": slices_healthy}, None

# ----------------------
# UI: screens
# ----------------------
if "screen" not in st.session_state:
    st.session_state.screen = "intro"

def show_intro():
    # show MRI hero (if available) - else show gradient hero
    if os.path.exists(MRI_HERO_PATH):
        b64 = base64.b64encode(open(MRI_HERO_PATH,"rb").read()).decode("utf-8")
        bg_style = f"background-image: url(data:image/png;base64,{b64});"
    else:
        bg_style = ""
    st.markdown(f"<div class='hero' style='{bg_style}'>"
                f"<div class='hero-overlay'></div>"
                f"<div class='hero-inner'>"
                f"<div style='margin-bottom:12px'>{LIVER_SVG}</div>"
                f"<div class='hero-title'>Liver MRI Cinematic</div>"
                f"<div class='hero-sub'>AI-assisted research support for cirrhosis screening — not a diagnosis.</div>"
                f"<div style='margin-top:18px'>"
                f"</div>"
                f"</div></div>", unsafe_allow_html=True)

    # Centered Start button (real Streamlit button — reliable)
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button("▶ Start", key="center_start", help="Click to proceed to upload and analysis", use_container_width=False):
            st.session_state.screen = "upload"

def show_upload():
    st.markdown("<div class='upload-card vibrant'><div style='display:flex; justify-content:space-between; align-items:center;'><div><div class='section-title'>Upload paired T1 and T2 MRI volumes (.nii / .nii.gz)</div><div class='small-muted'>Please upload DICOM-converted NIfTI volumes. Keep PHI out of public demos.</div></div><div style='display:flex; gap:10px; align-items:center;'>"
                f"<div style='text-align:center'>{DOCTOR_SVG}<div class='small-muted'>Dr. AI</div></div>"
                f"<div style='text-align:center'>{NURSE_SVG}<div class='small-muted'>Nurse AI</div></div>"
                "</div></div></div>", unsafe_allow_html=True)

    left, right = st.columns([1,1])
    with left:
        t1 = st.file_uploader("Upload T1 (.nii / .nii.gz)", type=["nii","nii.gz","gz"])
        t2 = st.file_uploader("Upload T2 (.nii / .nii.gz)", type=["nii","nii.gz","gz"])
        st.markdown("<div style='height:8px'></div>")
        demo = st.button("Use Synthetic Demo (fast demo)")
        start = st.button("Start AI Analysis")
    with right:
        status_slot = st.empty()
        progress_slot = st.empty()
        if st.checkbox("Show debug log (last 200 lines)"):
            st.text_area("Debug log", read_log(200), height=260)

    if demo:
        # create small synthetic volumes for UI/demo
        v1 = np.zeros((64,64,16), dtype=np.float32)
        v2 = np.zeros((64,64,16), dtype=np.float32)
        v1[18:46,18:46,6:10] = 0.6
        v2[20:44,20:44,6:10] = 0.4
        p1 = "/tmp/demo_t1.nii"; p2 = "/tmp/demo_t2.nii"
        nib.Nifti1Image(v1, affine=np.eye(4)).to_filename(p1)
        nib.Nifti1Image(v2, affine=np.eye(4)).to_filename(p2)
        st.success("Synthetic demo created. Click Start AI Analysis.")
        st.session_state._demo_t1 = p1; st.session_state._demo_t2 = p2

    if start:
        if hasattr(st.session_state, "_demo_t1"):
            p1 = st.session_state._demo_t1; p2 = st.session_state._demo_t2
        else:
            if t1 is None or t2 is None:
                st.error("Please upload both T1 and T2 or use Synthetic Demo.")
                return
            p1 = save_uploaded_preserve_name(t1); p2 = save_uploaded_preserve_name(t2)
        status_slot.info("Starting pipeline...")
        progress_slot.progress(2)
        result, guidance = run_pipeline_from_paths(p1, p2, status_slot, progress_slot)
        write_log(f"pipeline result: {result}, guidance: {guidance}")
        if result is None:
            status_slot.error("Pipeline failed. See debug log for details.")
            if guidance:
                st.error(str(guidance))
            return
        st.session_state._last_result = result
        st.session_state._last_guidance = guidance
        if result["prob"] < LOWER_THRESHOLD:
            st.session_state.screen = "result_healthy"
        elif result["prob"] > UPPER_THRESHOLD:
            st.session_state.screen = "result_cirrhosis"
        else:
            st.session_state.screen = "result_borderline"

def show_result_common():
    res = st.session_state.get("_last_result", {})
    sim = res.get("simulated", False)
    prob = res.get("prob", None)
    cirr = res.get("slices_cirr", 0)
    healthy = res.get("slices_healthy", 0)
    if sim:
        st.warning("This result is SIMULATED because the RandomForest model was unavailable or failed. Upload a working model for real predictions.")
    st.markdown(f"**Mean estimated cirrhosis probability:** {prob*100:.2f}%")
    st.markdown(f"- **Slices cirrhosis-leaning:** {cirr}  \n- **Slices healthy-leaning:** {healthy}")
    col1, col2 = st.columns([1,1])
    with col1:
        st.image(gauge_png(prob))
    with col2:
        st.image(bar_png(cirr, healthy))
    # medical + technical explanation
    st.markdown("### Clinical interpretation (concise)")
    if prob is not None:
        if prob > UPPER_THRESHOLD:
            st.markdown("**AI Radiology Note:** The model indicates an elevated probability of cirrhosis. Correlate with clinical status, LFTs (AST, ALT, bilirubin), platelet count, ultrasound elastography, and consider hepatology referral. Imaging signs suggestive of chronic liver disease include surface nodularity, caudate lobe hypertrophy, and heterogeneous parenchymal signal. This tool is an **assistive** device; decisions should be by clinicians.")
        elif prob < LOWER_THRESHOLD:
            st.markdown("**AI Radiology Note:** The model indicates a low probability of cirrhosis. If clinical suspicion remains high, consider further tests (elastography, LFTs), as imaging alone may miss early fibrosis.")
        else:
            st.markdown("**AI Radiology Note:** The model is inconclusive (borderline). Recommend hepatology/radiology review and additional diagnostic tests as indicated.")
    st.markdown("### Technical summary (how the AI decided)")
    st.markdown("- Input: Paired axial T1 & T2 NIfTI slices\n- Feature extraction: Vision Transformer (ViT) embeddings per slice\n- Fusion: Concatenate T1 & T2 embeddings slice-wise\n- Classifier: RandomForest (slice-level probabilities averaged to a study-level score)\n- Borderline band: results in a pre-defined inconclusive region to reduce false positives/negatives.")
    # Downloadable report contains both medical & technical details
    md_report = f"""# AI Liver MRI Report
Diagnosis: {'Cirrhosis' if prob>UPPER_THRESHOLD else ('Healthy' if prob<LOWER_THRESHOLD else 'Borderline / Inconclusive')}
Mean cirrhosis probability: {prob*100:.2f}%
Slices analysed: {res.get('n_slices', 'NA')}
Cirrhosis-leaning: {cirr}
Healthy-leaning: {healthy}

Clinical guidance:
{('Elevated probability; correlate clinically. Recommend LFTs, elastography, hepatology referral.' if prob>UPPER_THRESHOLD else ('Low probability; correlate with clinical findings.' if prob<LOWER_THRESHOLD else 'Inconclusive; specialist review advised.'))}

Technical summary:
- ViT features + RandomForest classifier; fused T1/T2 slice embeddings; averaged slice probabilities.
- Note: If result is simulated, re-save and upload the RandomForest pickle compatible with current scikit-learn version.
"""
    st.download_button("Download full report (Markdown)", md_report.encode("utf-8"), "liver_report.md", mime="text/markdown")
    if st.button("Analyze another study"):
        st.session_state.screen = "upload"
    if st.button("Back to Home"):
        st.session_state.screen = "intro"

def show_result_healthy():
    st.markdown("<div class='result-note result-healthy'><h3>✅ Healthy</h3><p>Low model-estimated probability of cirrhosis.</p></div>", unsafe_allow_html=True)
    show_result_common()

def show_result_borderline():
    st.markdown("<div class='result-note result-border'><h3>🔶 Borderline / Inconclusive</h3><p>Result is within the borderline range — specialist review recommended.</p></div>", unsafe_allow_html=True)
    show_result_common()

def show_result_cirrhosis():
    st.markdown("<div class='result-note result-cirr'><h3>⚠️ Cirrhosis (AI)</h3><p>Elevated model-estimated probability. Correlate with clinical and lab data.</p></div>", unsafe_allow_html=True)
    show_result_common()

# ----------------------
# Router
# ----------------------
if st.session_state.screen == "intro":
    show_intro()
elif st.session_state.screen == "upload":
    show_upload()
elif st.session_state.screen == "result_healthy":
    show_result_healthy()
elif st.session_state.screen == "result_borderline":
    show_result_borderline()
elif st.session_state.screen == "result_cirrhosis":
    show_result_cirrhosis()
else:
    st.session_state.screen = "intro"
    show_intro()

# Footer developer tip
st.markdown("---")
st.markdown("**Note:** If the Streamlit app shows a SIMULATED result, it means the RandomForest pickle was not loaded or failed during prediction. Re-save the model using the same scikit-learn version used during training and upload `RandomForest_Cirrhosis.pkl` to the repository root. Example Colab snippet:\n\n```python\nimport joblib\nm = joblib.load('/path/to/original.pkl')\njoblib.dump(m, 'RandomForest_Cirrhosis.pkl', compress=3)\n```")
if st.checkbox("Show debug log"):
    st.text_area("Debug log", read_log(400), height=280)

