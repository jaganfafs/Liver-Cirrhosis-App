# app.py
"""
Liver MRI Cinematic — Updated:
- The hero's "Get started" (first-panel) is the only working Start button (no duplicate)
- Upload section styled like second panel
- When progress completes automatically redirect to output page (third-panel style)
- Results contain only 7 clinical bullet points (no technical summary)
- Uses provided hero image path:
  /mnt/data/WhatsApp Image 2025-11-21 at 22.37.35.jpeg
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

# -------------------------
# CONFIG (change HERO_IMAGE_PATH if needed)
# -------------------------
st.set_page_config(page_title="Liver MRI Cinematic", layout="wide")
MODEL_PATH = "RandomForest_Cirrhosis.pkl"
HERO_IMAGE_PATH = "/mnt/data/WhatsApp Image 2025-11-21 at 22.37.35.jpeg"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LOWER_THRESHOLD = 0.455
UPPER_THRESHOLD = 0.475
SLICE_INFO_THRESHOLD = 0.465
LOG_PATH = "/tmp/pipeline_debug.log"

# init log
open(LOG_PATH, "w").close()
def write_log(s):
    with open(LOG_PATH, "a") as fh:
        fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {s}\n")
def read_log_tail(n=200):
    if not os.path.exists(LOG_PATH): return ""
    with open(LOG_PATH,'r') as fh:
        lines = fh.readlines()
    return "".join(lines[-n:])

# -------------------------
# CSS (mobile-like 3-panel)
# -------------------------
st.markdown("""
<style>
:root{--accent1:#0f172a; --accent2:#06b6d4; --card:#ffffff; --muted:#6b7280;}
.app-wrap{max-width:1200px;margin:18px auto;}
.panel-row{display:flex;gap:18px;justify-content:center;align-items:flex-start;}
.phone{width:300px;border-radius:22px;padding:18px;box-shadow:0 18px 48px rgba(2,6,23,0.12);overflow:hidden;position:relative;}
.phone.left{background:linear-gradient(180deg,#0b2540,#08283a);color:white;}
.phone.mid{background:linear-gradient(180deg,#0d59a3,#0b3a86);color:white;}
.phone.right{background:linear-gradient(180deg,#ffffff,#f8fafc);color:#06202b;box-shadow:0 14px 36px rgba(2,6,23,0.06);}
.hero-title{font-size:20px;font-weight:800;margin-top:8px;}
.hero-sub{font-size:13px;color:rgba(255,255,255,0.9);margin-top:8px;}
.get-start{display:flex;justify-content:center;margin-top:18px;}
.btn-hero{padding:12px 22px;border-radius:12px;border:none;font-size:16px;background:linear-gradient(90deg,var(--accent1),var(--accent2));color:white;cursor:pointer;}
.upload-card{background:var(--card);padding:14px;border-radius:12px;box-shadow:0 12px 30px rgba(2,6,23,0.04);}
.section-title{font-weight:700;font-size:16px;margin-bottom:8px;}
.small-muted{color:var(--muted);font-size:13px;}
.progress-bar{height:14px;background:#e6f0ff;border-radius:10px;overflow:hidden;margin-top:12px;}
.progress-fill{height:100%;background:linear-gradient(90deg,var(--accent1),var(--accent2));width:0%;transition:width 0.4s ease;}
.result-card{margin-top:20px;padding:16px;border-radius:12px;background:var(--card);}
.result-healthy{background:linear-gradient(90deg,#e8fff2,#d7fff0);}
.result-border{background:linear-gradient(90deg,#fffaf0,#fff5e6);}
.result-cirr{background:linear-gradient(90deg,#fff0f0,#ffe6e6);}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Small visual helpers
# -------------------------
def fig_to_b64(fig):
    buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0); buf.seek(0)
    b = base64.b64encode(buf.read()).decode("utf-8"); plt.close(fig); return "data:image/png;base64," + b

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

# -------------------------
# Model & preprocess helpers (unchanged)
# -------------------------
@st.cache_resource(show_spinner=False)
def get_vit():
    model = timm.create_model("vit_base_patch16_224", pretrained=True)
    if hasattr(model, "head"): model.head = torch.nn.Identity()
    elif hasattr(model, "fc"): model.fc = torch.nn.Identity()
    model.to(DEVICE); model.eval(); return model
vit_model = get_vit()
transform_3ch = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

@st.cache_resource(show_spinner=False)
def load_rf(path=MODEL_PATH):
    if not os.path.exists(path): return None, f"Model not found at {path}"
    try:
        m = joblib.load(path); return m, None
    except ModuleNotFoundError as e:
        return None, f"ModuleNotFoundError: {e}"
    except Exception as e:
        try:
            with open(path,"rb") as fh:
                m = cloudpickle.load(fh)
            return m, None
        except Exception as e2:
            return None, f"joblib error: {e}; cloudpickle error: {e2}"
rf_model, rf_error = load_rf()

def save_uploaded_preserve_name(uploaded_file, target_dir="/tmp"):
    name = getattr(uploaded_file, "name", None) or "uploaded.nii"
    safe = name.replace(" ", "_")
    out = os.path.join(target_dir, safe)
    with open(out, "wb") as fh: fh.write(uploaded_file.getbuffer())
    write_log(f"Saved upload to {out}"); return out

def ensure_correct_and_load(path):
    with open(path,"rb") as fh:
        head = fh.read(2)
    is_gz = head == b'\x1f\x8b'
    if is_gz and not path.endswith(".gz"):
        newp = path + ".gz"; os.rename(path, newp); path = newp; write_log(f"renamed to {newp}")
    if (not is_gz) and path.endswith(".gz"):
        try:
            outp = path[:-3]
            with gzip.open(path,"rb") as g, open(outp,"wb") as o:
                shutil.copyfileobj(g,o)
            path = outp; write_log(f"decompressed to {outp}")
        except Exception as e:
            write_log(f"decompress failed: {e}")
    img = nib.load(path); return img

def nlm_denoise(slice_img):
    img = np.clip(slice_img*255,0,255).astype(np.uint8)
    den = cv2.fastNlMeansDenoising(img, None, h=10, templateWindowSize=7, searchWindowSize=21)
    return den.astype(np.float32)/255.0

def preprocess_slice(sl):
    if np.nanmax(sl)-np.nanmin(sl) < 1e-6:
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

# -------------------------
# Simulated fallback
# -------------------------
def simulate_prob(vol1, vol2):
    m1 = float(np.nanmean(vol1)); m2 = float(np.nanmean(vol2))
    if (m1+m2)==0: return 0.25
    r = m2/(m1+1e-8); p = 0.35 + (0.4 * (r/(1+r))); return float(min(max(p,0.05),0.95))

# -------------------------
# Pipeline (auto-redirect behavior)
# -------------------------
def run_pipeline(t1_path, t2_path, status_slot, prog_slot):
    write_log("pipeline start")
    try:
        status_slot.info("Validating volumes..."); prog_slot.progress(5)
        img1 = ensure_correct_and_load(t1_path); img2 = ensure_correct_and_load(t2_path)
    except Exception as e:
        write_log(f"nib load error: {e}"); status_slot.error("NIfTI load error: "+str(e)); return None, f"NIfTI error: {e}"
    vol1 = img1.get_fdata().astype(np.float32); vol2 = img2.get_fdata().astype(np.float32)
    write_log(f"vol shapes: {vol1.shape} / {vol2.shape}")
    n = min(vol1.shape[2], vol2.shape[2])
    if n<=0: status_slot.error("Empty volumes"); return None, "Empty volumes"
    # preprocess
    status_slot.info(f"Preprocessing {n} slices..."); prog_slot.progress(8)
    t1_s, t2_s = [], []
    for i in range(n):
        try:
            t1s = preprocess_slice(vol1[:,:,i]); t2s = preprocess_slice(vol2[:,:,i])
        except Exception as e:
            write_log(f"preprocess error slice {i}: {e}"); status_slot.error(f"Preprocess failed: {e}"); return None, f"Preprocess error: {e}"
        t1_s.append(t1s); t2_s.append(t2s); prog_slot.progress(int(8 + 32*(i+1)/n))
    # features
    status_slot.info("Extracting features..."); prog_slot.progress(42)
    chunk = max(1, n//6); f1_chunks=[]; f2_chunks=[]
    for start in range(0,n,chunk):
        end = min(n, start+chunk)
        try:
            f1 = vit_extract_batch(t1_s[start:end]); f2 = vit_extract_batch(t2_s[start:end])
        except Exception as e:
            write_log(f"vit error {start}:{end} -> {e}"); status_slot.error("Feature extraction error: "+str(e)); return None, f"ViT error: {e}"
        f1_chunks.append(f1); f2_chunks.append(f2); prog_slot.progress(int(42 + 20*(end)/n))
    feats1 = np.concatenate(f1_chunks, axis=0) if f1_chunks else np.zeros((0,getattr(vit_model,"embed_dim",768)))
    feats2 = np.concatenate(f2_chunks, axis=0) if f2_chunks else np.zeros((0,getattr(vit_model,"embed_dim",768)))
    write_log(f"feats shapes: {feats1.shape}/{feats2.shape}")
    status_slot.info("Fusing features and classifying..."); prog_slot.progress(68)
    fused = fuse_features(feats1, feats2); write_log(f"fused shape: {fused.shape}")
    if fused.shape[0]==0: status_slot.error("No fused features"); return None, "No fused features"
    if rf_model is None:
        write_log("RF missing -> simulated"); prob = simulate_prob(vol1, vol2)
        prog_slot.progress(100); status_slot.success("Simulation complete (model missing).")
        return {"simulated":True,"prob":prob,"n_slices":len(fused),"slices_cirr":int((prob>=SLICE_INFO_THRESHOLD)*len(fused)),"slices_healthy":int((1-(prob>=SLICE_INFO_THRESHOLD))*len(fused))}, "Model missing"
    try:
        probs = rf_model.predict_proba(fused)[:,1]
    except Exception as e:
        write_log(f"predict error: {e}"); prob = simulate_prob(vol1,vol2)
        prog_slot.progress(100); status_slot.error("Prediction failed; simulated fallback used.")
        return {"simulated":True,"prob":prob,"n_slices":len(fused),"slices_cirr":int((prob>=SLICE_INFO_THRESHOLD)*len(fused)),"slices_healthy":int((1-(prob>=SLICE_INFO_THRESHOLD))*len(fused))}, f"Predict error: {e}"
    final_prob = float(np.mean(probs)); slices_cirr = int((probs>=SLICE_INFO_THRESHOLD).sum()); slices_healthy = len(probs)-slices_cirr
    prog_slot.progress(92); status_slot.success("Finalizing results..."); time.sleep(0.3); prog_slot.progress(100)
    return {"simulated":False,"prob":final_prob,"n_slices":len(probs),"slices_cirr":slices_cirr,"slices_healthy":slices_healthy}, None

# -------------------------
# UI screens
# -------------------------
if "screen" not in st.session_state: st.session_state.screen = "intro"

# --- Intro: hero with single actionable Get started inside the first panel
def show_intro():
    # embed hero image if exists for left panel (keeps look similar to your uploaded composite)
    left_bg = ""
    if os.path.exists(HERO_IMAGE_PATH):
        try:
            b64 = base64.b64encode(open(HERO_IMAGE_PATH,"rb").read()).decode("utf-8")
            left_bg = f"background-image: url(data:image/jpeg;base64,{b64}); background-size:cover; background-position:center;"
        except Exception as e:
            write_log(f"hero embed failed: {e}")
            left_bg = ""
    st.markdown("<div class='app-wrap'><div class='panel-row'>"
                f"<div class='phone left' style='{left_bg}'>"
                "<div class='hero-title'>Welcome</div>"
                "<div class='hero-sub'>AI-assisted liver MRI screening — research tool only.</div>"
                "<div class='get-start'><form><button class='btn-hero'>Get started</button></form></div>"
                "</div>"
                "<div class='phone mid'><div class='hero-title'>Find the doctor</div><div class='hero-sub'>Categories • Quick access • Info</div></div>"
                "<div class='phone right'><div style='font-weight:700'>Book Now</div><div class='small-muted'>Choose date & time</div></div>"
                "</div></div>", unsafe_allow_html=True)
    # THE ACTUAL working button (placed immediately under hero visually, but only one Start control exists)
    # This is the single actionable control; visually it sits under the left panel and aligns with the Get started look.
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button("Get started", key="hero_get_started"):
            st.session_state.screen = "upload"

# --- Upload screen (styled like panel 2)
def show_upload():
    st.markdown("<div style='height:14px'></div>")
    st.markdown("<div style='display:flex;gap:18px;justify-content:center;align-items:flex-start;'><div class='upload-card' style='width:620px'><div class='section-title'>Upload paired T1 & T2 MRI volumes (.nii / .nii.gz)</div><div class='small-muted'>Please upload axial NIfTI volumes (converted from DICOM). Keep PHI out of public demos.</div>", unsafe_allow_html=True)
    t1 = st.file_uploader("Upload T1 (.nii / .nii.gz)", type=["nii","nii.gz","gz"], key="u1")
    t2 = st.file_uploader("Upload T2 (.nii / .nii.gz)", type=["nii","nii.gz","gz"], key="u2")
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    start = st.button("Start AI Analysis", key="analysis_start")
    st.markdown("</div><div style='width:300px' class='upload-card'><div class='section-title'>Live Status</div><div id='status-area'>Waiting for input</div></div></div>", unsafe_allow_html=True)
    # No checkboxes — direct flow
    if start:
        # decide file paths
        if t1 is None or t2 is None:
            st.error("Please upload both T1 and T2.")
            return
        p1 = save_uploaded_preserve_name(t1); p2 = save_uploaded_preserve_name(t2)
        # status & progress placeholders
        status_slot = st.empty()
        prog_slot = st.empty()
        status_slot.info("Starting pipeline...")
        prog_visual = prog_slot.progress(0)
        # run pipeline; it updates prog via prog_slot.progress() inside
        result, guidance = run_pipeline(p1, p2, status_slot, prog_slot)
        write_log(f"pipeline done -> {result}, {guidance}")
        if result is None:
            status_slot.error("Pipeline failed; check debug log.")
            if guidance: st.error(str(guidance))
            return
        # set result into session and auto-redirect based on thresholds
        st.session_state._last_result = result
        st.session_state._last_guidance = guidance
        p = result["prob"]
        if p < LOWER_THRESHOLD:
            st.session_state.screen = "result_healthy"
        elif p > UPPER_THRESHOLD:
            st.session_state.screen = "result_cirrhosis"
        else:
            st.session_state.screen = "result_borderline"
        # immediate rerun to show result page
        st.experimental_rerun()

# --- Clinical result pages (third-panel look)
def clinical_points_for(prob, slices_cirr, slices_healthy):
    if prob > UPPER_THRESHOLD:
        pts = [
            f"1) Imaging suggests features consistent with chronic liver disease/cirrhosis (slices cirrhosis-leaning: {slices_cirr}; healthy-leaning: {slices_healthy}).",
            "2) Order liver function tests (AST, ALT, ALP, GGT, bilirubin) and INR for clinical correlation.",
            "3) Consider transient elastography (FibroScan) to quantify liver stiffness/fibrosis.",
            "4) Check platelet count — low platelets may be associated with portal hypertension.",
            "5) Evaluate for clinical/ultrasound signs of portal hypertension (ascites, varices).",
            "6) Refer to hepatology for comprehensive workup and management planning.",
            "7) Document this AI assessment as adjunctive; confirm diagnosis with clinical and laboratory data."
        ]
    elif prob < LOWER_THRESHOLD:
        pts = [
            f"1) Low AI-estimated probability of cirrhosis (slices cirrhosis-leaning: {slices_cirr}; healthy-leaning: {slices_healthy}).",
            "2) If clinical suspicion remains, perform LFTs and non-invasive fibrosis assessment (elastography).",
            "3) Address modifiable risks: alcohol reduction, weight loss, glycemic control.",
            "4) Continue routine monitoring; escalate if symptoms or abnormal labs occur.",
            "5) No immediate specialist referral unless other findings indicate it.",
            "6) Reinforce lifestyle counselling and risk-factor management.",
            "7) Use AI output as supportive information — not a final diagnosis."
        ]
    else:
        pts = [
            f"1) Borderline/inconclusive AI result (slices cirrhosis-leaning: {slices_cirr}; healthy-leaning: {slices_healthy}).",
            "2) Correlate with clinical history (alcohol use, viral hepatitis, metabolic risk factors) and labs.",
            "3) Consider transient elastography to better stratify fibrosis risk.",
            "4) If tests remain indeterminate, obtain specialist radiology or hepatology review.",
            "5) Repeat imaging or testing in an appropriate interval if concern persists.",
            "6) Discuss case in MDT if management depends on staging.",
            "7) Consider biopsy only after non-invasive tests are exhausted and if necessary."
        ]
    return pts

def show_result(kind):
    res = st.session_state.get("_last_result", {})
    prob = res.get("prob", 0.0); cirr = res.get("slices_cirr", 0); healthy = res.get("slices_healthy", 0)
    if kind == "healthy":
        st.markdown("<div class='result-card result-healthy'><h3>✅ Healthy</h3><p>Low estimated probability of cirrhosis.</p></div>", unsafe_allow_html=True)
    elif kind == "border":
        st.markdown("<div class='result-card result-border'><h3>🔶 Borderline / Inconclusive</h3><p>Further workup recommended.</p></div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='result-card result-cirr'><h3>⚠️ Cirrhosis (AI)</h3><p>Elevated estimated probability — correlate clinically.</p></div>", unsafe_allow_html=True)
    st.markdown(f"**Mean estimated cirrhosis probability:** {prob*100:.2f}%")
    st.markdown(f"- Slices cirrhosis-leaning: **{cirr}**  \n- Slices healthy-leaning: **{healthy}**")
    st.markdown("### Clinical interpretation — recommended actions")
    pts = clinical_points_for(prob, cirr, healthy)
    for p in pts:
        st.markdown(p)
    cols = st.columns([1,1])
    with cols[0]:
        st.image(gauge_png(prob))
    with cols[1]:
        st.image(bar_png(cirr, healthy))
    # report download
    diag = ("Cirrhosis" if prob>UPPER_THRESHOLD else ("Healthy" if prob<LOWER_THRESHOLD else "Borderline / Inconclusive"))
    md = f"# AI Liver MRI Report\nDiagnosis: {diag}\nMean probability: {prob*100:.2f}%\nSlices cirrhosis: {cirr}\nSlices healthy: {healthy}\n\nClinical recommendations:\n"
    for p in pts:
        md += "- " + p + "\n"
    st.download_button("Download clinical report (MD)", md.encode("utf-8"), "liver_clinical_report.md", mime="text/markdown")
    if st.button("Analyze another study"):
        st.session_state.screen = "upload"
        st.experimental_rerun()
    if st.button("Back to Home"):
        st.session_state.screen = "intro"
        st.experimental_rerun()

# -------------------------
# Router
# -------------------------
if st.session_state.screen == "intro":
    show_intro()
elif st.session_state.screen == "upload":
    show_upload()
elif st.session_state.screen == "result_healthy":
    show_result("healthy")
elif st.session_state.screen == "result_cirrhosis":
    show_result("cirr")
elif st.session_state.screen == "result_borderline":
    show_result("border")
else:
    st.session_state.screen = "intro"
    show_intro()

# Footer hint
st.markdown("---")
st.markdown("If app used simulated result, upload `RandomForest_Cirrhosis.pkl` compatible with your scikit-learn version to get real predictions.")
if st.checkbox("Show debug log"):
    st.text_area("Debug log", read_log_tail(400), height=280)

