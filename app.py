# app.py
"""
Liver MRI Cinematic — Mobile-styled UI
- Center Start button (Streamlit-native) inside mobile-like hero using user's uploaded image
- Vibrant upload screen (mobile-card style)
- Result screens with clinical-only descriptions (7 bullet points) for Healthy / Borderline / Cirrhosis
- Robust NIfTI handling and ViT+RF pipeline; simulated fallback if model not available
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
# CONFIG
# -------------------------
st.set_page_config(page_title="Liver MRI Cinematic", layout="wide")
MODEL_PATH = "RandomForest_Cirrhosis.pkl"
# Use the user's uploaded image path (from your session) per request:
HERO_IMAGE_PATH = "/mnt/data/WhatsApp Image 2025-11-21 at 22.37.35.jpeg"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LOWER_THRESHOLD = 0.455
UPPER_THRESHOLD = 0.475
SLICE_INFO_THRESHOLD = 0.465
LOG_PATH = "/tmp/pipeline_debug.log"

# init/clear debug log
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
# CSS (mobile-like look)
# -------------------------
st.markdown("""
<style>
:root{--accent1:#5b21b6;--accent2:#06b6d4;--bg:#f8fafc;--card:#ffffff;--muted:#6b7280;}
.mobile-hero { margin:20px auto; width:90%; max-width:1100px; border-radius:28px; overflow:hidden; display:flex; justify-content:center; align-items:center; height:62vh; box-shadow:0 20px 60px rgba(2,6,23,0.12); background:#071230; color:white; position:relative; background-size:cover; background-position:center;}
.hero-overlay { position:absolute; inset:0; background:linear-gradient(180deg, rgba(3,7,18,0.6), rgba(3,12,30,0.75)); }
.hero-inner { z-index:2; text-align:center; padding:36px; }
.hero-title { font-size:46px; font-weight:800; margin:0; letter-spacing:-0.02em; }
.hero-sub { color:#dbeefd; margin-top:10px; }
.center-start { margin-top:20px; padding:14px 28px; border-radius:14px; border:none; font-size:18px; color:white; background:linear-gradient(90deg,var(--accent1),var(--accent2)); cursor:pointer; box-shadow:0 12px 30px rgba(11,22,70,0.28); }
.upload-area { margin:26px auto; width:90%; max-width:1100px; display:flex; gap:20px; }
.card { background:var(--card); padding:18px; border-radius:14px; box-shadow:0 12px 30px rgba(2,6,23,0.04); flex:1; }
.vibrant { background:linear-gradient(90deg, rgba(91,33,182,0.06), rgba(6,182,212,0.04)); border-radius:12px; padding:10px; }
.section-title { font-weight:700; font-size:18px; margin-bottom:8px; }
.small-muted { color:var(--muted); font-size:14px; }
.progress-bar { height:14px; background:#e6f0ff; border-radius:10px; overflow:hidden; margin-top:10px; }
.progress-fill { height:100%; background:linear-gradient(90deg,var(--accent1),var(--accent2)); width:0%; transition:width 0.4s ease; }
.result-card { margin:20px auto; width:90%; max-width:1100px; padding:18px; border-radius:14px; background:var(--card); box-shadow:0 12px 30px rgba(2,6,23,0.04); }
.result-note { padding:14px; border-radius:10px; margin-bottom:12px; }
.result-healthy{ background:linear-gradient(90deg,#e8fff2,#d7fff0); }
.result-border{ background:linear-gradient(90deg,#fffaf0,#fff5e6); }
.result-cirr{ background:linear-gradient(90deg,#fff0f0,#ffe6e6); }
.btn { padding:10px 14px; border-radius:10px; border:1px solid #e6edf8; background:#fff; cursor:pointer; }
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
# Model & feature helpers
# -------------------------
@st.cache_resource(show_spinner=False)
def load_vit():
    model = timm.create_model("vit_base_patch16_224", pretrained=True)
    if hasattr(model,"head"): model.head = torch.nn.Identity()
    elif hasattr(model,"fc"): model.fc = torch.nn.Identity()
    model.to(DEVICE); model.eval(); return model
vit_model = load_vit()
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
    with open(path,"rb") as fh: head = fh.read(2)
    is_gz = head == b'\x1f\x8b'
    if is_gz and not path.endswith(".gz"):
        newp = path + ".gz"; os.rename(path, newp); path = newp; write_log(f"renamed to {newp}")
    if (not is_gz) and path.endswith(".gz"):
        try:
            outp = path[:-3]
            with gzip.open(path,"rb") as g, open(outp,"wb") as o: shutil.copyfileobj(g,o)
            path = outp; write_log(f"decompressed to {outp}")
        except Exception as e:
            write_log(f"decompress failed: {e}")
    img = nib.load(path); return img

def nlm_denoise(sl):
    img = np.clip(sl*255,0,255).astype(np.uint8)
    den = cv2.fastNlMeansDenoising(img, None, h=10, templateWindowSize=7, searchWindowSize=21)
    return den.astype(np.float32)/255.0

def preprocess_slice(sl):
    if np.nanmax(sl)-np.nanmin(sl) < 1e-6: sln = np.zeros_like(sl, dtype=np.float32)
    else:
        sl = np.nan_to_num(sl); sln = (sl - sl.min())/(sl.max()-sl.min()+1e-8)
    sln = nlm_denoise(sln); sln = resize(sln, (224,224), preserve_range=True).astype(np.float32); return sln

def vit_extract_batch(slices):
    if len(slices)==0: return np.zeros((0, getattr(vit_model,"embed_dim",768)), dtype=np.float32)
    batch = []
    for s in slices:
        img = np.clip(s*255,0,255).astype(np.uint8); rgb = np.stack([img]*3, axis=-1); pil = Image.fromarray(rgb)
        batch.append(transform_3ch(pil))
    xb = torch.stack(batch).to(DEVICE)
    with torch.no_grad(): feats = vit_model(xb)
    return feats.cpu().numpy()

def fuse_features(t1f, t2f):
    L = min(len(t1f), len(t2f))
    if L==0:
        d1 = t1f.shape[1] if len(t1f)>0 else 0; d2 = t2f.shape[1] if len(t2f)>0 else 0
        return np.zeros((0, d1+d2), dtype=np.float32)
    return np.concatenate([t1f[:L], t2f[:L]], axis=1)

# -------------------------
# Simulated fallback (if RF missing)
# -------------------------
def simulate_prob(vol1, vol2):
    m1 = float(np.nanmean(vol1)); m2 = float(np.nanmean(vol2))
    if (m1+m2)==0: return 0.25
    r = m2/(m1+1e-8); p = 0.35 + (0.4 * (r/(1+r))); return float(min(max(p,0.05),0.95))

# -------------------------
# Pipeline (robust)
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
        write_log("RF missing -> simulated")
        prob = simulate_prob(vol1, vol2)
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

def show_intro():
    # embed hero image if exists, otherwise gradient
    bg_style = ""
    if os.path.exists(HERO_IMAGE_PATH):
        try:
            b64 = base64.b64encode(open(HERO_IMAGE_PATH,"rb").read()).decode("utf-8")
            bg_style = f"background-image: url(data:image/jpeg;base64,{b64});"
        except Exception as e:
            write_log(f"hero embed failed: {e}")
            bg_style = ""
    st.markdown(f"<div class='mobile-hero' style='{bg_style}'><div class='hero-overlay'></div><div class='hero-inner'><div class='hero-title'>Liver MRI Cinematic</div><div class='hero-sub' style='max-width:760px;margin:auto;'>AI-assisted research support for liver cirrhosis screening. This tool assists clinicians — not a diagnosis.</div><div style='margin-top:22px;'><button class='center-start' onclick=''>{'▶ Start'}</button></div></div></div>", unsafe_allow_html=True)
    # Centered working Streamlit button
    col1,col2,col3 = st.columns([1,2,1])
    with col2:
        if st.button("▶ Start", key="start_center"):
            st.session_state.screen = "upload"

def show_upload():
    st.markdown("<div style='height:12px'></div>")
    st.markdown("<div class='upload-area'><div class='card vibrant'><div class='section-title'>Uploads</div><div class='small-muted'>Please upload paired axial T1 and T2 NIfTI volumes (.nii or .nii.gz). Keep PHI out of public demos.</div></div><div class='card'><div class='section-title'>Status</div><div id='status'>Waiting for uploads</div></div></div>", unsafe_allow_html=True)
    left,right = st.columns([1,1])
    with left:
        t1 = st.file_uploader("Upload T1 (.nii / .nii.gz)", type=["nii","nii.gz","gz"])
        t2 = st.file_uploader("Upload T2 (.nii / .nii.gz)", type=["nii","nii.gz","gz"])
        st.write("") 
        demo = st.button("Use Synthetic Demo (fast)")
        start = st.button("Start AI Analysis")
    with right:
        status_slot = st.empty()
        progress_slot = st.empty()
        if st.checkbox("Show debug log (tail)"):
            st.text_area("Debug log", read_log_tail(200), height=240)
    if demo:
        v1 = np.zeros((64,64,16), dtype=np.float32); v2 = np.zeros((64,64,16), dtype=np.float32)
        v1[18:46,18:46,6:10]=0.6; v2[20:44,20:44,6:10]=0.4
        p1="/tmp/demo_t1.nii"; p2="/tmp/demo_t2.nii"
        nib.Nifti1Image(v1, affine=np.eye(4)).to_filename(p1); nib.Nifti1Image(v2, affine=np.eye(4)).to_filename(p2)
        st.success("Synthetic demo created. Click Start AI Analysis.")
        st.session_state._demo_t1 = p1; st.session_state._demo_t2 = p2
    if start:
        if hasattr(st.session_state, "_demo_t1"):
            p1 = st.session_state._demo_t1; p2 = st.session_state._demo_t2
        else:
            if t1 is None or t2 is None:
                st.error("Please upload both T1 and T2 or use demo.")
                return
            p1 = save_uploaded_preserve_name(t1); p2 = save_uploaded_preserve_name(t2)
        status_slot.info("Starting pipeline..."); progress_slot.progress(2)
        result, guide = run_pipeline(p1, p2, status_slot, progress_slot)
        write_log(f"pipeline returned: {result}, {guide}")
        if result is None:
            status_slot.error("Pipeline failed. See debug log.")
            if guide: st.error(str(guide))
            return
        st.session_state._last_result = result; st.session_state._last_guidance = guide
        prob = result["prob"]
        if prob < LOWER_THRESHOLD: st.session_state.screen = "result_healthy"
        elif prob > UPPER_THRESHOLD: st.session_state.screen = "result_cirrhosis"
        else: st.session_state.screen = "result_borderline"

# Build 7-point clinical descriptions (no tech summary)
def clinical_points_for(prob, slices_cirr, slices_healthy):
    # return list of 7 items tailored by class
    if prob > UPPER_THRESHOLD:
        pts = [
            "1) Imaging suggests features consistent with chronic liver disease and cirrhosis — correlate with clinical history.",
            "2) Elevated AI-estimated probability; consider ordering liver function tests (AST, ALT, ALP, GGT, bilirubin) and INR.",
            "3) Recommend non-invasive fibrosis assessment (transient elastography / FibroScan) to quantify fibrosis burden.",
            "4) Check platelet count and complete blood count — thrombocytopenia may accompany portal hypertension.",
            "5) Evaluate for signs of portal hypertension (ascites, varices) clinically and with ultrasound if indicated.",
            "6) Refer to hepatology for further workup and discussion of management, including potential antiviral/antifibrotic therapy where appropriate.",
            "7) Document AI assessment as an adjunct; final diagnosis and management decisions must be made by treating clinicians."
        ]
    elif prob < LOWER_THRESHOLD:
        pts = [
            "1) Low AI-estimated probability of cirrhosis based on MRI appearance.",
            "2) If clinical suspicion persists, consider liver function tests and elastography — early fibrosis can be missed on imaging.",
            "3) Continue routine clinical follow-up and monitoring of LFTs if risk factors (alcohol, viral hepatitis, metabolic syndrome) are present.",
            "4) Counsel on modifiable risk factors: alcohol moderation, weight management, control of diabetes and dyslipidemia.",
            "5) If abnormal labs or symptoms arise, escalate to non-invasive fibrosis testing and specialist referral.",
            "6) Maintain surveillance for hepatocellular carcinoma only per guideline indications if chronic liver disease is known.",
            "7) Use this AI output as supportive information — not definitive diagnosis."
        ]
    else:
        pts = [
            "1) AI output is borderline/inconclusive — imaging features do not strongly favor either cirrhosis or normal liver.",
            "2) Recommend targeted clinical correlation: history (hepatitis risk, alcohol use), symptoms, and lab tests (LFTs).",
            "3) Consider transient elastography to measure liver stiffness and better stratify fibrosis risk.",
            "4) If labs or elastography are indeterminate, consider contrast-enhanced imaging or expert radiology review.",
            "5) Monitor closely; repeat imaging or testing in a follow-up interval if clinical concern persists.",
            "6) Discuss case with hepatology/radiology for consensus opinion if management decisions depend on result.",
            "7) Document uncertainty and discuss need for possible biopsy only when non-invasive workup remains inconclusive."
        ]
    # adjust first line to include slice counts briefly
    pts[0] = pts[0] + f" (slices cirrhosis-leaning: {slices_cirr}; healthy-leaning: {slices_healthy})"
    return pts

def show_result_screen(kind):
    res = st.session_state.get("_last_result", {})
    prob = res.get("prob", 0.0); cirr = res.get("slices_cirr", 0); healthy = res.get("slices_healthy", 0)
    if kind == "healthy":
        st.markdown("<div class='result-note result-healthy'><h3>✅ Healthy</h3><p>Low estimated probability of cirrhosis.</p></div>", unsafe_allow_html=True)
    elif kind == "border":
        st.markdown("<div class='result-note result-border'><h3>🔶 Borderline / Inconclusive</h3><p>Result lies in a borderline zone; further workup recommended.</p></div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='result-note result-cirr'><h3>⚠️ Cirrhosis (AI)</h3><p>Elevated estimated probability of cirrhosis. Correlate clinically.</p></div>", unsafe_allow_html=True)
    st.markdown(f"**Mean estimated cirrhosis probability:** {prob*100:.2f}%")
    st.markdown(f"- Slices cirrhosis-leaning: **{cirr}**  \n- Slices healthy-leaning: **{healthy}**")
    st.markdown("### Clinical interpretation — recommended actions")
    pts = clinical_points_for(prob, cirr, healthy)
    for p in pts:
        st.markdown(p)
    # visuals and report download
    cols = st.columns([1,1])
    with cols[0]:
        st.image(gauge_png(prob))
    with cols[1]:
        st.image(bar_png(cirr, healthy))
    # build report text (medical-only)
    diagnosis = ("Cirrhosis" if prob>UPPER_THRESHOLD else ("Healthy" if prob<LOWER_THRESHOLD else "Borderline / Inconclusive"))
    report = f"# AI Liver MRI Report\n\nDiagnosis: {diagnosis}\nMean probability: {prob*100:.2f}%\nSlices analysed: {res.get('n_slices','NA')}\nCirrhosis-leaning: {cirr}\nHealthy-leaning: {healthy}\n\nClinical recommendations:\n"
    for p in pts: report += "- " + p + "\n"
    st.download_button("Download clinical report (MD)", report.encode("utf-8"), "liver_clinical_report.md", mime="text/markdown")
    if st.button("Analyze another study"): st.session_state.screen = "upload"
    if st.button("Back to Home"): st.session_state.screen = "intro"

# -------------------------
# Router
# -------------------------
if st.session_state.screen == "intro": show_intro()
elif st.session_state.screen == "upload": show_upload()
elif st.session_state.screen == "result_healthy": show_result_screen("healthy")
elif st.session_state.screen == "result_cirrhosis": show_result_screen("cirr")
elif st.session_state.screen == "result_borderline": show_result_screen("border")
else: st.session_state.screen = "intro"; show_intro()

# footer dev tips
st.markdown("---")
st.markdown("**Note:** If you see a simulated result it means the RandomForest pickle was unavailable or incompatible. Re-save model in same scikit-learn version and upload `RandomForest_Cirrhosis.pkl` to repo root. Example:\n```\nimport joblib\nm = joblib.load('/path/to/orig.pkl')\njoblib.dump(m,'RandomForest_Cirrhosis.pkl',compress=3)\n```")
if st.checkbox("Show debug log"): st.text_area("Debug log", read_log_tail(400), height=280)

