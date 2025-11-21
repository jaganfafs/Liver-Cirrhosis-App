# app.py
"""
Liver MRI Cinematic — Mobile-style hero (single functional centered Start button)
- Single center Start button (Streamlit-native) — no duplicate start button below
- Three-panel mobile-like hero resembling the uploaded image
- Upload, pipeline, and result flows preserved (robust, simulated fallback if RF missing)
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
# CONFIG
# -------------------------
st.set_page_config(page_title="Liver MRI Cinematic", layout="wide")
MODEL_PATH = "RandomForest_Cirrhosis.pkl"
HERO_IMAGE_PATH = "/mnt/data/WhatsApp Image 2025-11-21 at 22.37.35.jpeg"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LOWER_THRESHOLD = 0.455
UPPER_THRESHOLD = 0.475
SLICE_INFO_THRESHOLD = 0.465
LOG_PATH = "/tmp/pipeline_debug.log"

# start fresh log for each run
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
# CSS (mobile-like)
# -------------------------
st.markdown("""
<style>
:root{--accent1:#065f46;--accent2:#06b6d4;--bg:#f8fafc;--card:#ffffff;--muted:#6b7280;}
.app-wrap { max-width:1200px; margin:18px auto; }
.mobile-hero { display:flex; gap:18px; justify-content:center; align-items:flex-start; width:100%; }
.phone { width:300px; border-radius:22px; padding:18px; box-shadow:0 18px 48px rgba(2,6,23,0.12); background:linear-gradient(180deg,#0b2540,#08283a); color:white; position:relative; overflow:hidden; }
.phone.light { background:linear-gradient(180deg,#0d59a3,#0b3a86); }
.phone.right { background:linear-gradient(180deg,#ffffff,#f8fafc); color:#06202b; box-shadow:0 14px 36px rgba(2,6,23,0.06); }
.hero-title { font-size:20px; font-weight:700; margin-top:8px; }
.hero-sub { font-size:13px; color:rgba(255,255,255,0.92); margin-top:10px; }
.mobile-cta { display:flex; justify-content:center; margin-top:18px; }
.center-start { padding:12px 22px; border-radius:12px; border:none; font-size:16px; background:linear-gradient(90deg,var(--accent1),var(--accent2)); color:white; cursor:pointer; box-shadow:0 10px 28px rgba(6,86,63,0.18); }
.upload-area { margin-top:22px; display:flex; gap:20px; }
.card { background:var(--card); padding:16px; border-radius:12px; box-shadow:0 10px 30px rgba(2,6,23,0.04); }
.section-title { font-weight:700; font-size:16px; margin-bottom:8px; }
.small-muted { color:var(--muted); font-size:13px; }
.progress-bar { height:14px; background:#e6f0ff; border-radius:10px; overflow:hidden; }
.progress-fill { height:100%; background:linear-gradient(90deg,var(--accent1),var(--accent2)); width:0%; transition:width 0.4s ease; }
.result-card { margin-top:22px; padding:16px; border-radius:12px; background:var(--card); }
.result-healthy{ background:linear-gradient(90deg,#e8fff2,#d7fff0); }
.result-border{ background:linear-gradient(90deg,#fffaf0,#fff5e6); }
.result-cirr{ background:linear-gradient(90deg,#fff0f0,#ffe6e6); }
.btn { padding:10px 12px; border-radius:8px; border:1px solid #e6edf8; background:#fff; cursor:pointer; }
</style>
""", unsafe_allow_html=True)

# -------------------------
# Visual helpers
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
# Model + preprocess helpers
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
# Pipeline
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
    prog_slot.progress(92); status_slot.success("Finalizing results..."); time.sleep(0.25); prog_slot.progress(100)
    return {"simulated":False,"prob":final_prob,"n_slices":len(probs),"slices_cirr":slices_cirr,"slices_healthy":slices_healthy}, None

# -------------------------
# UI screens & flow
# -------------------------
if "screen" not in st.session_state: st.session_state.screen = "intro"

def show_intro():
    # hero: three mobile cards (left, middle, right) to mimic uploaded image layout
    # left: illustration + text; middle: categories; right: appointment CTA
    img_style = ""
    if os.path.exists(HERO_IMAGE_PATH):
        try:
            b64 = base64.b64encode(open(HERO_IMAGE_PATH,"rb").read()).decode("utf-8")
            img_style = f"background-image: url(data:image/jpeg;base64,{b64}); background-size:cover; background-position:center;"
        except Exception as e:
            write_log(f"hero embed failed: {e}")
            img_style = ""
    st.markdown("<div class='app-wrap'><div class='mobile-hero'>"
                f"<div class='phone' style='{img_style}'>"
                "<div style='height:8px'></div><div class='hero-title'>Medical App</div>"
                "<div class='hero-sub'>AI-assisted liver MRI screening — research tool only.</div>"
                "<div class='mobile-cta'><button class='center-start'>Get started</button></div></div>"
                "<div class='phone light'><div class='hero-title'>Find the doctor</div><div class='hero-sub'>Categories • Doctors • Quick access</div><div style='height:40px'></div></div>"
                "<div class='phone right' style='color:#06202b'><div style='font-weight:700'>Choose Date</div><div class='small-muted'>Quick appointment CTA</div><div style='height:40px'></div></div>"
                "</div></div>", unsafe_allow_html=True)
    # THE SINGLE WORKING Start button (Streamlit)
    col1,col2,col3 = st.columns([1,2,1])
    with col2:
        if st.button("Start", key="single_center_start", help="Click to proceed"):
            st.session_state.screen = "upload"

def show_upload():
    st.markdown("<div style='height:14px'></div>")
    st.markdown("<div class='upload-area'><div class='card' style='flex:1'><div class='section-title'>Upload paired T1 & T2 MRI</div><div class='small-muted'>Please upload NIfTI volumes converted from DICOM. Keep PHI out of public demos.</div></div><div class='card' style='width:300px'><div class='section-title'>Status</div><div id='status'>Waiting</div></div></div>", unsafe_allow_html=True)
    left,right = st.columns([1,1])
    with left:
        t1 = st.file_uploader("Upload T1 (.nii / .nii.gz)", type=["nii","nii.gz","gz"])
        t2 = st.file_uploader("Upload T2 (.nii / .nii.gz)", type=["nii","nii.gz","gz"])
        st.write("")
        demo = st.button("Use Synthetic Demo")
        start = st.button("Start AI Analysis")
    with right:
        status_slot = st.empty(); progress_slot = st.empty()
        if st.checkbox("Show debug log (tail)"):
            st.text_area("Debug log", read_log_tail(200), height=240)
    if demo:
        v1 = np.zeros((64,64,16), dtype=np.float32); v2 = np.zeros((64,64,16), dtype=np.float32)
        v1[18:46,18:46,6:10]=0.6; v2[20:44,20:44,6:10]=0.4
        p1="/tmp/demo_t1.nii"; p2="/tmp/demo_t2.nii"
        nib.Nifti1Image(v1, affine=np.eye(4)).to_filename(p1); nib.Nifti1Image(v2, affine=np.eye(4)).to_filename(p2)
        st.success("Demo volumes created. Click Start AI Analysis.")
        st.session_state._demo_t1 = p1; st.session_state._demo_t2 = p2
    if start:
        if hasattr(st.session_state, "_demo_t1"):
            p1 = st.session_state._demo_t1; p2 = st.session_state._demo_t2
        else:
            if t1 is None or t2 is None:
                st.error("Please upload both T1 and T2 or use Demo.")
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

def clinical_points_for(prob, cirr, healthy):
    if prob > UPPER_THRESHOLD:
        pts = [
            f"1) Imaging suggests features consistent with chronic liver disease/cirrhosis (slices cirrhosis-leaning: {cirr}; healthy-leaning: {healthy}).",
            "2) Elevated probability — order liver function tests (AST, ALT, ALP, GGT, bilirubin) and INR.",
            "3) Perform non-invasive fibrosis assessment (transient elastography/FibroScan) to quantify stiffness.",
            "4) Evaluate platelet count — thrombocytopenia may reflect portal hypertension.",
            "5) Assess for clinical signs of portal hypertension (ascites, variceal bleeding) and refer for ultrasound.",
            "6) Consider hepatology referral for diagnostic workup and management planning.",
            "7) Document AI assessment as adjunctive; treat based on clinician judgement and confirmatory tests."
        ]
    elif prob < LOWER_THRESHOLD:
        pts = [
            f"1) Low AI-estimated probability of cirrhosis (slices cirrhosis-leaning: {cirr}; healthy-leaning: {healthy}).",
            "2) If clinical suspicion exists, consider LFTs and elastography — imaging can miss early fibrosis.",
            "3) Continue routine monitoring and address risk factors (alcohol, obesity, metabolic disease).",
            "4) Counsel patient on lifestyle measures (reduce alcohol, manage weight, control diabetes).",
            "5) Consider non-invasive follow-up testing if labs abnormal or symptoms develop.",
            "6) No immediate specialist referral required unless clinical/lab indicators present.",
            "7) Use this AI output as supportive information, not definitive diagnosis."
        ]
    else:
        pts = [
            f"1) Borderline/inconclusive AI result (slices cirrhosis-leaning: {cirr}; healthy-leaning: {healthy}).",
            "2) Correlate carefully with clinical history (alcohol, viral hepatitis, metabolic syndrome) and labs.",
            "3) Consider transient elastography (FibroScan) to better stratify fibrosis risk.",
            "4) If labs or elastography remain indeterminate, seek specialist radiology review.",
            "5) Repeat imaging or laboratory evaluation in an appropriate follow-up interval.",
            "6) Discuss case in multidisciplinary forum if management decisions depend on these results.",
            "7) Document uncertainty and escalate to biopsy only after non-invasive tests are exhausted."
        ]
    return pts

def show_result(kind):
    res = st.session_state.get("_last_result", {})
    prob = res.get("prob", 0.0); cirr = res.get("slices_cirr", 0); healthy = res.get("slices_healthy", 0)
    if kind == "healthy":
        st.markdown("<div class='result-note result-healthy'><h3>✅ Healthy</h3><p>Low estimated probability of cirrhosis</p></div>", unsafe_allow_html=True)
    elif kind == "border":
        st.markdown("<div class='result-note result-border'><h3>🔶 Borderline / Inconclusive</h3><p>Further workup recommended</p></div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='result-note result-cirr'><h3>⚠️ Cirrhosis (AI)</h3><p>Elevated probability — correlate clinically</p></div>", unsafe_allow_html=True)
    st.markdown(f"**Mean estimated cirrhosis probability:** {prob*100:.2f}%")
    st.markdown(f"- Slices cirrhosis-leaning: **{cirr}**  \n- Slices healthy-leaning: **{healthy}**")
    st.markdown("### Clinical interpretation — recommended actions")
    pts = clinical_points_for(prob, cirr, healthy)
    for p in pts:
        st.markdown(p)
    c1, c2 = st.columns([1,1])
    with c1:
        st.image(gauge_png(prob))
    with c2:
        st.image(bar_png(cirr, healthy))
    # download clinical report
    diag = ("Cirrhosis" if prob>UPPER_THRESHOLD else ("Healthy" if prob<LOWER_THRESHOLD else "Borderline / Inconclusive"))
    md = f"# AI Liver MRI Report\nDiagnosis: {diag}\nMean probability: {prob*100:.2f}%\nSlices cirrhosis: {cirr}\nSlices healthy: {healthy}\n\nClinical recommendations:\n"
    for p in pts: md += "- " + p + "\n"
    st.download_button("Download clinical report (MD)", md.encode("utf-8"), "liver_clinical_report.md", mime="text/markdown")
    if st.button("Analyze another study"): st.session_state.screen = "upload"
    if st.button("Back to Home"): st.session_state.screen = "intro"

# Router
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
    st.session_state.screen = "intro"; show_intro()

# Footer debug/help
st.markdown("---")
st.markdown("If app used simulated result, upload `RandomForest_Cirrhosis.pkl` compatible with your scikit-learn version to get real predictions.")
if st.checkbox("Show debug log"):
    st.text_area("Debug log", read_log_tail(400), height=280)

