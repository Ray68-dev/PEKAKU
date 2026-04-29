import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
import io
import os
import base64
from huggingface_hub import hf_hub_download
import gspread
from google.oauth2.service_account import Credentials
import datetime

def log_visit():
    try:
        sheet = connect_sheets()
        sheet.append_row([
            str(datetime.datetime.now()),
            "visit"
        ])
    except Exception as e:
        st.error(f"Gagal log: {e}")  # biar app tetap jalan
        
def connect_sheets():
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
    )
    client = gspread.authorize(creds)
    return client.open("Histori_Kunjungan_PEKAKU").sheet1
# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="PEKAKU — Pendeteksi Risiko Kanker Kulit",
    page_icon="pekaku_icon.png",
    layout="centered",
    initial_sidebar_state="collapsed"
)
if st.button("TEST SHEETS"):
    try:
        sheet = connect_sheets()
        sheet.append_row(["TEST", "OK"])
        st.success("Berhasil nulis ke Sheets!")
    except Exception as e:
        st.error(f"Gagal: {e}")

# Auto log visit
if "logged" not in st.session_state:
    try:
        log_visit()
        st.session_state.logged = True
    except Exception as e:
        st.error(f"Log gagal: {e}")
# ─────────────────────────────────────────────
# ICON BASE64
# ─────────────────────────────────────────────
def get_icon_b64():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pekaku_icon.png")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""

ICON_B64 = get_icon_b64()
ICON_TAG = (
    f'<img src="data:image/png;base64,{ICON_B64}" class="hero-logo" alt="Logo PEKAKU">'
    if ICON_B64 else ""
)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500;600&display=swap');
*,*::before,*::after{box-sizing:border-box;}
html,body,[class*="css"],.stApp{font-family:'Inter',sans-serif!important;background-color:#0e0b18!important;color:#ede8ff!important;}
header[data-testid="stHeader"],#MainMenu,footer,[data-testid="stToolbar"],[data-testid="stDecoration"],[data-testid="collapsedControl"]{display:none!important;}
section[data-testid="stSidebar"]{display:none!important;}
.block-container{padding:0!important;max-width:820px!important;}
.hero{background:linear-gradient(160deg,#170e32 0%,#1c1240 55%,#0e0b18 100%);border-bottom:1px solid rgba(168,85,247,0.15);padding:3.5rem 2rem 2.8rem;text-align:center;position:relative;overflow:hidden;}
.hero::before{content:'';position:absolute;top:-100px;left:50%;transform:translateX(-50%);width:520px;height:320px;background:radial-gradient(ellipse,rgba(124,58,237,0.2) 0%,transparent 70%);pointer-events:none;}
.hero::after{content:'';position:absolute;bottom:-60px;right:-40px;width:280px;height:280px;background:radial-gradient(ellipse,rgba(245,158,11,0.08) 0%,transparent 70%);pointer-events:none;}
.hero-logo{width:90px;height:90px;border-radius:50%;border:2px solid rgba(168,85,247,0.35);box-shadow:0 0 36px rgba(124,58,237,0.35),0 0 80px rgba(124,58,237,0.12);display:block;margin:0 auto 1.3rem;}
.hero-title{font-family:'Syne',sans-serif;font-size:clamp(2rem,6vw,3rem);font-weight:800;background:linear-gradient(90deg,#c084fc 0%,#f59e0b 50%,#c084fc 100%);background-size:200%;-webkit-background-clip:text;-webkit-text-fill-color:transparent;letter-spacing:-0.5px;line-height:1.1;}
.hero-sub{font-size:clamp(0.78rem,2.5vw,0.92rem);color:#8b7db0;margin-top:0.5rem;letter-spacing:0.1em;text-transform:uppercase;font-weight:300;}
.section{padding:2rem 2rem 0;}
.section-label{font-family:'Syne',sans-serif;font-size:0.68rem;text-transform:uppercase;letter-spacing:0.16em;color:#6b5b90;margin-bottom:0.75rem;}
.card{background:rgba(255,255,255,0.025);border:1px solid rgba(168,85,247,0.12);border-radius:18px;padding:1.5rem 1.8rem;margin-bottom:1.1rem;line-height:1.75;font-size:0.92rem;color:#c0b4e0;}
.card b{color:#ddd0ff;}
.warn{background:linear-gradient(135deg,rgba(245,158,11,0.09),rgba(180,100,0,0.05));border:1px solid rgba(245,158,11,0.25);border-left:4px solid #f59e0b;border-radius:14px;padding:1.1rem 1.4rem;font-size:0.87rem;color:#d4b870;line-height:1.7;margin-bottom:1rem;}
.warn-title{font-family:'Syne',sans-serif;font-size:0.88rem;font-weight:700;color:#f59e0b;margin-bottom:0.35rem;}
.steps{display:flex;gap:0.8rem;flex-wrap:wrap;margin-bottom:1.5rem;}
.step{flex:1;min-width:130px;background:rgba(124,58,237,0.07);border:1px solid rgba(124,58,237,0.16);border-radius:14px;padding:1rem 0.9rem;text-align:center;}
.step-n{font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:800;background:linear-gradient(90deg,#a855f7,#f59e0b);-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.step-t{font-size:0.78rem;color:#9480c0;margin-top:0.25rem;line-height:1.4;}
.divider{height:1px;background:linear-gradient(90deg,transparent,rgba(168,85,247,0.15),transparent);margin:1.8rem 2rem;}
.upload-title{font-family:'Syne',sans-serif;font-size:1.25rem;font-weight:700;color:#ddd0ff;margin-bottom:1rem;}
[data-testid="stFileUploader"]{background:rgba(124,58,237,0.05)!important;border:2px dashed rgba(124,58,237,0.22)!important;border-radius:16px!important;}
.stButton>button{background:linear-gradient(135deg,#7c3aed,#a855f7)!important;color:#fff!important;font-family:'Syne',sans-serif!important;font-weight:700!important;font-size:0.95rem!important;border:none!important;border-radius:12px!important;padding:0.72rem 2rem!important;width:100%!important;letter-spacing:0.04em!important;box-shadow:0 4px 22px rgba(124,58,237,0.35)!important;transition:all 0.2s!important;}
.stButton>button:hover{transform:translateY(-2px)!important;box-shadow:0 6px 30px rgba(124,58,237,0.5)!important;}
.res-card{border-radius:20px;padding:1.8rem 2rem;margin-bottom:1.2rem;}
.res-high{background:linear-gradient(135deg,rgba(220,38,38,0.11),rgba(160,20,20,0.05));border:1px solid rgba(220,38,38,0.28);border-top:3px solid #ef4444;}
.res-low{background:linear-gradient(135deg,rgba(34,197,94,0.09),rgba(16,140,60,0.05));border:1px solid rgba(34,197,94,0.26);border-top:3px solid #22c55e;}
.res-badge{display:inline-block;font-family:'Syne',sans-serif;font-size:0.68rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;padding:0.22rem 0.7rem;border-radius:100px;margin-bottom:0.75rem;}
.b-high{background:rgba(239,68,68,0.13);color:#f87171;border:1px solid rgba(239,68,68,0.28);}
.b-low{background:rgba(34,197,94,0.11);color:#4ade80;border:1px solid rgba(34,197,94,0.28);}
.res-label{font-family:'Syne',sans-serif;font-size:clamp(1.1rem,4vw,1.5rem);font-weight:800;color:#f0eaff;margin-bottom:0.25rem;}
.res-pct{font-family:'Syne',sans-serif;font-size:clamp(2rem,7vw,3rem);font-weight:800;line-height:1;}
.res-high .res-pct{color:#f87171;}
.res-low .res-pct{color:#4ade80;}
.res-desc{font-size:0.87rem;color:#9880c0;margin-top:0.6rem;line-height:1.65;}
.pills{display:flex;gap:0.65rem;flex-wrap:wrap;margin-top:0.9rem;}
.pill{background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);border-radius:9px;padding:0.4rem 0.9rem;font-size:0.78rem;color:#9880c0;}
.pill b{color:#ddd0ff;font-family:'Syne',sans-serif;}
.gc-title{font-family:'Syne',sans-serif;font-size:0.95rem;font-weight:700;color:#c084fc;margin-bottom:0.7rem;}
.gc-desc{font-size:0.79rem;color:#6b5b90;margin-top:0.45rem;line-height:1.55;}
[data-testid="stImage"] img{border-radius:14px!important;border:1px solid rgba(168,85,247,0.14)!important;}
.footer{text-align:center;padding:2.5rem 1.5rem 3.5rem;color:#3d3260;font-size:0.76rem;border-top:1px solid rgba(168,85,247,0.08);margin-top:2rem;line-height:1.9;}
@media(max-width:620px){.hero{padding:2.5rem 1.2rem 2rem;}.section{padding:1.5rem 1.2rem 0;}.divider{margin:1.5rem 1.2rem;}.steps{gap:0.5rem;}.step{min-width:110px;padding:0.75rem 0.6rem;}.res-card{padding:1.4rem 1.3rem;}.pills{gap:0.45rem;}.pill{font-size:0.72rem;padding:0.35rem 0.7rem;}}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
st.markdown(f'<div class="hero">{ICON_TAG}<div class="hero-title">PEKAKU</div><div class="hero-sub">Pendeteksi Risiko Kanker Kulit</div></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# INTRO - Tentang
# ─────────────────────────────────────────────
st.markdown('<div class="section">', unsafe_allow_html=True)

st.markdown('<div class="section-label">Tentang Aplikasi</div>', unsafe_allow_html=True)

st.markdown('<div class="card"><b>PEKAKU</b> adalah sistem skrining berbasis kecerdasan buatan yang membantu mendeteksi potensi risiko kanker kulit dari gambar lesi atau bercak pada kulit. Sistem ini menggunakan model <b>EfficientNet-B0</b> yang telah dilatih untuk mengenali karakteristik visual lesi berisiko tinggi maupun rendah, dilengkapi visualisasi <b>Grad-CAM</b> untuk menunjukkan area yang menjadi fokus analisis AI.</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# STEPS
# ─────────────────────────────────────────────
st.markdown('<div class="section-label">Cara Menggunakan</div>', unsafe_allow_html=True)

st.markdown("""
<div class="steps">
  <div class="step"><div class="step-n">01</div><div class="step-t">Unggah foto lesi atau bercak kulit</div></div>
  <div class="step"><div class="step-n">02</div><div class="step-t">Klik tombol Analisis Risiko</div></div>
  <div class="step"><div class="step-n">03</div><div class="step-t">Lihat hasil dan visualisasi AI</div></div>
  <div class="step"><div class="step-n">04</div><div class="step-t">Konsultasikan ke dokter kulit</div></div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# WARNING
# ─────────────────────────────────────────────
st.markdown('<div class="warn"><div class="warn-title">&#9888; Perhatian Penting</div>PEKAKU adalah alat bantu skrining awal, <b>bukan alat diagnosis medis</b>. Hasil analisis AI tidak dapat menggantikan pemeriksaan langsung oleh dokter. Akurasi dipengaruhi oleh kualitas foto, sudut pengambilan gambar, dan kondisi pencahayaan.</div>', unsafe_allow_html=True)

st.markdown('<div class="warn"><div class="warn-title">&#128203; Kapan Harus ke Dokter?</div>Jika hasil menunjukkan <b>risiko tinggi</b>, atau kamu menemukan perubahan pada kulit seperti bercak baru, warna tidak merata, tepi tidak beraturan, atau ukuran yang membesar — <b>jangan tunda, segera konsultasikan ke dokter spesialis kulit (dermatologis)</b> untuk evaluasi dan penanganan yang tepat.</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    path = hf_hub_download(repo_id="Ray-68/PEKAKU_01", filename="PEKAKU_0.1_model.keras")
    return tf.keras.models.load_model(path, compile=False)

with st.spinner("Memuat model AI dari Hugging Face..."):
    try:
        model = load_model()
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        st.stop()

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
THRESHOLD = 0.31

def preprocess_image(pil_img):
    img_rgb = np.array(pil_img.convert("RGB").resize((224, 224)))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    arr = preprocess_input(img_rgb.astype(np.float32).copy())
    return np.expand_dims(arr, axis=0), img_bgr

def make_gradcam(img_array, model, conv_layer="top_conv"):
    base = model.get_layer("efficientnetb0")
    gm = tf.keras.Model(
        inputs=base.input,
        outputs=[base.get_layer(conv_layer).output, base.output]
    )
    t = tf.cast(img_array, tf.float32)
    with tf.GradientTape() as tape:
        conv_out, _ = gm(t, training=False)
        tape.watch(conv_out)
        x = conv_out
        found = False
        for layer in base.layers:
            if found:
                x = layer(x, training=False)
            if layer.name == conv_layer:
                found = True
        for name in ["global_average_pooling2d", "batch_normalization", "dense", "dropout", "dense_1"]:
            x = model.get_layer(name)(x, training=False)
        loss = x[:, 0]
    grads = tape.gradient(loss, conv_out)
    if grads is None:
        raise ValueError("Gradien None.")
    pg = tf.reduce_mean(grads, axis=(0, 1, 2))
    hm = conv_out[0] @ pg[..., tf.newaxis]
    hm = tf.squeeze(hm)
    hm = tf.maximum(hm, 0) / (tf.reduce_max(hm) + 1e-8)
    return hm.numpy()

def make_gradcam_figure(img_bgr, heatmap, alpha=0.4):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h = cv2.resize(heatmap, (224, 224))
    hc = cv2.applyColorMap(np.uint8(255 * h), cv2.COLORMAP_JET)
    hc = cv2.cvtColor(hc, cv2.COLOR_BGR2RGB)
    ov = cv2.addWeighted(img_rgb, 1 - alpha, hc, alpha, 0)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    fig.patch.set_facecolor("#110c22")
    for ax, im, title, cmap in zip(
        axes,
        [img_rgb, h, ov],
        ["Gambar Asli", "Peta Panas", "Overlay"],
        [None, "jet", None]
    ):
        ax.imshow(im, cmap=cmap)
        ax.set_title(title, color="#9480c0", fontsize=9.5, pad=6, fontweight="bold")
        ax.axis("off")
    plt.tight_layout(pad=1.0)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="#110c22")
    plt.close(fig)
    buf.seek(0)
    return buf

# ─────────────────────────────────────────────
# UPLOAD
# ─────────────────────────────────────────────
st.markdown('<div class="section">', unsafe_allow_html=True)
st.markdown('<div class="upload-title">&#128228; Unggah Gambar Kulit</div>', unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Format yang didukung: JPG, JPEG, PNG",
    type=["jpg", "jpeg", "png"],
    label_visibility="visible"
)

if uploaded:
    image = Image.open(uploaded)
    st.image(image, use_column_width=True, caption=f"{uploaded.name}")
    st.markdown("<br>", unsafe_allow_html=True)
    do_analyze = st.button("&#128269; Analisis Risiko Sekarang")
else:
    do_analyze = False

st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# RESULT
# ─────────────────────────────────────────────
if uploaded and do_analyze:
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section">', unsafe_allow_html=True)

    with st.spinner("Menganalisis gambar..."):
        img_array, img_bgr = preprocess_image(image)
        pred = model.predict(img_array, verbose=0)[0][0]

    pct      = pred * 100
    is_high  = pred >= THRESHOLD
    card_cls = "res-high" if is_high else "res-low"
    bdg_cls  = "b-high" if is_high else "b-low"
    bdg_txt  = "&#9888; Risiko Tinggi" if is_high else "&#10003; Risiko Rendah"
    label    = "Risiko Tinggi Terdeteksi" if is_high else "Risiko Rendah Terdeteksi"
    desc     = (
        "Model mendeteksi karakteristik visual yang berkaitan dengan lesi kanker kulit. Segera konsultasikan ke dokter spesialis kulit untuk pemeriksaan lebih lanjut."
        if is_high else
        "Tidak ditemukan pola signifikan yang mencirikan kanker kulit pada gambar ini. Tetap lakukan pemeriksaan kulit secara rutin dan jaga kesehatan kulitmu."
    )

    st.markdown(f'<div class="res-card {card_cls}"><span class="res-badge {bdg_cls}">{bdg_txt}</span><div class="res-label">{label}</div><div class="res-pct">{pct:.1f}%</div><div class="res-desc">{desc}</div><div class="pills"><div class="pill">Skor: <b>{pred:.4f}</b></div><div class="pill">Threshold: <b>{THRESHOLD}</b></div><div class="pill">Model: <b>EfficientNet-B0</b></div></div></div>', unsafe_allow_html=True)

    st.markdown('<div class="gc-title">&#128293; Visualisasi Area Perhatian AI (Grad-CAM)</div>', unsafe_allow_html=True)

    with st.spinner("Membuat visualisasi Grad-CAM..."):
        try:
            heatmap = make_gradcam(img_array, model)
            fig_buf = make_gradcam_figure(img_bgr, heatmap)
            st.image(fig_buf, use_column_width=True)
            st.markdown('<div class="gc-desc">&#128308; <b>Area merah/hangat</b> = bagian yang paling mempengaruhi keputusan model. &#128309; <b>Biru/dingin</b> = pengaruh rendah.</div>', unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"Visualisasi Grad-CAM tidak tersedia: {e}")

    st.markdown('<div class="warn" style="margin-top:1.3rem;"><div class="warn-title">&#129658; Langkah Selanjutnya</div>Hasil di atas adalah <b>skrining awal berbasis AI</b> dan bukan diagnosis medis final. Jika kamu menemukan perubahan mencurigakan pada kulit, <b>segera periksakan diri ke dokter spesialis kulit (dermatologis)</b> untuk mendapatkan evaluasi klinis yang akurat.</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
footer_icon = (
    f'<img src="data:image/png;base64,{ICON_B64}" style="width:28px;height:28px;border-radius:50%;opacity:0.45;vertical-align:middle;margin-right:6px;">'
    if ICON_B64 else ""
)
st.markdown(f'<div class="footer">{footer_icon}<b style="color:#5c4a8a;">PEKAKU</b> — Pendeteksi Risiko Kanker Kulit<br>Ditenagai oleh TensorFlow &amp; EfficientNet-B0 &middot; Dibuat dengan Streamlit<br><span>Hasil analisis bukan pengganti diagnosis medis profesional.</span></div>', unsafe_allow_html=True)
