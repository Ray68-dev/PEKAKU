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
import time
from huggingface_hub import hf_hub_download

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="PEKAKU – Pendeteksi Risiko Kanker Kulit",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- CSS CUSTOM ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,wght@0,700;0,900;1,700&family=Outfit:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
    background-color: #faf7f4 !important;
    color: #1c1c1c;
}

/* Sembunyikan elemen bawaan Streamlit */
header[data-testid="stHeader"], [data-testid="stToolbar"] { display: none !important; }

/* ── POPUP STYLING ── */
.popup-overlay {
    position: fixed; inset: 0;
    background: rgba(10,10,10,0.7);
    backdrop-filter: blur(5px);
    z-index: 999;
    display: flex; align-items: center; justify-content: center;
}

.popup-box {
    background: #fff;
    border-radius: 22px;
    padding: 2rem;
    max-width: 400px; width: 90%;
    text-align: center;
    box-shadow: 0 20px 50px rgba(0,0,0,0.3);
}

.popup-icon { font-size: 3rem; display: block; margin-bottom: 1rem; }
.popup-title { font-family: 'Fraunces', serif; font-size: 1.4rem; font-weight: 700; margin-bottom: 0.5rem; }
.popup-body { font-size: 0.9rem; color: #555; line-height: 1.6; margin-bottom: 1rem; }

/* ── CARD & UI ── */
.card {
    background:#fff; border-radius:20px;
    padding:2rem 1.6rem; margin:1.2rem 0;
    box-shadow:0 2px 18px rgba(0,0,0,.06);
    border:1px solid rgba(0,0,0,.05);
}

.app-header { text-align:center; padding:2rem 1rem; }
.app-title { font-family:'Fraunces',serif; font-size: 2.5rem; font-weight:900; }

.result-card { border-radius:18px; padding:1.5rem; text-align:center; margin-top: 1rem; }
.result-high { background: #fff5f5; border: 1.5px solid #f5c6c2; }
.result-low  { background: #f0fff8; border: 1.5px solid #b2e4cc; }

.stButton > button {
    border-radius: 50px !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)

# --- INITIALIZE SESSION STATE ---
if "disclaimer_closed" not in st.session_state:
    st.session_state.disclaimer_closed = False
if "popup_start" not in st.session_state:
    st.session_state.popup_start = time.time()
if "show_high_risk_popup" not in st.session_state:
    st.session_state.show_high_risk_popup = False

# --- 1. POPUP DISCLAIMER (WAJIB BACA) ---
if not st.session_state.disclaimer_closed:
    elapsed = time.time() - st.session_state.popup_start
    remaining = max(0, 3 - int(elapsed))
    can_close = elapsed >= 3

    st.markdown(f"""
    <div class="popup-overlay">
        <div class="popup-box">
            <span class="popup-icon">⚕️</span>
            <div class="popup-title">Perhatian Penting</div>
            <div class="popup-body">
                PEKAKU adalah alat bantu skrining awal berbasis AI.<br><br>
                <b>Hasil ini BUKAN diagnosis medis.</b><br>
                Selalu konsultasikan dengan dokter spesialis kulit (dermatologis).
            </div>
    """, unsafe_allow_html=True)
    
    if can_close:
        if st.button("✓ Saya Mengerti", key="btn_disclaimer", type="primary", use_container_width=True):
            st.session_state.disclaimer_closed = True
            st.rerun()
    else:
        st.button(f"Tunggu ({remaining}s)", disabled=True, use_container_width=True)
        time.sleep(1)
        st.rerun()
        
    st.markdown("</div></div>", unsafe_allow_html=True)
    st.stop()

# --- 2. LOAD MODEL & FUNGSI ANALISIS ---
@st.cache_resource(show_spinner=False)
def load_model_hf():
    path = hf_hub_download(repo_id="Ray-68/PEKAKU_01", filename="PEKAKU_0.1_model.keras")
    return tf.keras.models.load_model(path, compile=False)

def preprocess_image(image_pil):
    img_rgb = np.array(image_pil.convert("RGB").resize((224, 224)))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    arr = preprocess_input(img_rgb.astype(np.float32).copy())
    arr = np.expand_dims(arr, axis=0)
    return arr, img_bgr

def make_gradcam_heatmap(img_array, model, last_conv="top_conv"):
    try:
        base = model.get_layer("efficientnetb0")
        grad_model = tf.keras.Model(inputs=base.input, outputs=[base.get_layer(last_conv).output, base.output])
        with tf.GradientTape() as tape:
            conv_out, _ = grad_model(img_array, training=False)
            tape.watch(conv_out)
            # Logika loss sederhana untuk visualisasi
            loss = model(img_array, training=False)[:, 0]
        grads = tape.gradient(loss, conv_out)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = conv_out[0] @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy()
    except: return None

# --- 3. UI UTAMA ---
st.markdown("""
<div class="app-header">
    <div class="app-title">🩺 PEKAKU</div>
    <div style="color:#888;">Pendeteksi Risiko Kanker Kulit</div>
</div>
""", unsafe_allow_html=True)

with st.spinner("Menghubungkan ke server AI..."):
    try:
        model = load_model_hf()
        model_ok = True
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        model_ok = False

if model_ok:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload Foto Gejala Kulit", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    
    if uploaded:
        image = Image.open(uploaded)
        st.image(image, use_container_width=True, caption=uploaded.name)
        run_btn = st.button("🔍 Periksa Sekarang", type="primary", use_container_width=True)
    else:
        st.info("Silakan pilih foto kulit yang ingin diperiksa.")
        run_btn = False
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded and run_btn:
        with st.spinner("AI sedang menganalisis..."):
            arr, img_bgr = preprocess_image(image)
            pred = model.predict(arr, verbose=0)[0][0]
            THRESHOLD = 0.31
            is_high = pred >= THRESHOLD
            conf = (pred if is_high else 1 - pred) * 100

            if is_high:
                st.markdown(f"""
                <div class="result-card result-high">
                    <h2 style="color:#c0392b;">⚠️ Risiko Tinggi</h2>
                    <h1 style="font-size:3rem; margin:0;">{conf:.1f}%</h1>
                    <p>Terdeteksi indikasi risiko. Harap segera ke dokter kulit!</p>
                </div>
                """, unsafe_allow_html=True)
                st.session_state.show_high_risk_popup = True
            else:
                st.markdown(f"""
                <div class="result-card result-low">
                    <h2 style="color:#27ae60;">✅ Risiko Rendah</h2>
                    <h1 style="font-size:3rem; margin:0;">{conf:.1f}%</h1>
                    <p>Tetap waspada dan periksa rutin secara mandiri.</p>
                </div>
                """, unsafe_allow_html=True)

            # Grad-CAM Visualisasi
            heatmap = make_gradcam_heatmap(arr, model)
            if heatmap is not None:
                st.write("---")
                st.subheader("🔥 Fokus Analisis AI")
                h_res = cv2.resize(heatmap, (img_bgr.shape[1], img_bgr.shape[0]))
                h_col = cv2.applyColorMap(np.uint8(255*h_res), cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), 0.6, cv2.cvtColor(h_col, cv2.COLOR_BGR2RGB), 0.4, 0)
                st.image(overlay, use_container_width=True, caption="Area merah menunjukkan bagian yang dicurigai model.")

# --- 4. POPUP RISIKO TINGGI (Hanya Muncul Jika Terdeteksi) ---
if st.session_state.show_high_risk_popup:
    st.markdown("""
    <div class="popup-overlay">
        <div class="popup-box">
            <span class="popup-icon">🚨</span>
            <div class="popup-title" style="color:#c0392b;">Peringatan Risiko</div>
            <div class="popup-body">
                Hasil menunjukkan potensi risiko tinggi.<br><br>
                <b>Jangan menunda pemeriksaan!</b> Temui dokter spesialis kulit untuk diagnosis profesional.
            </div>
    """, unsafe_allow_html=True)
    
    if st.button("✓ Saya Pahami, Akan Segera Konsultasi", key="btn_highrisk", type="primary", use_container_width=True):
        st.session_state.show_high_risk_popup = False
        st.rerun()
        
    st.markdown("</div></div>", unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("""
<div style="text-align:center; padding:2rem; color:#aaa; font-size:0.8rem;">
    PEKAKU v0.1 • Bukan Alat Diagnosis Medis Utama
</div>
""", unsafe_allow_html=True)
