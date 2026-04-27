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

st.set_page_config(
    page_title="PEKAKU – Pendeteksi Risiko Kanker Kulit",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,wght@0,700;0,900;1,700&family=Outfit:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
    background-color: #faf7f4 !important;
    color: #1c1c1c;
}
.stApp { background: #faf7f4 !important; }

/* Hide Streamlit chrome */
header[data-testid="stHeader"],
#MainMenu, footer,
[data-testid="stToolbar"],
section[data-testid="stSidebar"] { display: none !important; }

/* ── POPUP ── */
.popup-overlay {
    position: fixed; inset: 0;
    background: rgba(10,10,10,0.6);
    backdrop-filter: blur(5px);
    z-index: 9999;
    display: flex; align-items: center; justify-content: center;
    animation: fadeIn .25s ease;
}
@keyframes fadeIn { from{opacity:0} to{opacity:1} }

.popup-box {
    background: #fff;
    border-radius: 22px;
    padding: 2.4rem 2rem 2rem;
    max-width: 400px; width: 92%;
    box-shadow: 0 32px 80px rgba(0,0,0,0.2);
    text-align: center;
    animation: slideUp .35s cubic-bezier(.22,.68,0,1.15);
}
@keyframes slideUp {
    from{transform:translateY(36px);opacity:0}
    to{transform:translateY(0);opacity:1}
}
.popup-icon  { font-size:3rem; margin-bottom:.6rem; display:block; }
.popup-title { font-family:'Fraunces',serif; font-size:1.3rem; font-weight:700; color:#1c1c1c; margin-bottom:.7rem; }
.popup-body  { font-size:.88rem; color:#555; line-height:1.7; margin-bottom:1.2rem; }
.popup-timer { font-size:.78rem; color:#bbb; margin-bottom:.8rem; }
.popup-btn {
    display: inline-block;
    background: #c0392b; color: #fff;
    font-family:'Outfit',sans-serif; font-weight:600; font-size:.9rem;
    padding:.65rem 2rem; border-radius:50px; border:none;
    cursor:pointer; transition:background .2s;
}
.popup-btn:hover { background:#a93226; }
.popup-btn-disabled { background:#d5d5d5 !important; color:#aaa !important; cursor:not-allowed !important; }

/* ── HEADER ── */
.app-header { text-align:center; padding:2.5rem 1rem 1.5rem; }
.app-logo {
    display:inline-flex; align-items:center; justify-content:center;
    width:70px; height:70px;
    background:linear-gradient(135deg,#e74c3c,#c0392b);
    border-radius:20px; font-size:2.2rem; margin-bottom:1rem;
    box-shadow:0 10px 28px rgba(192,57,43,.25);
}
.app-title {
    font-family:'Fraunces',serif;
    font-size:clamp(2rem,7vw,2.8rem);
    font-weight:900; color:#1c1c1c; letter-spacing:-.5px;
}
.app-subtitle { font-size:.88rem; color:#999; font-weight:300; letter-spacing:.04em; margin-top:.3rem; }

/* ── CARD ── */
.card {
    background:#fff; border-radius:20px;
    padding:2rem 1.6rem; margin:1.2rem 0;
    box-shadow:0 2px 18px rgba(0,0,0,.06);
    border:1px solid rgba(0,0,0,.05);
}

/* ── UPLOAD PLACEHOLDER ── */
.upload-placeholder {
    text-align:center; padding:2rem 1rem; color:#ccc;
}
.upload-placeholder .ph-icon { font-size:2.8rem; }
.upload-placeholder p { font-size:.85rem; margin-top:.5rem; }

/* ── RESULT ── */
.result-card {
    border-radius:18px; padding:1.8rem 1.4rem;
    text-align:center; margin-bottom:1rem;
}
.result-high { background:linear-gradient(135deg,#fff5f5,#ffe8e6); border:1.5px solid #f5c6c2; }
.result-low  { background:linear-gradient(135deg,#f0fff8,#ddf5eb); border:1.5px solid #b2e4cc; }
.result-emoji { font-size:3rem; display:block; margin-bottom:.5rem; }
.result-label {
    font-family:'Fraunces',serif; font-size:1.45rem; font-weight:700; margin-bottom:.3rem;
}
.result-high .result-label { color:#c0392b; }
.result-low  .result-label { color:#1a7a4a; }
.result-conf {
    font-size:2.6rem; font-weight:700; letter-spacing:-1px; margin:.3rem 0;
}
.result-high .result-conf { color:#e74c3c; }
.result-low  .result-conf { color:#27ae60; }
.result-note { font-size:.82rem; color:#777; line-height:1.6; margin-top:.5rem; }

/* ── GRADCAM ── */
.gradcam-label {
    font-family:'Fraunces',serif; font-weight:700;
    font-size:.95rem; color:#555; margin-bottom:.6rem;
}
.gradcam-note { font-size:.76rem; color:#bbb; margin-top:.5rem; }

/* ── BUTTON ── */
.stButton > button {
    width:100%;
    background:linear-gradient(135deg,#e74c3c,#c0392b);
    color:#fff !important;
    font-family:'Outfit',sans-serif; font-weight:600; font-size:1rem;
    border:none; border-radius:50px; padding:.8rem 2rem;
    box-shadow:0 6px 20px rgba(192,57,43,.28);
    transition:all .2s; margin-top:.4rem;
}
.stButton > button:hover {
    transform:translateY(-2px);
    box-shadow:0 10px 28px rgba(192,57,43,.35);
}

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"] {
    background:#fdf7f5 !important;
    border:2px dashed #e5d8d3 !important;
    border-radius:14px !important;
    transition:border-color .2s;
}
[data-testid="stFileUploader"]:hover { border-color:#e74c3c !important; }

/* ── FOOTER ── */
.app-footer {
    text-align:center; padding:2rem 1rem 3.5rem;
    font-size:.78rem; color:#ccc;
}

/* ── MOBILE ── */
@media(max-width:600px){
    .card        { padding:1.4rem 1.1rem; }
    .result-card { padding:1.4rem 1rem; }
    .popup-box   { padding:1.8rem 1.2rem 1.6rem; }
    .app-header  { padding-top:1.5rem; }
}
</style>
""", unsafe_allow_html=True)

if "disclaimer_closed" not in st.session_state:
    st.session_state.disclaimer_closed = False
if "popup_start" not in st.session_state:
    st.session_state.popup_start = time.time()
if "show_high_risk_popup" not in st.session_state:
    st.session_state.show_high_risk_popup = False

if not st.session_state.disclaimer_closed:
    elapsed   = time.time() - st.session_state.popup_start
    remaining = max(0, 3 - int(elapsed))
    can_close = elapsed >= 3

    timer_html = (
        f'<div class="popup-timer">Dapat ditutup dalam <b>{remaining}</b> detik...</div>'
        if not can_close else ""
    )
    btn_class = "popup-btn" if can_close else "popup-btn popup-btn-disabled"

    st.markdown(f"""
    <div class="popup-overlay">
        <div class="popup-box">
            <span class="popup-icon">⚕️</span>
            <div class="popup-title">Perhatian Penting</div>
            <div class="popup-body">
                PEKAKU adalah alat bantu skrining awal berbasis kecerdasan buatan.<br><br>
                <b>Hasil analisis ini BUKAN diagnosis medis.</b><br>
                Selalu konsultasikan kondisi kulit Anda kepada dokter spesialis kulit
                (dermatologis) untuk diagnosis dan penanganan yang tepat.
            </div>
            {timer_html}
        </div>
    </div>
    """, unsafe_allow_html=True)

    if can_close:
        if st.button("✓ Saya Mengerti", key="btn_disclaimer"):
            st.session_state.disclaimer_closed = True
            st.rerun()
    else:
        time.sleep(1)
        st.rerun()

    st.stop()

if st.session_state.show_high_risk_popup:
    st.markdown("""
    <div class="popup-overlay">
        <div class="popup-box">
            <span class="popup-icon">🚨</span>
            <div class="popup-title">Risiko Tinggi Terdeteksi</div>
            <div class="popup-body">
                Model mendeteksi <b>kemungkinan risiko kanker kulit yang tinggi</b>
                pada gambar ini.<br><br>
                Jangan abaikan hasil ini. Segera jadwalkan pemeriksaan dengan
                dokter spesialis kulit sesegera mungkin.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("✓ Saya Pahami, Akan Segera Konsultasi", key="btn_highrisk"):
        st.session_state.show_high_risk_popup = False
        st.rerun()

    st.stop()

@st.cache_resource(show_spinner=False)
def load_model_hf():
    path = hf_hub_download(
        repo_id="Ray-68/PEKAKU_01",
        filename="PEKAKU_0.1_model.keras"
    )
    return tf.keras.models.load_model(path, compile=False)

def preprocess_image(image_pil):
    img_rgb  = np.array(image_pil.convert("RGB").resize((224, 224)))
    img_bgr  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    arr      = preprocess_input(img_rgb.astype(np.float32).copy())
    arr      = np.expand_dims(arr, axis=0)
    return arr, img_bgr, img_rgb


def make_gradcam_heatmap(img_array, model, last_conv="top_conv"):
    base = model.get_layer("efficientnetb0")
    grad_model = tf.keras.Model(
        inputs=base.input,
        outputs=[base.get_layer(last_conv).output, base.output]
    )
    img_t = tf.cast(img_array, tf.float32)
    with tf.GradientTape() as tape:
        conv_out, _ = grad_model(img_t, training=False)
        tape.watch(conv_out)
        x = conv_out; found = False
        for layer in base.layers:
            if found: x = layer(x, training=False)
            if layer.name == last_conv: found = True
        x = model.get_layer("global_average_pooling2d")(x, training=False)
        x = model.get_layer("batch_normalization")(x, training=False)
        x = model.get_layer("dense")(x, training=False)
        x = model.get_layer("dropout")(x, training=False)
        x = model.get_layer("dense_1")(x, training=False)
        loss = x[:, 0]
    grads = tape.gradient(loss, conv_out)
    if grads is None: return None
    pg       = tf.reduce_mean(grads, axis=(0,1,2))
    heatmap  = conv_out[0] @ pg[..., tf.newaxis]
    heatmap  = tf.squeeze(heatmap)
    heatmap  = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def gradcam_figure(img_bgr, heatmap, alpha=0.4):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h_res   = cv2.resize(heatmap, (img_rgb.shape[1], img_rgb.shape[0]))
    h_col   = cv2.applyColorMap(np.uint8(255*h_res), cv2.COLORMAP_JET)
    h_col   = cv2.cvtColor(h_col, cv2.COLOR_BGR2RGB)
    over    = cv2.addWeighted(img_rgb, 1-alpha, h_col, alpha, 0)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    fig.patch.set_facecolor("#ffffff")
    for ax, img, title, cmap in zip(
        axes,
        [img_rgb, h_res, over],
        ["Gambar Asli", "Heatmap", "Overlay"],
        [None, "jet", None]
    ):
        ax.imshow(img, cmap=cmap); ax.set_title(title, fontsize=9, color="#888", pad=6); ax.axis("off")
    plt.tight_layout(pad=1)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="#fff")
    plt.close(fig); buf.seek(0)
    return buf

st.markdown("""
<div class="app-header">
    <div class="app-logo">🩺</div>
    <div class="app-title">PEKAKU</div>
    <div class="app-subtitle">Pendeteksi Risiko Kanker Kulit</div>
</div>
""", unsafe_allow_html=True)

with st.spinner("Memuat model AI dari Hugging Face..."):
    try:
        model    = load_model_hf()
        model_ok = True
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        st.info("Pastikan nama file `.keras` di repo `Ray-68/PEKAKU_01` sudah benar, lalu refresh halaman.")
        model_ok = False

if model_ok:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "upload",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded:
        image = Image.open(uploaded)
        st.image(image, use_column_width=True, caption=f"📎 {uploaded.name}")
        run_btn = st.button("🔍 Periksa Sekarang")
    else:
        st.markdown("""
        <div class="upload-placeholder">
            <div class="ph-icon">📷</div>
            <p>Seret gambar ke sini atau klik untuk memilih</p>
            <p style="font-size:.76rem;color:#ddd;margin-top:.3rem;">Format: JPG · JPEG · PNG</p>
        </div>
        """, unsafe_allow_html=True)
        run_btn = False

    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded and run_btn:
        with st.spinner("Menganalisis gambar..."):
            arr, img_bgr, _ = preprocess_image(image)
            pred      = model.predict(arr, verbose=0)[0][0]
            THRESHOLD = 0.31
            is_high   = pred >= THRESHOLD
            conf      = (pred if is_high else 1 - pred) * 100

        if is_high:
            st.markdown(f"""
            <div class="result-card result-high">
                <span class="result-emoji">⚠️</span>
                <div class="result-label">Risiko Tinggi</div>
                <div class="result-conf">{conf:.1f}%</div>
                <div class="result-note">
                    Model mendeteksi kemungkinan kanker kulit.<br>
                    <b>Segera konsultasikan ke dokter spesialis kulit.</b>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.session_state.show_high_risk_popup = True
        else:
            st.markdown(f"""
            <div class="result-card result-low">
                <span class="result-emoji">✅</span>
                <div class="result-label">Risiko Rendah</div>
                <div class="result-conf">{conf:.1f}%</div>
                <div class="result-note">
                    Tidak terdeteksi tanda risiko tinggi pada gambar ini.<br>
                    Tetap lakukan pemeriksaan rutin ke dokter kulit.
                </div>
            </div>
            """, unsafe_allow_html=True)

        with st.spinner("Membuat visualisasi area fokus..."):
            try:
                heatmap = make_gradcam_heatmap(arr, model)
                if heatmap is not None:
                    fig_buf = gradcam_figure(img_bgr, heatmap)
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown('<div class="gradcam-label">🔥 Area Fokus Model (Grad-CAM)</div>', unsafe_allow_html=True)
                    st.image(fig_buf, use_column_width=True)
                    st.markdown('<div class="gradcam-note">Area merah = bagian yang paling mempengaruhi prediksi model.</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
            except Exception:
                pass

        if st.session_state.show_high_risk_popup:
            st.rerun()

st.markdown("""
<div class="app-footer">
    PEKAKU · Pendeteksi Risiko Kanker Kulit<br>
    Hasil bukan pengganti diagnosis medis profesional
</div>
""", unsafe_allow_html=True)
