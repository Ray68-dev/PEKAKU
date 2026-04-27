import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
import io
import os

st.set_page_config(
    page_title="PEKAKU (Pendeteksi Resiko Kanker Kulit)",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .main {
        background-color: #0d0f14;
        color: #e8eaf0;
    }

    .stApp {
        background: linear-gradient(135deg, #0d0f14 0%, #111420 100%);
    }

    /* Header */
    .hero-title {
        font-family: 'Space Mono', monospace;
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00e5ff, #7c4dff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -1px;
        margin-bottom: 0.2rem;
    }

    .hero-sub {
        font-size: 1rem;
        color: #8892a4;
        margin-bottom: 2rem;
        font-weight: 300;
        letter-spacing: 0.05em;
    }

    /* Result cards */
    .result-card {
        border-radius: 16px;
        padding: 1.5rem 2rem;
        margin: 1rem 0;
        border: 1px solid rgba(255,255,255,0.08);
    }

    .result-high {
        background: linear-gradient(135deg, rgba(255,59,59,0.12), rgba(255,100,0,0.08));
        border-color: rgba(255,59,59,0.3);
    }

    .result-low {
        background: linear-gradient(135deg, rgba(0,229,255,0.10), rgba(0,200,120,0.08));
        border-color: rgba(0,229,255,0.3);
    }

    .result-label {
        font-family: 'Space Mono', monospace;
        font-size: 1.4rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }

    .result-high .result-label { color: #ff5c5c; }
    .result-low  .result-label { color: #00e5c8; }

    .result-score {
        font-family: 'Space Mono', monospace;
        font-size: 3rem;
        font-weight: 700;
        line-height: 1;
        margin: 0.5rem 0;
    }

    .result-high .result-score { color: #ff5c5c; }
    .result-low  .result-score { color: #00e5c8; }

    .result-meta {
        font-size: 0.85rem;
        color: #8892a4;
        margin-top: 0.5rem;
    }

    /* Divider */
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
        margin: 1.5rem 0;
    }

    /* Info box */
    .info-box {
        background: rgba(124, 77, 255, 0.08);
        border: 1px solid rgba(124, 77, 255, 0.25);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        font-size: 0.88rem;
        color: #b0b8cc;
        line-height: 1.6;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #0a0c11 !important;
        border-right: 1px solid rgba(255,255,255,0.06);
    }

    .sidebar-section {
        background: rgba(255,255,255,0.03);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(255,255,255,0.06);
    }

    .sidebar-title {
        font-family: 'Space Mono', monospace;
        font-size: 0.75rem;
        color: #5c6478;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-bottom: 0.8rem;
    }

    /* Button override */
    .stButton > button {
        background: linear-gradient(135deg, #00e5ff, #7c4dff);
        color: #0d0f14;
        font-family: 'Space Mono', monospace;
        font-weight: 700;
        font-size: 0.9rem;
        border: none;
        border-radius: 10px;
        padding: 0.65rem 2rem;
        letter-spacing: 0.05em;
        transition: opacity 0.2s;
    }

    .stButton > button:hover {
        opacity: 0.85;
        color: #0d0f14;
    }

    /* Metric */
    [data-testid="metric-container"] {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 12px;
        padding: 0.8rem 1rem;
    }

    /* Hide default streamlit header */
    header[data-testid="stHeader"] { display: none; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model(model_path):
    """Load model dengan cache agar tidak reload setiap saat."""
    model = tf.keras.models.load_model(model_path, compile=False)
    return model


def preprocess_image(image_pil):
    """Preprocess PIL Image untuk input model."""
    img_rgb = np.array(image_pil.convert("RGB").resize((224, 224)))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    img_array = preprocess_input(img_rgb.astype(np.float32).copy())
    img_array = np.expand_dims(img_array, axis=0)
    return img_array, img_bgr, img_rgb


def make_gradcam_heatmap(img_array, model, last_conv_layer_name="top_conv"):
    base_model = model.get_layer("efficientnetb0")
    grad_model = tf.keras.Model(
        inputs=base_model.input,
        outputs=[
            base_model.get_layer(last_conv_layer_name).output,
            base_model.output
        ]
    )
    img_tensor = tf.cast(img_array, tf.float32)

    with tf.GradientTape() as tape:
        conv_outputs, _ = grad_model(img_tensor, training=False)
        tape.watch(conv_outputs)
        x = conv_outputs
        found = False
        for layer in base_model.layers:
            if found:
                x = layer(x, training=False)
            if layer.name == last_conv_layer_name:
                found = True
        x = model.get_layer("global_average_pooling2d")(x, training=False)
        x = model.get_layer("batch_normalization")(x, training=False)
        x = model.get_layer("dense")(x, training=False)
        x = model.get_layer("dropout")(x, training=False)
        x = model.get_layer("dense_1")(x, training=False)
        loss = x[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        raise ValueError("Gradien None!")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_map = conv_outputs[0]
    heatmap = conv_map @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def generate_gradcam_figure(img_bgr, heatmap, alpha=0.4):
    """Generate Grad-CAM figure dan return sebagai bytes untuk Streamlit."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    heatmap_resized = cv2.resize(heatmap, (img_rgb.shape[1], img_rgb.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    superimposed = cv2.addWeighted(img_rgb, 1 - alpha, heatmap_colored, alpha, 0)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.patch.set_facecolor("#0d0f14")

    titles = ["Gambar Asli", "Grad-CAM Heatmap", "Overlay"]
    images = [img_rgb, heatmap_resized, superimposed]
    cmaps  = [None, "jet", None]

    for ax, img, title, cmap in zip(axes, images, titles, cmaps):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, color="#8892a4", fontsize=11, pad=8)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.tight_layout(pad=1.5)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="#0d0f14", transparent=False)
    plt.close(fig)
    buf.seek(0)
    return buf

with st.sidebar:
    st.markdown('<div class="hero-title" style="font-size:1.4rem;">⚙️ Konfigurasi</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="sidebar-title">📂 Model Path</div>', unsafe_allow_html=True)
    model_path = st.text_input(
        "Path file .keras",
        value="skin_cancer_model.keras",
        label_visibility="collapsed"
    )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-title">🎚️ Threshold</div>', unsafe_allow_html=True)
    threshold = st.slider(
        "Batas deteksi kanker",
        min_value=0.0, max_value=1.0,
        value=0.31, step=0.01,
        label_visibility="collapsed"
    )
    st.markdown(f'<div class="result-meta">Threshold saat ini: <b style="color:#00e5ff">{threshold:.2f}</b></div>', unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-title">🔥 Grad-CAM</div>', unsafe_allow_html=True)
    show_gradcam = st.toggle("Tampilkan Grad-CAM", value=True)
    if show_gradcam:
        gradcam_alpha = st.slider("Opacity overlay", 0.1, 0.8, 0.4, 0.05)
        conv_layer = st.text_input("Nama conv layer", value="top_conv")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    ⚠️ <b>Disclaimer</b><br>
    Hasil ini bukan diagnosis medis. Selalu konsultasikan dengan dokter spesialis kulit untuk diagnosis yang akurat.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────
st.markdown('<div class="hero-title">🔬 SkinScan AI</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Deteksi Kanker Kulit dengan EfficientNet + Grad-CAM</div>', unsafe_allow_html=True)
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Load model
model = None
model_status = st.empty()

if os.path.exists(model_path):
    with st.spinner("Memuat model..."):
        try:
            model = load_model(model_path)
            model_status.success(f"✅ Model berhasil dimuat: `{model_path}`")
        except Exception as e:
            model_status.error(f"❌ Gagal memuat model: {e}")
else:
    model_status.warning(f"⚠️ File model tidak ditemukan: `{model_path}` — Ubah path di sidebar.")

st.markdown("<br>", unsafe_allow_html=True)

# Layout 2 kolom
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📤 Upload Gambar")
    uploaded_file = st.file_uploader(
        "Pilih gambar lesi kulit (JPG / PNG)",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar input", use_column_width=True)
        st.markdown(f'<div class="result-meta">📄 {uploaded_file.name} &nbsp;|&nbsp; {image.size[0]}×{image.size[1]} px</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("🔍 Analisis Sekarang", use_container_width=True, disabled=(model is None))
    else:
        st.markdown("""
        <div style="border: 2px dashed rgba(255,255,255,0.1); border-radius:14px;
                    padding: 3rem 1rem; text-align:center; color:#5c6478;">
            <div style="font-size:2.5rem;">🖼️</div>
            <div style="margin-top:0.5rem; font-size:0.9rem;">Belum ada gambar diupload</div>
        </div>
        """, unsafe_allow_html=True)
        run_btn = False

with col2:
    st.markdown("### 📊 Hasil Analisis")

    if uploaded_file and run_btn:
        with st.spinner("Memproses gambar..."):
            img_array, img_bgr, img_rgb = preprocess_image(image)
            pred = model.predict(img_array, verbose=0)[0][0]
            confidence = pred * 100
            is_cancer  = pred >= threshold
            label      = "RISIKO TINGGI (Cancer)" if is_cancer else "RISIKO RENDAH (Non-Cancer)"
            card_class = "result-high" if is_cancer else "result-low"
            emoji      = "⚠️" if is_cancer else "✅"

        st.markdown(f"""
        <div class="result-card {card_class}">
            <div class="result-label">{emoji} {label}</div>
            <div class="result-score">{confidence:.1f}%</div>
            <div class="result-meta">
                Score mentah: <b>{pred:.4f}</b> &nbsp;|&nbsp; Threshold: <b>{threshold}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Score", f"{pred:.4f}")
        m2.metric("Confidence", f"{confidence:.1f}%")
        m3.metric("Threshold", f"{threshold}")

        # Grad-CAM
        if show_gradcam:
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown("#### 🔥 Grad-CAM Visualization")
            with st.spinner("Membuat Grad-CAM..."):
                try:
                    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name=conv_layer)
                    fig_buf = generate_gradcam_figure(img_bgr, heatmap, alpha=gradcam_alpha)
                    st.image(fig_buf, use_column_width=True)
                    st.markdown('<div class="result-meta">🔴 Area merah = region paling berpengaruh pada prediksi model</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Grad-CAM gagal: {e}")
                    st.code("Cek nama conv layer di sidebar (default: top_conv)")

    elif not uploaded_file:
        st.markdown("""
        <div style="border: 1px solid rgba(255,255,255,0.06); border-radius:14px;
                    padding: 3rem 1rem; text-align:center; color:#5c6478;">
            <div style="font-size:2rem;">⬅️</div>
            <div style="margin-top:0.5rem; font-size:0.9rem;">Upload gambar terlebih dahulu</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="border: 1px solid rgba(255,255,255,0.06); border-radius:14px;
                    padding: 3rem 1rem; text-align:center; color:#5c6478;">
            <div style="font-size:2rem;">🔍</div>
            <div style="margin-top:0.5rem; font-size:0.9rem;">Klik tombol <b>Analisis Sekarang</b></div>
        </div>
        """, unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []

if uploaded_file and run_btn and model:
    st.session_state.history.append({
        "file": uploaded_file.name,
        "label": label,
        "score": f"{pred:.4f}",
        "confidence": f"{confidence:.1f}%"
    })

if st.session_state.history:
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("### 🗂️ Riwayat Analisis Sesi Ini")
    import pandas as pd
    df = pd.DataFrame(st.session_state.history[::-1])
    df.columns = ["File", "Diagnosis", "Score", "Confidence"]
    df.index = range(1, len(df) + 1)
    st.dataframe(df, use_container_width=True)

    if st.button("🗑️ Hapus Riwayat"):
        st.session_state.history = []
        st.rerun()