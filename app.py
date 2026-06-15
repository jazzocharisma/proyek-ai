import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import time
import os

# ── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EcoScan — Waste Classifier",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.stApp { background: #f0f7f0; color: #1a2e1a; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem; font-weight: 800;
    letter-spacing: -0.03em; line-height: 1.1;
    background: linear-gradient(135deg, #2d7a3a 0%, #4caf50 60%, #81c784 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin-bottom: 0.1rem;
}
.hero-sub {
    font-size: 0.85rem; color: #7a9e7a;
    letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 1.5rem;
}
.result-card {
    border-radius: 20px; padding: 1.8rem; margin: 0.5rem 0;
    border: 2px solid; box-shadow: 0 4px 24px rgba(0,0,0,0.06);
}
.result-organic  { background: linear-gradient(135deg,#e8f5e9,#f1f8f1); border-color:#a5d6a7; }
.result-inorganic{ background: linear-gradient(135deg,#fff3e0,#fdf6ee); border-color:#ffcc80; }
.result-label    { font-family:'Syne',sans-serif; font-size:2rem; font-weight:700; margin-bottom:0.3rem; }
.result-label-organic  { color:#2e7d32; }
.result-label-inorganic{ color:#e65100; }
.result-desc { font-size:0.88rem; color:#5a7a5a; line-height:1.6; margin-top:0.8rem; }

.conf-bar-container { background:rgba(0,0,0,0.07); border-radius:999px; height:10px; margin:0.8rem 0 0.3rem; overflow:hidden; }
.conf-bar-fill-organic  { height:100%; border-radius:999px; background:linear-gradient(90deg,#43a047,#81c784); }
.conf-bar-fill-inorganic{ height:100%; border-radius:999px; background:linear-gradient(90deg,#fb8c00,#ffcc02); }
.conf-label { font-size:0.72rem; color:#8aaa8a; letter-spacing:0.1em; text-transform:uppercase; }
.conf-pct         { font-family:'Syne',sans-serif; font-size:1.8rem; font-weight:700; }
.conf-pct-organic { color:#2e7d32; }
.conf-pct-inorganic{ color:#e65100; }

.chip-row { display:flex; gap:0.4rem; flex-wrap:wrap; margin-top:0.8rem; }
.chip        { display:inline-block; padding:0.25rem 0.7rem; border-radius:999px; font-size:0.76rem; font-weight:500; background:#c8e6c9; color:#1b5e20; border:1px solid #a5d6a7; }
.chip-orange { background:#ffe0b2; color:#bf360c; border:1px solid #ffcc80; }

.live-badge {
    display:inline-flex; align-items:center; gap:6px;
    background:#e8f5e9; border:1.5px solid #a5d6a7; border-radius:999px;
    padding:4px 14px; font-size:0.78rem; font-weight:600; color:#2e7d32; margin-bottom:1rem;
}
.live-dot {
    width:8px; height:8px; border-radius:50%; background:#4caf50;
    animation:pulse 1.2s infinite; display:inline-block;
}
@keyframes pulse {
    0%,100%{ opacity:1; transform:scale(1); }
    50%    { opacity:0.5; transform:scale(0.8); }
}
.custom-divider { height:1px; background:linear-gradient(90deg,transparent,#a5d6a7,transparent); margin:1.2rem 0; }

.stTabs [data-baseweb="tab-list"] {
    gap:0.5rem; background:#e0f0e0; border-radius:12px;
    padding:0.3rem; border:1px solid #b2d8b2;
}
.stTabs [data-baseweb="tab"] { border-radius:8px; font-weight:500; color:#5a7a5a; padding:0.5rem 1.5rem; }
.stTabs [aria-selected="true"] { background:white !important; color:#2e7d32 !important; box-shadow:0 2px 8px rgba(0,0,0,0.08); }

[data-testid="stSidebar"] { background:#e8f5e9; border-right:1px solid #c8e6c9; }
[data-testid="stSidebar"] * { color:#2d4a2d; }
[data-testid="stMetric"] { background:white; border:1px solid #c8e6c9; border-radius:12px; padding:1rem; box-shadow:0 2px 8px rgba(0,0,0,0.04); }
</style>
""", unsafe_allow_html=True)


# ── Load Model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = "model/waste_classifier_final.keras"
    if not os.path.exists(model_path):
        return None
    return tf.keras.models.load_model(model_path)

model = load_model()


# ── Helper Functions ─────────────────────────────────────────────────────────
def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def predict(img: Image.Image):
    processed = preprocess_image(img)
    pred = float(model.predict(processed, verbose=0)[0][0])
    is_organic = pred < 0.5
    confidence = (1 - pred) * 100 if is_organic else pred * 100
    return is_organic, confidence, pred

def render_result(is_organic: bool, confidence: float):
    if is_organic:
        examples  = ["Sisa makanan", "Daun", "Kertas", "Kulit buah"]
        tips      = "Dapat dikompos! Buang ke tempat sampah <b>hijau</b>."
        card_cls  = "result-organic"
        label_cls = "result-label-organic"
        bar_cls   = "conf-bar-fill-organic"
        pct_cls   = "conf-pct-organic"
        chip_cls  = "chip"
        icon      = "🌿"; label = "ORGANIK"
    else:
        examples  = ["Botol plastik", "Kaleng", "Kaca", "Baterai"]
        tips      = "Perlu didaur ulang! Buang ke tempat sampah <b>kuning</b>."
        card_cls  = "result-inorganic"
        label_cls = "result-label-inorganic"
        bar_cls   = "conf-bar-fill-inorganic"
        pct_cls   = "conf-pct-inorganic"
        chip_cls  = "chip chip-orange"
        icon      = "🔴"; label = "ANORGANIK"

    chips_html = "".join([f'<span class="{chip_cls}">{e}</span>' for e in examples])
    st.markdown(f"""
    <div class="result-card {card_cls}">
        <div class="result-label {label_cls}">{icon} {label}</div>
        <div class="conf-label">Confidence</div>
        <div class="conf-bar-container">
            <div class="{bar_cls}" style="width:{int(confidence)}%"></div>
        </div>
        <div class="conf-pct {pct_cls}">{confidence:.1f}%</div>
        <div class="custom-divider"></div>
        <div class="result-desc">{tips}</div>
        <div class="chip-row">{chips_html}</div>
    </div>
    """, unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div style="font-family:Syne,sans-serif;font-size:1.3rem;font-weight:700;color:#2e7d32;margin-bottom:1rem;">♻️ EcoScan</div>', unsafe_allow_html=True)
    st.markdown("**Tentang Aplikasi**")
    st.markdown("Sistem klasifikasi sampah berbasis AI menggunakan **Transfer Learning MobileNetV2**.")
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    st.markdown("**Model Info**")
    st.markdown("- 🧠 MobileNetV2\n- 📦 Transfer Learning\n- 🎯 Binary Classification\n- 📊 ~25.000 gambar")
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    st.markdown("**Kategori**")
    st.markdown("🌿 **Organik** — Sisa makanan, daun, kertas\n\n🔴 **Anorganik** — Plastik, logam, kaca, baterai")
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    st.caption("Dikembangkan oleh kelompok 10")


# ── Main ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">EcoScan</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">AI-Powered Waste Classification System</div>', unsafe_allow_html=True)

if model is None:
    st.error("Model belum ditemukan! Pastikan `model/waste_classifier.h5` sudah ada.")
    st.stop()
else:
    st.success("Model loaded — Siap mengklasifikasikan sampah!")

st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Kamera Live", "Upload Gambar"])

# ── TAB 1: LIVE CAMERA ───────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="live-badge"><span class="live-dot"></span> LIVE — Prediksi otomatis setiap frame</div>', unsafe_allow_html=True)
    st.caption("Arahkan kamera ke sampah. Hasil klasifikasi muncul otomatis secara real-time.")

    col_cam, col_result = st.columns([3, 2], gap="large")

    with col_cam:
        frame_placeholder = st.empty()
    with col_result:
        result_placeholder = st.empty()

    c1, c2 = st.columns(2)
    with c1:
        start_btn = st.button("▶ Mulai Kamera", type="primary", use_container_width=True)
    with c2:
        stop_btn = st.button("⏹ Stop Kamera", use_container_width=True)

    if "camera_running" not in st.session_state:
        st.session_state.camera_running = False
    if start_btn:
        st.session_state.camera_running = True
    if stop_btn:
        st.session_state.camera_running = False

    if st.session_state.camera_running:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not cap.isOpened():
            st.error("❌ Kamera tidak dapat dibuka.")
            st.session_state.camera_running = False
        else:
            frame_count      = 0
            last_is_organic  = None
            last_confidence  = 0.0

            while st.session_state.camera_running:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)

                # Prediksi setiap 10 frame
                if frame_count % 10 == 0:
                    pil_img = Image.fromarray(frame_rgb)
                    last_is_organic, last_confidence, _ = predict(pil_img)

                # Overlay label pada frame
                if last_is_organic is not None:
                    label_text = f"{'ORGANIK' if last_is_organic else 'ANORGANIK'}  {last_confidence:.1f}%"
                    color      = (46, 125, 50)   if last_is_organic else (230, 81, 0)
                    bg_color   = (232, 245, 233) if last_is_organic else (255, 243, 224)
                    cv2.rectangle(frame_rgb, (10, 10), (460, 65), bg_color, -1)
                    cv2.rectangle(frame_rgb, (10, 10), (460, 65), color, 2)
                    cv2.putText(frame_rgb, label_text, (22, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2, cv2.LINE_AA)

                frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

                # Update panel hasil
                if frame_count % 10 == 0 and last_is_organic is not None:
                    with result_placeholder.container():
                        render_result(last_is_organic, last_confidence)

                frame_count += 1
                time.sleep(0.01)

            cap.release()
    else:
        empty_html = lambda icon, text: f"""
        <div style="min-height:380px;border:2px dashed #c8e6c9;border-radius:16px;
                    display:flex;align-items:center;justify-content:center;
                    flex-direction:column;gap:1rem;background:#f9fbf9;color:#8aaa8a;">
            <div style="font-size:3.5rem;">{icon}</div>
            <div style="font-size:0.92rem;">{text}</div>
        </div>"""
        frame_placeholder.markdown(
            empty_html("📷", "Klik <b style='color:#2e7d32'>▶ Mulai Kamera</b> untuk memulai"),
            unsafe_allow_html=True)
        result_placeholder.markdown(
            empty_html("🔍", "Hasil klasifikasi muncul di sini"),
            unsafe_allow_html=True)


# ── TAB 2: UPLOAD ────────────────────────────────────────────────────────────
with tab2:
    st.markdown("#### Upload gambar sampah dari perangkat kamu")
    st.caption("Format yang didukung: JPG, JPEG, PNG")

    uploaded = st.file_uploader("", type=["jpg","jpeg","png"], label_visibility="collapsed")

    if uploaded is not None:
        img = Image.open(uploaded)
        col_img, col_res = st.columns([1, 1], gap="large")
        with col_img:
            st.image(img, caption="Gambar yang diupload", use_column_width=True)
        with col_res:
            with st.spinner("🔍 Menganalisis gambar..."):
                time.sleep(0.4)
                is_organic, confidence, raw_pred = predict(img)
            render_result(is_organic, confidence)
            st.markdown("<br>", unsafe_allow_html=True)
            m1, m2 = st.columns(2)
            with m1: st.metric("Raw Score", f"{raw_pred:.4f}")
            with m2: st.metric("Threshold", "0.5000")