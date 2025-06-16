import streamlit as st
import cv2
import tempfile
from PIL import Image
from ultralytics import YOLO

# Load model YOLO
@st.cache_resource
def load_model():
    return YOLO("models/best.pt")

model = load_model()

st.title("🚧 Deteksi Kerusakan Jalan dengan YOLOv11")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)

# Buat 3 tab: Kamera, Gambar, Video
tab1, tab2, tab3 = st.tabs(["📷 Kamera", "🖼️ Gambar", "🎥 Video"])

# 📷 Kamera
with tab1:
    run_cam = st.checkbox("Aktifkan Kamera")
    cam_frame = st.empty()
    if run_cam:
        cap = cv2.VideoCapture(0)
        while run_cam:
            ret, frame = cap.read()
            if not ret:
                st.error("Tidak bisa mengakses kamera.")
                break

            frame = cv2.resize(frame, (640, 640))
            results = model.predict(frame, conf=conf_threshold, verbose=False)
            annotated = results[0].plot()
            cam_frame.image(annotated, channels="BGR", use_container_width=True)

        cap.release()

# 🖼️ Gambar
with tab2:
    img_file = st.file_uploader("Upload gambar", type=["jpg", "jpeg", "png"])
    if img_file:
        image = Image.open(img_file).convert("RGB")
        st.image(image, caption="Input Gambar", use_container_width=True)

        # --- TAMBAHKAN LANGKAH RESIZE DI SINI ---
        # Kecilkan gambar untuk mempercepat prediksi
        resized_image = image.resize((640, 640)) 
        # -----------------------------------------

        st.info("Sedang memproses, harap tunggu...")
        results = model.predict(resized_image, conf=conf_threshold) # Gunakan gambar yang sudah di-resize
        annotated = results[0].plot()
        st.image(annotated, caption="Hasil Deteksi", use_container_width=True)

# 🎥 Video
with tab3:
    vid_file = st.file_uploader("Upload video (.mp4)", type=["mp4"])
    if vid_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(vid_file.read())
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (640, 640))
            results = model.predict(frame, conf=conf_threshold, verbose=False)
            annotated = results[0].plot()
            stframe.image(annotated, channels="BGR", use_container_width=True)
        cap.release()
        st.success("✅ Video selesai diproses.")
