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
    st.warning("⚠️ Fitur deteksi via kamera tidak tersedia pada versi web ini karena server tidak memiliki akses ke perangkat keras kamera. Fitur ini hanya berfungsi saat aplikasi dijalankan di komputer lokal.")
    # run_cam = st.checkbox("Aktifkan Kamera")
    # cam_frame = st.empty()
    # if run_cam:
    #     cap = cv2.VideoCapture(0)
    #     while run_cam:
    #         ret, frame = cap.read()
    #         if not ret:
    #             st.error("Tidak bisa mengakses kamera.")
    #             break

    #         frame = cv2.resize(frame, (640, 640))
    #         results = model.predict(frame, conf=conf_threshold, verbose=False)
    #         annotated = results[0].plot()
    #         cam_frame.image(annotated, channels="BGR", use_container_width=True)

    #     cap.release()

# 🖼️ Gambar
with tab2:
    img_file = st.file_uploader("Upload gambar", type=["jpg", "jpeg", "png"])
    if img_file:
        image = Image.open(img_file).convert("RGB")
        st.image(image, caption="Input Gambar", use_container_width=True)

        resized_image = image.resize((640, 640)) 
        
        st.info("Sedang memproses, harap tunggu...")
        results = model.predict(resized_image, conf=conf_threshold) 
        annotated = results[0].plot()
        
        # --- PERUBAHAN WARNA ---
        # Konversi dari BGR (output .plot()) ke RGB untuk ditampilkan dengan benar di Streamlit
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        # --- AKHIR PERUBAHAN ---

        st.image(annotated_rgb, caption="Hasil Deteksi", use_container_width=True)

# 🎥 Video
with tab3:
    vid_file = st.file_uploader("Upload video (.mp4)", type=["mp4"])
    if vid_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(vid_file.read())
        
        st.info("Video sedang diproses. Proses ini akan memakan waktu sangat lama tergantung durasi video. Harap jangan tutup tab ini.")

        # Buka video input
        cap = cv2.VideoCapture(tfile.name)
        
        # Dapatkan properti video untuk membuat video output
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Buat video writer untuk menyimpan hasil
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        # Progress bar
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress_bar = st.progress(0, text="Memproses frame...")
        
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            # Jalankan deteksi
            results = model.predict(frame, conf=conf_threshold, verbose=False)
            annotated_frame = results[0].plot()
            
            # Tulis frame ke video output
            out.write(annotated_frame)
            
            # Update progress bar
            progress_bar.progress((i + 1) / total_frames, text=f"Memproses frame {i+1}/{total_frames}")

        # Tutup semua file video
        cap.release()
        out.release()
        
        progress_bar.empty()
        st.success("✅ Video selesai diproses!")
        
        # Tampilkan video hasil
        st.video(output_path)
        
        # Sediakan link download
        with open(output_path, "rb") as file:
            st.download_button(
                label="Download Video Hasil",
                data=file,
                file_name="hasil_deteksi.mp4",
                mime="video/mp4"
            )
