import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import pandas as pd

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Deteksi Telur Parasit",
    page_icon="🔬",
    layout="wide"
)

# --- Judul dan Deskripsi ---
st.title("🔬 Deteksi Telur Parasit Mikroskopis")
st.markdown("""
Aplikasi ini menggunakan model **YOLO** untuk mendeteksi dan menghitung jumlah telur parasit pada citra mikroskop.
""")

st.markdown("---")

# --- Sidebar: Konfigurasi Model ---
st.sidebar.header("⚙️ Konfigurasi Model")

# Upload File Model (.pt)
model_file = st.sidebar.file_uploader("Upload Model (.pt)", type=['pt'])

# Slider Confidence dan IOU
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.01)
iou_threshold = st.sidebar.slider("IoU Threshold", 0.0, 1.0, 0.45, 0.01)

# --- Fungsi Load Model ---
@st.cache_resource
def load_model(model_path):
    """
    Load model YOLO dengan caching agar tidak reload setiap kali ada interaksi.
    """
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# --- Main App Logic ---

if model_file is not None:
    # Simpan file sementara agar bisa dibaca oleh Ultralytics
    with open("temp_model.pt", "wb") as f:
        f.write(model_file.getbuffer())
    
    # Load Model
    model = load_model("temp_model.pt")
    
    if model:
        st.sidebar.success("✅ Model berhasil dimuat!")
        
        # Tampilkan kelas yang dikenali model
        class_names = model.names
        with st.sidebar.expander("Daftar Kelas Parasit"):
            st.write(class_names)

        # Upload Gambar untuk Deteksi
        st.subheader("1. Upload Citra Mikroskop")
        uploaded_image = st.file_uploader("Pilih gambar...", type=['jpg', 'jpeg', 'png', 'bmp'])

        if uploaded_image is not None:
            # Buka gambar dengan PIL
            image = Image.open(uploaded_image)
            
            # Buat 2 kolom: Original vs Hasil
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption="Gambar Asli", use_container_width=True)

            # Tombol Deteksi
            if st.button("🔍 Deteksi Parasit", type="primary"):
                with st.spinner('Sedang memproses...'):
                    # Lakukan Inference
                    results = model.predict(image, conf=conf_threshold, iou=iou_threshold)
                    
                    # Ambil hasil plot (gambar dengan kotak bounding box)
                    # Result.plot() mengembalikan array numpy (BGR), perlu convert ke RGB
                    res_plotted = results[0].plot()
                    res_plotted_rgb = res_plotted[:, :, ::-1]

                with col2:
                    st.image(res_plotted_rgb, caption="Hasil Deteksi", use_container_width=True)

                # --- Bagian Statistik / Perhitungan ---
                st.subheader("2. Hasil Perhitungan")
                
                # Menghitung jumlah deteksi per kelas
                boxes = results[0].boxes
                if len(boxes) > 0:
                    cls_ids = boxes.cls.cpu().numpy().astype(int)
                    class_counts = {}
                    
                    for cls_id in cls_ids:
                        cls_name = class_names[cls_id]
                        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                    
                    # Tampilkan Metric Cards
                    st.write(f"Total Objek Terdeteksi: **{len(boxes)}**")
                    
                    # Tampilkan Tabel
                    df_counts = pd.DataFrame(list(class_counts.items()), columns=['Jenis Parasit', 'Jumlah'])
                    st.table(df_counts)
                    
                    # Visualisasi Chart Sederhana
                    st.bar_chart(df_counts.set_index('Jenis Parasit'))
                    
                else:
                    st.warning("Tidak ada parasit yang terdeteksi dengan threshold saat ini.")

else:
    st.info("👈 Silakan upload file model YOLO (`.pt`) Anda di sidebar sebelah kiri untuk memulai.")
    
    # Penjelasan singkat jika model belum diupload
    st.markdown("""
    ### Cara Penggunaan:
    1. Upload file **model weights** hasil training (biasanya `best.pt` atau `last.pt`).
    2. Atur tingkat **Confidence** (keyakinan model) di sidebar.
    3. Upload gambar mikroskop yang ingin dianalisis.
    4. Klik tombol **Deteksi Parasit**.
    """)