import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import joblib
import requests
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go

# Konfigurasi halaman
st.set_page_config(
    page_title="Deteksi Jenis Sampah",
    page_icon="🗑️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS custom
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
    }
    .organic {
        background-color: #90EE90;
        border: 2px solid #2E8B57;
    }
    .inorganic {
        background-color: #FFB6C1;
        border: 2px solid #DC143C;
    }
</style>
""", unsafe_allow_html=True)

class WasteDetector:
    def __init__(self):
        self.model = self.load_model()
    
    def load_model(self):
        """Load model machine learning (simulasi)"""
        # Dalam implementasi nyata, ini akan load model yang sudah di-train
        try:
            # Simulasi model sederhana
            model = {
                'predict': self.predict_waste_type
            }
            return model
        except:
            return None
    
    def predict_waste_type(self, features):
        """Fungsi prediksi jenis sampah"""
        # Fitur: [berat_gram, volume_cm3, kadar_air, suhu_celcius]
        berat, volume, kadar_air, suhu = features
        
        # Logika sederhana untuk klasifikasi
        if kadar_air > 60 and suhu > 25:
            return "ORGANIK", 0.85
        elif berat < 50 and volume < 100:
            return "ANORGANIK", 0.78
        else:
            # Random forest simulation
            organic_score = (kadar_air * 0.4 + suhu * 0.3) / 100
            inorganic_score = 1 - organic_score
            
            if organic_score > inorganic_score:
                return "ORGANIK", organic_score
            else:
                return "ANORGANIK", inorganic_score

def main():
    # Header
    st.markdown('<h1 class="main-header">🗑️ Deteksi Jenis Sampah</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("Pengaturan")
        st.info("Aplikasi ini menggunakan machine learning untuk mengklasifikasikan jenis sampah menjadi organik dan anorganik.")
        
        st.subheader("Metode Input")
        input_method = st.radio(
            "Pilih metode input:",
            ["Manual Input", "Upload Data CSV"]
        )
        
        st.markdown("---")
        st.subheader("Informasi")
        st.write("**Sampah Organik:**")
        st.write("- Sisa makanan, daun, kayu, dll.")
        st.write("- Mudah terurai")
        
        st.write("**Sampah Anorganik:**")
        st.write("- Plastik, kaca, logam, dll.")
        st.write("- Sulit terurai")
    
    # Inisialisasi detector
    detector = WasteDetector()
    
    if input_method == "Manual Input":
        manual_input(detector)
    else:
        csv_upload(detector)

def manual_input(detector):
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Input Karakteristik Sampah")
        
        with st.form("input_form"):
            berat = st.slider("Berat (gram):", 1, 1000, 100)
            volume = st.slider("Volume (cm³):", 1, 1000, 200)
            kadar_air = st.slider("Kadar Air (%):", 0, 100, 60)
            suhu = st.slider("Suhu (°C):", 10, 50, 25)
            warna = st.selectbox("Warna Dominan:", ["Hijau", "Coklat", "Putih", "Transparan", "Warna-warni"])
            tekstur = st.selectbox("Tekstur:", ["Lunak", "Keras", "Elastis", "Rapuh"])
            
            submitted = st.form_submit_button("Deteksi Jenis Sampah")
    
    with col2:
        st.subheader("Visualisasi & Hasil")
        
        if submitted:
            # Mapping fitur tambahan ke nilai numerik
            warna_map = {"Hijau": 1, "Coklat": 2, "Putim": 3, "Transparan": 4, "Warna-warni": 5}
            tekstur_map = {"Lunak": 1, "Keras": 2, "Elastis": 3, "Rapuh": 4}
            
            features = [berat, volume, kadar_air, suhu, warna_map[warna], tekstur_map[tekstur]]
            
            # Prediksi
            waste_type, confidence = detector.model['predict'](features[:4])
            
            # Tampilkan hasil
            display_result(waste_type, confidence)
            
            # Visualisasi
            fig = create_visualization(berat, volume, kadar_air, suhu, waste_type)
            st.plotly_chart(fig, use_container_width=True)
            
            # Rekomendasi penanganan
            display_recommendation(waste_type)
        else:
            st.info("Silakan isi form di sebelah kiri dan klik 'Deteksi Jenis Sampah'")
            
            # Chart placeholder
            fig = go.Figure()
            fig.add_layout_image(
                dict(
                    source="https://images.unsplash.com/photo-1587332066588-15b65c89d7d3?w=500",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    sizex=0.8, sizey=0.8,
                    xanchor="center", yanchor="middle"
                )
            )
            fig.update_layout(
                title="Ilustrasi Sistem Deteksi Sampah",
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

def csv_upload(detector):
    st.subheader("Upload Data CSV")
    
    uploaded_file = st.file_uploader("Upload file CSV dengan kolom: berat, volume, kadar_air, suhu", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Data Preview:")
            st.dataframe(df.head())
            
            if st.button("Proses Batch Detection"):
                results = []
                for _, row in df.iterrows():
                    features = [row['berat'], row['volume'], row['kadar_air'], row['suhu']]
                    waste_type, confidence = detector.model['predict'](features)
                    results.append({
                        'Jenis_Sampah': waste_type,
                        'Confidence': confidence,
                        'Rekomendasi': 'Kompos' if waste_type == 'ORGANIK' else 'Daur Ulang'
                    })
                
                results_df = pd.DataFrame(results)
                final_df = pd.concat([df, results_df], axis=1)
                
                st.subheader("Hasil Deteksi Batch")
                st.dataframe(final_df)
                
                # Statistik
                col1, col2, col3 = st.columns(3)
                with col1:
                    organic_count = len(final_df[final_df['Jenis_Sampah'] == 'ORGANIK'])
                    st.metric("Sampah Organik", organic_count)
                
                with col2:
                    inorganic_count = len(final_df[final_df['Jenis_Sampah'] == 'ANORGANIK'])
                    st.metric("Sampah Anorganik", inorganic_count)
                
                with col3:
                    avg_confidence = final_df['Confidence'].mean()
                    st.metric("Rata-rata Confidence", f"{avg_confidence:.2%}")
                
                # Download hasil
                csv = final_df.to_csv(index=False)
                st.download_button(
                    label="Download Hasil CSV",
                    data=csv,
                    file_name="hasil_deteksi_sampah.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"Error membaca file: {e}")

def display_result(waste_type, confidence):
    if waste_type == "ORGANIK":
        st.markdown(f"""
        <div class="prediction-box organic">
            <h2>🥦 HASIL: SAMPAH ORGANIK</h2>
            <h3>Tingkat Kepercayaan: {confidence:.2%}</h3>
            <p>Sampah ini mudah terurai secara alami</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="prediction-box inorganic">
            <h2>🧴 HASIL: SAMPAH ANORGANIK</h2>
            <h3>Tingkat Kepercayaan: {confidence:.2%}</h3>
            <p>Sampah ini sulit terurai, perlu didaur ulang</p>
        </div>
        """, unsafe_allow_html=True)

def create_visualization(berat, volume, kadar_air, suhu, waste_type):
    # Radar chart
    categories = ['Berat', 'Volume', 'Kadar Air', 'Suhu']
    values = [berat/1000, volume/1000, kadar_air, suhu/50]  # Normalisasi
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Karakteristik Sampah',
        line=dict(color='green' if waste_type == 'ORGANIK' else 'red')
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=False,
        title="Profil Karakteristik Sampah"
    )
    
    return fig

def display_recommendation(waste_type):
    st.subheader("💡 Rekomendasi Penanganan")
    
    if waste_type == "ORGANIK":
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info("**Kompos**")
            st.write("Ubah menjadi pupuk organik")
            
        with col2:
            st.info("**Biopori**")
            st.write("Teknik resapan air")
            
        with col3:
            st.info("**Magot**")
            st.write("Pakan ternak alternatif")
    else:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info("**Daur Ulang**")
            st.write("Kirim ke bank sampah")
            
        with col2:
            st.info("**Reuse**")
            st.write("Gunakan kembali")
            
        with col3:
            st.info("**Dropbox**")
            st.write("Titik pengumpulan sampah")

if __name__ == "__main__":
    main()