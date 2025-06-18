import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import os
import io
from scipy.optimize import curve_fit
import plotly.express as px
import plotly.graph_objects as go

# Pastikan untuk mengimpor logistic_model, karena masih digunakan untuk prediksi
from models.logistic_functions import logistic_model, estimate_logistic_params_auto 

# --- FUNGSI PEMBANTU UNTUK FORMAT ANGKA INDONESIA ---
def format_indo_number(number, decimal_places=0):
    """
    Memformat angka dengan titik (.) sebagai pemisah ribuan dan koma (,) sebagai pemisah desimal.
    """
    if np.isinf(number) or np.isnan(number):
        return "Tidak Terdefinisi"
    
    num_float = float(number)
    
    # Pisahkan bagian integer dan desimal
    integer_part = int(num_float)
    
    # Format bagian integer dengan titik sebagai pemisah ribuan
    formatted_integer = f"{integer_part:,}".replace(",", ".")
    
    if decimal_places > 0:
        # Format bagian desimal dengan jumlah tempat desimal yang ditentukan
        formatted_decimal = f"{abs(num_float) - abs(integer_part):.{decimal_places}f}".lstrip('0.')
        
        if not formatted_decimal and num_float == integer_part:
            return formatted_integer
        
        if formatted_decimal: 
            return f"{formatted_integer},{formatted_decimal.replace('.', ',')}"
        else: 
            return f"{formatted_integer},{'0' * decimal_places}" 
    else:
        return formatted_integer
# --- AKHIR FUNGSI PEMBANTU ---


# --- Konfigurasi Halaman ---
st.set_page_config(layout="wide", page_title="Prediksi Populasi Logistik", page_icon="📈") 
st.title('Sistem Estimasi dan Prediksi Populasi')
st.subheader('Menggunakan Model Pertumbuhan Logistik (K otomatis atau manual)')

# --- Bagian Penjelasan Umum Aplikasi ---
st.markdown("---") # Pemisah visual
st.markdown("### ℹ Tentang Aplikasi Ini")
st.markdown("""
Aplikasi ini dirancang untuk melakukan estimasi dan prediksi populasi menggunakan Model Pertumbuhan Logistik.
Model ini dipilih karena kemampuannya untuk menggambarkan pertumbuhan populasi yang realistis,
yang mempertimbangkan batasan sumber daya lingkungan.
""")

st.markdown("#### Mengapa Model Logistik?")
st.markdown("""
Model logistik (atau kurva S) adalah pilihan yang tepat untuk memprediksi populasi karena beberapa alasan:
* Realisme: Tidak seperti model pertumbuhan eksponensial yang tak terbatas, model logistik mengakui adanya daya tampung (carrying capacity, K) lingkungan. Ini berarti populasi tidak akan tumbuh tanpa batas, melainkan akan melambat dan stabil seiring mendekati batas sumber daya yang tersedia.
* Pola Umum: Banyak populasi biologis (termasuk manusia dalam skala tertentu dan kondisi tertentu) menunjukkan pola pertumbuhan berbentuk 'S' ini, di mana pertumbuhan awal cepat, kemudian melambat, dan akhirnya mencapai dataran (plateau) di sekitar daya tampung.
* Parameter yang Mudah Diinterpretasi: Parameter model ($N₀$, $r$, $K$) memiliki makna biologis yang jelas, sehingga hasil prediksi mudah dipahami.
""")

st.markdown("#### Rumus Model Pertumbuhan Logistik")
st.markdown("""
Model ini didasarkan pada persamaan diferensial logistik yang memiliki solusi analitik:

$$ N(t) = \\frac{K}{1 + (\\frac{K}{N_0} - 1) e^{-rt}} $$

Di mana:
* $N(t)$ = Ukuran populasi pada waktu $t$
* $K$ = Daya tampung lingkungan (populasi maksimum yang dapat ditampung)
* $N_0$ = Ukuran populasi awal (pada waktu $t=0$)
* $r$ = Laju pertumbuhan intrinsik populasi
* $e$ = Basis logaritma natural (sekitar 2.71828)
* $t$ = Waktu (dalam tahun, relatif terhadap tahun awal data)
""")

st.markdown("#### Alur Kerja Aplikasi")
st.markdown("""
Aplikasi ini memandu Anda melalui langkah-langkah berikut:
1. Input Data (Tab 'Input Data & Konfigurasi'): Anda dapat mengunggah data historis populasi dari file CSV atau memasukkannya secara manual.
 * Jika CSV, Anda akan memilih kolom 'Tahun' dan 'Populasi' dari file Anda.
 * Data waktu akan dinormalisasi sehingga tahun pertama data menjadi $t=0$.2. Konfigurasi Estimasi Model (Tab 'Input Data & Konfigurasi'):
 * Estimasi Otomatis (Direkomendasikan): Sistem akan secara otomatis mengestimasi ketiga parameter model ($N₀$, $r$, $K$) yang paling sesuai dengan data historis Anda. Ini adalah pendekatan terbaik untuk mendapatkan model yang paling pas.
 * K Manual: Anda dapat memilih untuk memasukkan nilai daya tampung (K) secara manual. Dalam kasus ini, sistem hanya akan mengestimasi $N₀$ dan $r$, sementara $K$ akan dipertahankan sesuai input Anda. Ini berguna jika Anda memiliki informasi eksternal tentang daya tampung atau ingin melakukan analisis skenario.
3. Hasil Model & Prediksi (Tab 'Hasil Model & Prediksi'):
 * Parameter Model: Menampilkan nilai $N₀$, $r$, dan $K$ yang telah diestimasi.
 * Visualisasi Kurva: Menampilkan grafik yang membandingkan data historis dengan kurva logistik yang diprediksi, termasuk proyeksi ke masa depan hingga daya tampung.
 * Prediksi Tahun Spesifik: Anda dapat memasukkan tahun tertentu, dan aplikasi akan menghitung perkiraan populasi untuk tahun tersebut berdasarkan model yang telah dibangun.
""")

st.markdown("---") # Pemisah visual
# --- Akhir Bagian Penjelasan Umum Aplikasi ---
    
# --- Bagian Panduan Pengguna ---
with st.expander("Panduan Pengguna Aplikasi (Klik untuk melihat)"):
    st.markdown("""
    Selamat datang di **Sistem Estimasi dan Prediksi Populasi**!
    Panduan ini akan membantu Anda menggunakan aplikasi ini langkah demi langkah.

    ---
    #### 1. Input Data (Tab '📊 Input Data & Konfigurasi')

    Anda memiliki dua pilihan untuk memasukkan data historis populasi:

    * **a. Unggah dari File CSV:**
        * Pilih opsi "CSV" di sidebar kiri.
        * Jika Anda tidak yakin dengan format file CSV, klik tombol **"Unduh Template CSV"** di bawah untuk mendapatkan contoh.
        * Klik tombol "Unggah file CSV" dan pilih file `.csv` Anda.
        * Aplikasi akan menampilkan *preview* data Anda.
        * Pilih `Kolom Tahun` dan `Kolom Populasi` yang sesuai dari *dropdown*. Pastikan kolom-kolom ini berisi angka.
        * Setelah berhasil diunggah, tabel data yang akan digunakan untuk analisis akan ditampilkan di bawahnya.

    * **b. Masukkan Data Manual:**
        * Pilih opsi "Manual" di sidebar kiri.
        * Tentukan `Jumlah baris data` yang ingin Anda masukkan.
        * Masukkan nilai `Tahun Data` dan `Populasi Data` untuk setiap baris.
        * Tabel data yang telah Anda masukkan akan ditampilkan untuk verifikasi.

    *Catatan: Data waktu Anda akan dinormalisasi secara otomatis sehingga tahun pertama data menjadi $t=0$ untuk perhitungan model.*

    ---
    #### 2. Konfigurasi Estimasi Model (Tab '📊 Input Data & Konfigurasi')

    Setelah data diinput, Anda perlu mengkonfigurasi bagaimana model logistik akan mengestimasi parameternya:

    * **Estimasi Otomatis (Direkomendasikan):**
        * Biarkan kotak "Gunakan nilai K (daya tampung) manual?" **tidak dicentang**.
        * Sistem akan secara otomatis mencari nilai terbaik untuk $N_0$ (Populasi Awal), $r$ (Laju Pertumbuhan), dan $K$ (Daya Tampung) yang paling sesuai dengan data historis Anda. Ini adalah cara terbaik jika Anda tidak memiliki informasi spesifik tentang $K$.

    * **K Manual:**
        * **Centang** kotak "Gunakan nilai K (daya tampung) manual?".
        * Masukkan nilai `K (daya tampung)` yang Anda inginkan secara manual.
        * Dalam mode ini, sistem hanya akan mengestimasi $N_0$ dan $r$, sementara nilai $K$ yang Anda masukkan akan digunakan sebagai konstanta. Ini berguna jika Anda memiliki informasi eksternal tentang batas populasi atau ingin mencoba skenario tertentu.

    ---
    #### 3. Hasil Model & Prediksi (Tab '📈 Hasil Model & Prediksi')

    Beralihlah ke tab ini untuk melihat hasil analisis model Anda:

    * **Parameter Model:**
        * Anda akan melihat nilai-nilai $N_0$, $r$, dan $K$ yang telah diestimasi atau diinput secara manual.

    * **Metrik Evaluasi Model (Kesesuaian Data Historis):**
        * **RMSE (Root Mean Squared Error)**: Menunjukkan rata-rata kesalahan absolut prediksi model Anda dalam satuan populasi (misal: "jiwa"). Semakin kecil nilainya, semakin baik.
        * **MAPE (Mean Absolute Percentage Error)**: Menunjukkan rata-rata kesalahan prediksi dalam persentase relatif terhadap populasi sebenarnya. Semakin kecil persentasenya, semakin akurat model Anda secara relatif.
        * **R² (R-squared)**: Mengukur seberapa baik model Anda menjelaskan variasi dalam data populasi. Nilai mendekati 1 menunjukkan kecocokan yang sangat baik.
        * Ada **interpretasi otomatis** di bawah metrik ini yang akan memberi tahu Anda apakah hasil model Anda termasuk kategori "Sangat Baik", "Baik", atau "Perlu Tinjauan Lebih Lanjut" berdasarkan nilai MAPE.

    * **Kurva Model & Prediksi:**
        * Grafik interaktif akan menampilkan titik-titik `Data Historis` Anda (biru) dan `Garis Model & Prediksi` (merah putus-putus).
        * `Garis Daya Tampung (K)` (hijau titik-titik) juga akan ditampilkan.
        * **Interaktivitas:** Arahkan kursor Anda ke titik-titik data historis (biru) untuk melihat detail `Tahun` dan `Populasi` pada titik tersebut.
        * Gunakan `slider "Prediksi ke depan (tahun)"` untuk mengatur seberapa jauh ke masa depan kurva prediksi akan ditampilkan.

    * **Prediksi Populasi di Tahun Spesifik:**
        * Masukkan `Tahun` tertentu yang ingin Anda prediksi.
        * Aplikasi akan menghitung dan menampilkan perkiraan populasi untuk tahun tersebut berdasarkan model yang telah dibangun.

    ---
    Semoga panduan ini membantu Anda memanfaatkan aplikasi ini sepenuhnya!
    """)
# --- Akhir Bagian Panduan Pengguna ---


# --- Sidebar: Input Data & Model ---
st.sidebar.header('1. Input Data') # Pindah ke sini
# Pindahkan pilihan sumber data ke sidebar
data_source = st.sidebar.radio("Pilih Sumber Data:", ("CSV", "Manual"), index=0, help="Pilih apakah data akan diunggah dari CSV atau dimasukkan secara manual.") # Pindah ke sini


df_filtered = pd.DataFrame()
col_tahun = 'Tahun'
col_populasi = 'Populasi'

# --- Menggunakan Tab untuk Struktur Halaman yang Lebih Baik ---
tab1, tab2 = st.tabs(["📊 Input Data & Konfigurasi", "📈 Hasil Model & Prediksi"])

with tab1: # Konten untuk Tab "Input Data & Konfigurasi"
    st.header('Detail Input Data') # Ubah judul karena pilihan sumber data sudah di sidebar

    if data_source == "CSV":
        # --- Bagian Unduh Template CSV ---
        st.markdown("---") 
        st.markdown("#### Unduh Template CSV (Opsional)")
        st.markdown("Jika Anda tidak yakin dengan format file CSV, Anda dapat mengunduh template di bawah ini:")

        def create_csv_template():
            template_data = {
                'Tahun': [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009],
                'Jumlah Penduduk': [100000, 105000, 110250, 115762, 121551, 127629, 134010, 140710, 147746, 155138]
            }
            df_template = pd.DataFrame(template_data)
            csv_buffer = io.StringIO()
            df_template.to_csv(csv_buffer, index=False)
            return csv_buffer.getvalue()

        st.download_button(
            label="Unduh Template CSV",
            data=create_csv_template(),
            file_name="template_populasi.csv",
            mime="text/csv",
            help="Klik untuk mengunduh contoh format file CSV yang diperlukan."
        )
        st.markdown("---") 
        # --- Akhir Bagian Unduh Template CSV ---

                # --- AWAL BAGIAN TAMBAHAN WARNING/TIPS UNTUK CSV ---
        st.warning("""
        **Penting:** Perhatikan format file CSV Anda!
        * Pastikan **tidak ada spasi ekstra** setelah koma (`,`) pemisah antar kolom, atau di awal/akhir nilai data. Contoh:
            * **Benar:** `Tahun,Jumlah Penduduk` atau `2018,2034567`
            * **Salah:** `Tahun, Jumlah Penduduk` (ada spasi setelah koma header) atau `2018, 2034567` (ada spasi setelah koma data)
        * Kolom `Tahun` dan `Jumlah Penduduk` (atau nama kolom yang Anda pilih) harus **hanya berisi angka**.
        * Hindari baris atau sel yang **kosong** di tengah data.
        * Nama kolom yang Anda pilih di bawah (`Pilih Kolom Tahun`, `Pilih Kolom Populasi`) harus **persis sama** dengan nama kolom di file CSV Anda (peka huruf besar/kecil).
        """)
        st.markdown("---")
        # --- AKHIR BAGIAN TAMBAHAN WARNING/TIPS UNTUK CSV ---

        uploaded_file = st.file_uploader("Unggah file CSV", type="csv")
        if uploaded_file is not None:
            try:
                df_raw = pd.read_csv(uploaded_file)
                st.write("Preview Data CSV:")
                st.dataframe(df_raw.head())

                available_columns = df_raw.columns.tolist()
                st.write("Kolom yang tersedia di CSV:", available_columns)

                default_tahun_idx = 0
                if 'Tahun' in available_columns: default_tahun_idx = available_columns.index('Tahun')
                elif 'tahun' in available_columns: default_tahun_idx = available_columns.index('tahun')

                default_populasi_idx = 0
                if 'Jumlah Penduduk' in available_columns: default_populasi_idx = available_columns.index('Jumlah Penduduk')
                elif 'Populasi' in available_columns: default_populasi_idx = available_columns.index('Populasi')
                elif 'jumlah penduduk' in available_columns: default_populasi_idx = available_columns.index('jumlah penduduk')


                col_tahun = st.selectbox(
                    "Pilih Kolom Tahun:",
                    available_columns,
                    index=default_tahun_idx
                )
                col_populasi = st.selectbox(
                    "Pilih Kolom Populasi:",
                    available_columns,
                    index=default_populasi_idx
                )

                if col_tahun not in df_raw.columns or col_populasi not in df_raw.columns:
                    st.error("Kolom yang dipilih tidak ditemukan dalam CSV. Pastikan nama kolom sesuai.")
                    st.stop()

                df_filtered = df_raw[[col_tahun, col_populasi]].copy()
                df_filtered[col_tahun] = pd.to_numeric(df_filtered[col_tahun], errors='coerce')
                df_filtered[col_populasi] = pd.to_numeric(df_filtered[col_populasi], errors='coerce')
                df_filtered.dropna(inplace=True)

                if df_filtered.empty:
                    st.error("Data setelah filtering kosong. Pastikan kolom yang dipilih berisi angka dan tidak ada baris kosong.")
                    st.stop()
                st.success("CSV berhasil diunggah dan diproses.")

                # --- Menampilkan Tabel Data yang Diunggah ---
                st.subheader("Data yang Digunakan untuk Analisis")
                st.dataframe(df_filtered) # Menampilkan DataFrame yang sudah difilter dan diproses
                # --- Akhir Menampilkan Tabel Data ---

            except Exception as e:
                st.error(f"Gagal membaca atau memproses CSV: {e}")
                st.stop()
        else:
            st.info("Silakan unggah file CSV Anda.")
            st.stop()
    elif data_source == "Manual":
        st.subheader("Masukkan Data Manual")
        # Menggunakan kolom untuk input manual agar lebih rapi
        col1_manual, col2_manual = st.columns(2)
        num_rows = st.number_input("Jumlah baris data:", min_value=2, value=5)
        manual_data_list = []
        for i in range(num_rows):
            with col1_manual:
                tahun_val = st.number_input(f"Tahun Data {i+1}:", value=1950+i, key=f"tahun_{i}")
            with col2_manual:
                populasi_val = st.number_input(f"Populasi Data {i+1}:", value=100*(i+1), key=f"populasi_{i}")
            manual_data_list.append({col_tahun: tahun_val, col_populasi: populasi_val})
        df_filtered = pd.DataFrame(manual_data_list)
        st.write("Preview Data Manual:")
        st.dataframe(df_filtered)
        st.success("Data manual siap.")

        # --- Menampilkan Tabel Data yang Digunakan (Untuk Manual) ---
        st.subheader("Data yang Digunakan untuk Analisis")
        st.dataframe(df_filtered)
        # --- Akhir Menampilkan Tabel Data ---

    # Cek jika df_filtered kosong setelah input data (penting sebelum kalkulasi)
    if df_filtered.empty:
        st.warning("Tidak ada data yang tersedia untuk analisis. Harap unggah CSV atau masukkan data manual.")
        st.stop()

    t_data = df_filtered[col_tahun].values - df_filtered[col_tahun].min()
    N_data = df_filtered[col_populasi].values

    st.divider() # Pemisah visual
    st.header('2. Konfigurasi Estimasi Model')
    use_manual_K = st.checkbox("Gunakan nilai K (daya tampung) manual?", value=False, help="Centang untuk memasukkan nilai K secara manual, jika tidak, sistem akan mengestimasi K secara otomatis.")

    N0_fit, r_fit, K_fit = 0, 0, 0 

    if use_manual_K:
        K_manual = st.number_input("Masukkan K (daya tampung):", value=float(max(N_data)*1.2), min_value=1.0, help="Nilai kapasitas maksimum yang dapat ditampung lingkungan.")

        # --- Definisi logistic_model_fixed_K yang diperbarui ---
        # Ini penting agar fungsi ini konsisten dengan logistic_model di file terpisah
        def logistic_model_fixed_K(t, N0, r):
            N0 = float(N0)
            r = float(r)
            
            # --- PERBAIKAN DI SINI UNTUK MENANGANI RETURN INF/NAN AGAR BERBENTUK ARRAY ---
            if N0 <= 0 or r <= 0:
                return np.full_like(t, np.inf, dtype=float) 
            # --- AKHIR PERBAIKAN ---

            val = (K_manual / N0 - 1) * np.exp(-r * t)
            denominator = 1 + val
            # Hindari pembagian nol dengan mengganti 0 di denominator dengan nilai yang sangat kecil
            denominator = np.where(denominator == 0, np.finfo(float).eps, denominator) 
            result = K_manual / denominator
            # Mengganti NaN atau inf dengan nilai yang dapat dihandle oleh curve_fit (misalnya np.inf)
            return np.nan_to_num(result, nan=np.inf, posinf=np.inf, neginf=-np.inf) 
        # --- Akhir Definisi logistic_model_fixed_K ---

        try:
            initial_guesses_N0_r = [N_data[0] if N_data[0] > 0 else 1, 0.1]
            bounds_N0_r = ([1e-6, 1e-6], [np.inf, np.inf])

            popt_fixed_K, pcov_fixed_K = curve_fit(logistic_model_fixed_K, t_data, N_data,
                                                    p0=initial_guesses_N0_r,
                                                    bounds=bounds_N0_r,
                                                    maxfev=10000,
                                                    method='trf') # 'trf' often more robust for bounded problems

            N0_fit, r_fit = popt_fixed_K
            K_fit = K_manual 

            st.success(f"Estimasi berhasil dengan K manual: N₀={format_indo_number(N0_fit)} jiwa, r={r_fit:.3f}") # Menggunakan format_indo_number

        except RuntimeError as e:
            st.error(f"Gagal mengestimasi parameter dengan K manual: {e}")
            st.stop()
        except Exception as e:
            st.error(f"Terjadi kesalahan tak terduga saat estimasi dengan K manual: {e}")
            st.stop()

    else:
        try:
            auto_params, _ = estimate_logistic_params_auto(t_data, N_data)
            N0_fit, r_fit, K_fit = auto_params 
            st.success(f"Estimasi otomatis berhasil: N₀={format_indo_number(N0_fit)} jiwa, r={r_fit:.3f}, K={format_indo_number(K_fit)} jiwa") # Menggunakan format_indo_number
        except Exception as e:
            st.error(f"Gagal estimasi otomatis: {e}")
            st.stop()

with tab2: # Konten untuk Tab "Hasil Model & Prediksi"
    st.header('🛠 Hasil Estimasi Model')
    st.markdown("""
    Berikut adalah parameter model logistik yang berhasil diestimasi berdasarkan data Anda:
    """)

    # Menggunakan kolom untuk menampilkan metrik parameter
    col_n0, col_r, col_k = st.columns(3)
    with col_n0:
        st.metric("N₀ (Populasi Awal)", format_indo_number(N0_fit)) # Terapkan format_indo_number
    with col_r:
        # r (Laju Pertumbuhan) biasanya ditampilkan dengan titik desimal, jika ingin koma, gunakan format_indo_number
        st.metric("r (Laju Pertumbuhan)", format_indo_number(r_fit, decimal_places=4)) # Diubah ke format_indo_number
    with col_k:
        st.metric("K (Daya Tampung)", format_indo_number(K_fit)) # Terapkan format_indo_number

    # --- Bagian Baru: Perhitungan dan Tampilan RMSE & MAPE ---
    st.subheader("📊 Metrik Evaluasi Model (Kesesuaian Data Historis)")

    # Hitung prediksi model untuk data historis yang ada
    N_pred_historical = logistic_model(t_data, N0_fit, r_fit, K_fit)

    # Pastikan kedua array memiliki tipe data float untuk perhitungan
    N_data_float = np.asarray(N_data, dtype=float)
    N_pred_historical_float = np.asarray(N_pred_historical, dtype=float)

    # Menghitung RMSE (Root Mean Squared Error)
    rmse = np.sqrt(np.mean((N_data_float - N_pred_historical_float)**2))
    
    # Menghitung MAPE
    non_zero_actuals_mask = N_data_float != 0
    if np.any(non_zero_actuals_mask):
        mape = np.mean(np.abs((N_data_float[non_zero_actuals_mask] - N_pred_historical_float[non_zero_actuals_mask]) / N_data_float[non_zero_actuals_mask])) * 100
    else:
        mape = 0.0
    
    # Menghitung R-squared (R^2)
    ss_total = np.sum((N_data_float - np.mean(N_data_float))**2) # Total sum of squares
    ss_residual = np.sum((N_data_float - N_pred_historical_float)**2) # Sum of squares of residuals
    r_squared = 1 - (ss_residual / ss_total) if ss_total > 0 else 0.0 # Hindari pembagian nol

    col_rmse, col_mape, col_r_squared = st.columns(3) 
    with col_rmse:
        if np.isinf(rmse) or np.isnan(rmse):
            st.metric("RMSE (Root Mean Squared Error)", "Tidak Terdefinisi / Sangat Besar")
        else:
            st.metric("RMSE (Root Mean Squared Error)", format_indo_number(rmse)) # Terapkan format_indo_number
    with col_mape:
        if np.isinf(mape) or np.isnan(mape):
            st.metric("MAPE (Mean Absolute Percentage Error)", "Tidak Terdefinisi / Sangat Besar")
        else:
            st.metric("MAPE (Mean Absolute Percentage Error)", format_indo_number(mape, decimal_places=2) + "%") # Terapkan format_indo_number
    with col_r_squared: 
        if np.isinf(r_squared) or np.isnan(r_squared):
            st.metric("R² (R-squared)", "Tidak Terdefinisi")
        else:
            st.metric("R² (R-squared)", format_indo_number(r_squared, decimal_places=4)) # Terapkan format_indo_number

    st.markdown("---")
    st.markdown("""
    **Penjelasan Metrik:**
    * RMSE (Root Mean Squared Error): Mengukur rata-rata besar kesalahan absolut antara prediksi model dan nilai aktual, dalam satuan yang sama dengan data populasi (misal: "jiwa"). Semakin kecil nilainya, semakin dekat prediksi model secara absolut.
    * MAPE (Mean Absolute Percentage Error): Mengukur rata-rata kesalahan dalam persentase relatif terhadap populasi sebenarnya (misal: "model meleset 2% dari populasi aktual"). Semakin kecil persentasenya, semakin akurat model secara relatif.
    * R² (R-squared): Mengukur proporsi variasi dalam data aktual yang dapat dijelaskan oleh model. Nilai mendekati 1 (atau 100%) menunjukkan model sangat cocok dengan data.
    """)

    st.markdown("---")
    st.markdown("#### Interpretasi Hasil:")

    # Tentukan ambang batas (threshold) untuk interpretasi. Anda bisa menyesuaikan angka ini.
    # Untuk MAPE: < 2.5% = Sangat Baik, < 5% = Baik, > 5% = Perlu Tinjauan
    mape_threshold_excellent = 2.5
    mape_threshold_good = 5.0
    
    # Menampilkan interpretasi berdasarkan MAPE
    if mape <= mape_threshold_excellent:
        st.success(f"**Hasil yang Sangat Baik!** Model Anda sangat akurat.")
        st.write(f"MAPE sebesar **{format_indo_number(mape, decimal_places=2)}%** menunjukkan bahwa model Anda memprediksi dengan akurasi persentase yang luar biasa tinggi (kesalahan rata-rata hanya sekitar {format_indo_number(mape, decimal_places=2)}% dari populasi aktual).")
        st.write(f"RMSE sekitar **{format_indo_number(rmse)} jiwa** berarti model Anda secara absolut sangat dekat dengan data historis. Ini mengindikasikan bahwa prediksi model Anda sangat **valid** dan memiliki kesalahan yang **sangat kecil** relatif terhadap nilai populasi sebenarnya.")
        # --- Tambahan interpretasi R^2 untuk hasil Sangat Baik ---
        if r_squared > 0.95: # Contoh ambang batas R^2 yang sangat tinggi (95% ke atas)
            st.write(f"Nilai R² yang sangat tinggi sebesar **{format_indo_number(r_squared, decimal_places=4)}** (yaitu sekitar **{format_indo_number(r_squared*100, decimal_places=2)}%**) semakin menguatkan bahwa model ini sangat baik dalam menjelaskan sebagian besar variasi dalam data populasi historis.")
        elif r_squared > 0.90: # Ambang batas R^2 yang kuat (90% ke atas)
            st.write(f"Nilai R² yang kuat sebesar **{format_indo_number(r_squared, decimal_places=4)}** (yaitu sekitar **{format_indo_number(r_squared*100, decimal_places=2)}%**) menunjukkan model sangat baik dalam menjelaskan variasi data historis.")
        else: # Jika R^2 tidak setinggi itu meskipun MAPE excellent (kasus jarang, mungkin data sangat seragam)
            st.write(f"Meskipun RMSE dan MAPE sangat rendah, nilai R² sebesar **{format_indo_number(r_squared, decimal_places=4)}** mengindikasikan model mungkin masih belum sepenuhnya menangkap semua variasi kecil dalam data historis.")

    elif mape <= mape_threshold_good:
        st.info(f"**Hasil yang Baik!** Model Anda cukup akurat.")
        st.write(f"MAPE sebesar **{format_indo_number(mape, decimal_places=2)}%** menunjukkan bahwa model Anda memprediksi dengan persentase kesalahan yang baik.")
        st.write(f"RMSE sekitar **{format_indo_number(rmse)} jiwa** menunjukkan tingkat kesalahan absolut rata-rata. Secara keseluruhan, model Anda terlihat **valid** dan memiliki kesalahan yang **kecil**.")
        # --- Tambahan interpretasi R^2 untuk hasil Baik ---
        if r_squared > 0.75: # Contoh ambang batas R^2 yang baik (75% ke atas)
            st.write(f"Nilai R² sebesar **{format_indo_number(r_squared, decimal_places=4)}** (yaitu sekitar **{format_indo_number(r_squared*100, decimal_places=2)}%**) menunjukkan bahwa model Anda mampu menjelaskan sebagian besar variasi dalam data historis.")
        elif r_squared > 0.5: # Ambang batas R^2 yang moderat
            st.write(f"Nilai R² sebesar **{format_indo_number(r_squared, decimal_places=4)}** (yaitu sekitar **{format_indo_number(r_squared*100, decimal_places=2)}%**) menunjukkan model menjelaskan variasi data pada tingkat moderat. Model mungkin bisa ditingkatkan.")
        else: # Jika R^2 relatif rendah meskipun MAPE/RMSE lumayan
            st.write(f"Namun, perhatikan nilai R² yang relatif rendah sebesar **{format_indo_number(r_squared, decimal_places=4)}**. Ini mengindikasikan model belum sepenuhnya menjelaskan variasi data. Pertimbangkan untuk meninjau data Anda.")

    else: # MAPE > mape_threshold_good (Hasil Perlu Tinjauan Lebih Lanjut)
        st.warning(f"**Perhatian: Hasil Mungkin Perlu Tinjauan Lebih Lanjut.** Model mungkin belum optimal.")
        st.write(f"MAPE sebesar **{format_indo_number(mape, decimal_places=2)}%** dan RMSE sekitar **{format_indo_number(rmse)} jiwa** menunjukkan bahwa model mungkin mengalami tantangan dalam mencocokkan data historis secara akurat.")
        # --- Tambahan interpretasi R^2 untuk hasil Perlu Tinjauan ---
        if r_squared < 0.5: # Contoh ambang batas R^2 yang buruk (<50%)
            st.write(f"Nilai R² yang rendah atau bahkan negatif sebesar **{format_indo_number(r_squared, decimal_places=4)}** mengindikasikan bahwa model ini **tidak mampu menjelaskan variasi populasi historis dengan baik**. Kinerjanya mungkin tidak lebih baik dari sekadar menggunakan rata-rata populasi sebagai prediksi.")
        else: # R^2 mungkin moderat/baik tapi MAPE/RMSE buruk (kasus jarang, bisa karena outlier besar)
            st.write(f"Meskipun R² sebesar **{format_indo_number(r_squared, decimal_places=4)}** menunjukkan adanya penjelasan variasi, nilai RMSE dan MAPE yang tinggi menandakan model masih sering meleset secara absolut atau persentase.")
        
        st.write(f"Ini bisa terjadi karena beberapa alasan:")
        st.markdown("""
        * Data historis mungkin tidak sepenuhnya mengikuti pola pertumbuhan logistik.
        * Ada outlier (nilai data ekstrem) atau anomali yang signifikan dalam data.
        * Jumlah data terlalu sedikit atau tidak cukup bervariasi untuk menangkap pola dengan baik.
        * Nilai 'K' manual yang dipilih mungkin tidak sesuai (jika digunakan).
        * **Saran:** Pertimbangkan untuk meninjau kembali kualitas data, apakah data benar-benar mengikuti pola 'S', atau coba sesuaikan nilai 'K' jika Anda menggunakan mode manual.
        """)

    st.subheader("📈 Kurva Model & Prediksi")
    st.markdown("Visualisasi data historis dan kurva model logistik, termasuk proyeksi ke masa depan.")

    # --- AWAL BAGIAN KODE PLOTLY YANG BARU ---
    future_years = st.slider("Prediksi ke depan (tahun)", 0, 100, 20, 5, help="Atur berapa tahun ke depan kurva prediksi akan ditampilkan.")
    
    # Buat DataFrame untuk data historis
    df_history_for_plot = df_filtered.copy()

    # Buat DataFrame untuk data model/prediksi
    years_for_pred_plot = df_filtered[col_tahun].min() + np.linspace(0, t_data.max() + future_years, 500)
    t_plot_normalized = np.linspace(0, t_data.max() + future_years, 500) 
    N_pred_plot = logistic_model(t_plot_normalized, N0_fit, r_fit, K_fit)
    df_model_for_plot = pd.DataFrame({col_tahun: years_for_pred_plot, 'Populasi Prediksi': N_pred_plot})

    # Inisialisasi figur Plotly
    fig = go.Figure()

    # 1. Tambahkan Data Historis (titik biru)
    fig.add_trace(go.Scatter(
        x=df_history_for_plot[col_tahun],
        y=df_history_for_plot[col_populasi],
        mode='markers', 
        name='Data Historis', 
        marker=dict(color='blue', size=8, opacity=0.7),
        hoverinfo='text', 
        hovertext=df_history_for_plot.apply(lambda row: f"Tahun: {int(row[col_tahun])}<br>Populasi: {format_indo_number(row[col_populasi])} jiwa", axis=1)
    ))

    # 2. Tambahkan Garis Model & Prediksi (garis putus-putus merah)
    fig.add_trace(go.Scatter(
        x=df_model_for_plot[col_tahun],
        y=df_model_for_plot['Populasi Prediksi'],
        mode='lines', 
        name='Model & Prediksi', 
        line=dict(color='red', dash='dash', width=2)
    ))

    # 3. Tambahkan Garis Daya Tampung (K) (garis putus-putus hijau) sebagai trace terpisah
    fig.add_trace(go.Scatter(
        x=[years_for_pred_plot.min(), years_for_pred_plot.max()], 
        y=[K_fit, K_fit], 
        mode='lines',
        name=f'Daya Tampung (K) ({format_indo_number(K_fit)})', # Nama untuk legend, sertakan juga nilai K
        line=dict(color='green', dash='dot', width=2),
        showlegend=True, 
        hoverinfo='skip' 
    ))

    # Perbarui layout figur (judul, label sumbu, legend, dll.)
    fig.update_layout(
        title="Model Pertumbuhan Populasi Logistik",
        title_font_size=20,
        xaxis_title="Tahun",
        yaxis_title="Populasi",
        xaxis_title_font_size=14,
        yaxis_title_font_size=14,
        legend=dict(
            orientation="h", 
            yanchor="bottom",
            y=1.02, 
            xanchor="right",
            x=1
        ),
        legend_font_size=12,
        hovermode="x unified" 
    )

    # Tampilkan plot menggunakan Streamlit
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📌 Interpretasi Singkat")
    st.write(f"Dengan kapasitas daya dukung (K) $\\approx$ {format_indo_number(K_fit)}, populasi diperkirakan akan mendekati titik jenuh di level tersebut dalam jangka panjang.")

    st.divider()

    st.header('🗓 Prediksi Populasi di Tahun Spesifik')
    st.markdown("Masukkan tahun spesifik untuk mendapatkan estimasi populasi berdasarkan model.")

    tahun_awal_data = df_filtered[col_tahun].min()
    tahun_akhir_data = df_filtered[col_tahun].max()

    tahun_prediksi_min_input = int(tahun_akhir_data)
    tahun_prediksi_max_input = int(tahun_akhir_data + 50)

    col_input_tahun, col_hasil_prediksi = st.columns([1, 2])

    with col_input_tahun:
        tahun_input_user = st.number_input(
            "Masukkan Tahun untuk Prediksi:",
            min_value=int(tahun_awal_data),
            max_value=int(tahun_prediksi_max_input),
            value=int(tahun_akhir_data + 5),
            step=1,
            help=f"Masukkan tahun antara {tahun_prediksi_min_input} dan {tahun_prediksi_max_input}."
        )

    t_prediksi_spesifik = tahun_input_user - tahun_awal_data
    populasi_prediksi_spesifik = logistic_model(t_prediksi_spesifik, N0_fit, r_fit, K_fit)

    with col_hasil_prediksi:
        st.write(f"Prediksi Populasi pada Tahun {int(tahun_input_user)}:")
        st.success(f"{format_indo_number(populasi_prediksi_spesifik)} jiwa")

    if populasi_prediksi_spesifik > K_fit * 1.05:
        st.warning("Catatan: Prediksi ini melebihi daya tampung (K) secara signifikan. Model logistik menunjukkan populasi akan mendekati K dalam jangka panjang.")
    elif populasi_prediksi_spesifik > K_fit:
        st.info("Catatan: Prediksi ini sedikit di atas daya tampung (K), hal ini wajar karena model logistik adalah aproksimasi.")


st.write("---") # Garis pemisah untuk estetika
st.markdown(
    """
    <div style="text-align: center; font-size: 0.8em; color: #808080;">
        &copy; Mei Lammi Malau - 2025
        <br>
        Dibuat sebagai proyek Persamaan Diferensial.
    </div>
    """,
    unsafe_allow_html=True
)
