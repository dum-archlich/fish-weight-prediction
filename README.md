# Prediksi Berat Ikan

Proyek ini menggunakan machine learning untuk memprediksi berat ikan berdasarkan ukuran fisiknya menggunakan model regresi linear.

## Deskripsi Proyek

Aplikasi ini menganalisis dataset ikan dan membangun model prediktif untuk memperkirakan berat ikan dari pengukuran seperti panjang, tinggi, dan lebar.

## Fitur

- Analisis atribut data ikan
- Pelatihan model regresi linear
- Evaluasi performa model
- Visualisasi data dan hasil prediksi
- Output tabel dan grafik yang dapat disimpan

## Persyaratan Sistem

- Python 3.7+
- Library yang diperlukan tercantum dalam `requirements.txt`

## Instalasi

1. Clone repositori ini
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Penggunaan

Jalankan script utama:
```bash
python fish_predictor.py
```

Script akan:
1. Memuat data dari `fishers_maket.csv`
2. Menampilkan analisis atribut dalam format tabel
3. Melatih model regresi linear
4. Menampilkan koefisien model dalam bentuk tabel dan grafik batang
5. Mengevaluasi model dan menampilkan metrik performa
6. Membuat visualisasi data dan hasil prediksi

## Output

- Tabel analisis data (console)
- Histogram distribusi atribut (fish_attributes_histogram.png)
- Grafik batang koefisien (coefficients_bar_chart.png)
- Scatter plot aktual vs prediksi (actual_vs_predicted.png)

## Struktur Proyek

```
fish-weight-prediction/
├── fish_predictor.py      # Script utama
├── fishers_maket.csv      # Dataset ikan
├── requirements.txt       # Dependencies
├── README.md              # Dokumentasi ini
└── .gitignore            # File yang diabaikan Git
```

## Metrik Evaluasi

Model dievaluasi menggunakan:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R-squared (R²)

## Lisensi

Proyek ini dibuat untuk tujuan edukasi.