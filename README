# Fundamen Sains Data
## Implementasi Linear Regression dan Logistic Regression

Repository ini berisi implementasi dua algoritma *Machine Learning* menggunakan Python untuk memenuhi tugas mata kuliah **Fundamen Sains Data**. Algoritma yang digunakan adalah Linear Regression untuk memprediksi harga rumah dan Logistic Regression untuk mengklasifikasikan pesan SMS sebagai spam atau bukan spam.

## Anggota Kelompok

| No. | Nama | NIM |
|---|---|---|
| 1 | HANAFI RAIHAN FIRMANSYAH PUTRA | 25523052 |
| 2 | PURNAMA ANNISA | 25523279 |
| 3 | CHEALSEA | ...... |

## Struktur Repository

```text
fundamen-sains-data/
├── README.md
├── A_linear_regression.ipynb
├── B_logistic_regression.ipynb
└── data/
    └── dataset-harga-rumah-yogyakarta.json
```

Dataset SMS Spam Collection perlu disediakan sesuai sumbernya atau diunduh saat menjalankan notebook. Pastikan file dataset tersedia dengan nama yang sesuai dengan kode pada notebook.

## A. Linear Regression — Prediksi Harga Rumah Yogyakarta

### Deskripsi

Bagian ini menggunakan algoritma Linear Regression untuk memprediksi harga rumah berdasarkan karakteristik properti di Yogyakarta.

- **Sumber dataset:** [Dataset Harga Rumah Yogyakarta — Kaggle](https://www.kaggle.com/datasets/jasaaplikasiskripsi/dataset-harga-rumah-yogyakarta)
- **Format data awal:** JSON
- **Jumlah data awal:** 562 baris
- **Jumlah data setelah preprocessing:** 437 baris
- **Variabel target:** `price`
- **Fitur prediktor:**
  - `Luas_bangunan`
  - `Luas_tanah`
  - `Kamar_tidur`
  - `Kamar_Mandi`

### Metodologi

Tahapan yang dilakukan meliputi:

1. Memuat dan memahami struktur dataset.
2. Membersihkan nilai harga serta mengubah fitur numerik ke tipe data yang sesuai.
3. Menghapus data kosong dan duplikat serta menangani pencilan harga menggunakan metode Interquartile Range (IQR).
4. Melakukan Exploratory Data Analysis (EDA) melalui histogram harga, heatmap korelasi, dan scatter plot luas bangunan terhadap harga.
5. Membagi data menjadi 80% data latih dan 20% data uji menggunakan `random_state=42`.
6. Melatih model Linear Regression menggunakan Scikit-learn.
7. Mengevaluasi model menggunakan MAE, MSE, RMSE, dan R², serta visualisasi nilai aktual terhadap prediksi dan residual.

### Hasil Evaluasi

Hasil evaluasi model pada notebook:

| Metrik | Hasil |
|---|---:|
| MAE | Rp876.842.176,97 |
| MSE | 8.563.642.327.789.814.784,00 |
| RMSE | Rp2.926.370.162,47 |
| R² data latih | 0,4435 |
| R² data uji | -8,2904 |

### Analisis

Nilai R² pada data latih menunjukkan bahwa model dapat menjelaskan sebagian variasi harga rumah. Namun, nilai R² negatif pada data uji menunjukkan bahwa model belum mampu memprediksi data baru dengan baik. Perbedaan performa antara data latih dan data uji mengindikasikan bahwa kemampuan generalisasi model masih rendah.

Performa model dapat dipengaruhi oleh pencilan, karakteristik properti yang beragam, hubungan fitur dengan harga yang tidak sepenuhnya linear, serta faktor lain yang belum dimasukkan, seperti lokasi dan kondisi bangunan. Model ini masih memerlukan evaluasi dan pengembangan lebih lanjut.

## B. Logistic Regression — Klasifikasi SMS Spam

### Deskripsi

Bagian ini menggunakan Logistic Regression untuk mengklasifikasikan pesan SMS ke dalam dua kelas, yaitu pesan normal (*ham*) dan pesan spam.

- **Sumber dataset:** [SMS Spam Collection Dataset — Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- **Jumlah data awal:** 5.572 baris
- **Jumlah data setelah preprocessing:** 5.169 baris
- **Variabel target:** `label`
- **Fitur teks:** `text`
- **Label kelas:**
  - `ham` (0): pesan normal atau bukan spam.
  - `spam` (1): pesan spam, seperti penipuan atau promosi yang tidak diinginkan.

### Metodologi

Tahapan yang dilakukan meliputi:

1. Memuat dataset dan memahami distribusi label.
2. Melakukan EDA melalui visualisasi distribusi kelas, panjang karakter pesan, dan jumlah kata.
3. Mengubah label `ham` dan `spam` menjadi nilai numerik 0 dan 1.
4. Menghapus data kosong dan duplikat.
5. Mengubah teks menjadi representasi numerik menggunakan TF-IDF dengan maksimal 1.000 fitur dan mengabaikan *stop words* bahasa Inggris.
6. Membagi data menjadi 80% data latih dan 20% data uji dengan `stratify=y` dan `random_state=42`.
7. Melatih Logistic Regression menggunakan `class_weight='balanced'` untuk membantu menangani ketidakseimbangan kelas.
8. Mengevaluasi model menggunakan accuracy, precision, recall, F1-score, confusion matrix, serta kurva ROC dan AUC.

### Hasil Evaluasi

Hasil evaluasi pada data uji di notebook:

| Metrik | Hasil |
|---|---:|
| Accuracy | 97,58% |
| Precision | 91,41% |
| Recall | 89,31% |
| F1-score | 90,35% |
| ROC-AUC | 0,9907 |

**Confusion matrix:**

| Aktual / Prediksi | Ham (0) | Spam (1) |
|---|---:|---:|
| Ham (0) | 892 | 11 |
| Spam (1) | 14 | 117 |

### Analisis

Model menghasilkan accuracy sebesar 97,58% dan ROC-AUC sebesar 0,9907 pada data uji. Hasil tersebut menunjukkan kemampuan klasifikasi yang baik pada dataset yang digunakan.

Dalam deteksi spam, precision dan recall sama-sama penting. Precision berkaitan dengan ketepatan prediksi spam, sedangkan recall menunjukkan seberapa banyak pesan spam yang berhasil ditemukan. Berdasarkan confusion matrix, terdapat 14 pesan spam yang masih diklasifikasikan sebagai pesan normal dan 11 pesan normal yang salah diklasifikasikan sebagai spam.

Model ini memiliki keterbatasan karena representasi TF-IDF dan Logistic Regression tidak sepenuhnya memahami konteks kalimat, perubahan bahasa, singkatan, atau pola spam baru.

## Teknologi yang Digunakan

- Python
- Google Colab / Jupyter Notebook
- Pandas dan NumPy
- Matplotlib dan Seaborn
- Scikit-learn
- TF-IDF Vectorizer
- Git dan GitHub

## Pembagian Tugas Anggota

| Anggota | Kontribusi |
|---|---|
| [Nama Anggota 1] | [Contoh: preprocessing data dan Linear Regression] |
| [Nama Anggota 2] | [Contoh: EDA dan Logistic Regression] |
| [Nama Anggota 3] | [Contoh: evaluasi model, dokumentasi, dan README] |

Sesuaikan tabel ini dengan kontribusi nyata setiap anggota kelompok.

## Cara Menjalankan

1. Clone repository:

   ```bash
   git clone [URL_REPOSITORY]
   cd fundamen-sains-data
   ```

2. Buka notebook menggunakan Google Colab atau Jupyter Notebook.
3. Siapkan dataset sesuai sumber dan nama file yang digunakan di dalam notebook.
4. Jalankan setiap sel notebook secara berurutan dari awal sampai akhir.

Pastikan seluruh sel dapat dijalankan tanpa error dan output evaluasi tersimpan sebelum notebook dikumpulkan.

## Catatan

Hasil yang dicantumkan di README ini mengacu pada output notebook saat ini. Perubahan preprocessing atau pemodelan dapat menghasilkan metrik yang berbeda.

Dataset SMS mentah memiliki satu kolom prediktor berupa teks. TF-IDF mengubah teks menjadi banyak fitur numerik, tetapi hal ini tidak otomatis berarti persyaratan minimal tiga fitur pada dataset mentah telah terpenuhi apabila persyaratan tersebut ditafsirkan berdasarkan kolom asli dataset. Hal ini perlu dipastikan sesuai ketentuan tugas.

---

**Mata Kuliah:** Fundamen Sains Data  
**Topik:** Linear Regression dan Logistic Regression  
**Bahasa Pemrograman:** Python
