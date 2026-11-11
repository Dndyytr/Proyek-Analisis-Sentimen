# Proyek Analisis Sentimen Ulasan Steam - Counter-Strike 2

**Nama:** Dandy Taufiqurrochman  
**NIM:** 7020230002  
**Program Studi:** Sistem Informasi  
**Perguruan Tinggi:** Universitas Galuh

## Deskripsi
Proyek ini mengembangkan model klasifikasi dua kelas untuk ulasan pengguna Steam pada game Counter-Strike 2 (App ID: 730). Data diperoleh melalui Steam `GetAppReviews` API menggunakan Python.

## Dataset
- Dataset mentah hasil scraping: **10.000 ulasan** (5.000 label positif dan 5.000 label negatif menurut kategori rekomendasi Steam).
- Dataset final untuk pemodelan: **3.518 ulasan**.
- Training: 2.462 ulasan; validation: 528; testing: 528.
- Kolom teks yang digunakan: `text`; target: `label`.
- Pembagian split tidak memiliki `review_id` yang saling tumpang tindih.

Dataset mentah tersedia di `data/raw/`. Dataset final dan pembagian training/validation/testing tersedia di `data/final/`.

## Definisi Label dan Penggunaan AI
Label `positif`/`negatif` berasal dari keputusan rekomendasi Steam (`voted_up`). Oleh karena itu, model memprediksi **apakah ulasan merekomendasikan game menurut keputusan pengguna Steam**, bukan polaritas sentimen teks yang telah diverifikasi oleh anotator manusia.

ChatGPT digunakan sebagai alat bantu penyusunan dan debugging kode serta anotasi tambahan untuk audit bahasa/kualitas data. Anotasi berbantuan AI tidak digunakan sebagai ground truth target model dan tidak diklaim sebagai anotasi manual manusia.

## Metode dan eksperimen
Tiga skema TF-IDF dibandingkan pada validation set:

| Skema | Fitur dan algoritma | Akurasi validation |
|---|---|---:|
| S1 | Word TF-IDF + LinearSVC | 84,09% |
| S2 | Character TF-IDF + LinearSVC | 83,14% |
| S3 | Hybrid Word-Character TF-IDF + Logistic Regression | 84,85% |

Eksperimen deep learning mencakup BERT base uncased, optimasi hyperparameter BERT, dan fine-tuning RoBERTa base. Model akhir menggunakan ensemble BERT fine-tuned dan RoBERTa base (bobot 50:50; threshold 0,585).

## Hasil evaluasi model akhir
- Validation accuracy: **89,20%**.
- Test accuracy: **85,04%** (449 benar dari 528 sampel).
- Test balanced accuracy: 81,39%.
- Test macro F1: 0,8200.
- Test weighted F1: 0,8488.
- Confusion matrix (urutan label negatif, positif): `[[333, 34], [45, 116]]`.

Sebagai pembanding, eksperimen S3 TF-IDF menghasilkan akurasi test 81,06%; BERT baseline 84,66%; BERT tuned 84,66%; dan RoBERTa 84,28%. Ensemble dipilih berdasarkan kombinasi bobot dan threshold yang dioptimalkan pada validation set.

**Catatan evaluasi:** Test set pernah digunakan untuk membandingkan beberapa kandidat model dalam rangkaian eksperimen. Karena itu, angka test 85,04% adalah hasil yang teramati pada rangkaian eksperimen ini dan bukan estimasi dari holdout yang sepenuhnya independen.

## Struktur folder
- `training.ipynb`: proses preprocessing final, eksperimen TF-IDF, training/evaluasi BERT dan RoBERTa, ensemble, dan inference dengan output tersimpan.
- `src/scraping.py`: script scraping Steam `GetAppReviews` API.
- `src/inference.py`: modul prediksi ensemble BERT-RoBERTa.
- `data/raw/`: dataset mentah 10.000 ulasan.
- `data/final/`: dataset final dan split data.
- `reports/`: hasil eksperimen, metrik, prediksi, konfigurasi ensemble, dan contoh inference.
- `requirements.txt`: dependensi Python yang digunakan.

## Cara menjalankan
1. Ekstrak folder `Proyek_Analisis_Sentimen` ke Google Drive `MyDrive`.
2. Buka `training.ipynb` menggunakan Google Colab.
3. Jalankan notebook secara berurutan. Untuk mengulang training transformer, pilih runtime GPU (misalnya T4).
4. Notebook akan menghubungkan Google Drive, membaca dataset final, menjalankan eksperimen, dan menyimpan checkpoint serta laporan pada folder proyek.
5. `src/inference.py` memerlukan checkpoint BERT di `models/bert_tuned_best`, checkpoint RoBERTa di `models/roberta_base_cs2`, dan konfigurasi di `reports/ensemble_bert_roberta/ensemble_best_config.json`. Checkpoint tidak disertakan di ZIP karena ukuran file model besar; checkpoint dihasilkan oleh sel training pada notebook.

Instal dependensi dengan `pip install -r requirements.txt` bila menjalankan di lingkungan Python sendiri. Untuk Colab, notebook juga memuat instalasi/pengecekan library transformer.

## Catatan terhadap saran opsional Dicoding
Proyek menerapkan deep learning dan inference. Dataset final yang digunakan training berjumlah 3.518 sampel dengan dua kelas; akurasi belum mencapai >92%. Tiga skema TF-IDF dibandingkan pada validation set, tetapi tidak seluruhnya mencapai akurasi test 85%. Hal ini dicantumkan agar capaian proyek tidak dilebih-lebihkan.
