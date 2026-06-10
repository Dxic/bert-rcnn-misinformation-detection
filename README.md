# Peningkatan Deteksi Misinformasi Multi-Kelas Menggunakan BERT-Augmented RCNN dan Focal Loss

## 1. Informasi Tugas

| Informasi | Keterangan |
|---|---|
| Mata Kuliah | Data Mining |
| Dosen Pengampu | Galih Mahalisa |
| Program Studi | Teknik Informatika |
| Tahun Akademik | 2025/2026 |
| Jenis Tugas | Computer Science Research Analysis & Improvement |
| Bentuk Pengerjaan | Kelompok |

### Anggota Kelompok

| No. | Nama | NPM |
|---:|---|---|
| 1 | Dicky Nugraha Febriano | 2410010146 |
| 2 | **Marchella Diva Gantari** | **2410010642** |
| 3 | **Nor Indah Sari** | **2410010427** |
| 4 | **Rasyid** | **2410010032** |



---

## 2. Ringkasan Proyek

Proyek ini merupakan eksperimen *research improvement* dari paper:

> V.-I. Ilie, C.-O. Truică, E.-S. Apostol, dan A. Paschke,  
> “Context-Aware Misinformation Detection: A Benchmark of Deep Learning Architectures Using Word Embeddings,”  
> *IEEE Access*, vol. 9, 2021.  
> DOI: `10.1109/ACCESS.2021.3132502`

Paper acuan membandingkan beberapa arsitektur *deep learning* dan *word embedding* untuk mendeteksi misinformasi multi-kelas. Konfigurasi terbaik paper memperoleh akurasi **87,56%** menggunakan RCNN dengan *Specific Word2Vec* dan *Lemma Text Preprocessing*.

Eksperimen kelompok ini berfokus pada dua perbaikan:

1. Mengganti *static embedding* Word2Vec menjadi *contextual embedding* BERT.
2. Menguji Focal Loss pada kondisi data seimbang dan tidak seimbang.

Eksperimen menunjukkan bahwa **BERT-RCNN dengan Cross Entropy** menjadi konfigurasi lokal terbaik. Focal Loss tidak meningkatkan seluruh metrik, tetapi pada kondisi *class imbalance* meningkatkan **Macro Recall** dan **G-Mean**. Dengan demikian, hasil Focal Loss menunjukkan adanya *trade-off* antara performa global dan keseimbangan sensitivitas antarkelas.

---

## 3. Dataset

### 3.1 Sumber Dataset

Dataset mentah berasal dari **FakeNewsCorpus**:

```text
https://github.com/several27/FakeNewsCorpus/releases/tag/v1.0
```

Dataset mentah tidak disertakan di folder pengumpulan karena ukurannya sangat besar. Corpus mentah diekstrak dari arsip multi-part dan diproses menggunakan kolom:

| Kolom | Fungsi |
|---|---|
| `content` | Isi artikel berita |
| `type` | Label kelas artikel |

### 3.2 Dataset Balanced

Dataset balanced digunakan untuk replikasi terbatas baseline dan eksperimen utama.

| Keterangan | Nilai |
|---|---:|
| Total artikel | 100.000 |
| Jumlah kelas | 10 |
| Artikel per kelas | 10.000 |
| Train | 72.000 |
| Validation | 8.000 |
| Test | 20.000 |

Distribusi setiap kelas setelah *stratified split*:

| Split | Jumlah per kelas |
|---|---:|
| Train | 7.200 |
| Validation | 800 |
| Test | 2.000 |

Daftar kelas:

```text
fake
satire
bias
conspiracy
junksci
hate
clickbait
unreliable
political
reliable
```

### 3.3 Controlled Imbalanced Stress-Test Dataset

Dataset tambahan dibuat secara terkontrol untuk menguji efek Focal Loss pada kondisi tidak seimbang. Dataset ini **bukan** dataset original paper.

| Kelas | Jumlah Artikel |
|---|---:|
| reliable | 10.000 |
| clickbait | 10.000 |
| political | 8.000 |
| unreliable | 6.000 |
| bias | 4.000 |
| fake | 3.000 |
| conspiracy | 2.000 |
| satire | 1.500 |
| junksci | 1.000 |
| hate | 1.000 |
| **Total** | **46.500** |

Rasio kelas terbesar dan terkecil adalah **10:1**.

---

## 4. Lingkungan Eksperimen

| Komponen | Keterangan |
|---|---|
| Platform | Google Colaboratory |
| GPU | NVIDIA Tesla T4 |
| Framework | PyTorch |
| Library NLP | Hugging Face Transformers |
| Model BERT | `bert-base-uncased` |
| Random seed | `42` |

### 4.1 Konfigurasi Baseline

| Parameter | Nilai |
|---|---:|
| Model | RCNN + Specific Word2Vec |
| Loss | Cross Entropy |
| Vocabulary size | 100.000 |
| Dimensi Word2Vec | 300 |
| Window Word2Vec | 5 |
| Epoch Word2Vec | 5 |
| Hidden size RCNN | 256 |
| Dropout | 0,5 |
| Learning rate | `1e-4` |
| Weight decay | `5e-4` |
| Batch size | 8 |
| Maksimum epoch | 10 |
| Early stopping | Berdasarkan validation F1-Macro |

### 4.2 Konfigurasi BERT-RCNN

| Parameter | Nilai |
|---|---:|
| Model dasar | `bert-base-uncased` |
| Freeze lower BERT layers | 6 layer |
| BERT learning rate | `2e-5` |
| Head learning rate | `1e-3` |
| Warmup ratio | 10% |
| Dropout | 0,5 |
| Maximum sequence length | 64 token |
| Batch size | 4 |
| Gradient accumulation | 4 |
| Epoch | 3 |
| Focal Loss gamma | 2,0 |
| Alpha Focal Loss | Inverse frequency kelas |

> Sequence length 64 dipilih karena keterbatasan resource GPU Colab T4 dan kebutuhan efisiensi waktu.

---

## 5. Struktur Source Code

```text
bert_rcnn_original_experiment/
├── main.py               # Entry point eksperimen
├── config.py             # Konfigurasi label dan hyperparameter
├── data_loader.py        # Load CSV, filtering, split, class alpha
├── dataset.py            # PyTorch Dataset untuk baseline dan BERT
├── focal_loss.py         # Implementasi Focal Loss multi-kelas
├── models.py             # BaselineRCNN dan BERTAugmentedRCNN
├── trainer.py            # Training loop, early stopping, checkpoint
├── evaluate.py           # Accuracy, precision, recall, F1, G-Mean
├── experiments.py        # Skenario A, B1, B2, C1, dan C2
├── logger_excel.py       # Logging otomatis ke Excel
├── requirements.txt      # Dependency Python
└── checkpoints/          # Checkpoint model terbaik
```

---

## 6. Skenario Eksperimen

| ID | Dataset | Model | Loss | Tujuan |
|---|---|---|---|---|
| A | Balanced 100k | RCNN + Specific Word2Vec | Cross Entropy | Replikasi terbatas baseline |
| B1 | Balanced 100k | BERT-RCNN | Cross Entropy | Mengukur dampak BERT |
| B2 | Balanced 100k | BERT-RCNN | Focal Loss | Mengukur dampak Focal Loss pada data balanced |
| C1 | Imbalanced controlled | BERT-RCNN | Cross Entropy | Pembanding loss pada data imbalanced |
| C2 | Imbalanced controlled | BERT-RCNN | Focal Loss | Mengukur dampak Focal Loss pada data imbalanced |

---

## 7. Hasil Eksperimen

### 7.1 Ringkasan Hasil

| ID | Dataset | Model | Loss | Accuracy | Macro Precision | Macro Recall | F1-Macro | F1-Weighted | G-Mean | Training Time |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| A | Balanced 100k | RCNN + Specific Word2Vec | Cross Entropy | 71,94% | 79,66% | 71,94% | 74,03% | 74,03% | 71,19% | 1.801,49 detik |
| B1 | Balanced 100k | BERT-RCNN | Cross Entropy | **83,41%** | **84,15%** | **83,41%** | **83,61%** | **83,61%** | **82,98%** | 2.728,86 detik |
| B2 | Balanced 100k | BERT-RCNN | Focal Loss | 82,94% | 83,84% | 82,94% | 83,17% | 83,17% | 82,46% | 2.912,89 detik |
| C1 | Imbalanced 46.500 | BERT-RCNN | Cross Entropy | **82,69%** | **84,29%** | 77,34% | **80,21%** | **82,69%** | 75,58% | 1.282,50 detik |
| C2 | Imbalanced 46.500 | BERT-RCNN | Focal Loss | 77,60% | 74,09% | **78,74%** | 75,08% | 78,33% | **78,13%** | 1.322,57 detik |

### 7.2 Interpretasi Singkat

Pada dataset balanced, BERT-RCNN dengan Cross Entropy meningkatkan accuracy sebesar **11,47 poin persentase** dibanding baseline lokal, dari 71,94% menjadi 83,41%. F1-Macro meningkat sebesar **9,58 poin persentase**, sedangkan G-Mean meningkat sebesar **11,79 poin persentase**.

Pada dataset balanced, Focal Loss tidak memberikan peningkatan karena distribusi kelas telah dibuat sama. B2 sedikit lebih rendah dibanding B1.

Pada dataset imbalanced, Focal Loss meningkatkan Macro Recall dari 77,34% menjadi 78,74% dan G-Mean dari 75,58% menjadi 78,13%. Namun, accuracy, precision, dan F1-Macro mengalami penurunan. Hasil ini menunjukkan adanya *trade-off*.

Model lokal terbaik adalah:

```text
BERT-RCNN + Cross Entropy
```

dengan:

```text
Accuracy : 83,41%
F1-Macro : 83,61%
G-Mean   : 82,98%
```

> Hasil baseline A didokumentasikan melalui output dan screenshot Google Colab. Hasil B1, B2, C1, dan C2 tercatat pada file `hasil_eksperimen.xlsx`.

---

## 8. Cara Menjalankan Project

## Dataset Eksperimen

Dataset mentah berasal dari FakeNewsCorpus:

https://github.com/several27/FakeNewsCorpus/releases/tag/v1.0

Untuk memudahkan reproduksi eksperimen, subset dataset yang sudah diproses tersedia pada Google Drive:

[Download dataset eksperimen](https://drive.google.com/drive/folders/1CpIXxqQMgjPsIL1OtLKBwX-Aw2VLDz0M?usp=sharing)

Isi folder:

| File | Keterangan | Ukuran |
|---|---|---:|
| `fnc_balanced_100k.csv` | Dataset balanced untuk eksperimen A, B1, dan B2 | ±336,8 MB |
| `fnc_imbalanced_controlled.csv` | Dataset controlled imbalanced untuk eksperimen C1 dan C2 | ±157,5 MB |

Letakkan file pada struktur berikut di Google Drive:

```text
MyDrive/
└── CS_RI_2026/
    └── data/
        ├── fnc_balanced_100k.csv
        └── fnc_imbalanced_controlled.csv
### 8.1 Persiapan Dataset

Letakkan file berikut pada Google Drive:

```text
MyDrive/
└── CS_RI_2026/
    └── data/
        ├── fnc_balanced_100k.csv
        └── fnc_imbalanced_controlled.csv
```

File `fnc_imbalanced_controlled.csv` hanya diperlukan untuk eksperimen C1 dan C2.

### 8.2 Mount Google Drive

```python
from google.colab import drive
drive.mount("/content/drive")
```

### 8.3 Upload dan Extract Source Code

```python
from google.colab import files
uploaded = files.upload()
```

Upload ZIP source code, lalu extract:

```bash
%cd /content
!unzip -o bert_rcnn_original_experiment_final.zip
%cd /content/bert_rcnn_original_experiment
```

### 8.4 Install Dependencies

```bash
!pip install -q -r requirements.txt
!pip install -q --upgrade "transformers==4.44.0"
```

### 8.5 Definisikan Path Dataset

```python
from main import run_colab

BALANCED_CSV = (
    "/content/drive/MyDrive/"
    "CS_RI_2026/data/fnc_balanced_100k.csv"
)

IMBALANCED_CSV = (
    "/content/drive/MyDrive/"
    "CS_RI_2026/data/fnc_imbalanced_controlled.csv"
)
```

### 8.6 Jalankan Eksperimen A

```python
result_a = run_colab(
    csv_path=BALANCED_CSV,
    experiment="baseline_rcnn_w2v_ce",
    epochs=10,
    batch_size=8,
    seed=42,
)
```

### 8.7 Jalankan Eksperimen B1

```python
result_b1 = run_colab(
    csv_path=BALANCED_CSV,
    experiment="bert_rcnn_ce",
    epochs=3,
    batch_size=4,
    max_seq_len=64,
    grad_accum_steps=4,
    seed=42,
)
```

### 8.8 Jalankan Eksperimen B2

```python
result_b2 = run_colab(
    csv_path=BALANCED_CSV,
    experiment="bert_rcnn_focal",
    epochs=3,
    batch_size=4,
    max_seq_len=64,
    grad_accum_steps=4,
    focal_gamma=2.0,
    seed=42,
)
```

### 8.9 Jalankan Eksperimen C1

```python
result_c1 = run_colab(
    csv_path=IMBALANCED_CSV,
    experiment="bert_rcnn_ce_imbalanced",
    epochs=3,
    batch_size=4,
    max_seq_len=64,
    grad_accum_steps=4,
    seed=42,
)
```

### 8.10 Jalankan Eksperimen C2

```python
result_c2 = run_colab(
    csv_path=IMBALANCED_CSV,
    experiment="bert_rcnn_focal_imbalanced",
    epochs=3,
    batch_size=4,
    max_seq_len=64,
    grad_accum_steps=4,
    focal_gamma=2.0,
    seed=42,
)
```

---

## 9. Output Program

Setelah eksperimen dijalankan, program menghasilkan:

```text
hasil_eksperimen.xlsx
training.log
checkpoints/
```

| Output | Fungsi |
|---|---|
| `hasil_eksperimen.xlsx` | Rekap metrik eksperimen |
| `training.log` | Log training dan validation setiap epoch |
| `checkpoints/` | Model terbaik berdasarkan validation F1-Macro |

---

## 10. Keterbatasan Eksperimen

Eksperimen ini masih memiliki beberapa keterbatasan yang dapat menjadi ruang pengembangan lanjutan:

1. Replikasi baseline dilakukan secara terbatas sehingga hasilnya belum identik dengan paper acuan.
2. Subset artikel lokal tidak dapat dipastikan sama persis dengan subset yang digunakan pada paper.
3. Preprocessing lokal dibuat lebih ringan dibandingkan konfigurasi preprocessing pada paper.
4. Sequence length BERT dibatasi menjadi 64 token untuk menyesuaikan keterbatasan resource Google Colab T4.
5. Focal Loss baru diuji menggunakan nilai gamma 2,0.
6. Alpha dihitung menggunakan inverse frequency tanpa tuning lanjutan.
7. Eksperimen belum dijalankan menggunakan beberapa random seed.
8. Analisis confusion matrix per kelas belum dilakukan.
9. Uji statistik seperti McNemar Test belum dilakukan.

## 11. Saran Pengembangan

Pengembangan berikutnya dapat menguji:

1. Nilai gamma Focal Loss lain.
2. Strategi alpha yang berbeda.
3. Weighted Cross Entropy.
4. Oversampling kelas minoritas.
5. Sequence length 128 atau 256 token.
6. Model RoBERTa, DistilBERT, atau domain-specific BERT.
7. Repeated runs dengan beberapa random seed.
8. Confusion matrix dan error analysis per kelas.
9. Uji statistik untuk mengukur signifikansi perbedaan model.

---

## 12. Deklarasi Penggunaan AI

Dalam proses pengerjaan tugas kelompok ini, AI generatif digunakan sebagai alat bantu untuk memahami dokumentasi teknis, melakukan brainstorming desain eksperimen, menyusun panduan eksekusi Google Colaboratory, membantu debugging kode, serta merapikan struktur draf laporan. Seluruh eksperimen dijalankan secara langsung oleh kelompok pada Google Colaboratory. Verifikasi data, interpretasi hasil, revisi isi laporan, dan tanggung jawab akhir atas laporan tetap dilakukan oleh seluruh anggota kelompok.



## 14. Tautan Tambahan

| Keterangan | Tautan |
|---|---|
| Dataset FakeNewsCorpus | `https://github.com/several27/FakeNewsCorpus/releases/tag/v1.0` |
| Paper utama | DOI: `10.1109/ACCESS.2021.3132502` |
| Repository source code kelompok | **https://github.com/Dxic/bert-rcnn-misinformation-detection** |
