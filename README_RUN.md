# Cara Menjalankan Paket Source Code

Paket ini sudah diubah dari skema Kaggle `Fake.csv + True.csv` menjadi Fake News Corpus original:

- input CSV: `fnc_balanced_100k.csv`
- kolom teks: `content`
- kolom label: `type`
- jumlah kelas: 10

## Mount Google Drive

```python
from google.colab import drive
drive.mount("/content/drive")
```

## Install dependencies

```bash
!pip install -q -r requirements.txt
```

## Jalankan baseline RCNN + Specific Word2Vec

```python
from main import run_colab

BALANCED_CSV = "/content/drive/MyDrive/CS_RI_2026/data/fnc_balanced_100k.csv"

run_colab(
    csv_path=BALANCED_CSV,
    experiment="baseline_rcnn_w2v_ce",
    epochs=10,
    batch_size=8,
    seed=42,
)
```

## Jalankan BERT-RCNN + Cross Entropy

```python
run_colab(
    csv_path=BALANCED_CSV,
    experiment="bert_rcnn_ce",
    epochs=3,
    batch_size=4,
    max_seq_len=128,
    grad_accum_steps=4,
    seed=42,
)
```

## Jalankan BERT-RCNN + Focal Loss

```python
run_colab(
    csv_path=BALANCED_CSV,
    experiment="bert_rcnn_focal",
    epochs=3,
    batch_size=4,
    max_seq_len=128,
    grad_accum_steps=4,
    focal_gamma=2.0,
    seed=42,
)
```

## Jalankan stress-test pada data imbalanced

```python
IMBALANCED_CSV = "/content/drive/MyDrive/CS_RI_2026/data/fnc_imbalanced_controlled.csv"

run_colab(
    csv_path=IMBALANCED_CSV,
    experiment="bert_rcnn_ce_imbalanced",
    epochs=3,
    batch_size=4,
    max_seq_len=128,
    grad_accum_steps=4,
    seed=42,
)

run_colab(
    csv_path=IMBALANCED_CSV,
    experiment="bert_rcnn_focal_imbalanced",
    epochs=3,
    batch_size=4,
    max_seq_len=128,
    grad_accum_steps=4,
    focal_gamma=2.0,
    seed=42,
)
```

## Output

Hasil otomatis disimpan ke:

- `hasil_eksperimen.xlsx`
- `training.log`
- folder `checkpoints/`
