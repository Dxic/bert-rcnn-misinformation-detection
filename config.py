from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any
import json

LABEL_TO_IDX = {
    "fake": 0,
    "satire": 1,
    "bias": 2,
    "conspiracy": 3,
    "junksci": 4,
    "hate": 5,
    "clickbait": 6,
    "unreliable": 7,
    "political": 8,
    "reliable": 9,
}

IDX_TO_LABEL = {idx: label for label, idx in LABEL_TO_IDX.items()}
NUM_CLASSES = len(LABEL_TO_IDX)


@dataclass
class ExperimentConfig:
    csv_path: str
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"
    log_excel_path: str = "hasil_eksperimen.xlsx"
    training_log_path: str = "training.log"

    # Data
    seed: int = 42
    test_size: float = 0.20
    val_size_from_train: float = 0.10
    text_column: str = "content"
    label_column: str = "type"

    # General training
    epochs: int = 5
    batch_size: int = 8
    num_workers: int = 2
    early_stopping_patience: int = 3
    grad_accum_steps: int = 1

    # Baseline RCNN + Word2Vec
    max_vocab_size: int = 100_000
    min_word_freq: int = 2
    max_words_baseline: int = 512
    word2vec_dim: int = 300
    word2vec_window: int = 5
    word2vec_epochs: int = 5
    word2vec_workers: int = 4
    baseline_hidden_size: int = 256
    baseline_dropout: float = 0.5
    baseline_lr: float = 1e-4
    baseline_weight_decay: float = 5e-4

    # BERT-RCNN
    bert_model_name: str = "bert-base-uncased"
    max_seq_len: int = 128
    bert_hidden_size: int = 128
    bert_dropout: float = 0.5
    freeze_bert_layers: int = 6
    bert_lr: float = 2e-5
    head_lr: float = 1e-3
    bert_weight_decay: float = 0.01
    warmup_ratio: float = 0.10

    # Focal Loss
    focal_gamma: float = 2.0

    def ensure_dirs(self) -> None:
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_json(self, path: str) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2),
            encoding="utf-8",
        )
