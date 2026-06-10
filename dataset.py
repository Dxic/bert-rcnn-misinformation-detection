from __future__ import annotations

import re
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def basic_tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9']+", str(text).lower())


def build_vocab(
    texts: Iterable[str],
    max_vocab_size: int = 100_000,
    min_word_freq: int = 2,
) -> Dict[str, int]:
    counter: Counter = Counter()

    for text in texts:
        counter.update(basic_tokenize(text))

    vocab = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1,
    }

    for token, freq in counter.most_common(max_vocab_size - len(vocab)):
        if freq < min_word_freq:
            break
        vocab[token] = len(vocab)

    return vocab


class FakeNewsDatasetBaseline(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        vocab: Dict[str, int],
        max_words: int = 512,
    ) -> None:
        self.texts = df["text"].astype(str).tolist()
        self.labels = df["label"].astype(int).tolist()
        self.vocab = vocab
        self.max_words = max_words

    def __len__(self) -> int:
        return len(self.labels)

    def encode(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = basic_tokenize(text)[: self.max_words]
        ids = [self.vocab.get(token, self.vocab[UNK_TOKEN]) for token in tokens]
        length = max(1, len(ids))

        if not ids:
            ids = [self.vocab[UNK_TOKEN]]

        ids = ids[: self.max_words]
        padded = ids + [self.vocab[PAD_TOKEN]] * (self.max_words - len(ids))

        return (
            torch.tensor(padded, dtype=torch.long),
            torch.tensor(min(length, self.max_words), dtype=torch.long),
        )

    def __getitem__(self, index: int):
        input_ids, length = self.encode(self.texts[index])

        return {
            "input_ids": input_ids,
            "length": length,
            "labels": torch.tensor(self.labels[index], dtype=torch.long),
        }


class FakeNewsDatasetBERT(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        model_name: str = "bert-base-uncased",
        max_seq_len: int = 128,
        tokenizer: Optional[AutoTokenizer] = None,
    ) -> None:
        self.texts = df["text"].astype(str).tolist()
        self.labels = df["label"].astype(int).tolist()
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        encoded = self.tokenizer(
            self.texts[index],
            truncation=True,
            padding="max_length",
            max_length=self.max_seq_len,
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[index], dtype=torch.long),
        }


def create_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 2,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
