from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import LABEL_TO_IDX, NUM_CLASSES


def normalize_text(text: str) -> str:
    """
    Pembersihan ringan agar cocok untuk BERT dan baseline.
    Untuk replikasi yang sangat ketat, tambahkan pipeline lemma preprocessing
    sesuai implementasi paper original.
    """
    text = str(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_fnc_dataset(
    csv_path: str,
    text_column: str = "content",
    label_column: str = "type",
) -> pd.DataFrame:
    df = pd.read_csv(
        csv_path,
        usecols=[text_column, label_column],
        low_memory=False,
        on_bad_lines="skip",
    )

    df = df.dropna(subset=[text_column, label_column]).copy()

    df[label_column] = (
        df[label_column]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df[text_column] = (
        df[text_column]
        .astype(str)
        .map(normalize_text)
    )

    df = df[
        (df[label_column].isin(LABEL_TO_IDX))
        & (df[text_column].str.len() > 0)
    ].copy()

    df["label"] = df[label_column].map(LABEL_TO_IDX).astype(int)
    df["text"] = df[text_column]

    result = df[["text", "label", label_column]].reset_index(drop=True)

    if result.empty:
        raise ValueError(
            "Dataset kosong setelah filtering. Pastikan CSV memiliki kolom "
            f"'{text_column}' dan '{label_column}' serta label sesuai config.py."
        )

    return result


def stratified_split(
    df: pd.DataFrame,
    test_size: float = 0.20,
    val_size_from_train: float = 0.10,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_pool, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df["label"],
    )

    train_df, val_df = train_test_split(
        train_pool,
        test_size=val_size_from_train,
        random_state=seed,
        stratify=train_pool["label"],
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def compute_class_alpha(train_df: pd.DataFrame) -> List[float]:
    """
    Alpha Focal Loss berdasarkan inverse frequency yang dinormalisasi.
    Pada dataset balanced, nilainya akan hampir sama untuk semua kelas.
    """
    counts = (
        train_df["label"]
        .value_counts()
        .reindex(range(NUM_CLASSES), fill_value=0)
        .astype(float)
    )

    if (counts <= 0).any():
        missing = counts[counts <= 0].index.tolist()
        raise ValueError(f"Label berikut tidak muncul pada train set: {missing}")

    inv = 1.0 / counts.to_numpy()
    alpha = inv / inv.sum()
    return alpha.tolist()


def summarize_split(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Dict[str, Dict[str, int]]:
    def counts(df: pd.DataFrame) -> Dict[str, int]:
        raw = df["label"].value_counts().sort_index()
        return {str(idx): int(raw.get(idx, 0)) for idx in range(NUM_CLASSES)}

    return {
        "train": counts(train_df),
        "validation": counts(val_df),
        "test": counts(test_df),
    }
