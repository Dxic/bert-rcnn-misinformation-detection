from __future__ import annotations

import math
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def geometric_mean_recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> float:
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
    ).astype(float)

    recalls = []

    for idx in range(num_classes):
        denom = cm[idx].sum()
        recalls.append(cm[idx, idx] / denom if denom > 0 else 0.0)

    recalls = np.clip(np.array(recalls, dtype=float), 1e-12, 1.0)

    return float(np.exp(np.mean(np.log(recalls))))


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    num_classes: int,
) -> Dict[str, object]:
    model.eval()

    y_true: List[int] = []
    y_pred: List[int] = []

    started = time.perf_counter()

    for batch in loader:
        labels = batch["labels"].to(device)

        if "attention_mask" in batch:
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
        else:
            logits = model(
                input_ids=batch["input_ids"].to(device),
                length=batch["length"].to(device),
            )

        preds = logits.argmax(dim=1)

        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

    elapsed = time.perf_counter() - started

    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    return {
        "accuracy": accuracy_score(y_true_arr, y_pred_arr),
        "macro_precision": precision_score(
            y_true_arr,
            y_pred_arr,
            average="macro",
            zero_division=0,
        ),
        "macro_recall": recall_score(
            y_true_arr,
            y_pred_arr,
            average="macro",
            zero_division=0,
        ),
        "f1_macro": f1_score(
            y_true_arr,
            y_pred_arr,
            average="macro",
            zero_division=0,
        ),
        "f1_weighted": f1_score(
            y_true_arr,
            y_pred_arr,
            average="weighted",
            zero_division=0,
        ),
        "gmean": geometric_mean_recall(
            y_true_arr,
            y_pred_arr,
            num_classes=num_classes,
        ),
        "inference_seconds": elapsed,
        "samples": len(y_true_arr),
        "y_true": y_true_arr,
        "y_pred": y_pred_arr,
    }
