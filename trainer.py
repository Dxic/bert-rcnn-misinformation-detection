from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, Optional

import torch

from evaluate import evaluate_model


def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("bert_rcnn_training")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def train_model(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device: torch.device,
    epochs: int,
    checkpoint_path: str,
    log_path: str,
    num_classes: int,
    scheduler=None,
    grad_accum_steps: int = 1,
    patience: int = 3,
) -> Dict[str, float]:
    logger = setup_logger(log_path)

    model.to(device)

    best_f1 = -1.0
    wait = 0
    started_total = time.perf_counter()

    Path(checkpoint_path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    for epoch in range(1, epochs + 1):
        model.train()

        optimizer.zero_grad(set_to_none=True)

        running_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
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

            loss = criterion(logits, labels)
            loss = loss / max(1, grad_accum_steps)

            loss.backward()

            if step % grad_accum_steps == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )

                optimizer.step()

                if scheduler is not None:
                    scheduler.step()

                optimizer.zero_grad(set_to_none=True)

            running_loss += float(loss.item()) * max(1, grad_accum_steps)

        val_metrics = evaluate_model(
            model,
            val_loader,
            device=device,
            num_classes=num_classes,
        )

        avg_loss = running_loss / max(1, len(train_loader))

        logger.info(
            "Epoch %s/%s | train_loss=%.6f | val_accuracy=%.4f | "
            "val_f1_macro=%.4f | val_gmean=%.4f",
            epoch,
            epochs,
            avg_loss,
            val_metrics["accuracy"],
            val_metrics["f1_macro"],
            val_metrics["gmean"],
        )

        current_f1 = float(val_metrics["f1_macro"])

        if current_f1 > best_f1:
            best_f1 = current_f1
            wait = 0

            torch.save(
                model.state_dict(),
                checkpoint_path,
            )

            logger.info("Checkpoint terbaik disimpan: %s", checkpoint_path)
        else:
            wait += 1

            if wait >= patience:
                logger.info("Early stopping aktif.")
                break

    total_seconds = time.perf_counter() - started_total

    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=device,
        )
    )

    return {
        "training_seconds": total_seconds,
        "best_val_f1_macro": best_f1,
    }
