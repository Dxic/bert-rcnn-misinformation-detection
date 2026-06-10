from __future__ import annotations

import gc
import os
import random
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
from gensim.models import Word2Vec
from transformers import get_linear_schedule_with_warmup

from config import ExperimentConfig, NUM_CLASSES
from data_loader import (
    load_fnc_dataset,
    stratified_split,
    compute_class_alpha,
    summarize_split,
)
from dataset import (
    build_vocab,
    basic_tokenize,
    FakeNewsDatasetBaseline,
    FakeNewsDatasetBERT,
    create_loader,
)
from evaluate import evaluate_model
from focal_loss import FocalLoss
from logger_excel import append_result
from models import BaselineRCNN, BERTAugmentedRCNN
from trainer import train_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cleanup_memory() -> None:
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _build_specific_word2vec(
    train_texts,
    vocab,
    cfg: ExperimentConfig,
) -> np.ndarray:
    sentences = [
        basic_tokenize(text)
        for text in train_texts
    ]

    w2v = Word2Vec(
        sentences=sentences,
        vector_size=cfg.word2vec_dim,
        window=cfg.word2vec_window,
        min_count=cfg.min_word_freq,
        workers=cfg.word2vec_workers,
        epochs=cfg.word2vec_epochs,
        sg=1,
        seed=cfg.seed,
    )

    rng = np.random.default_rng(cfg.seed)

    matrix = rng.normal(
        loc=0.0,
        scale=0.05,
        size=(len(vocab), cfg.word2vec_dim),
    ).astype(np.float32)

    matrix[0] = 0.0

    for token, idx in vocab.items():
        if token in w2v.wv:
            matrix[idx] = w2v.wv[token]

    return matrix


def _load_and_split(cfg: ExperimentConfig):
    df = load_fnc_dataset(
        cfg.csv_path,
        text_column=cfg.text_column,
        label_column=cfg.label_column,
    )

    train_df, val_df, test_df = stratified_split(
        df,
        test_size=cfg.test_size,
        val_size_from_train=cfg.val_size_from_train,
        seed=cfg.seed,
    )

    print("Distribusi split:")
    print(summarize_split(train_df, val_df, test_df))

    return train_df, val_df, test_df


def run_baseline_rcnn_w2v_ce(
    cfg: ExperimentConfig,
    dataset_name: str = "balanced_100k",
) -> Dict[str, Any]:
    set_seed(cfg.seed)
    cleanup_memory()
    cfg.ensure_dirs()

    train_df, val_df, test_df = _load_and_split(cfg)

    vocab = build_vocab(
        train_df["text"],
        max_vocab_size=cfg.max_vocab_size,
        min_word_freq=cfg.min_word_freq,
    )

    print("Vocabulary size:", len(vocab))

    embedding_matrix = _build_specific_word2vec(
        train_df["text"],
        vocab=vocab,
        cfg=cfg,
    )

    train_loader = create_loader(
        FakeNewsDatasetBaseline(
            train_df,
            vocab=vocab,
            max_words=cfg.max_words_baseline,
        ),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )

    val_loader = create_loader(
        FakeNewsDatasetBaseline(
            val_df,
            vocab=vocab,
            max_words=cfg.max_words_baseline,
        ),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    test_loader = create_loader(
        FakeNewsDatasetBaseline(
            test_df,
            vocab=vocab,
            max_words=cfg.max_words_baseline,
        ),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = BaselineRCNN(
        embedding_matrix=embedding_matrix,
        hidden_size=cfg.baseline_hidden_size,
        dropout=cfg.baseline_dropout,
        num_classes=NUM_CLASSES,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.baseline_lr,
        weight_decay=cfg.baseline_weight_decay,
    )

    train_info = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=nn.CrossEntropyLoss(),
        device=device,
        epochs=cfg.epochs,
        checkpoint_path=str(
            Path(cfg.checkpoint_dir)
            / "best_baseline_rcnn_w2v_ce.pt"
        ),
        log_path=cfg.training_log_path,
        num_classes=NUM_CLASSES,
        patience=cfg.early_stopping_patience,
    )

    metrics = evaluate_model(
        model,
        test_loader,
        device=device,
        num_classes=NUM_CLASSES,
    )

    result = {
        "experiment_id": "A",
        "dataset": dataset_name,
        "model": "RCNN + Specific Word2Vec",
        "loss": "CrossEntropy",
        **metrics,
        **train_info,
        "notes": "Baseline replikasi terbatas paper Ilie et al. (2021)",
    }

    append_result(cfg.log_excel_path, result)

    return result


def run_bert_rcnn(
    cfg: ExperimentConfig,
    use_focal_loss: bool,
    dataset_name: str,
    experiment_id: str,
) -> Dict[str, Any]:
    set_seed(cfg.seed)
    cleanup_memory()
    cfg.ensure_dirs()

    train_df, val_df, test_df = _load_and_split(cfg)

    train_dataset = FakeNewsDatasetBERT(
        train_df,
        model_name=cfg.bert_model_name,
        max_seq_len=cfg.max_seq_len,
    )

    tokenizer = train_dataset.tokenizer

    val_dataset = FakeNewsDatasetBERT(
        val_df,
        model_name=cfg.bert_model_name,
        max_seq_len=cfg.max_seq_len,
        tokenizer=tokenizer,
    )

    test_dataset = FakeNewsDatasetBERT(
        test_df,
        model_name=cfg.bert_model_name,
        max_seq_len=cfg.max_seq_len,
        tokenizer=tokenizer,
    )

    train_loader = create_loader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )

    val_loader = create_loader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    test_loader = create_loader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = BERTAugmentedRCNN(
        model_name=cfg.bert_model_name,
        hidden_size=cfg.bert_hidden_size,
        dropout=cfg.bert_dropout,
        num_classes=NUM_CLASSES,
        freeze_bert_layers=cfg.freeze_bert_layers,
    )

    bert_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if name.startswith("bert."):
            bert_params.append(param)
        else:
            head_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {
                "params": bert_params,
                "lr": cfg.bert_lr,
            },
            {
                "params": head_params,
                "lr": cfg.head_lr,
            },
        ],
        weight_decay=cfg.bert_weight_decay,
    )

    updates_per_epoch = max(
        1,
        len(train_loader) // max(1, cfg.grad_accum_steps),
    )

    total_steps = updates_per_epoch * cfg.epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    if use_focal_loss:
        alpha = compute_class_alpha(train_df)

        criterion = FocalLoss(
            alpha=alpha,
            gamma=cfg.focal_gamma,
        )

        loss_name = f"FocalLoss(gamma={cfg.focal_gamma})"
        checkpoint_name = (
            f"best_bert_rcnn_focal_{dataset_name}.pt"
        )
    else:
        criterion = nn.CrossEntropyLoss()
        loss_name = "CrossEntropy"
        checkpoint_name = (
            f"best_bert_rcnn_ce_{dataset_name}.pt"
        )

    train_info = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=cfg.epochs,
        checkpoint_path=str(
            Path(cfg.checkpoint_dir)
            / checkpoint_name
        ),
        log_path=cfg.training_log_path,
        num_classes=NUM_CLASSES,
        scheduler=scheduler,
        grad_accum_steps=cfg.grad_accum_steps,
        patience=cfg.early_stopping_patience,
    )

    metrics = evaluate_model(
        model,
        test_loader,
        device=device,
        num_classes=NUM_CLASSES,
    )

    result = {
        "experiment_id": experiment_id,
        "dataset": dataset_name,
        "model": "BERT-RCNN",
        "loss": loss_name,
        **metrics,
        **train_info,
        "notes": (
            "Improvement utama"
            if use_focal_loss
            else "Ablation study tanpa Focal Loss"
        ),
    }

    append_result(cfg.log_excel_path, result)

    return result


def print_result(result: Dict[str, Any]) -> None:
    print("\n===== HASIL EKSPERIMEN =====")
    print("ID              :", result["experiment_id"])
    print("Dataset         :", result["dataset"])
    print("Model           :", result["model"])
    print("Loss            :", result["loss"])
    print("Accuracy        :", round(result["accuracy"], 4))
    print("Macro Precision :", round(result["macro_precision"], 4))
    print("Macro Recall    :", round(result["macro_recall"], 4))
    print("F1 Macro        :", round(result["f1_macro"], 4))
    print("F1 Weighted     :", round(result["f1_weighted"], 4))
    print("G-Mean          :", round(result["gmean"], 4))
    print("Training sec    :", round(result["training_seconds"], 2))
    print("Inference sec   :", round(result["inference_seconds"], 2))
