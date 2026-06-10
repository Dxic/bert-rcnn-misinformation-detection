from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel

from config import NUM_CLASSES


class BaselineRCNN(nn.Module):
    """
    Baseline praktis:
    Specific Word2Vec embedding -> BiLSTM -> Conv1D -> global max pooling -> FC
    """

    def __init__(
        self,
        embedding_matrix: np.ndarray,
        hidden_size: int = 256,
        dropout: float = 0.5,
        num_classes: int = NUM_CLASSES,
        freeze_embeddings: bool = False,
    ) -> None:
        super().__init__()

        vocab_size, embedding_dim = embedding_matrix.shape

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0,
        )

        self.embedding.weight.data.copy_(
            torch.tensor(embedding_matrix, dtype=torch.float32)
        )

        self.embedding.weight.requires_grad = not freeze_embeddings

        self.bilstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )

        self.conv = nn.Conv1d(
            in_channels=embedding_dim + 2 * hidden_size,
            out_channels=256,
            kernel_size=3,
            padding=1,
        )

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(
            256,
            num_classes,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        length: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        recurrent, _ = self.bilstm(embedded)

        features = torch.cat([embedded, recurrent], dim=2)
        features = features.transpose(1, 2)

        conv = torch.relu(self.conv(features))
        pooled = torch.max(conv, dim=2).values

        return self.classifier(self.dropout(pooled))


class BERTAugmentedRCNN(nn.Module):
    """
    BERT contextual embedding -> BiLSTM -> Conv1D multi-kernel -> pooling -> FC
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        hidden_size: int = 128,
        dropout: float = 0.5,
        num_classes: int = NUM_CLASSES,
        freeze_bert_layers: int = 6,
    ) -> None:
        super().__init__()

        self.bert = AutoModel.from_pretrained(model_name)
        bert_dim = int(self.bert.config.hidden_size)

        self._freeze_lower_bert_layers(freeze_bert_layers)

        self.bilstm = nn.LSTM(
            input_size=bert_dim,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )

        combined_dim = bert_dim + 2 * hidden_size

        self.conv3 = nn.Conv1d(
            combined_dim,
            128,
            kernel_size=3,
            padding=1,
        )

        self.conv5 = nn.Conv1d(
            combined_dim,
            128,
            kernel_size=5,
            padding=2,
        )

        self.conv7 = nn.Conv1d(
            combined_dim,
            128,
            kernel_size=7,
            padding=3,
        )

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(128 * 3, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def _freeze_lower_bert_layers(self, n_layers: int) -> None:
        if n_layers <= 0:
            return

        if hasattr(self.bert, "embeddings"):
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False

        encoder = getattr(self.bert, "encoder", None)
        if encoder is None or not hasattr(encoder, "layer"):
            return

        for layer in encoder.layer[:n_layers]:
            for param in layer.parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state

        recurrent, _ = self.bilstm(bert_output)

        features = torch.cat(
            [bert_output, recurrent],
            dim=2,
        ).transpose(1, 2)

        pooled = []

        for conv in (self.conv3, self.conv5, self.conv7):
            activated = torch.relu(conv(features))
            pooled.append(torch.max(activated, dim=2).values)

        merged = torch.cat(pooled, dim=1)

        return self.classifier(self.dropout(merged))
