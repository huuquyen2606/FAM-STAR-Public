"""Hybrid BLSTM-GRU classifier used by FAM-STAR."""

from __future__ import annotations

import torch.nn as nn


class HybridBLSTM_GRU(nn.Module):
    """BLSTM -> GRU -> dense classifier used in the original notebook."""

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        blstm_hidden: int = 300,
        gru_hidden: int = 100,
        dense_hidden: int = 80,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.blstm = nn.LSTM(
            input_size=input_size,
            hidden_size=blstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.ln_blstm = nn.LayerNorm(blstm_hidden * 2)

        self.gru = nn.GRU(
            input_size=blstm_hidden * 2,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
        )
        self.ln_gru = nn.LayerNorm(gru_hidden)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

        self.fc1 = nn.Linear(gru_hidden, dense_hidden)
        self.ln_fc = nn.LayerNorm(dense_hidden)
        self.fc_out = nn.Linear(dense_hidden, num_classes)

    def forward(self, x, return_embedding: bool = False):
        out, _ = self.blstm(x)
        out = out[:, -1, :]
        out = self.ln_blstm(out)
        out = self.relu(out)
        out = out.unsqueeze(1)

        out, _ = self.gru(out)
        out = out[:, -1, :]
        out = self.ln_gru(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc1(out)
        out = self.ln_fc(out)
        embedding = self.relu(out)
        logits = self.fc_out(embedding)

        if return_embedding:
            return logits, embedding
        return logits
