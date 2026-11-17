"""Modelo LSTM para previsão da temperatura no dia t+k."""

import torch
import torch.nn as nn


class LSTMForecaster(nn.Module):
    def __init__(self, hidden: int = 64, layers: int = 1, dropout: float = 0.0):
        super().__init__()
        # LSTM: lê a janela dia a dia (1 valor por passo)
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, num_layers=layers,
                            batch_first=True, dropout=dropout if layers > 1 else 0.0)
        # Camada de saída: último estado da LSTM -> 1 valor (temperatura em t+k)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x):                # x: (lote, W, 1)
        h, _ = self.lstm(x)              # h: (lote, W, hidden)
        return self.out(h[:, -1]).squeeze(-1)  # usa só o último passo da janela


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
