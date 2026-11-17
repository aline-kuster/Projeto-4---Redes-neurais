"""Busca em grade dos hiperparâmetros da LSTM, escolhendo pelo erro na validação.

Hiperparâmetros: tamanho da janela, neurônios da LSTM, número de camadas e taxa de aprendizado.
Cada combinação é treinada com parada antecipada (paciência na validação).

Uso: python src/search.py --k 5
"""

import argparse
import copy
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from data import make_windows
from model import LSTMForecaster, count_params

ROOT = Path(__file__).resolve().parent.parent / "results"
K = 5  # distância da previsão (t+k); pode ser trocada com --k

# ESPAÇO DE BUSCA
GRID = {
    "window": [14, 30, 60],   # dias anteriores usados como entrada
    "hidden": [32, 64],       # neurônios da LSTM
    "layers": [1, 2],         # camadas LSTM empilhadas
    "lr": [1e-3, 5e-3],       # taxa de aprendizado
}
MAX_EPOCHS, PATIENCE, BATCH, SEED = 100, 10, 64, 0


def to_tensor(split):
    X, y, _, _ = split
    return torch.tensor(X), torch.tensor(y)


@torch.no_grad()
def mse(model, X, y):
    model.eval()
    return nn.functional.mse_loss(model(X), y).item()


# TREINO de uma combinação
def train(cfg, data):
    torch.manual_seed(SEED)
    Xtr, ytr = to_tensor(data["treino"])
    Xva, yva = to_tensor(data["validacao"])
    model = LSTMForecaster(cfg["hidden"], cfg["layers"], dropout=0.2)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    hist = {"treino": [], "validacao": []}
    best, best_state, bad = float("inf"), None, 0
    # Época 0: erro antes de treinar (pesos aleatórios)
    hist["treino"].append(mse(model, Xtr, ytr))
    hist["validacao"].append(mse(model, Xva, yva))

    for ep in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), BATCH):
            idx = perm[i:i + BATCH]
            loss = nn.functional.mse_loss(model(Xtr[idx]), ytr[idx])  # erro quadrático médio
            opt.zero_grad()
            loss.backward()
            opt.step()
        hist["treino"].append(mse(model, Xtr, ytr))
        hist["validacao"].append(mse(model, Xva, yva))

        # PARADA ANTECIPADA: guarda a melhor época e para após PATIENCE épocas sem melhorar
        if hist["validacao"][-1] < best:
            best, best_state, bad = hist["validacao"][-1], copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, hist


def main():
    global K
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=5)
    K = p.parse_args().k
    RESULTS = ROOT / f"k{K}"          # resultados de cada k numa pasta própria
    RESULTS.mkdir(parents=True, exist_ok=True)
    runs, best_run = [], None
    cache = {}
    for values in itertools.product(*GRID.values()):
        cfg = dict(zip(GRID, values))
        if cfg["window"] not in cache:
            cache[cfg["window"]] = make_windows(cfg["window"], K)
        data, mean, std = cache[cfg["window"]]

        t0 = time.time()
        model, hist = train(cfg, data)
        best_ep = int(np.argmin(hist["validacao"]))  # índice 0 = antes de treinar
        val_rmse = float(np.sqrt(min(hist["validacao"])) * std)  # erro em °C
        run = {**cfg, "val_rmse_C": val_rmse, "best_epoch": best_ep, "epochs_run": len(hist["validacao"]) - 1,
               "params": count_params(model), "seconds": time.time() - t0, "history": hist}
        runs.append(run)
        print(f"{cfg} -> RMSE val {val_rmse:.3f} °C (época {best_ep}/{run['epochs_run']}, {run['seconds']:.0f}s)",
              flush=True)

        if best_run is None or val_rmse < best_run["val_rmse_C"]:
            best_run = run
            torch.save({"config": cfg, "k": K, "state_dict": model.state_dict()}, RESULTS / "lstm_melhor.pt")

    print(f"\nMelhor: { {k: best_run[k] for k in GRID} } -> RMSE val {best_run['val_rmse_C']:.3f} °C")
    (RESULTS / "busca.json").write_text(json.dumps({"k": K, "grid": GRID, "runs": runs,
                                                   "melhor": {k: best_run[k] for k in GRID}}, indent=2))


if __name__ == "__main__":
    main()
