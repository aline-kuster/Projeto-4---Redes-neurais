"""Avaliação do melhor modelo no teste e figuras do relatório.

Métricas em °C: MAE, RMSE e R². Comparação com o baseline de persistência
(prever que a temperatura em t+k será igual à de hoje, dia t).

Uso: python src/evaluate.py --k 5
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from data import load_series, make_windows
from model import LSTMForecaster

ROOT = Path(__file__).resolve().parent.parent / "results"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 160, "savefig.bbox": "tight"})


def metrics(y, yhat):
    err = yhat - y
    return {"MAE": float(np.abs(err).mean()),
            "RMSE": float(np.sqrt((err ** 2).mean())),
            "R2": float(1 - (err ** 2).sum() / ((y - y.mean()) ** 2).sum())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=5)
    RESULTS = ROOT / f"k{p.parse_args().k}"
    FIG = RESULTS / "figs"
    FIG.mkdir(parents=True, exist_ok=True)
    busca = json.loads((RESULTS / "busca.json").read_text())
    ck = torch.load(RESULTS / "lstm_melhor.pt", weights_only=False)
    cfg, k = ck["config"], ck["k"]
    model = LSTMForecaster(cfg["hidden"], cfg["layers"], dropout=0.2)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    data, mean, std = make_windows(cfg["window"], k)
    out = {"config": cfg, "k": k}
    preds = {}
    for name in ("treino", "validacao", "teste"):
        X, y, d, last = data[name]
        with torch.no_grad():
            yhat = model(torch.tensor(X)).numpy() * std + mean   # volta para °C
        y_c = y * std + mean
        preds[name] = (d, y_c, yhat, last)
        out[name] = {"LSTM": metrics(y_c, yhat), "persistencia": metrics(y_c, last)}

    # ---------- Figura: série e divisão ----------
    df = load_series()
    fig, ax = plt.subplots(figsize=(10, 3))
    cores = {"treino": "#4c72b0", "validacao": "#dd8452", "teste": "#55a868"}
    for name, (a, b) in {"treino": (1981, 1986), "validacao": (1987, 1988), "teste": (1989, 1990)}.items():
        m = df["Date"].dt.year.between(a, b)
        ax.plot(df["Date"][m], df["Temp"][m], lw=0.6, color=cores[name], label=f"{ {'validacao': 'validação'}.get(name, name)} ({a}-{b})")
    ax.set(ylabel="Temperatura mínima (°C)", title="Série completa e divisão dos dados")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.savefig(FIG / "serie.png"); plt.close(fig)

    # ---------- Figura: busca de hiperparâmetros ----------
    runs = sorted(busca["runs"], key=lambda r: r["val_rmse_C"])
    fig, ax = plt.subplots(figsize=(7, 6))
    lab = [f"janela {r['window']} | {r['hidden']} neur. | {r['layers']} cam. | lr {r['lr']:g}" for r in runs]
    ax.barh(range(len(runs)), [r["val_rmse_C"] for r in runs],
            color=["#d62728" if i == 0 else "#9aa5b1" for i in range(len(runs))])
    ax.set_yticks(range(len(runs)), lab, fontsize=8)
    ax.invert_yaxis()
    v = [r["val_rmse_C"] for r in runs]
    ax.set_xlim(min(v) - 0.05, max(v) + 0.05)
    ax.set(xlabel="RMSE na validação (°C)", title="Busca de hiperparâmetros (em vermelho, a escolhida)")
    fig.savefig(FIG / "busca.png"); plt.close(fig)

    # ---------- Figura: curva de aprendizagem do melhor ----------
    best = runs[0]
    h = best["history"]
    ep = np.arange(0, len(h["treino"]))  # época 0 = antes de treinar
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.plot(ep, np.sqrt(h["treino"]) * std, label="treino")
    ax.plot(ep, np.sqrt(h["validacao"]) * std, label="validação")
    ax.axvline(best["best_epoch"], ls=":", c="gray", label=f"melhor época ({best['best_epoch']})")
    ax.set(xlabel="Época", ylabel="RMSE (°C)", title="Curva de aprendizagem do modelo escolhido")
    ax.legend(frameon=False)
    fig.savefig(FIG / "curva_aprendizagem.png"); plt.close(fig)

    # ---------- Figura: previsão no teste ----------
    d, y_c, yhat, last = preds["teste"]
    d = pd.to_datetime(d)
    fig, axs = plt.subplots(2, 1, figsize=(10, 5.6))
    axs[0].plot(d, y_c, lw=0.7, c="k", label="real")
    axs[0].plot(d, yhat, lw=0.9, c="#d62728", label=f"previsão LSTM (t+{k})")
    axs[0].set(ylabel="°C", title="Conjunto de teste (1989-1990)")
    axs[0].legend(frameon=False, ncol=2)
    m = (d >= "1990-05-01") & (d < "1990-08-01")
    axs[1].plot(d[m], y_c[m], "o-", ms=3, lw=0.8, c="k", label="real")
    axs[1].plot(d[m], yhat[m], "o-", ms=3, lw=0.8, c="#d62728", label="previsão LSTM")
    axs[1].plot(d[m], last[m], "--", lw=0.8, c="#9aa5b1", label=f"persistência (valor de {k} dias antes)")
    axs[1].set(ylabel="°C", title="Detalhe: maio a julho de 1990")
    axs[1].legend(frameon=False, ncol=3, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "previsao_teste.png"); plt.close(fig)

    (RESULTS / "metricas.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
