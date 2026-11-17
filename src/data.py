"""Dados: temperatura mínima diária de Melbourne (1981-1990).

Monta janelas de W dias anteriores e o alvo k dias à frente.
Divisão cronológica pela data do alvo:
  treino 1981-1986 | validação 1987-1988 | teste 1989-1990
"""

from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CSV = DATA_DIR / "daily-min-temperatures.csv"
URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"


def load_series() -> pd.DataFrame:
    if not CSV.exists():
        DATA_DIR.mkdir(exist_ok=True)
        urllib.request.urlretrieve(URL, CSV)
    return pd.read_csv(CSV, parse_dates=["Date"])


def make_windows(window: int, k: int):
    """Retorna X (N, window, 1), y (N,), datas do alvo, último valor da janela e média/desvio do treino."""
    df = load_series()
    temp = df["Temp"].to_numpy(dtype=np.float32)
    dates = df["Date"].to_numpy()

    # Normalização com média e desvio padrão só do período de treino
    train_mask = df["Date"].dt.year <= 1986
    mean, std = temp[train_mask].mean(), temp[train_mask].std()
    z = (temp - mean) / std

    X, y, d, last = [], [], [], []
    for t in range(window - 1, len(z) - k):
        X.append(z[t - window + 1:t + 1])   # dias t-W+1 ... t
        y.append(z[t + k])                  # dia t+k
        d.append(dates[t + k])
        last.append(temp[t])                # temperatura no dia t (usada no baseline)
    X = np.array(X)[..., None]
    y, d, last = np.array(y), np.array(d), np.array(last)

    year = pd.DatetimeIndex(d).year
    splits = {
        "treino": year <= 1986,
        "validacao": (year >= 1987) & (year <= 1988),
        "teste": year >= 1989,
    }
    return {name: (X[m], y[m], d[m], last[m]) for name, m in splits.items()}, float(mean), float(std)
