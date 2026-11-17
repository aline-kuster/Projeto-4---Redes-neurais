# Projeto 4 — Previsão de temperatura com LSTM

Aline Breda Kuster

Uma LSTM recebe a temperatura mínima dos últimos W dias e prevê a temperatura do dia t+5.
Os hiperparâmetros (janela, neurônios, camadas e taxa de aprendizado) são escolhidos pelo
erro no conjunto de validação.

Dados: temperatura mínima diária de Melbourne, 1981-1990 (3 650 dias), baixados de
`github.com/jbrownlee/Datasets`.

## Estrutura

```
src/
  data.py      # baixa a série, normaliza e monta as janelas (treino 1981-86, validação 1987-88, teste 1989-90)
  model.py     # modelo LSTM
  search.py    # busca em grade dos hiperparâmetros + parada antecipada
  evaluate.py  # métricas no teste (MAE, RMSE, R²) e figuras
results/
  k5/, k1/     # busca.json, metricas.json, lstm_melhor.pt e figs/ de cada k
```

## Como rodar

```bash
pip install torch numpy pandas matplotlib
python src/search.py --k 5     # cerca de 5 minutos em CPU
python src/evaluate.py --k 5
python src/search.py --k 1     # comparação com previsão de 1 dia à frente
python src/evaluate.py --k 1
```
