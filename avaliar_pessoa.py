"""
Avalia o modelo JÁ SALVO em uma pessoa do CSV, sem retreinar.

Uso:
    python avaliar_pessoa.py --pessoa maria

Rode ANTES de `treinar_gestos.py`: depois de retreinar, a pessoa entra no
treino e o teste deixa de ser honesto.
"""
import argparse
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score)

from gestos_comum import CLASSES, classificar_por_regras, colunas_features

BASE = Path(__file__).parent
CSV_PATH = BASE / "dados" / "gestos.csv"
MODELO_PATH = BASE / "modelos" / "mlp_gestos.joblib"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pessoa", required=True)
    args = ap.parse_args()

    df = pd.read_csv(CSV_PATH)
    df = df[df["pessoa"] == args.pessoa]
    if df.empty:
        raise SystemExit(f"Nenhum quadro de '{args.pessoa}' em {CSV_PATH}.")
    X = df[colunas_features()].to_numpy(np.float32)
    y = df["classe"].to_numpy()

    pred_regras = np.array([classificar_por_regras(v) for v in X])
    pred_mlp = joblib.load(MODELO_PATH).predict(X)

    print(f"{len(df)} quadros de '{args.pessoa}' | mão esquerda/direita: "
          f"{(df['lado'] == 'Left').sum()}/{(df['lado'] == 'Right').sum()}\n")
    print(f"{'modelo':10s} {'acurácia':>9s} {'F1 macro':>9s}")
    for nome, pred in [("regras", pred_regras), ("MLP", pred_mlp)]:
        print(f"{nome:10s} {accuracy_score(y, pred):9.3f} "
              f"{f1_score(y, pred, average='macro', labels=CLASSES):9.3f}")
    print("\nMLP por classe:")
    print(classification_report(y, pred_mlp, labels=CLASSES, zero_division=0))

    fig, eixos = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, (nome, pred) in zip(eixos, [("Regras", pred_regras), ("MLP", pred_mlp)]):
        ConfusionMatrixDisplay.from_predictions(
            y, pred, labels=CLASSES, ax=ax, colorbar=False,
            xticks_rotation=45, normalize="true", values_format=".2f")
        ax.set_title(f"{nome} - {args.pessoa}")
    fig.tight_layout()
    saida = BASE / f"matriz_{args.pessoa}.png"
    fig.savefig(saida, dpi=120)
    print(f"Matriz de confusão em {saida.name}")


if __name__ == "__main__":
    main()
