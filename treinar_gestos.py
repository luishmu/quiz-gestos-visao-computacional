"""
Treina o classificador de gestos e compara com a baseline por regras.

Uso:
    python treinar_gestos.py

Split por GRUPO (módulo 6 da disciplina): quadros vizinhos de um vídeo são
quase idênticos, então dividir quadro a quadro vaza informação e infla a acurácia.
- Com 2+ pessoas no CSV: testa em uma pessoa que o modelo nunca viu.
- Com 1 pessoa só: testa em uma sessão de gravação que o modelo nunca viu.
"""
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score)
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from gestos_comum import (CLASSES, classificar_por_regras, colunas_features,
                          features_invariantes)

BASE = Path(__file__).parent
CSV_PATH = BASE / "dados" / "gestos.csv"
MODELO_OUT = BASE / "modelos" / "mlp_gestos.joblib"


def main():
    df = pd.read_csv(CSV_PATH)
    X = df[colunas_features()].to_numpy(np.float32)
    y = df["classe"].to_numpy()

    if df["pessoa"].nunique() >= 2:
        grupos, tipo = df["pessoa"], "pessoa"
    elif df["sessao"].nunique() >= 2:
        grupos, tipo = df["sessao"], "sessão"
    else:
        raise SystemExit("Grave pelo menos 2 sessões (ou 2 pessoas) para ter um teste honesto.")

    split = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    tr, te = next(split.split(X, y, grupos))
    print(f"{len(df)} quadros | split por {tipo} | "
          f"treino: {sorted(set(grupos.iloc[tr]))} | teste: {sorted(set(grupos.iloc[te]))}\n")

    # ---- baseline: regras ----
    pred_regras = np.array([classificar_por_regras(v) for v in X[te]])

    # ---- MLP: coordenadas cruas vs. medidas invariantes ----
    # As duas recebem o mesmo vetor de 63; a segunda converte em ângulos/distâncias
    # dentro do pipeline, então o app chama predict() igual nos dois casos.
    def criar_mlp(features=None):
        etapas = [FunctionTransformer(features)] if features else []
        return make_pipeline(*etapas, StandardScaler(),
                             MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=600,
                                           early_stopping=True, random_state=42))

    mlp_coords = criar_mlp().fit(X[tr], y[tr])
    mlp = criar_mlp(features_invariantes).fit(X[tr], y[tr])
    pred_coords = mlp_coords.predict(X[te])
    pred_mlp = mlp.predict(X[te])

    resultados = [("Regras", pred_regras), ("MLP coords", pred_coords),
                  ("MLP invariantes", pred_mlp)]
    print(f"{'modelo':16s} {'acurácia':>9s} {'F1 macro':>9s}")
    for nome, pred in resultados:
        print(f"{nome:16s} {accuracy_score(y[te], pred):9.3f} "
              f"{f1_score(y[te], pred, average='macro', labels=CLASSES):9.3f}")

    print("\nMLP invariantes por classe:")
    print(classification_report(y[te], pred_mlp, labels=CLASSES, zero_division=0))

    # ---- validação cruzada: cada pessoa (ou sessão) vira teste uma vez ----
    # O split acima depende de quem foi sorteado para o teste; aqui todo grupo é
    # testado, e a média ± desvio mostra se a diferença entre os modelos é estável.
    print(f"Validação cruzada deixando um(a) {tipo} de fora (acurácia | F1 macro):")
    nomes = [nome for nome, _ in resultados]
    print(f"{tipo:12s} " + " ".join(f"{n:>22s}" for n in nomes))
    metricas = {n: [] for n in nomes}
    for tr_cv, te_cv in LeaveOneGroupOut().split(X, y, grupos):
        preds = [np.array([classificar_por_regras(v) for v in X[te_cv]]),
                 criar_mlp().fit(X[tr_cv], y[tr_cv]).predict(X[te_cv]),
                 criar_mlp(features_invariantes).fit(X[tr_cv], y[tr_cv]).predict(X[te_cv])]
        linha = []
        for nome, pred in zip(nomes, preds):
            acc = accuracy_score(y[te_cv], pred)
            f1 = f1_score(y[te_cv], pred, average="macro", labels=CLASSES)
            metricas[nome].append((acc, f1))
            linha.append(f"{acc:10.3f} | {f1:.3f}")
        print(f"{str(grupos.iloc[te_cv[0]]):12s} " + " ".join(f"{c:>22s}" for c in linha))
    medias = []
    for nome in nomes:
        m = np.array(metricas[nome])
        medias.append(f"{m[:, 0].mean():.3f}±{m[:, 0].std():.3f} | {m[:, 1].mean():.3f}")
    print(f"{'média±desvio':12s} " + " ".join(f"{c:>22s}" for c in medias) + "\n")

    fig, eixos = plt.subplots(1, 3, figsize=(19, 5.5))
    for ax, (nome, pred) in zip(eixos, resultados):
        ConfusionMatrixDisplay.from_predictions(
            y[te], pred, labels=CLASSES, ax=ax, colorbar=False,
            xticks_rotation=45, normalize="true", values_format=".2f")
        ax.set_title(nome)
    fig.tight_layout()
    fig.savefig(BASE / "matriz_confusao.png", dpi=120)

    # modelo final treinado com tudo, para o app
    mlp.fit(X, y)
    MODELO_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(mlp, MODELO_OUT)
    print(f"Modelo salvo em {MODELO_OUT} | matriz de confusão em matriz_confusao.png")


if __name__ == "__main__":
    main()
