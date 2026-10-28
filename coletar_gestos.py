"""
Coletor de dados do quiz por gestos.

Uso:
    python coletar_gestos.py --pessoa pessoa5

Teclas:
    1..7     escolhe o gesto que você vai gravar (lista aparece na tela)
    ESPAÇO   começa / pausa a gravação
    q        sai e salva

Cada quadro com mão detectada vira uma linha em dados/gestos.csv
(63 features normalizadas + pessoa + sessão + classe).
Dica: enquanto grava, mexa a mão (mais perto, mais longe, gire um pouco,
troque de mão). Variedade aqui = modelo que generaliza depois.
"""
import argparse
import csv
import time
from pathlib import Path

import cv2

from gestos_comum import (CLASSES, SIGNIFICADO, colunas_features, criar_detector,
                          desenhar_mao, detectar, normalizar)

CSV_PATH = Path(__file__).parent / "dados" / "gestos.csv"
ALVO_POR_CLASSE = 300  # quadros por gesto por sessão (~10 s a 30 fps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pessoa", required=True, help="quem está gravando (vira o grupo do split)")
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    sessao = time.strftime("%Y%m%d-%H%M%S")
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    novo = not CSV_PATH.exists()
    arq = open(CSV_PATH, "a", newline="", encoding="utf-8")
    escritor = csv.writer(arq)
    if novo:
        escritor.writerow(colunas_features() + ["lado", "pessoa", "sessao", "classe"])

    detector = criar_detector()
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Não consegui abrir a webcam. Tente --camera 1.")

    atual = 0
    gravando = False
    contagem = {c: 0 for c in CLASSES}
    t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # efeito espelho, mais natural
        pts, lado = detectar(detector, frame, int((time.time() - t0) * 1000))

        classe = CLASSES[atual]
        if pts is not None:
            desenhar_mao(frame, pts)
            if gravando:
                escritor.writerow(list(normalizar(pts, lado)) + [lado, args.pessoa, sessao, classe])
                contagem[classe] += 1
                if contagem[classe] >= ALVO_POR_CLASSE:
                    gravando = False  # para sozinho ao bater a meta

        # ---- painel ----
        cor = (0, 0, 255) if gravando else (200, 200, 200)
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(frame, f"{'GRAVANDO' if gravando else 'pausado'}  |  gesto: {classe} "
                    f"({SIGNIFICADO[classe]})", (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2)
        for i, c in enumerate(CLASSES):
            marca = ">" if i == atual else " "
            cv2.putText(frame, f"{marca}{i + 1} {c}: {contagem[c]}/{ALVO_POR_CLASSE}",
                        (10, 70 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 255, 0) if contagem[c] >= ALVO_POR_CLASSE else (255, 255, 255), 1)
        if pts is None:
            cv2.putText(frame, "nenhuma mao detectada", (10, frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow("Coleta de gestos", frame)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord("q"):
            break
        if tecla == ord(" "):
            gravando = not gravando
        if ord("1") <= tecla <= ord(str(len(CLASSES))):
            atual = tecla - ord("1")
            gravando = False

    arq.close()
    cap.release()
    cv2.destroyAllWindows()
    print(f"Sessão {sessao} salva em {CSV_PATH}:")
    for c, n in contagem.items():
        print(f"  {c:8s} {n}")


if __name__ == "__main__":
    main()
