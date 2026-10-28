"""
Quiz controlado por gestos (webcam + modelo treinado).

Uso:
    python app_quiz.py [--camera 1] [--perguntas perguntas.json]

Em cada pergunta:
    1..4 dedos   escolhe a alternativa A..D
    palma        confirma e mostra a resposta / avança
    joinha       "fácil": vai para a próxima
    deslike      "errei": a pergunta volta para o fim da fila

Teclado (reserva para a apresentação): 1-4, ESPAÇO = palma, j = joinha,
d = deslike, q = sair.

Para evitar disparos acidentais, um gesto só vale depois de ficar estável por
SEGURAR_S segundos, e só vale de novo depois que a mão muda de gesto ou sai da tela.
"""
import argparse
import json
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import joblib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from gestos_comum import (SIGNIFICADO, criar_detector, desenhar_mao, detectar,
                          normalizar)

BASE = Path(__file__).parent
MODELO_PATH = BASE / "modelos" / "mlp_gestos.joblib"

CONF_MIN = 0.70     # probabilidade mínima do MLP para aceitar o gesto do quadro
JANELA = 7          # quadros considerados na votação
VOTOS_MIN = 5       # quantos deles precisam concordar
SEGURAR_S = 0.6     # tempo segurando o gesto até disparar

LETRAS = "ABCD"
GESTO_ALTERNATIVA = {"um": 0, "dois": 1, "tres": 2, "quatro": 3}

CAM_W, CAM_H, PAINEL_W, FAIXA_H = 640, 480, 640, 160
FUNDO, CARTAO, BRANCO, CINZA = (24, 26, 32), (44, 48, 58), (240, 240, 240), (160, 165, 175)
AZUL, VERDE, VERMELHO, AMARELO = (70, 130, 230), (60, 170, 90), (210, 70, 70), (240, 190, 60)


def fonte(tamanho, negrito=False):
    for nome in (("segoeuib.ttf", "arialbd.ttf") if negrito else ("segoeui.ttf", "arial.ttf")):
        try:
            return ImageFont.truetype(nome, tamanho)
        except OSError:
            pass
    return ImageFont.load_default(tamanho)


F_PEQ, F_TXT, F_TIT, F_GRANDE = fonte(16), fonte(20), fonte(24, True), fonte(40, True)


def quebrar(d, texto, f, largura):
    linhas, atual = [], ""
    for palavra in texto.split():
        teste = f"{atual} {palavra}".strip()
        if d.textlength(teste, font=f) <= largura:
            atual = teste
        else:
            linhas.append(atual)
            atual = palavra
    return linhas + [atual]


# ---------------------------------------------------------------------------
# Gesto do quadro -> evento
# ---------------------------------------------------------------------------
class Estabilizador:
    """Votação nos últimos quadros + tempo segurando + rearme ao trocar de gesto."""

    def __init__(self):
        self.votos = deque(maxlen=JANELA)
        self.atual, self.desde, self.disparado = None, 0.0, False

    def atualizar(self, gesto, agora):
        """Recebe o gesto do quadro (ou None). Devolve (evento ou None, gesto estável, progresso 0..1)."""
        self.votos.append(gesto)
        mais_comum, n = Counter(self.votos).most_common(1)[0]
        estavel = mais_comum if n >= VOTOS_MIN else None

        if estavel != self.atual:
            self.atual, self.desde, self.disparado = estavel, agora, False
        if estavel is None:
            return None, None, 0.0
        if self.disparado:
            return None, estavel, 1.0
        progresso = (agora - self.desde) / SEGURAR_S
        if progresso >= 1:
            self.disparado = True
            return estavel, estavel, 1.0
        return None, estavel, progresso


# ---------------------------------------------------------------------------
# Estado do quiz
# ---------------------------------------------------------------------------
class Quiz:
    def __init__(self, perguntas):
        self.perguntas = perguntas
        self.reiniciar()

    def reiniciar(self):
        self.fila = list(range(len(self.perguntas)))
        self.tentadas, self.acertos, self.revisoes = set(), 0, 0
        self.escolha, self.revelada = None, False
        self.aviso = ""

    @property
    def terminou(self):
        return not self.fila

    @property
    def pergunta(self):
        return self.perguntas[self.fila[0]]

    def correta(self):
        return LETRAS.index(self.pergunta["correta"])

    def acao(self, gesto):
        self.aviso = ""
        if self.terminou:
            if gesto == "palma":
                self.reiniciar()
            return
        if not self.revelada:
            if gesto in GESTO_ALTERNATIVA:
                self.escolha = GESTO_ALTERNATIVA[gesto]
            elif gesto == "palma":
                if self.escolha is None:
                    self.aviso = "Escolha uma alternativa antes (1 a 4 dedos)."
                    return
                self.revelada = True
                if self.fila[0] not in self.tentadas:  # placar conta só a 1ª tentativa
                    self.tentadas.add(self.fila[0])
                    self.acertos += self.escolha == self.correta()
            return
        if gesto in ("joinha", "palma"):
            self.fila.pop(0)
        elif gesto == "deslike":
            self.fila.append(self.fila.pop(0))
            self.revisoes += 1
        else:
            return
        self.escolha, self.revelada = None, False


# ---------------------------------------------------------------------------
# Desenho
# ---------------------------------------------------------------------------
def desenhar_painel(d, quiz):
    x, w, total = CAM_W + 28, PAINEL_W - 56, len(quiz.perguntas)
    altura = CAM_H + FAIXA_H
    if quiz.terminou:
        d.text((x, 120), "Fim do quiz!", font=F_GRANDE, fill=BRANCO)
        d.text((x, 190), f"Acertos de primeira: {quiz.acertos} de {total}", font=F_TIT, fill=VERDE)
        d.text((x, 235), f"Perguntas revisadas: {quiz.revisoes}", font=F_TXT, fill=CINZA)
        d.text((x, altura - 60), "Palma: recomeçar   ·   q: sair", font=F_TXT, fill=AMARELO)
        return

    p = quiz.pergunta
    y = 22
    d.text((x, y), f"Faltam {len(quiz.fila)} de {total}   ·   acertos de primeira: {quiz.acertos}",
           font=F_PEQ, fill=CINZA)
    y += 34
    for linha in quebrar(d, p["pergunta"], F_TIT, w):
        d.text((x, y), linha, font=F_TIT, fill=BRANCO)
        y += 32
    y += 12

    certa = quiz.correta()
    for i, alt in enumerate(p["alternativas"]):
        linhas = quebrar(d, f"{LETRAS[i]})  {alt}", F_TXT, w - 110)
        h = 18 + 27 * len(linhas)
        borda, fundo = CARTAO, CARTAO
        if quiz.revelada and i == certa:
            borda, fundo = VERDE, (40, 80, 55)
        elif quiz.revelada and i == quiz.escolha:
            borda, fundo = VERMELHO, (90, 45, 45)
        elif i == quiz.escolha:
            borda = AZUL
        d.rounded_rectangle([x, y, x + w, y + h], radius=10, fill=fundo, outline=borda, width=3)
        for k, linha in enumerate(linhas):
            d.text((x + 14, y + 9 + 27 * k), linha, font=F_TXT, fill=BRANCO)
        dica = f"{i + 1} dedo" + ("s" if i else "")
        d.text((x + w - 14 - d.textlength(dica, font=F_PEQ), y + 11), dica, font=F_PEQ, fill=CINZA)
        y += h + 10

    if quiz.revelada:
        acertou = quiz.escolha == certa
        y += 4
        d.text((x, y), "Acertou!" if acertou else f"Errou. A certa é {LETRAS[certa]}.",
               font=F_TIT, fill=VERDE if acertou else VERMELHO)
        y += 36
        for linha in quebrar(d, p.get("explicacao", ""), F_PEQ, w):
            d.text((x, y), linha, font=F_PEQ, fill=CINZA)
            y += 22
        rodape = "Joinha: fácil, próxima   ·   Deslike: revisar depois   ·   Palma: próxima"
    elif quiz.escolha is None:
        rodape = "Mostre 1 a 4 dedos (polegar dobrado) para escolher"
    else:
        rodape = f"Escolheu {LETRAS[quiz.escolha]}. Mostre a palma para confirmar"

    if quiz.aviso:
        d.text((x, altura - 90), quiz.aviso, font=F_TXT, fill=VERMELHO)
    for k, linha in enumerate(quebrar(d, rodape, F_TXT, w)):
        d.text((x, altura - 58 + 26 * k), linha, font=F_TXT, fill=AMARELO)


def desenhar_faixa(d, gesto, progresso, conf):
    y0 = CAM_H
    if gesto is None:
        d.text((20, y0 + 18), "Gesto: —", font=F_TIT, fill=CINZA)
    else:
        d.text((20, y0 + 18), f"Gesto: {gesto} ({conf:.0%})", font=F_TIT, fill=BRANCO)
        d.text((20, y0 + 52), SIGNIFICADO[gesto], font=F_TXT, fill=CINZA)
    barra_w = CAM_W - 40
    d.rounded_rectangle([20, y0 + 88, 20 + barra_w, y0 + 102], radius=7, fill=CARTAO)
    if progresso > 0:
        d.rounded_rectangle([20, y0 + 88, 20 + int(barra_w * min(progresso, 1)), y0 + 102],
                            radius=7, fill=VERDE if progresso >= 1 else AZUL)
    d.text((20, y0 + 118), "1–4 dedos: A–D   ·   palma: confirmar   ·   joinha / deslike",
           font=F_PEQ, fill=CINZA)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--perguntas", default=str(BASE / "perguntas.json"))
    args = ap.parse_args()

    perguntas = json.loads(Path(args.perguntas).read_text(encoding="utf-8"))
    if not MODELO_PATH.exists():
        raise SystemExit("Modelo não encontrado. Rode antes: python treinar_gestos.py")
    modelo = joblib.load(MODELO_PATH)
    classes = list(modelo.classes_)

    detector = criar_detector()
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Não consegui abrir a webcam. Tente --camera 1.")

    quiz, estab = Quiz(perguntas), Estabilizador()
    teclas = {ord(str(i + 1)): g for g, i in GESTO_ALTERNATIVA.items()}
    teclas.update({ord(" "): "palma", ord("j"): "joinha", ord("d"): "deslike"})
    t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # mesmo espelhamento da coleta
        agora = time.time()
        pts, lado = detectar(detector, frame, int((agora - t0) * 1000))

        gesto_quadro, conf = None, 0.0
        if pts is not None:
            desenhar_mao(frame, pts)
            proba = modelo.predict_proba(normalizar(pts, lado)[None])[0]
            conf = float(proba.max())
            if conf >= CONF_MIN:
                gesto_quadro = classes[int(proba.argmax())]

        evento, estavel, progresso = estab.atualizar(gesto_quadro, agora)
        if evento:
            quiz.acao(evento)

        tela = Image.new("RGB", (CAM_W + PAINEL_W, CAM_H + FAIXA_H), FUNDO)
        cam = cv2.cvtColor(cv2.resize(frame, (CAM_W, CAM_H)), cv2.COLOR_BGR2RGB)
        tela.paste(Image.fromarray(cam), (0, 0))
        d = ImageDraw.Draw(tela)
        desenhar_faixa(d, estavel, progresso, conf)
        desenhar_painel(d, quiz)
        cv2.imshow("Quiz por gestos", cv2.cvtColor(np.asarray(tela), cv2.COLOR_RGB2BGR))

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord("q"):
            break
        if tecla in teclas:
            quiz.acao(teclas[tecla])

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
