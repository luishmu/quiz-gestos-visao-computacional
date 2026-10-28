"""
Funções compartilhadas do quiz por gestos (coleta, treino e app).

- Carrega o detector de mãos do MediaPipe (Tasks API).
- Transforma os 21 pontos da mão em um vetor de 63 números normalizado
  (independe de posição na tela, distância da câmera e mão esquerda/direita).
- Tem um classificador por REGRAS, que serve de baseline para comparar com a rede.
"""
from pathlib import Path
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ---------------------------------------------------------------------------
# Classes do MVP
# ---------------------------------------------------------------------------
# Dedos: mostre com o polegar DOBRADO (1 = indicador, 2 = indicador+médio...)
CLASSES = ["um", "dois", "tres", "quatro", "palma", "joinha", "deslike"]

# Significado no quiz
SIGNIFICADO = {
    "um": "Alternativa A",
    "dois": "Alternativa B",
    "tres": "Alternativa C",
    "quatro": "Alternativa D",
    "palma": "Mostrar resposta / próximo",
    "joinha": "Fácil (acertei)",
    "deslike": "Errei",
}

MODELO_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
MODELO_PATH = Path(__file__).parent / "modelos" / "hand_landmarker.task"

# Ligações entre os 21 pontos, só para desenhar o "esqueleto" da mão
CONEXOES = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # polegar
    (0, 5), (5, 6), (6, 7), (7, 8),          # indicador
    (5, 9), (9, 10), (10, 11), (11, 12),     # médio
    (9, 13), (13, 14), (14, 15), (15, 16),   # anelar
    (13, 17), (17, 18), (18, 19), (19, 20),  # mínimo
    (0, 17),
]


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------
def baixar_modelo():
    if MODELO_PATH.exists():
        return MODELO_PATH
    MODELO_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Baixando o detector de mãos para {MODELO_PATH} ...")
    urllib.request.urlretrieve(MODELO_URL, MODELO_PATH)
    return MODELO_PATH


def criar_detector():
    opcoes = vision.HandLandmarkerOptions(
        # passa os bytes em vez do caminho: o MediaPipe não abre caminhos com acento ("PÓS", "Visão")
        base_options=mp_python.BaseOptions(model_asset_buffer=baixar_modelo().read_bytes()),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(opcoes)


def detectar(detector, frame_bgr, timestamp_ms):
    """Devolve (pontos_px, lado) ou (None, None) se não houver mão.

    pontos_px: array (21, 3) em pixels (x, y) e z na mesma escala de x.
    lado: 'Left' ou 'Right' (como o MediaPipe enxerga a imagem recebida).
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    imagem = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    res = detector.detect_for_video(imagem, timestamp_ms)
    if not res.hand_landmarks:
        return None, None
    h, w = frame_bgr.shape[:2]
    pts = np.array([[p.x * w, p.y * h, p.z * w] for p in res.hand_landmarks[0]],
                   dtype=np.float32)
    lado = res.handedness[0][0].category_name
    return pts, lado


def desenhar_mao(frame, pts):
    for a, b in CONEXOES:
        cv2.line(frame, tuple(pts[a, :2].astype(int)), tuple(pts[b, :2].astype(int)),
                 (255, 255, 255), 2)
    for p in pts:
        cv2.circle(frame, tuple(p[:2].astype(int)), 4, (0, 140, 255), -1)


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
def normalizar(pts, lado):
    """21 pontos em pixels -> vetor de 63 números.

    1. Translação: o pulso (ponto 0) vira a origem.
    2. Espelhamento: mão esquerda é espelhada para parecer direita.
    3. Escala: divide pela maior distância ao pulso (perto/longe da câmera não importa).
    """
    p = pts - pts[0]
    if lado == "Left":
        p[:, 0] *= -1
    escala = np.linalg.norm(p[:, :2], axis=1).max()
    if escala > 0:
        p = p / escala
    return p.flatten()


def colunas_features():
    return [f"{eixo}{i}" for i in range(21) for eixo in ("x", "y", "z")]


# cadeia de pontos de cada dedo, do pulso até a ponta
DEDOS = [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12],
         [0, 13, 14, 15, 16], [0, 17, 18, 19, 20]]
PONTAS = [4, 8, 12, 16, 20]


def features_invariantes(X):
    """(N, 63) normalizado -> (N, 34) medidas que dependem do FORMATO do gesto,
    não do tamanho/proporção da mão de quem fez.

    - 15 ângulos de dobra (3 juntas x 5 dedos, cosseno em 3D): dedo esticado ~ 1.
    - 5 distâncias ponta -> pulso, relativas ao tamanho da palma (pulso -> base do médio).
    - 4 distâncias entre pontas vizinhas (dedos juntos ou abertos).
    - 1 distância ponta do polegar -> base do indicador (polegar dobrado ou não).
    - 2 direção da palma (pulso -> base do médio) e 2 direção do polegar
      (base -> ponta), em 2D: separa joinha de deslike.
    - 5 altura de cada ponta em relação ao pulso (y), relativa à palma.
    """
    p = np.asarray(X, dtype=np.float32).reshape(-1, 21, 3)
    palma = np.linalg.norm(p[:, 9] - p[:, 0], axis=1, keepdims=True) + 1e-6

    def unit(v):
        return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-6)

    angulos = []
    for cadeia in DEDOS:
        for a, b, c in zip(cadeia, cadeia[1:], cadeia[2:]):
            angulos.append(np.sum(unit(p[:, b] - p[:, a]) * unit(p[:, c] - p[:, b]), axis=1))
    angulos = np.stack(angulos, axis=1)

    dist = lambda i, j: np.linalg.norm(p[:, i] - p[:, j], axis=1)
    ponta_pulso = np.stack([dist(t, 0) for t in PONTAS], axis=1) / palma
    entre_pontas = np.stack([dist(a, b) for a, b in zip(PONTAS, PONTAS[1:])], axis=1) / palma
    polegar_indicador = (dist(4, 5) / palma[:, 0])[:, None]

    dir_palma = unit(p[:, 9, :2] - p[:, 0, :2])
    dir_polegar = unit(p[:, 4, :2] - p[:, 2, :2])
    altura_pontas = (p[:, PONTAS, 1] - p[:, [0], 1]) / palma

    return np.hstack([angulos, ponta_pulso, entre_pontas, polegar_indicador,
                      dir_palma, dir_polegar, altura_pontas]).astype(np.float32)


# ---------------------------------------------------------------------------
# Baseline por regras (sem aprendizado)
# ---------------------------------------------------------------------------
def classificar_por_regras(vetor63):
    """Conta dedos esticados e decide por if/else. Serve de comparação."""
    p = np.asarray(vetor63).reshape(21, 3)[:, :2]
    d = lambda i, j: np.linalg.norm(p[i] - p[j])

    # dedo esticado = ponta mais longe do pulso que a articulação do meio
    esticados = [d(ponta, 0) > 1.1 * d(meio, 0)
                 for ponta, meio in [(8, 6), (12, 10), (16, 14), (20, 18)]]
    n = sum(esticados)
    # polegar esticado = ponta do polegar mais longe da base do médio que a junta anterior
    polegar = d(4, 9) > 1.15 * d(3, 9)

    if n == 4 and polegar:
        return "palma"
    if n == 0 and polegar:
        # y cresce para BAIXO na imagem
        return "joinha" if p[4, 1] < p[0, 1] - 0.2 * d(5, 0) else "deslike"
    return {1: "um", 2: "dois", 3: "tres", 4: "quatro"}.get(n, "nenhum")
