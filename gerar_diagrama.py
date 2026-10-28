"""
Gera o diagrama do pipeline para a apresentação.

Uso:
    python gerar_diagrama.py

Salva diagrama_pipeline.svg (texto editável) e diagrama_pipeline.png (200 dpi).
Os números do quadro de resultados são copiados da saída do treinar_gestos.py
(validação cruzada deixando uma pessoa de fora); atualize RESULTADOS ao retreinar.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

BASE = Path(__file__).parent

# Segoe UI se existir (Windows); senão fica a fonte padrão do matplotlib
for arq in ("segoeui.ttf", "segoeuib.ttf"):
    caminho = Path("C:/Windows/Fonts") / arq
    if caminho.exists():
        font_manager.fontManager.addfont(str(caminho))
plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]
plt.rcParams["svg.fonttype"] = "none"  # texto continua texto no SVG

TEXTO = "#1f2937"
SUAVE = "#6b7280"
CORES = {  # (preenchimento, borda)
    "pre": ("#e3ecfb", "#3b6fd8"),      # pré-treinado, congelado
    "treinado": ("#fdebd3", "#d9730d"),  # treinado no projeto
    "fixo": ("#f1f2f4", "#8a919c"),      # processamento sem aprendizado
}

ETAPAS = [
    ("Webcam", "quadro espelhado\n(cv2.flip)", "fixo"),
    ("MediaPipe\nHand Landmarker", "21 pontos 3D\nda mão = 63 valores", "pre"),
    ("Normalização", "pulso na origem,\nmão esquerda → direita,\nescala pela mão", "fixo"),
    ("Features\ninvariantes", "34 medidas: ângulos\nde dobra, distâncias,\ndireção do polegar", "fixo"),
    ("MLP\n(64 → 32)", "7 gestos + confiança\n(aceita se ≥ 0,70)", "treinado"),
    ("Estabilizador", "5 de 7 quadros\nconcordam e gesto\nsegurado por 0,6 s", "fixo"),
    ("Quiz", "1–4 dedos: A–D\npalma: revela\njoinha: próxima\ndeslike: rever depois", "fixo"),
]

# acurácia por pessoa testada: pessoa1, pessoa2, pessoa3, pessoa4
RESULTADOS = [
    ("Regras (baseline)", [0.936, 0.868, 0.902, 0.810], False),
    ("MLP coordenadas", [0.929, 0.931, 0.908, 0.881], False),
    ("MLP invariantes (usado no app)", [0.949, 0.919, 0.949, 0.896], True),
]
CONCLUSOES = [
    "MLP invariantes tem a melhor média e vence\nem 3 das 4 pessoas testadas.",
    "A vantagem sobre coordenadas (~1,6 pp) fica\ndentro da variação entre pessoas.",
    "As regras são as mais instáveis: de 81% a 94%\nconforme quem faz o gesto.",
]

LARG, ALT, VAO, X0 = 19.0, 24.0, 3.5, 3.0
Y_CAIXA = 18.0


def centro(i):
    return X0 + i * (LARG + VAO) + LARG / 2


def chave(ax, i_ini, i_fim, y, texto, cor, acima=True):
    """Colchete horizontal sobre (ou sob) as caixas i_ini..i_fim, com rótulo."""
    x1 = X0 + i_ini * (LARG + VAO)
    x2 = X0 + i_fim * (LARG + VAO) + LARG
    d = -1.2 if acima else 1.2
    ax.plot([x1, x1, x2, x2], [y + d, y, y, y + d], color=cor, lw=1.4,
            solid_capstyle="round")
    ax.text((x1 + x2) / 2, y - d * 1.1, texto, ha="center",
            va="bottom" if acima else "top", fontsize=11, color=cor, fontweight="bold")


def pct(v):
    return f"{100 * v:.1f}".replace(".", ",") + "%"


def desenhar_resultados(fig):
    """Quadro inferior: dot plot da validação cruzada + conclusões."""
    fig.patches.append(FancyBboxPatch((0.02, 0.03), 0.96, 0.33,
                                      boxstyle="round,pad=0,rounding_size=0.012",
                                      transform=fig.transFigure, fc="#fafafa",
                                      ec="#d1d5db", lw=1.2, zorder=-1))
    fig.text(0.04, 0.33, "Resultados: validação cruzada deixando uma pessoa de fora",
             fontsize=14, fontweight="bold", color=TEXTO, va="top")
    fig.text(0.04, 0.297, "4 participantes · 8.228 quadros · cada pessoa é testada "
             "uma vez com o modelo treinado nas outras três",
             fontsize=10, color=SUAVE, va="top")

    ax = fig.add_axes([0.205, 0.085, 0.42, 0.175])
    for i, (nome, accs, destaque) in enumerate(RESULTADOS):
        cor = CORES["treinado"][1] if destaque else SUAVE
        media, desvio = np.mean(accs), np.std(accs)
        ax.plot([media - desvio, media + desvio], [i, i], color=cor, lw=2,
                solid_capstyle="round", zorder=2)
        ax.scatter(accs, [i] * len(accs), s=34, fc="white", ec=cor, lw=1.4, zorder=3)
        ax.scatter([media], [i], s=120, fc=cor, ec="white", lw=2, zorder=4)
        ax.text(0.985, i, f"{pct(media)} ± {pct(desvio)}", va="center", ha="left",
                fontsize=11, fontweight="bold" if destaque else "normal", color=TEXTO)
        ax.text(0.785, i, nome, va="center", ha="right", fontsize=11,
                fontweight="bold" if destaque else "normal", color=TEXTO)
    ax.set_xlim(0.79, 0.98)
    ax.set_ylim(-0.6, len(RESULTADOS) - 0.4)
    ax.set_yticks([])
    ax.set_xticks([0.80, 0.85, 0.90, 0.95])
    ax.set_xticklabels(["80%", "85%", "90%", "95%"], fontsize=9.5, color=SUAVE)
    ax.grid(axis="x", color="#e5e7eb", lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    for lado in ax.spines.values():
        lado.set_visible(False)
    ax.set_facecolor("none")
    ax.set_xlabel("acurácia  ·  ● média   ○ cada pessoa   — média ± desvio",
                  fontsize=9.5, color=SUAVE, labelpad=6)

    y = 0.245
    for texto in CONCLUSOES:
        fig.text(0.735, y, "•", fontsize=13, color=CORES["treinado"][1], va="top")
        fig.text(0.748, y, texto, fontsize=11, color=TEXTO, va="top", linespacing=1.3)
        y -= 0.062


def main():
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_axes([0, 0.38, 1, 0.62])
    ax.set_xlim(0, 160)
    ax.set_ylim(5, 58)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(80, 54.5, "Quiz por gestos: do quadro da webcam à ação no quiz",
            ha="center", va="center", fontsize=19, fontweight="bold", color=TEXTO)

    for i, (titulo, detalhe, tipo) in enumerate(ETAPAS):
        fundo, borda = CORES[tipo]
        x = X0 + i * (LARG + VAO)
        ax.add_patch(FancyBboxPatch((x, Y_CAIXA), LARG, ALT,
                                    boxstyle="round,pad=0,rounding_size=1.6",
                                    fc=fundo, ec=borda, lw=2))
        ax.text(x + 1.4, Y_CAIXA + ALT - 1.6, str(i + 1), ha="left", va="top",
                fontsize=10, color=borda, fontweight="bold")
        ax.text(x + LARG / 2, Y_CAIXA + ALT - 6.0, titulo, ha="center", va="center",
                fontsize=12.5, fontweight="bold", color=TEXTO, linespacing=1.15)
        ax.text(x + LARG / 2, Y_CAIXA + 8.0, detalhe, ha="center", va="center",
                fontsize=9.6, color=TEXTO, linespacing=1.35)
        if i < len(ETAPAS) - 1:
            ax.add_patch(FancyArrowPatch((x + LARG + 0.4, Y_CAIXA + ALT / 2),
                                         (x + LARG + VAO - 0.4, Y_CAIXA + ALT / 2),
                                         arrowstyle="-|>", mutation_scale=16,
                                         color=SUAVE, lw=1.6))

    # transfer learning: backbone congelado (2) + cabeça treinada (5)
    chave(ax, 1, 4, 47.5, "Transfer learning por extração de características", TEXTO)
    ax.text(centro(1), 43.6, "pré-treinado pelo Google · congelado", ha="center",
            fontsize=9, color=CORES["pre"][1], fontweight="bold")
    ax.text(centro(4), 43.6, "único componente treinado", ha="center",
            fontsize=9, color=CORES["treinado"][1], fontweight="bold")

    # ritmo: previsão por quadro vs. eventos ao longo do tempo
    chave(ax, 0, 4, 14.5, "a cada quadro da webcam", SUAVE, acima=False)
    chave(ax, 5, 6, 14.5, "ao longo do tempo: quadros → eventos", SUAVE, acima=False)

    legenda = [Patch(fc=CORES[t][0], ec=CORES[t][1], lw=2, label=r)
               for t, r in (("pre", "Modelo pré-treinado (pesos congelados)"),
                            ("treinado", "Treinado com os dados do projeto"),
                            ("fixo", "Processamento sem aprendizado"))]
    ax.legend(handles=legenda, loc="center", bbox_to_anchor=(80, 7.8),
              bbox_transform=ax.transData,
              ncol=3, frameon=False, fontsize=10.5, handlelength=1.6,
              handleheight=1.1, columnspacing=2.5)

    desenhar_resultados(fig)
    for ext in ("svg", "png"):
        fig.savefig(BASE / f"diagrama_pipeline.{ext}", dpi=200, facecolor="white")
    print("Salvo: diagrama_pipeline.svg e diagrama_pipeline.png")


if __name__ == "__main__":
    main()
