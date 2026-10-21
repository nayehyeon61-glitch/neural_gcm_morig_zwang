"""Draw docs/figures/model_structure.png: the closure inside NeuralGCM.

Needs matplotlib and a Hangul font (e.g. NanumGothic); it is not a runtime
dependency of the package. Fonts are passed as files, regular then bold.
Usage: python docs/figures/make_model_figure.py docs/figures/model_structure.png \
           [NanumGothic.ttf NanumGothicBold.ttf]
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from matplotlib import font_manager

INK, MUTED, LINE = "#1f2328", "#59636e", "#8c959f"
MARKOV, MEMORY, NOISE, NEW, TRAIN = "#dbe9f6", "#e3f1e0", "#f8e6d2", "#f3dcef", "#f2f2f0"


def box(ax, x, y, w, h, title, lines, face, edge=LINE, dashed=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012",
                                fc=face, ec=edge, lw=1.6 if dashed else 1.0, ls="--" if dashed else "-"))
    ax.text(x + 0.012, y + h - 0.03, title, fontsize=12.5, weight="bold", color=INK, va="top")
    for i, line in enumerate(lines):
        ax.text(x + 0.012, y + h - 0.068 - i * 0.027, line, fontsize=9.6, color=MUTED, va="top")


def arrow(ax, a, b, text=None, rad=0.0, color=LINE, offset=(0, 0.012)):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=13, lw=1.4, color=color,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))
    if text:
        ax.text((a[0] + b[0]) / 2 + offset[0], (a[1] + b[1]) / 2 + offset[1], text, fontsize=9,
                color=MUTED, ha="center")


def main(path):
    fig = plt.figure(figsize=(16, 9.4), dpi=170)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.02, 0.965, "NeuralGCM + Mori–Zwanzig 근사 클로저: 현재 모델 구조", fontsize=19, weight="bold", color=INK, va="top")
    ax.text(0.02, 0.925, "dx/dt = M(x)  +  ∫ K(s) x(t−s) ds  +  F(t)        "
            "Markov = 고정 NeuralGCM · 메모리 = 이력 MLP · 노이즈 = DDPM(빠른) + AR(1)(느린)",
            fontsize=12, color=MUTED, va="top")

    # Offline training band.
    ax.add_patch(FancyBboxPatch((0.02, 0.70), 0.96, 0.19, boxstyle="round,pad=0.004,rounding_size=0.01",
                                fc=TRAIN, ec="none"))
    ax.text(0.03, 0.88, "오프라인 학습 (train / validation 분할만 사용)", fontsize=11, weight="bold", color=INK, va="top")
    steps = [("ERA5 → 인코딩", ["σ 좌표 상태 x(t)", "frozen NeuralGCM 1구간 적분"]),
             ("잔차 r", ["(x_true(t+Δt) − Φ(x_true(t)))/Δt", "K/s, m/s², 1/s"]),
             ("메모리 μθ 학습", ["r 전체 블록(K×6h)에 MSE", "검증 손실로 선택"]),
             ("DDPM 학습", ["r−μ 의 시간 DCT, k≥2 (≤48h)", "ε-예측, cosine 스케줄"]),
             ("노이즈 통계 (PR #1)", ["모드별 진폭 이득 (검증)", "블록평균 AR(1): φ, Σ=LLᵀ", "상관 길이 L (fast/slow)"])]
    for i, (title, lines) in enumerate(steps):
        x = 0.03 + i * 0.19
        box(ax, x, 0.715, 0.17, 0.14, title, lines, "white", dashed=(i == 4), edge=LINE if i < 4 else "#a3489a")
        if i:
            arrow(ax, (x - 0.02, 0.785), (x - 0.002, 0.785))

    # Forecast loop.
    ax.text(0.03, 0.665, "예측 루프 (48 h 블록마다 생성, 6 h 구간마다 적용; 첫 구간은 순수 NeuralGCM)",
            fontsize=11, weight="bold", color=INK, va="top")
    box(ax, 0.03, 0.40, 0.18, 0.22, "인과적 이력 h",
        ["최근 H개 상태 x", "직전 H개 보정 a", "(scale 곱하기 전 값, PR #1)", "train 통계로 정규화"], MEMORY)
    box(ax, 0.27, 0.53, 0.22, 0.11, "메모리 항  μθ(h)", ["유한 이력 MLP → K×C 블록", "closure.py"], MEMORY)
    box(ax, 0.27, 0.375, 0.22, 0.14, "빠른 노이즈  DDPM", ["h 조건, DCT k≥2 → IDCT", "블록 합 = 0 · 모드별 이득",
                                                          "공간 상관 L_fast · × fast_noise_scale"], NOISE)
    box(ax, 0.27, 0.215, 0.22, 0.14, "느린 노이즈  AR(1)  (PR #1)", ["s' = φ s + √(1−φ²) L z", "블록 상수, 블록 간 지속",
                                                                   "공간 상관 L_slow · × slow_noise_scale"], NEW,
        edge="#a3489a", dashed=True)
    box(ax, 0.55, 0.40, 0.16, 0.17, "합산 · 제한", ["μ + 빠른 + 느린", "열·채널별 5σ 클리핑", "단위 복원 → a (SI)",
                                               "× correction_scale"], "white")
    box(ax, 0.76, 0.27, 0.21, 0.36, "NeuralGCM  (Markov 항, 고정)",
        ["① learned physics tendency G(x)", "② + a_n  ← 가로채기로 주입", "    DycoreWithPhysicsCorrector",
         "③ dycore IMEX substep 적분", "    (a_n 은 구간 내내 고정)", "", "neuralgcm_backend.py"], MARKOV)
    box(ax, 0.42, 0.06, 0.26, 0.11, "다음 상태  x(t+Δt)", ["observe_generated(): 상태와", "scale 전 보정을 이력에 추가"], "white")

    arrow(ax, (0.21, 0.585), (0.268, 0.585))
    arrow(ax, (0.21, 0.46), (0.268, 0.45))
    arrow(ax, (0.21, 0.42), (0.268, 0.31), rad=-0.15, text="z 만 사용", offset=(-0.03, -0.03))
    arrow(ax, (0.49, 0.585), (0.548, 0.53), rad=-0.1)
    arrow(ax, (0.49, 0.45), (0.548, 0.48))
    arrow(ax, (0.49, 0.30), (0.548, 0.42), rad=0.12)
    arrow(ax, (0.71, 0.485), (0.758, 0.485), text="a_n  (6 h)", offset=(0, 0.02))
    arrow(ax, (0.865, 0.27), (0.68, 0.12), rad=-0.15, text="적분 결과", offset=(0.07, 0.0))
    arrow(ax, (0.42, 0.11), (0.12, 0.398), rad=-0.25, text="이력 갱신 (피드백)", offset=(-0.15, 0.0), color="#4d7c4a")

    ax.text(0.02, 0.015, "보정은 디코딩된 출력이 아니라 learned physics tendency에 더해져 dycore가 함께 적분합니다. "
            "점선 상자 = PR #1 추가. 사영 연산자·직교 동역학은 계산하지 않는 MZ 영감 근사입니다.",
            fontsize=9.5, color=MUTED)
    fig.savefig(path, facecolor="white")


if __name__ == "__main__":
    for font in sys.argv[2:]:
        font_manager.fontManager.addfont(font)
    if len(sys.argv) > 2:
        plt.rcParams["font.family"] = [font_manager.FontProperties(fname=sys.argv[2]).get_name(), "DejaVu Sans"]
    main(sys.argv[1] if len(sys.argv) > 1 else "model_structure.png")
