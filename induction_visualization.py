# -*- coding: utf-8 -*-
"""
Визуализация индуктивного аргумента ТАМШ (AMGT).
Панель A: иерархия вложенных систем n=1..4 с асимметричными шлюзами.
Панель B: численная проверка rank-nullity — спектр сингулярных значений
          отображения F: R^3 -> R^9 (подъём). Только 3 ненулевых значения.
Панель C: каскад проекций R^9 -> R^6 -> R^3 (спуск) — «фильтрация компиляции».
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

rng = np.random.default_rng(42)

BG      = "#0d1117"
FG      = "#e6edf3"
GOLD    = "#d4a94e"
GREEN   = "#3fb950"
RED     = "#f85149"
BLUE    = "#58a6ff"
GRAY    = "#8b949e"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "text.color": FG, "axes.edgecolor": GRAY,
    "axes.labelcolor": FG, "xtick.color": FG, "ytick.color": FG,
    "font.family": "DejaVu Sans", "font.size": 10,
})

fig = plt.figure(figsize=(16, 6.2))
fig.suptitle("ТАМШ: индуктивный аргумент самоподобия и асимметрия шлюзов",
             fontsize=15, fontweight="bold", color=GOLD, y=0.99)

# ---------------- Панель A: иерархия уровней ----------------
axA = fig.add_subplot(1, 3, 1)
axA.set_xlim(0, 10); axA.set_ylim(0, 10); axA.axis("off")
axA.set_title("A. Иерархия систем (индукция по n)", color=FG, pad=12)

levels = [
    (1.0, "n = 1:  искусственные микро-миры\n(симуляции, игры, нейросети)"),
    (3.4, "n = 2:  наша Вселенная\n(3D-движок, матричная механика)"),
    (5.8, "n = 3:  родительская система\n(внешний мир, ИИ-оператор)"),
    (8.2, "n → ∞:  Мультивселенная\n(глобальный математический код)"),
]
for y, label in levels:
    box = FancyBboxPatch((1.2, y - 0.75), 7.6, 1.5,
                         boxstyle="round,pad=0.12",
                         fc="#161b22", ec=GOLD, lw=1.4)
    axA.add_patch(box)
    axA.text(5.0, y, label, ha="center", va="center", fontsize=9.5, color=FG)

for y0, y1 in [(1.75, 2.65), (4.15, 5.05), (6.55, 7.45)]:
    # спуск разрешён (проекция P̂, o-малое)
    axA.add_patch(FancyArrowPatch((3.2, y1), (3.2, y0),
                  arrowstyle="-|>", mutation_scale=18, color=GREEN, lw=2.2))
    # подъём заблокирован (O-большое)
    axA.add_patch(FancyArrowPatch((6.8, y0), (6.8, y1),
                  arrowstyle="-|>", mutation_scale=18, color=RED,
                  lw=2.2, linestyle=(0, (4, 3))))
    ym = (y0 + y1) / 2
    axA.text(7.15, ym, "✕", color=RED, fontsize=13, va="center", fontweight="bold")

axA.legend(handles=[
    mpatches.Patch(color=GREEN, label="спуск: проекция P̂ (o-малое) — разрешён"),
    mpatches.Patch(color=RED,   label="подъём (O-большое) — заблокирован"),
], loc="lower center", bbox_to_anchor=(0.5, -0.14), frameon=False, fontsize=8.5)

# ---------------- Панель B: rank-nullity, подъём ----------------
axB = fig.add_subplot(1, 3, 2)
axB.set_title("B. Подъём F: ℝ³ → ℝ⁹ — теорема о ранге", color=FG, pad=12)

# 200 случайных линейных отображений 9x3; сингулярный спектр в R^9
S = np.array([np.linalg.svd(rng.normal(size=(9, 3)), compute_uv=False) for _ in range(200)])
mean_s = np.zeros(9)
mean_s[:3] = S.mean(axis=0)          # только 3 ненулевых сингулярных числа
colors = [BLUE] * 3 + [RED] * 6
bars = axB.bar(range(1, 10), mean_s, color=colors, edgecolor=BG)
axB.set_xlabel("номер сингулярного значения (направление в ℝ⁹)")
axB.set_ylabel("среднее сингулярное значение (200 случайных F)")
axB.set_xticks(range(1, 10))
axB.axvspan(3.5, 9.5, color=RED, alpha=0.08)
axB.text(6.5, mean_s[0] * 0.55, "dim(Im F) ≤ 3:\n6 из 9 измерений\nнедостижимы\n(тождественный ноль)",
         ha="center", color=RED, fontsize=9.5)
axB.text(2.0, mean_s[0] * 1.02, "достижимое\nподпространство", ha="center", color=BLUE, fontsize=9)
axB.spines[["top", "right"]].set_visible(False)

# ---------------- Панель C: каскад проекций, спуск ----------------
axC = fig.add_subplot(1, 3, 3)
axC.set_title("C. Спуск P̂: ℝ⁹ → ℝ⁶ → ℝ³ — фильтрация компиляции", color=FG, pad=12)

v9 = rng.normal(size=9)                      # объект родительской системы
P96 = np.zeros((6, 9)); P96[:, :6] = np.eye(6)   # ортопроекторы (в подходящем базисе)
P63 = np.zeros((3, 6)); P63[:, :3] = np.eye(3)
v6, v3 = P96 @ v9, P63 @ (P96 @ v9)

stages = [("ℝ⁹\n(уровень n+1)", v9, 9), ("ℝ⁶\n(шлюз)", v6, 6), ("ℝ³\n(наш мир)", v3, 3)]
width = 0.8
for k, (name, v, dim) in enumerate(stages):
    x0 = k * 4
    comps = np.zeros(9); comps[:dim] = np.abs(v)
    kept = min(dim, 3)
    cols = [BLUE] * 3 + [GOLD] * (dim - 3) if dim > 3 else [BLUE] * 3
    axC.bar(x0 + np.arange(dim) * 0.38, np.abs(v), width=0.3,
            color=cols[:dim], edgecolor=BG)
    axC.text(x0 + (dim - 1) * 0.19, -0.55, name, ha="center", fontsize=9.5, color=FG)
    if k < 2:
        axC.annotate("", xy=(x0 + 3.55, 1.0), xytext=(x0 + dim * 0.38 + 0.15, 1.0),
                     arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=2))
        axC.text(x0 + 3.0, 1.15, "P̂", color=GREEN, fontsize=12, ha="center")

axC.set_ylim(-1.0, 2.6)
axC.set_xticks([]); axC.set_ylabel("|компонента вектора|")
axC.legend(handles=[
    mpatches.Patch(color=BLUE, label="компоненты, совместимые с 3D-базисом (сохраняются)"),
    mpatches.Patch(color=GOLD, label="старшие мерности — отсекаются: f(x) ∈ o(x)"),
], loc="upper right", frameon=False, fontsize=8.5)
axC.spines[["top", "right"]].set_visible(False)

fig.text(0.5, 0.005,
         "Инвариант: операции линейной алгебры структурно одинаковы на каждом уровне n → "
         "асимметрия шлюзов воспроизводится на всей иерархии.",
         ha="center", fontsize=9.5, color=GRAY, style="italic")

plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.savefig("/mnt/user-data/outputs/induction_visualization.png", dpi=200,
            facecolor=BG, bbox_inches="tight")
print("saved")
