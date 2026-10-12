"""Graphical abstract (3 panels), drawn in code; numbers read from results/results_v2.json."""
import json, numpy as np, matplotlib.pyplot as plt
import os
os.makedirs("figures", exist_ok=True)
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Ellipse, FancyArrowPatch
R = json.load(open("results/results_v2.json"))
plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
INK, GREY, DARK = "#1f2937", "#6b7280", "#4b5563"
RED, RED_BG, BLUE, BLUE_BG, GREEN, GREEN_BG = "#d64545", "#fdecec", "#2f6fd0", "#e8f0fc", "#2e9d5b", "#e7f6ec"
fig = plt.figure(figsize=(13.28, 5.31))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 132.8); ax.set_ylim(0, 53.1); ax.axis("off")
def T(x, y, s, size=10, c=INK, ha="center", va="center", **k): ax.text(x, y, s, fontsize=size, color=c, ha=ha, va=va, **k)
def arrow(x1, y1, x2, y2, c=INK, lw=1.6, ms=14):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=ms, color=c, lw=lw))
def panel(x, w, bg, edge, title):
    ax.add_patch(FancyBboxPatch((x, 1.2), w, 50.4, boxstyle="round,pad=0,rounding_size=2.2", fc=bg, ec=edge, lw=1.4))
    ax.add_patch(FancyBboxPatch((x, 45.4), w, 6.2, boxstyle="round,pad=0,rounding_size=2.2", fc=edge, ec=edge))
    ax.add_patch(Rectangle((x, 45.4), w, 2.5, fc=edge, ec=edge)); T(x + w/2, 48.5, title, 14, "white")
rng = np.random.default_rng(3)
def flakes(x, y, w, h, n, r):
    for _ in range(n):
        ax.add_patch(Ellipse((rng.uniform(x + r, x + w - r), rng.uniform(y + r*0.6, y + h - r*0.6)), 2.2*r, 1.1*r,
                             angle=rng.uniform(-45, 45), fc=DARK, ec="none"))
def slab(x, y, w, h):
    ax.add_patch(Rectangle((x, y - 1.2), w, 1.2, fc="#c8a27a", ec="none"))
    ax.add_patch(Rectangle((x, y), w, h, fc="white", ec=DARK, lw=0.8))

# ---- 1. pitfalls ----
panel(1.2, 40.8, RED_BG, RED, "1. Unconstrained designs mislead")
slab(5, 27.0, 12, 12.5); flakes(5, 27.0, 12, 12.5, 30, 1.1); T(11, 24.2, "Baseline", 10, GREY)
slab(25, 27.0, 12, 5.6); flakes(25, 27.0, 12, 5.6, 13, 1.1); T(31, 24.2, "Thinned", 10, GREY)
arrow(18.2, 32.5, 23.8, 32.5, RED)
T(31, 37.2, f"−{-R['unconstrained']['loading_change_pct']:.0f}% active\nmaterial", 12, RED)
T(21.6, 19.4, "high retention only because\nless material is stored", 10.5, INK)
ax.plot([4, 39.2], [14.6, 14.6], color=RED, lw=0.6, alpha=0.5)
gd, gs = R["gain"]["Q_ratio_x"], R["spme"]["apparent_gain_spme"]
for i, (lab, v, c) in enumerate([("DFN", gd, BLUE), ("SPMe", gs, RED)]):
    y = 10.8 - i*3.0; L = 21 * v / gs
    ax.add_patch(Rectangle((13, y - 1.0), L, 2.0, fc=c, ec="none"))
    T(11.8, y, lab, 10.5, ha="right"); T(13.8 + L, y, f"{v:.2f}×" if v < 2 else f"{v:.1f}×", 10.5, ha="left")
T(21.6, 3.6, f"reduced-order model inflates the gain ≈{R['spme']['overstatement']:.0f}×", 10.5, RED)

# ---- 2. method ----
panel(45.2, 40.8, BLUE_BG, BLUE, "2. Fixed-loading ML inverse design")
def step(y, h, title, sub, fc="white", tc=INK, sc=GREY):
    ax.add_patch(FancyBboxPatch((49, y), 33.2, h, boxstyle="round,pad=0,rounding_size=1.2", fc=fc, ec=BLUE, lw=1.1))
    T(65.6, y + h*0.66, title, 11.5, tc); T(65.6, y + h*0.28, sub, 10, sc)
step(35.5, 8.0, "DFN (P2D) simulations", f"N = {R['dataset']['n_dfn']} designs")
arrow(65.6, 35.3, 65.6, 32.9, BLUE)
step(24.5, 8.0, "Tree-ensemble surrogates", f"CV R² = {R['cv']['Q_ratio']['RandomForest']['R2']:.2f} / {R['cv']['dV_3C']['RandomForest']['R2']:.2f}")
arrow(65.6, 24.3, 65.6, 21.9, BLUE)
ax.add_patch(FancyBboxPatch((49, 12.5), 33.2, 9.0, boxstyle="round,pad=0,rounding_size=1.2", fc=BLUE, ec=BLUE))
T(65.6, 18.9, "Optimize at fixed loading", 11.5, "white")
T(65.6, 15.0, r"$S_0=(1-\varepsilon)\,L=65\ \mu\mathrm{m}$", 13.5, "white")
arrow(65.6, 12.3, 65.6, 9.9, BLUE)
step(2.6, 7.2, "DFN re-validation", f"within {R['validation']['Q_err_pct']:.1f}% / {R['validation']['dV_err_pct']:.1f}%")
ax.add_patch(Circle((53.0, 6.2), 1.6, fc=GREEN, ec="none")); ax.plot([52.2, 52.8, 53.9], [6.2, 5.5, 7.0], color="white", lw=1.6)

# ---- 3. result ----
panel(89.2, 42.4, GREEN_BG, GREEN, "3. DFN-validated gain")
def electrode(x, y, w, h, aligned):
    ax.add_patch(Rectangle((x, y - 1.1), w, 1.1, fc="#c8a27a", ec="none"))
    ax.add_patch(Rectangle((x, y), w, h, fc="white", ec=DARK, lw=0.9))
    ax.add_patch(Rectangle((x, y + h), w, 1.0, fc="#d1d5db", ec="none"))
    if aligned:
        for cx in np.arange(x + 1.5, x + w - 0.5, 2.8):
            for cy in np.arange(y + 1.0, y + h - 0.5, 1.15): ax.add_patch(Ellipse((cx, cy), 2.0, 0.75, fc=DARK, ec="none"))
        for cx in np.arange(x + 2.9, x + w - 1.0, 2.8): arrow(cx, y + h + 0.6, cx, y + 1.0, BLUE, lw=1.3, ms=9)
    else:
        flakes(x, y, w, h, 30, 1.35)
        xs = np.array([0.45, 0.20, 0.62, 0.30, 0.70, 0.42]) * w + x; ys = np.linspace(y + h + 0.6, y + 1.0, 6)
        ax.plot(xs, ys, color=RED, lw=1.5, zorder=5); arrow(xs[-2], ys[-2], xs[-1], ys[-1], RED, lw=1.5, ms=9)
s = R["matched"]["selected"]
electrode(93.0, 25.0, 15, 14, False); electrode(113.0, 25.0, 15, 14, True)
arrow(108.6, 32, 112.4, 32, GREEN, lw=1.8)
T(100.5, 21.4, r"Baseline, $b$ = 1.5", 10.5); T(120.5, 21.4, rf"Optimized, $b$ = {s['b']:.2f}", 10.5)
T(110.4, 42.3, "schematic: lower tortuosity, same loading", 9.5, GREY, style="italic")
for i, (big, small) in enumerate([(f"{R['gain']['Q_ratio_x']:.2f}×", "3C capacity\nretention"),
                                   (f"−{R['gain']['dV_reduction_pct']:.0f}%", "3C\npolarization"),
                                   ("1C → 2C", "plating-free\nCC charging*")]):
    xc = 97.6 + i*13.4
    ax.add_patch(FancyBboxPatch((xc - 6.1, 5.2), 12.2, 11.4, boxstyle="round,pad=0,rounding_size=1.2", fc="white", ec=GREEN, lw=1.1))
    T(xc, 12.9, big, 16 if i < 2 else 14, GREEN); T(xc, 8.2, small, 9.5)
T(110.4, 2.8, "*thermodynamic plating criterion (DFN)", 8.5, GREY)

arrow(42.2, 26, 45.0, 26, INK, lw=2, ms=18); arrow(86.2, 26, 89.0, 26, INK, lw=2, ms=18)
fig.savefig("figures/graphical_abstract.png", dpi=300); fig.savefig("figures/graphical_abstract.pdf")
print("saved")
