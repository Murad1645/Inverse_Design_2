"""Regenerate the data figures from the v2 datasets and results_v2.json.
House style: no in-plot titles (captions carry them), non-bold labels,
mathtext subscripts, no internal dataset-version labels."""
import json, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from electrode_model import FEAT, RANGES, S0

plt.rcParams.update({"axes.titleweight": "normal", "axes.labelweight": "normal", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300,
                     "legend.frameon": False})
BLUE, ORANGE, GREY = "#3b7dd8", "#e4572e", "#6b6b6b"
QLAB, DVLAB = r"$Q_{3\mathrm{C}}/Q_{0.5\mathrm{C}}$", r"$\Delta V_{3\mathrm{C}}$ (V)"
R = json.load(open("results_v2.json"))
dfn = pd.read_csv("dataset_dfn_v2.csv"); dfn = dfn[dfn.solver_ok == True]
spm = pd.read_csv("dataset_spme_v2.csv"); spm = spm[spm.solver_ok == True]
def panel(ax, s): ax.text(-0.14, 1.02, s, transform=ax.transAxes, fontsize=11)

# --- SPMe vs DFN: (a) distribution, (b) paired parity -------------------
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
bins = np.linspace(0, 1, 26)
ax[0].hist(spm.Q_ratio, bins=bins, alpha=0.6, color=ORANGE, label=f"SPMe: {100*R['dataset']['below02_spme']:.0f}% below 0.2")
ax[0].hist(dfn.Q_ratio, bins=bins, alpha=0.6, color=BLUE, label=f"DFN: {100*R['dataset']['below02_dfn']:.0f}% below 0.2")
ax[0].axvline(0.2, ls="--", color="black", lw=1)
ax[0].set_xlabel(f"Rate capability {QLAB}"); ax[0].set_ylabel("Number of designs"); ax[0].legend(fontsize=9)
m = dfn.merge(spm, on="sim_id", suffixes=("_d", "_s"))
ax[1].scatter(m.Q_ratio_d, m.Q_ratio_s, s=5, color=BLUE, alpha=0.4, edgecolors="none")
ax[1].plot([0, 1], [0, 1], "--", color="black", lw=1)
ax[1].set_xlabel(f"DFN {QLAB}"); ax[1].set_ylabel(f"SPMe {QLAB}")
ax[1].text(0.04, 0.93, rf"Spearman $\rho$ = {R['spme']['spearman']:.2f}", transform=ax[1].transAxes)
ax[1].set_xlim(0, 1); ax[1].set_ylim(0, 1); ax[1].set_aspect("equal")
panel(ax[0], "(a)"); panel(ax[1], "(b)")
plt.tight_layout(); plt.savefig("fig_spme_vs_dfn.png", bbox_inches="tight"); plt.close()

# --- Pareto front -----------------------------------------------------------
P = np.load("pareto_v2.npz"); qp, dp, pf, k = P["qp"], P["dp"], P["pf"], int(P["k"])
fig, ax = plt.subplots(figsize=(6, 4.4))
ax.scatter(dp[::40], qp[::40], s=5, color="#c8c8c8", alpha=0.6, edgecolors="none", label="Loading-matched candidates")
o = pf[np.argsort(dp[pf])]
ax.plot(dp[o], qp[o], "-o", color=BLUE, ms=4, lw=1.5, label="Pareto front")
ax.scatter(dp[k], qp[k], s=160, marker="*", color=ORANGE, edgecolor="black", zorder=5, label="Selected design")
ax.set_xlabel(f"Polarization {DVLAB}"); ax.set_ylabel(f"Rate capability {QLAB}")
ax.legend(fontsize=9, loc="upper right")
ins = ax.inset_axes([0.42, 0.34, 0.5, 0.4])
x0, x1 = dp[pf].min() - 0.01, dp[pf].max() + 0.01; y0, y1 = qp[pf].min() - 0.01, qp[pf].max() + 0.005
w = (dp > x0) & (dp < x1) & (qp > y0) & (qp < y1)
ins.scatter(dp[w], qp[w], s=6, color="#c8c8c8", edgecolors="none")
ins.plot(dp[o], qp[o], "-o", color=BLUE, ms=4, lw=1.5)
ins.scatter(dp[k], qp[k], s=160, marker="*", color=ORANGE, edgecolor="black", zorder=5)
ins.set_xlim(x0, x1); ins.set_ylim(y0, y1); ins.tick_params(labelsize=8)
ins.spines["top"].set_visible(True); ins.spines["right"].set_visible(True)
ax.indicate_inset_zoom(ins, edgecolor=GREY)
plt.tight_layout(); plt.savefig("fig_pareto_front.png", bbox_inches="tight"); plt.close()

# --- Rate capability: (a) discharge retention, (b) charge plating margin ----
rs = R["rate_sweep"]; ch = R["charging"]; s = R["matched"]["selected"]
sel_lab = rf"Optimized ($\varepsilon$={s['eps']:.2f}, $b$={s['b']:.2f}, $R_p$={s['Rp_um']:.1f} µm)"
base_lab = r"Baseline ($\varepsilon$=0.35, $b$=1.5, $R_p$=7 µm)"
fig, ax = plt.subplots(1, 2, figsize=(10.5, 4))
ax[0].plot([r["C"] for r in rs], [r["base_ret"] for r in rs], "o-", color=GREY, lw=1.8, ms=6, label=base_lab)
ax[0].plot([r["C"] for r in rs], [r["sel_ret"] for r in rs], "s-", color=BLUE, lw=1.8, ms=6, label=sel_lab)
ax[0].set_xlabel("Discharge C-rate"); ax[0].set_ylabel(r"Capacity retention $Q_{\mathrm{C}}/Q_{0.5\mathrm{C}}$")
ax[0].legend(fontsize=8); ax[0].grid(alpha=0.25)
ax[1].axhline(0, color="black", lw=1, ls="--")
ax[1].plot([c["C"] for c in ch], [c["base_plating_mV"] for c in ch], "o-", color=GREY, lw=1.8, ms=6, label="Baseline")
ax[1].plot([c["C"] for c in ch], [c["sel_plating_mV"] for c in ch], "s-", color=BLUE, lw=1.8, ms=6, label="Optimized")
ax[1].set_xlabel("Charge C-rate"); ax[1].set_ylabel(r"min($\phi_s-\phi_e$) at anode/separator (mV)")
ax[1].text(0.97, 0.93, "plating possible below 0 mV", transform=ax[1].transAxes, ha="right", fontsize=8)
ax[1].legend(fontsize=8, loc="lower left"); ax[1].grid(alpha=0.25); ax[1].set_xticks([1, 2, 3, 4, 5])
panel(ax[0], "(a)"); panel(ax[1], "(b)")
plt.tight_layout(); plt.savefig("fig_rate_and_plating.png", bbox_inches="tight"); plt.close()

# --- SHAP: loading-aware parameterisation -------------------------------------
LAB = {"eps": r"$\varepsilon$ (porosity)", "b": r"$b$ (Bruggeman)", "Rp_um": r"$R_p$ (µm)",
       "L_um": r"$L$ (µm)", "S0_um": r"$S_0$ (loading, µm)"}
for key, fname in [("shap_loading_param", "fig_shap_loading_param.png"), ("shap_full", "fig_shap_full.png")]:
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for a, t, lab in [(ax[0], "Q_ratio", f"mean |SHAP| on {QLAB}"), (ax[1], "dV_3C", f"mean |SHAP| on {DVLAB}")]:
        imp = R[key][t]; names = sorted(imp, key=imp.get)
        a.barh([LAB[n] for n in names], [imp[n] for n in names], color=BLUE, height=0.6)
        a.set_xlabel(lab); a.xaxis.set_major_locator(plt.MaxNLocator(5))
    panel(ax[0], "(a)"); panel(ax[1], "(b)")
    plt.tight_layout(); plt.savefig(fname, bbox_inches="tight"); plt.close()

# --- Gain vs achievable tortuosity -----------------------------------------------
bf = R["b_floor"]
fig, ax = plt.subplots(figsize=(5.2, 3.8))
ax.plot([r["b_min"] for r in bf], [r["gain"] for r in bf], "o-", color=BLUE, lw=1.8, ms=6)
ax.axhline(1, color=GREY, ls="--", lw=1); ax.text(bf[-1]["b_min"], 1.02, "baseline", ha="right", fontsize=8, color=GREY)
ax.set_xlabel(r"Lowest achievable Bruggeman exponent $b_{\min}$")
ax.set_ylabel(r"3C rate-capability gain (DFN, $\times$ baseline)")
ax.grid(alpha=0.25)
plt.tight_layout(); plt.savefig("fig_gain_vs_tortuosity.png", bbox_inches="tight"); plt.close()
print("figures written")
