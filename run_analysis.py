"""
Full v2 analysis: surrogates, loading-matched inverse design, DFN re-validation,
robustness, SPMe fidelity comparison, SHAP, charging/plating, checks.
Writes every manuscript number to results_v2.json and a readable results_v2.txt.
Requires dataset_dfn_v2.csv and dataset_spme_v2.csv (generate_dataset.py).
"""
import json, numpy as np, pandas as pd, pybamm, shap
from scipy.stats import spearmanr, kendalltau
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from electrode_model import (FEAT, RANGES, BASELINE, S0, evaluate, capacity_at,
                             charge_plating, params_for, make_model, V_CUT)

R = {}; LOG = []
def log(s=""): print(s, flush=True); LOG.append(s)
def rnd(d, n=3): return {k: round(float(v), n) for k, v in d.items()}

dfn = pd.read_csv("dataset_dfn_v2.csv"); dfn = dfn[dfn.solver_ok == True].reset_index(drop=True)
spm = pd.read_csv("dataset_spme_v2.csv"); spm = spm[spm.solver_ok == True].reset_index(drop=True)

# ---------------------------------------------------------------- dataset
R["dataset"] = dict(n_dfn=len(dfn), fail_dfn=2000 - len(dfn), n_spme=len(spm), fail_spme=2000 - len(spm),
                    below02_dfn=float((dfn.Q_ratio < 0.2).mean()), below02_spme=float((spm.Q_ratio < 0.2).mean()),
                    corr_Q05_L=float(np.corrcoef(dfn.Q05_Ah, dfn.L_um)[0, 1]),
                    corr_Q05_loading=float(np.corrcoef(dfn.Q05_Ah, (1 - dfn.eps) * dfn.L_um)[0, 1]))
log("== DATASET ==");  log(json.dumps(rnd(R["dataset"], 4)))

# ---------------------------------------------------------------- surrogates
def xgb(): return XGBRegressor(n_estimators=800, learning_rate=0.03, max_depth=4, subsample=0.9,
                               colsample_bytree=0.9, reg_lambda=1.0, random_state=42, n_jobs=-1, tree_method="hist")
def rf(): return RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)
MAKERS = {"XGBoost": xgb, "RandomForest": rf}

def cv(df, target):
    X, y = df[FEAT].values, df[target].values; out = {}
    for name, mk in MAKERS.items():
        r2, rm = [], []
        for tr, te in KFold(5, shuffle=True, random_state=42).split(X):
            p = mk().fit(X[tr], y[tr]).predict(X[te])
            r2.append(r2_score(y[te], p)); rm.append(np.sqrt(mean_squared_error(y[te], p)))
        out[name] = dict(R2=np.mean(r2), R2_sd=np.std(r2), RMSE=np.mean(rm), RMSE_sd=np.std(rm))
    return out

R["cv"] = {t: cv(dfn, t) for t in ["Q_ratio", "dV_3C"]}
log("\n== 5-FOLD CV (DFN) ==")
for t, v in R["cv"].items():
    for m, s in v.items(): log(f" {t:8s} {m:12s} R2={s['R2']:.3f}±{s['R2_sd']:.3f} RMSE={s['RMSE']:.4f}±{s['RMSE_sd']:.4f}")
best = {t: max(R["cv"][t], key=lambda m: R["cv"][t][m]["R2"]) for t in R["cv"]}
R["chosen_model"] = best; log(f" chosen for optimization: {best}")
mQ = MAKERS[best["Q_ratio"]]().fit(dfn[FEAT], dfn.Q_ratio)
mD = MAKERS[best["dV_3C"]]().fit(dfn[FEAT], dfn.dV_3C)

# ---------------------------------------------------------------- inverse design
def pareto(q, d):
    o = np.argsort(-q); f = []; b = np.inf
    for i in o:
        if d[i] < b - 1e-9: f.append(i); b = d[i]
    return np.array(f)

def knee(qp, dp, idx):
    pf = idx[pareto(qp[idx], dp[idx])]
    qn = (qp[pf] - qp[pf].min()) / np.ptp(qp[pf]); dn = (dp[pf] - dp[pf].min()) / np.ptp(dp[pf])
    return pf[np.argmax(qn - dn)], pf

rng = np.random.default_rng(0); N = 200000
eps = rng.uniform(*RANGES["eps"], N); b = rng.uniform(*RANGES["b"], N); Rp = rng.uniform(*RANGES["Rp_um"], N)
L = S0 / (1 - eps); m = (L >= RANGES["L_um"][0]) & (L <= RANGES["L_um"][1])
C = pd.DataFrame(np.column_stack([eps[m], b[m], Rp[m], L[m]]), columns=FEAT)
qp, dp = mQ.predict(C), mD.predict(C)
k, pf = knee(qp, dp, np.arange(len(C)))
SEL = dict(zip(FEAT, C.values[k]))
R["matched"] = dict(n_candidates=len(C), n_pareto=len(pf), selected=SEL,
                    surrogate=dict(Q_ratio=float(qp[k]), dV_3C=float(dp[k])),
                    front_Q_min=float(qp[pf].min()), front_Q_max=float(qp[pf].max()))
log("\n== LOADING-MATCHED INVERSE DESIGN ==")
log(f" candidates={len(C)} pareto={len(pf)} selected={rnd(SEL)} surrogate Q={qp[k]:.3f} dV={dp[k]:.3f}")
np.savez("pareto_v2.npz", qp=qp, dp=dp, pf=pf, k=k)

eb, es = evaluate(**BASELINE), evaluate(**SEL)
R["dfn_baseline"], R["dfn_selected"] = eb, es
R["validation"] = dict(Q_err_pct=100 * abs(qp[k] - es["Q_ratio"]) / es["Q_ratio"],
                       dV_err_pct=100 * abs(dp[k] - es["dV_3C"]) / es["dV_3C"])
R["gain"] = dict(Q_ratio_x=es["Q_ratio"] / eb["Q_ratio"], Q3_x=es["Q3_Ah"] / eb["Q3_Ah"],
                 dV_reduction_pct=100 * (1 - es["dV_3C"] / eb["dV_3C"]))
log(f" DFN baseline : {rnd(eb)}"); log(f" DFN selected : {rnd(es)}")
log(f" surrogate vs DFN: Q {qp[k]:.3f} vs {es['Q_ratio']:.3f} ({R['validation']['Q_err_pct']:.1f}%), "
    f"dV {dp[k]:.3f} vs {es['dV_3C']:.3f} ({R['validation']['dV_err_pct']:.1f}%)")
log(f" GAIN: {rnd(R['gain'], 3)}")

# multi-point validation along the front
pts = pf[np.linspace(0, len(pf) - 1, min(10, len(pf))).astype(int)]
rows = []
for i in pts:
    e = evaluate(**dict(zip(FEAT, C.values[i])))
    rows.append(dict(Q_s=qp[i], Q_d=e["Q_ratio"], dV_s=dp[i], dV_d=e["dV_3C"]))
pv = pd.DataFrame(rows)
R["pareto_validation"] = dict(n=len(pv), Q_mae=float((pv.Q_s - pv.Q_d).abs().mean()), Q_max=float((pv.Q_s - pv.Q_d).abs().max()),
                              dV_mae=float((pv.dV_s - pv.dV_d).abs().mean()), dV_max=float((pv.dV_s - pv.dV_d).abs().max()))
R["pareto_validation_points"] = pv.round(4).to_dict("records")
log(f" {len(pv)}-point Pareto validation: {rnd(R['pareto_validation'], 4)}")

# unconstrained comparison (same procedure, no loading constraint)
U = pd.DataFrame({k2: rng.uniform(*RANGES[k2], N) for k2 in FEAT})
qu, du = mQ.predict(U), mD.predict(U)
ku, _ = knee(qu, du, np.arange(N)); UNC = dict(zip(FEAT, U.values[ku])); eu = evaluate(**UNC)
R["unconstrained"] = dict(design=UNC, S0=(1 - UNC["eps"]) * UNC["L_um"], loading_change_pct=100 * ((1 - UNC["eps"]) * UNC["L_um"] / S0 - 1),
                          surrogate_Q=float(qu[ku]), dfn=eu)
log(f"\n== UNCONSTRAINED == {rnd(UNC)} S0={R['unconstrained']['S0']:.1f} ({R['unconstrained']['loading_change_pct']:+.0f}%) "
    f"surrQ={qu[ku]:.3f} DFN={rnd(eu)}")

# tortuosity-floor sensitivity
R["b_floor"] = []
log("\n== GAIN vs ACHIEVABLE TORTUOSITY (Bruggeman floor) ==")
for bmin in (1.0, 1.25, 1.5, 1.75, 2.0):
    kb, _ = knee(qp, dp, np.where(C.b.values >= bmin)[0]); d = dict(zip(FEAT, C.values[kb])); e = evaluate(**d)
    row = dict(b_min=bmin, design=d, Q_ratio=e["Q_ratio"], gain=e["Q_ratio"] / eb["Q_ratio"],
               dV=e["dV_3C"], dV_red_pct=100 * (1 - e["dV_3C"] / eb["dV_3C"]))
    R["b_floor"].append(row); log(f" b>={bmin}: {rnd(d, 2)} Q={e['Q_ratio']:.3f} gain={row['gain']:.2f}x dV={e['dV_3C']:.3f} ({row['dV_red_pct']:+.0f}%)")

# ---------------------------------------------------------------- robustness (discharge)
rates = [0.5, 1, 2, 3, 4, 5]
cb = {c: capacity_at(BASELINE, c) for c in rates}; cs = {c: capacity_at(SEL, c) for c in rates}
R["rate_sweep"] = [dict(C=c, base_Q=cb[c], sel_Q=cs[c], base_ret=cb[c] / cb[0.5], sel_ret=cs[c] / cs[0.5],
                        gain=(cs[c] / cs[0.5]) / (cb[c] / cb[0.5])) for c in rates]
log("\n== DISCHARGE RATE SWEEP (DFN) ==")
for r in R["rate_sweep"]: log(f" {r['C']}C: base {r['base_ret']:.3f} sel {r['sel_ret']:.3f} gain {r['gain']:.2f}x")

# ---------------------------------------------------------------- charging / plating
R["charging"] = []
log("\n== CC CHARGE 0%SOC -> 4.2 V: capacity, min(phi_s - phi_e) at anode/separator ==")
for c in [1, 2, 3, 4, 5]:
    qb, pb = charge_plating(BASELINE, c); qs, ps = charge_plating(SEL, c)
    R["charging"].append(dict(C=c, base_Q=qb, base_plating_mV=1e3 * pb, sel_Q=qs, sel_plating_mV=1e3 * ps))
    log(f" {c}C: base {qb:.3f} Ah, {1e3*pb:+.0f} mV | sel {qs:.3f} Ah, {1e3*ps:+.0f} mV")

# ---------------------------------------------------------------- SPMe fidelity
mm = dfn.merge(spm, on="sim_id", suffixes=("_d", "_s"))
top = int(0.05 * len(mm))
eb_s, es_s = evaluate(**BASELINE, kind="SPMe"), evaluate(**SEL, kind="SPMe")
R["spme"] = dict(n=len(mm), spearman=spearmanr(mm.Q_ratio_s, mm.Q_ratio_d)[0], kendall=kendalltau(mm.Q_ratio_s, mm.Q_ratio_d)[0],
                 top5_overlap=len(set(mm.nlargest(top, "Q_ratio_d").sim_id) & set(mm.nlargest(top, "Q_ratio_s").sim_id)), top5_n=top,
                 frac_spme_lower=float((mm.Q_ratio_s < mm.Q_ratio_d).mean()),
                 mae_all=float((mm.Q_ratio_s - mm.Q_ratio_d).abs().mean()),
                 mae_good=float((mm.Q_ratio_s - mm.Q_ratio_d)[mm.Q_ratio_d > 0.8].abs().mean()),
                 mae_poor=float((mm.Q_ratio_s - mm.Q_ratio_d)[mm.Q_ratio_d < 0.5].abs().mean()),
                 baseline_spme=eb_s, selected_spme=es_s,
                 apparent_gain_spme=es_s["Q_ratio"] / eb_s["Q_ratio"])
R["spme"]["overstatement"] = R["spme"]["apparent_gain_spme"] / R["gain"]["Q_ratio_x"]
# an SPMe-trained workflow end to end
sQ = MAKERS[best["Q_ratio"]]().fit(spm[FEAT], spm.Q_ratio); sD = MAKERS[best["dV_3C"]]().fit(spm[FEAT], spm.dV_3C)
ks, _ = knee(sQ.predict(C), sD.predict(C), np.arange(len(C))); SSEL = dict(zip(FEAT, C.values[ks]))
R["spme_workflow"] = dict(selected=SSEL, dfn=evaluate(**SSEL), spme=evaluate(**SSEL, kind="SPMe"))
log("\n== SPMe vs DFN =="); log(json.dumps({k2: (round(v, 4) if isinstance(v, float) else v) for k2, v in R["spme"].items() if not isinstance(v, dict)}))
log(f" SPMe baseline {rnd(eb_s)} | SPMe selected {rnd(es_s)}")
log(f" SPMe-trained workflow selects {rnd(SSEL,2)} -> DFN {rnd(R['spme_workflow']['dfn'])}")

# ---------------------------------------------------------------- SHAP
def mean_abs_shap(model, X):
    return dict(zip(X.columns, np.abs(shap.TreeExplainer(model).shap_values(X)).mean(0)))
xQ, xD = xgb().fit(dfn[FEAT], dfn.Q_ratio), xgb().fit(dfn[FEAT], dfn.dV_3C)
R["shap_full"] = {"Q_ratio": mean_abs_shap(xQ, dfn[FEAT]), "dV_3C": mean_abs_shap(xD, dfn[FEAT])}
# loading-aware parameterisation: (eps, b, Rp, S0) where S0 = (1-eps) L
F2 = ["eps", "b", "Rp_um", "S0_um"]; D2 = dfn.assign(S0_um=(1 - dfn.eps) * dfn.L_um)
R["shap_loading_param"] = {"Q_ratio": mean_abs_shap(xgb().fit(D2[F2], D2.Q_ratio), D2[F2]),
                           "dV_3C": mean_abs_shap(xgb().fit(D2[F2], D2.dV_3C), D2[F2])}
# on the loading-matched candidate set (L is set by eps there)
sub = C.sample(3000, random_state=0)
R["shap_matched_set"] = {"Q_ratio": mean_abs_shap(xQ, sub), "dV_3C": mean_abs_shap(xD, sub)}
inter = shap.TreeExplainer(xQ).shap_interaction_values(dfn[FEAT].iloc[:400])
ri, li = FEAT.index("Rp_um"), FEAT.index("L_um")
R["shap_Rp_interaction"] = dict(main=float(np.abs(inter[:, ri, ri]).mean()), RpxL=float(2 * np.abs(inter[:, ri, li]).mean()))
log("\n== SHAP mean|SHAP| ==")
for kk in ["shap_full", "shap_loading_param", "shap_matched_set"]:
    for t in ["Q_ratio", "dV_3C"]: log(f" {kk:20s} {t:8s} {rnd(R[kk][t], 3)}")
log(f" Rp interaction: {rnd(R['shap_Rp_interaction'], 4)}")

# ---------------------------------------------------------------- scaling, cost, mesh
p = params_for(**BASELINE); Ds = p["Negative particle diffusivity [m2.s-1]"]
R["timescales_s"] = {f"Rp{r}um": dict(Rp2_over_Ds=(r * 1e-6) ** 2 / Ds, Rp2_over_15Ds=(r * 1e-6) ** 2 / (15 * Ds)) for r in (2, 7, 12)}
R["t_3C_s"] = 3600 / 3
area_cm2 = p["Electrode height [m]"] * p["Electrode width [m]"] * 1e4
R["areal_capacity_mAh_cm2"] = dict(baseline=1e3 * eb["Q05_Ah"] / area_cm2, selected=1e3 * es["Q05_Ah"] / area_cm2)
log(f"\n== SCALING == {json.dumps({k2: rnd(v, 0) for k2, v in R['timescales_s'].items()})} t3C=1200 s")
log(f" areal capacity at 0.5C (mAh/cm2): {rnd(R['areal_capacity_mAh_cm2'], 2)}")

def q_mesh(d, c, scale):
    model = make_model("DFN"); vp = {k2: v * scale for k2, v in model.default_var_pts.items()}
    sim = pybamm.Simulation(model, parameter_values=params_for(**d), var_pts=vp,
                            experiment=pybamm.Experiment([f"Discharge at {c}C until {V_CUT} V"]))
    return float(sim.solve()["Discharge capacity [A.h]"].data[-1])
R["mesh"] = []
for nm, d in [("baseline", BASELINE), ("selected", SEL)]:
    for c in (3, 5):
        q1, q2 = q_mesh(d, c, 1), q_mesh(d, c, 2)
        R["mesh"].append(dict(design=nm, C=c, Q_default=q1, Q_2x=q2, diff_pct=100 * abs(q2 - q1) / q2))
log(" mesh check: " + "; ".join(f"{r['design']} {r['C']}C {r['diff_pct']:.2f}%" for r in R["mesh"]))
R["default_var_pts"] = {str(k2): v for k2, v in make_model("DFN").default_var_pts.items()}

json.dump(R, open("results_v2.json", "w"), indent=1, default=float)
open("results_v2.txt", "w").write("\n".join(LOG))
print("\nDONE -> results_v2.json / results_v2.txt")
