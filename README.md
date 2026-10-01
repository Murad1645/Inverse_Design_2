# Loading-Constrained Inverse Design of High-Rate Lithium-Ion Electrodes

Code and data accompanying the article:

> Md. Murad Hossen, *Loading-Constrained Inverse Design of High-Rate Lithium-Ion
> Electrodes via Physics-Validated Machine Learning*, submitted to the
> *Journal of Energy Storage* (2026).

The study designs a porous graphite negative electrode for high-rate operation.
A Latin-hypercube design space (porosity, Bruggeman exponent, particle radius,
thickness) is simulated with the Doyle–Fuller–Newman (DFN) model in PyBaMM,
tree-ensemble surrogates are trained on the unfiltered results, inverse design is
performed at fixed areal loading, and the selected design is re-validated with
the DFN model across discharge rates and under constant-current charging.

## Repository layout

| Path | Contents |
|---|---|
| `electrode_model.py` | Shared physics: parameter mapping, DFN/SPMe runs, metrics, plating indicator |
| `generate_dataset.py` | Latin-hypercube sampling and DFN or SPMe evaluation of 2000 designs |
| `run_analysis.py` | Cross-validation, loading-matched inverse design, DFN validation, rate sweep, charging, SPMe comparison, SHAP, mesh check |
| `make_figures_v2.py` | Figures 2–5 |
| `fig_workflow.py` | Figure 1 (workflow schematic) |
| `data/dataset_dfn_v2.csv` | DFN dataset (2000 designs; `solver_ok` marks the 1996 converged designs) |
| `data/dataset_spme_v2.csv` | SPMe results for the same designs |
| `results/results_v2.json` | Every number reported in the article |
| `results/results_v2.txt` | Human-readable summary of the same results |
| `figures/` | Rendered figures |

Dataset columns: `eps`, `b`, `Rp_um`, `L_um` (design variables); `Q05_Ah`,
`Q3_Ah` (discharge capacity at 0.5C and 3C); `Q_ratio` (= Q3/Q05);
`dV_3C` (polarization at 25% depth of discharge, V); `solver_ok`.

## Reproducing the results

Python 3.10 or newer. Run all commands from the repository root.

```bash
pip install -r requirements.txt

# Optional: regenerate the datasets (about 15 min on two CPU cores)
python generate_dataset.py DFN
python generate_dataset.py SPMe

# Full analysis (about 2 min); writes results/results_v2.json and results_v2.txt
python run_analysis.py

# Figures
python make_figures_v2.py
python fig_workflow.py
```

The datasets are included, so the analysis and figures can be reproduced without
regenerating them.

## Software

PyBaMM 26.8 with the Chen2020 parameter set, scikit-learn, XGBoost, SHAP.
Exact versions are pinned in `requirements.txt`. Please also cite PyBaMM
(Sulzer et al., *J. Open Res. Softw.* 9 (2021) 14) and the Chen2020 parameter set
(Chen et al., *J. Electrochem. Soc.* 167 (2020) 080534) if you use this code.

## License

MIT; see `LICENSE`.
