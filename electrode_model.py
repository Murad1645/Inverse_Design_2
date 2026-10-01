"""
Electrode-design physics shared by all scripts.

Design parameters are mapped onto the Chen2020 parameter set with the
negative-electrode active-material volume fraction set to 1 - eps (the
Chen2020 negative electrode has no separate inactive phase), so that the
loading proxy S0 = (1 - eps) L is proportional to the active-material loading.

Polarization dV_3C is evaluated at a fixed depth of discharge,
q_ref = 0.25 * Q_0.5C. If the 3C discharge reaches the 2.5 V cut-off before
q_ref, V_3C(q_ref) is taken as the cut-off voltage, so dV_3C is a lower bound.

C-rates follow PyBaMM's convention (current = C-rate x nominal 5 A h).
"""
import numpy as np
import pybamm

pybamm.set_logging_level("ERROR")

RANGES = {"eps": (0.28, 0.45), "b": (1.0, 2.5), "Rp_um": (2.0, 12.0), "L_um": (50.0, 140.0)}
FEAT = ["eps", "b", "Rp_um", "L_um"]
BASELINE = dict(eps=0.35, b=1.5, Rp_um=7.0, L_um=100.0)
S0 = (1 - BASELINE["eps"]) * BASELINE["L_um"]      # 65 um loading proxy
V_CUT = 2.5
QREF_FRAC = 0.25

_BASE = pybamm.ParameterValues("Chen2020")


def params_for(eps, b, Rp_um, L_um):
    p = _BASE.copy()
    p.update({
        "Negative electrode porosity": eps,
        "Negative electrode active material volume fraction": 1.0 - eps,
        "Negative electrode Bruggeman coefficient (electrolyte)": b,
        "Negative electrode Bruggeman coefficient (electrode)": b,
        "Negative particle radius [m]": Rp_um * 1e-6,
        "Negative electrode thickness [m]": L_um * 1e-6,
    })
    return p


def make_model(kind="DFN"):
    return {"DFN": pybamm.lithium_ion.DFN, "SPMe": pybamm.lithium_ion.SPMe}[kind]()


def discharge(model, p, crate):
    sim = pybamm.Simulation(model, parameter_values=p,
                            experiment=pybamm.Experiment([f"Discharge at {crate}C until {V_CUT} V"]))
    sol = sim.solve()
    return sol["Discharge capacity [A.h]"].data, sol["Terminal voltage [V]"].data


def dV_matched(Q05, V05, Q3, V3):
    q = QREF_FRAC * Q05[-1]
    v3 = np.interp(q, Q3, V3) if q <= Q3[-1] else V_CUT
    return float(np.interp(q, Q05, V05) - v3)


def evaluate(eps, b, Rp_um, L_um, kind="DFN"):
    """0.5C and 3C discharge -> metrics dict, or {'solver_ok': False}."""
    p = params_for(eps, b, Rp_um, L_um)
    try:
        Q05, V05 = discharge(make_model(kind), p, 0.5)
        Q3, V3 = discharge(make_model(kind), p, 3.0)
        if len(Q05) < 2 or len(Q3) < 2 or Q05[-1] <= 0 or Q3[-1] <= 0:
            return {"solver_ok": False}
    except Exception:
        return {"solver_ok": False}
    return {"solver_ok": True, "Q05_Ah": float(Q05[-1]), "Q3_Ah": float(Q3[-1]),
            "Q_ratio": float(Q3[-1] / Q05[-1]),
            "dV_3C": dV_matched(Q05, V05, Q3, V3)}


def capacity_at(d, crate, kind="DFN"):
    Q, _ = discharge(make_model(kind), params_for(**d), crate)
    return float(Q[-1])


def charge_plating(d, crate, v_max=4.2):
    """CC charge from 0% SOC. Returns charged capacity and the minimum
    solid-electrolyte potential difference at the anode/separator interface
    (plating is thermodynamically possible when it falls below 0 V)."""
    sim = pybamm.Simulation(make_model("DFN"), parameter_values=params_for(**d),
                            experiment=pybamm.Experiment([f"Charge at {crate}C until {v_max} V"]))
    sol = sim.solve(initial_soc=0)
    dphi = sol["Negative electrode surface potential difference [V]"].entries
    return float(abs(sol["Discharge capacity [A.h]"].data[-1])), float(dphi[-1, :].min())
