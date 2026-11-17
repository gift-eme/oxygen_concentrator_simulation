import argparse
import json
import os
from multiprocessing import Pool

import numpy as np

from data import ISOTHERMS, PARAMS
from isotherms import fit_all, langmuir
from breakthrough import (simulate, metrics, equilibrium_theory,
                          equilibrium_velocity_ahead, mass_balance_error,
                          ext_langmuir)
from plotting import make_figure

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
ADS = ["Romulus", "Remus"]
V_FACTORS = [0.5, 1.0, 2.0, 4.0]
LENGTHS = [0.16, 0.32, 0.64]
Y_BT = 0.05   # breakthrough defined at 5 % N2 at the outlet (>= 95 % O2)


def ergun_dp(prm, mu=1.78e-5):
    """Ergun pressure drop over the column [Pa] (N2 viscosity at 303 K)."""
    eps, dp = prm["eps"], 2 * prm["r_p"]
    u = eps * prm["v_in"]                       # superficial velocity
    rho = prm["P"] * 0.028 / (prm["R"] * prm["T"])
    grad = (150 * mu * (1 - eps) ** 2 * u / (eps ** 3 * dp ** 2)
            + 1.75 * rho * (1 - eps) * u ** 2 / (eps ** 3 * dp))
    return grad * prm["L"]


def run_case(args):
    ads, iso1, iso2, prm, n_cells, tag = args
    t_eq = equilibrium_theory(iso1, iso2, prm)[0]
    res = simulate(iso1, iso2, prm, n_cells=n_cells, t_end=1.6 * t_eq)
    m = metrics(res, prm, Y_BT)
    # outlet velocity on the pre-breakthrough plateau vs equilibrium theory
    i_half = np.searchsorted(res["t"], 0.5 * m["t_b"])
    m["v_ahead_model"] = res["v_out"][i_half]
    m["v_ahead_eq"] = equilibrium_velocity_ahead(iso1, iso2, prm)
    m.update({"adsorbent": ads, "tag": tag, "L": prm["L"], "v_in": prm["v_in"],
              "t_eq": t_eq, "mb_error": mass_balance_error(res, prm),
              "k_N2": res["k"][0], "k_O2": res["k"][1], "D_L": res["D_L"]})
    curve = {"t": res["t"], "y_out": res["y_out"], "v_out": res["v_out"]}
    return m, curve


def main(quick=False):
    os.makedirs(OUT, exist_ok=True)
    n_cells = 50 if quick else 100

    fits = fit_all()
    fit_rows = []
    for (ads, gas), fm in fits.items():
        for model, r in fm.items():
            fit_rows.append({"adsorbent": ads, "gas": gas, "model": model,
                             "R2": r["R2"], "RMSE": r["RMSE"],
                             "AARD_pct": r["AARD%"], "AICc": r["AICc"],
                             **{k: float(v) for k, v in r["params"].items()}})
    iso = {a: (tuple(fits[(a, "N2")]["Langmuir"]["x"]),
               tuple(fits[(a, "O2")]["Langmuir"]["x"])) for a in ADS}

    P = PARAMS["P"]
    static = {}
    for a in ADS:
        (qs1, b1), (qs2, b2) = iso[a]
        H1, H2 = qs1 * b1, qs2 * b2                     # Henry constants
        q1_air, q2_air = ext_langmuir(0.79 * P, 0.21 * P, iso[a][0], iso[a][1])
        static[a] = {
            "qs_N2": qs1, "b_N2": b1, "qs_O2": qs2, "b_O2": b2,
            "H_N2_mol_kg_Pa": H1, "H_O2_mol_kg_Pa": H2,
            "alpha_Henry": H1 / H2,
            "q_N2_pure_1bar": langmuir(P, qs1, b1),
            "q_O2_pure_1bar": langmuir(P, qs2, b2),
            "alpha_air_1bar": (q1_air / q2_air) / (0.79 / 0.21),
        }

    cases = []
    for a in ADS:
        for f in V_FACTORS:
            prm = dict(PARAMS, v_in=PARAMS["v_in"] * f)
            cases.append((a, *iso[a], prm, n_cells, f"v x{f}"))
        for L in LENGTHS:
            if L == PARAMS["L"]:
                continue
            prm = dict(PARAMS, L=L)
            cases.append((a, *iso[a], prm, n_cells, f"L={L}"))
    with Pool(min(os.cpu_count() or 1, len(cases))) as pool:
        out = pool.map(run_case, cases)
    sens = [m for m, _ in out]
    base_curves = {m["adsorbent"]: c for m, c in out if m["tag"] == "v x1.0"}

    import csv
    with open(os.path.join(OUT, "isotherm_fits.csv"), "w", newline="") as fh:
        keys = sorted({k for r in fit_rows for k in r}, key=lambda k: (
            ["adsorbent", "gas", "model", "R2", "RMSE", "AARD_pct", "AICc"]
            + ["qs", "b", "n", "t", "qs1", "b1", "qs2", "b2"]).index(k))
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(fit_rows)
    with open(os.path.join(OUT, "sensitivity.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(sens[0].keys()))
        w.writeheader()
        w.writerows(sens)
    for a, c in base_curves.items():
        np.savetxt(os.path.join(OUT, f"breakthrough_{a}.csv"),
                   np.column_stack([c["t"], c["y_out"], c["v_out"]]),
                   delimiter=",", header="t_s,y_N2_out,v_out_m_s", comments="")
    summary = {"static": static,
               "ergun_pressure_drop_Pa": ergun_dp(PARAMS),
               "breakthrough_level": Y_BT,
               "base_case": [m for m in sens if m["tag"] == "v x1.0"]}
    with open(os.path.join(OUT, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=float)

    make_figure(ISOTHERMS, iso, base_curves, sens, PARAMS,
                os.path.join(OUT, "figure.png"))

    print("\nLangmuir parameters and selectivity")
    for a, s in static.items():
        print(f"  {a:8s} N2 qs={s['qs_N2']:.3f} b={s['b_N2']:.3e} | "
              f"O2 qs={s['qs_O2']:.3f} b={s['b_O2']:.3e} | "
              f"q_N2(1bar)={s['q_N2_pure_1bar']:.3f} "
              f"alpha_H={s['alpha_Henry']:.2f} alpha_air={s['alpha_air_1bar']:.2f}")
    print(f"\nErgun pressure drop (base case): {summary['ergun_pressure_drop_Pa']:.2f} Pa")
    print("\nBreakthrough results (t in s, MTZ in m)")
    for m in sens:
        print(f"  {m['adsorbent']:8s} {m['tag']:9s} t_b={m['t_b']:8.1f} "
              f"t_50={m['t_50']:8.1f} t_st={m['t_st']:8.1f} t_eq={m['t_eq']:8.1f} "
              f"util={m['utilisation']:.3f} MTZ={m['MTZ']:.4f} "
              f"v_ahead={m['v_ahead_model']:.3e}/{m['v_ahead_eq']:.3e} "
              f"MBerr={m['mb_error']:.1e}")
    print(f"\nResults written to {OUT}")


def plot_only():
    """Redraw results/figure.png from saved results (no re-simulation)."""
    import csv
    fits = fit_all()
    iso = {a: (tuple(fits[(a, "N2")]["Langmuir"]["x"]),
               tuple(fits[(a, "O2")]["Langmuir"]["x"])) for a in ADS}
    with open(os.path.join(OUT, "sensitivity.csv")) as fh:
        sens = []
        for r in csv.DictReader(fh):
            sens.append({k: (v if k in ("adsorbent", "tag") else float(v))
                         for k, v in r.items()})
    curves = {}
    for a in ADS:
        d = np.loadtxt(os.path.join(OUT, f"breakthrough_{a}.csv"),
                       delimiter=",", skiprows=1)
        curves[a] = {"t": d[:, 0], "y_out": d[:, 1], "v_out": d[:, 2]}
    make_figure(ISOTHERMS, iso, curves, sens, PARAMS,
                os.path.join(OUT, "figure.png"))
    print("Figure redrawn:", os.path.join(OUT, "figure.png"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="coarser grid (50 cells) for a fast test run")
    ap.add_argument("--plot-only", action="store_true",
                    help="redraw the figure from saved results")
    args = ap.parse_args()
    plot_only() if args.plot_only else main(args.quick)
