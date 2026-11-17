# Fit the experimental isotherm data to all models and return a list of results.
""" Langmuir            q = qs b P / (1 + b P)
    Sips (Langmuir-Freundlich)  q = qs (bP)^n / (1 + (bP)^n)
    Toth                q = qs b P / (1 + (bP)^t)^(1/t)
    Dual-site Langmuir  q = qs1 b1 P/(1+b1 P) + qs2 b2 P/(1+b2 P)
"""
# Freundlich isotherm is not included because it is a special case of Sips with qs=1.0.

import numpy as np
from scipy.optimize import least_squares

from data import ISOTHERMS

# define the isotherm models
def langmuir(P, qs, b):
    return qs * b * P / (1.0 + b * P)
def sips(P, qs, b, n):
    x = (b * np.maximum(P, 0.0)) ** n
    return qs * x / (1.0 + x)
def toth(P, qs, b, t):
    bp = b * np.maximum(P, 0.0)
    return qs * bp / (1.0 + bp ** t) ** (1.0 / t)
def dsl(P, qs1, b1, qs2, b2):
    return langmuir(P, qs1, b1) + langmuir(P, qs2, b2)


# MODELS is a lookup table;
# Each entry is a tuple of (function, parameter names, initial guess, lower bounds, upper bounds)
MODELS = {
    "Langmuir": (langmuir, ["qs", "b"], [2.0, 1e-5], [1e-3, 1e-12], [50.0, 1e-2]),
    "Sips": (sips, ["qs", "b", "n"], [2.0, 1e-5, 1.0], [1e-3, 1e-12, 0.2], [50.0, 1e-2, 3.0]),
    "Toth": (toth, ["qs", "b", "t"], [2.0, 1e-5, 1.0], [1e-3, 1e-12, 0.1], [50.0, 1e-2, 3.0]),
    "DSL": (dsl, ["qs1", "b1", "qs2", "b2"], [1.0, 1e-5, 1.0, 1e-6],
            [1e-3, 1e-12, 1e-3, 1e-12], [50.0, 1e-2, 50.0, 1e-2]),
}


def fit_model(P, q, name):
    """Fit one model; returns dict with parameters and goodness-of-fit."""
    f, pnames, x0, lb, ub = MODELS[name]
    # Scale b-type parameters by working in log space for robustness
    best = None
    for scale in [1e-7, 1e-6, 1e-5, 1e-4]:
        guess = [scale if pn.startswith("b") else g for pn, g in zip(pnames, x0)]
        res = least_squares(lambda x: f(P, *x) - q, guess, bounds=(lb, ub),
                            x_scale="jac", method="trf", max_nfev=20000)
        if best is None or res.cost < best.cost:
            best = res
    x = best.x
    resid = f(P, *x) - q
    n_pts, k = len(q), len(x)
    sse = float(np.sum(resid ** 2))
    sst = float(np.sum((q - q.mean()) ** 2))
    r2 = 1.0 - sse / sst
    aic = n_pts * np.log(sse / n_pts) + 2 * k
    aicc = aic + 2 * k * (k + 1) / max(n_pts - k - 1, 1)
    rmse = np.sqrt(sse / n_pts)
    mask = q > 0
    aard = 100 * np.mean(np.abs(resid[mask] / q[mask]))
    return {"model": name, "params": dict(zip(pnames, x)), "x": x,
            "R2": r2, "RMSE": rmse, "AARD%": aard, "AICc": aicc}


def fit_all():
    """Fit every model to every adsorbent/gas pair."""
    out = {}
    for ads, gases in ISOTHERMS.items():
        for gas, (P, q) in gases.items():
            out[(ads, gas)] = {m: fit_model(P, q, m) for m in MODELS}
    return out


if __name__ == "__main__":
    res = fit_all()
    for key, fits in res.items():
        print(f"\n=== {key[0]} / {key[1]} ===")
        for m, r in fits.items():
            ps = ", ".join(f"{k}={v:.4g}" for k, v in r["params"].items())
            print(f"{m:9s} R2={r['R2']:.5f} RMSE={r['RMSE']:.4f} "
                  f"AARD={r['AARD%']:.2f}% AICc={r['AICc']:.1f} | {ps}")
