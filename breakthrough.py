"""Dynamic breakthrough model for a fixed bed (Eqs. 1-5 of the brief)."""
import numpy as np
from scipy.integrate import solve_ivp


def ext_langmuir(p1, p2, iso1, iso2):
    """Extended Langmuir. iso = (qs, b). Partial pressures in Pa."""
    d = 1.0 + iso1[1] * p1 + iso2[1] * p2
    return iso1[0] * iso1[1] * p1 / d, iso2[0] * iso2[1] * p2 / d


def ldf_coefficients(iso1, iso2, prm):
    """Macropore-controlled LDF constants (Glueckauf, 15 Dp eps_p / rp^2)."""
    c = prm["P"] / (prm["R"] * prm["T"])
    Dp = prm["D_m"] / prm["tau"]
    base = 15.0 * prm["eps_p"] * Dp / prm["r_p"] ** 2
    q1 = iso1[0] * iso1[1] * prm["P"] / (1 + iso1[1] * prm["P"])
    q2 = iso2[0] * iso2[1] * prm["P"] / (1 + iso2[1] * prm["P"])
    return (c / (prm["rho_s"] * q1)) * base, (c / (prm["rho_s"] * q2)) * base


def van_leer(r):
    return (r + np.abs(r)) / (1.0 + np.abs(r))


def simulate(iso1, iso2, prm, n_cells=100, t_end=None, y_feed=1.0,
             n_out=1500, method="RK45"):
    """Pure N2 fed to a bed saturated with pure O2."""
    L, eps, P, T, R = prm["L"], prm["eps"], prm["P"], prm["T"], prm["R"]
    v_in, rho_s = prm["v_in"], prm["rho_s"]
    N = n_cells
    dz = L / N
    S = (R * T / P) * rho_s * (1 - eps) / eps
    D_L = 0.7 * prm["D_m"] + v_in * prm["r_p"]          # Eq. (2)
    k1, k2 = ldf_coefficients(iso1, iso2, prm)

    # Initial state: gas pure O2, solid at equilibrium with pure O2
    y0 = np.zeros(N)
    q1_0 = np.zeros(N)
    q2_0 = np.full(N, ext_langmuir(0.0, P, iso1, iso2)[1])
    x0 = np.concatenate([y0, q1_0, q2_0])

    if t_end is None:  # 2.5 x equilibrium-theory breakthrough time
        q1f = ext_langmuir(y_feed * P, (1 - y_feed) * P, iso1, iso2)[0]
        t_end = 2.5 * L / v_in * (1 + S * q1f / y_feed)

    def rhs(t, x):
        y = np.clip(x[:N], 0.0, 1.0)
        q1, q2 = x[N:2 * N], x[2 * N:]
        qs1, qs2 = ext_langmuir(y * P, (1 - y) * P, iso1, iso2)
        dq1 = k1 * (qs1 - q1)
        dq2 = k2 * (qs2 - q2)

        # velocity at faces from total mass balance (face 0 = inlet)
        v = np.empty(N + 1)
        v[0] = v_in
        v[1:] = v_in - dz * S * np.cumsum(dq1 + dq2)

        # convective face values (van Leer TVD, upwind since v > 0)
        yf = np.empty(N + 1)
        yf[0] = y_feed
        yf[N] = y[-1]
        dy = np.diff(y)
        yf[1] = y[0]                          # first-order next to inlet
        num = y[1:-1] - y[:-2]
        den = dy[1:]
        r = num / np.where(np.abs(den) < 1e-12, 1e-12, den)
        yf[2:N] = y[1:-1] + 0.5 * van_leer(r) * den

        conv = v * yf
        disp = np.zeros(N + 1)
        disp[1:N] = -D_L * dy / dz
        flux = conv + disp
        flux[0] = v_in * y_feed               # Danckwerts
        dydt = -(flux[1:] - flux[:-1]) / dz - S * dq1
        return np.concatenate([dydt, dq1, dq2])

    # only mildly stiff, so explicit RK45 is fine
    t_eval = np.linspace(0, t_end, n_out)
    sol = solve_ivp(rhs, (0, t_end), x0, method=method, t_eval=t_eval,
                    rtol=1e-6, atol=1e-8)
    if not sol.success:
        raise RuntimeError(sol.message)

    y = sol.y[:N].T
    q1 = sol.y[N:2 * N].T
    q2 = sol.y[2 * N:].T
    v_out = np.empty(len(sol.t))
    for i, t in enumerate(sol.t):
        qs1, qs2 = ext_langmuir(y[i] * P, (1 - y[i]) * P, iso1, iso2)
        v_out[i] = v_in - dz * S * np.sum(k1 * (qs1 - q1[i]) + k2 * (qs2 - q2[i]))
    z = (np.arange(N) + 0.5) * dz
    return {"t": sol.t, "y_out": y[:, -1], "v_out": v_out, "y": y,
            "q1": q1, "q2": q2, "z": z, "k": (k1, k2), "D_L": D_L, "S": S,
            "t_end": t_end}


def first_crossing(t, y, level):
    idx = np.argmax(y >= level)
    if y[idx] < level:
        return np.nan
    if idx == 0:
        return t[0]
    return np.interp(level, [y[idx - 1], y[idx]], [t[idx - 1], t[idx]])


def metrics(res, prm, y_bt=0.05):
    """Breakthrough time, stoichiometric time, MTZ length, utilisation."""
    t, y, v = res["t"], res["y_out"], res["v_out"]
    t_b = first_crossing(t, y, y_bt)
    t_50 = first_crossing(t, y, 0.5)
    t_95 = first_crossing(t, y, 0.95)
    # stoichiometric time from N2 molar flow out (accounts for velocity change)
    t_st = np.trapezoid(1 - y * v / prm["v_in"], t)
    mtz = prm["L"] * (t_95 - t_b) / t_st
    return {"t_b": t_b, "t_50": t_50, "t_95": t_95, "t_st": t_st,
            "MTZ": mtz, "utilisation": t_b / t_st}


def equilibrium_theory(iso1, iso2, prm, y_feed=1.0):
    """Shock breakthrough time from an overall N2 balance."""
    P, T, R, eps = prm["P"], prm["T"], prm["R"], prm["eps"]
    q1f = ext_langmuir(y_feed * P, (1 - y_feed) * P, iso1, iso2)[0]
    t_sh = prm["L"] / prm["v_in"] * (
        1 + (1 - eps) / eps * prm["rho_s"] * q1f * R * T / (P * y_feed))
    return t_sh, prm["L"] / t_sh, q1f


def equilibrium_velocity_ahead(iso1, iso2, prm):
    """Gas velocity downstream of the shock (pure O2 region)."""
    P, T, R, eps = prm["P"], prm["T"], prm["R"], prm["eps"]
    S = (R * T / P) * prm["rho_s"] * (1 - eps) / eps
    q1f = ext_langmuir(P, 0.0, iso1, iso2)[0]
    q20 = ext_langmuir(0.0, P, iso1, iso2)[1]
    w = prm["v_in"] / (1 + S * q1f)
    return prm["v_in"] - w * S * (q1f - q20)


def mass_balance_error(res, prm):
    """N2 in - N2 out - N2 accumulated, relative to N2 in (per unit area/eps)."""
    t, y, v = res["t"], res["y_out"], res["v_out"]
    c = prm["P"] / (prm["R"] * prm["T"])
    eps, rho_s = prm["eps"], prm["rho_s"]
    dz = prm["L"] / len(res["z"])
    n_in = eps * c * prm["v_in"] * t[-1]
    n_out = eps * c * np.trapezoid(y * v, t)
    hold = (eps * c * np.sum(res["y"][-1]) * dz
            + (1 - eps) * rho_s * np.sum(res["q1"][-1]) * dz)
    return (n_in - n_out - hold) / n_in
