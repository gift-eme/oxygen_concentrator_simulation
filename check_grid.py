"""Grid-independence check: base case with 50, 100 and 200 cells."""
from data import PARAMS
from isotherms import fit_all
from breakthrough import simulate, metrics, equilibrium_theory

fits = fit_all()
for ads in ["Remus", "Romulus"]:
    iso1 = tuple(fits[(ads, "N2")]["Langmuir"]["x"])
    iso2 = tuple(fits[(ads, "O2")]["Langmuir"]["x"])
    t_eq = equilibrium_theory(iso1, iso2, PARAMS)[0]
    for n in (50, 100, 200):
        res = simulate(iso1, iso2, PARAMS, n_cells=n, t_end=1.6 * t_eq)
        m = metrics(res, PARAMS)
        print(f"{ads:8s} N={n:3d}  t_b={m['t_b']:7.1f} s  t_50={m['t_50']:7.1f} s"
              f"  t_st={m['t_st']:7.1f} s  utilisation={m['utilisation']:.3f}")
