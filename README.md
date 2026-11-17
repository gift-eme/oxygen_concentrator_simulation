# Mission O₂ – Adsorbent selection for an oxygen concentrator on Europa

A personal modelling project: isotherm fitting for the two candidate
adsorbents (*Romulus* and *Remus*), a dynamic breakthrough model of a packed
column (Eqs. 1–5 of `project_brief.docx`), a check against equilibrium theory, and a
sensitivity study on feed velocity and column length.

## Quick start (VS Code)

1. Open this folder in VS Code (`File → Open Folder…`).
2. Open a terminal (`Terminal → New Terminal`) and create an environment:

   ```bash
   python -m venv .venv
   # Windows:   .venv\Scripts\activate
   # macOS/Linux: source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. Run the whole study:

   ```bash
   python main.py            # full run, 100 cells (about 10-15 min)
   python main.py --quick    # 50 cells, faster test run
   python main.py --plot-only  # redraw the figure from saved results
   python check_grid.py      # grid-independence check (optional)
   python isotherms.py       # print all isotherm fits only
   ```

Results are written to `results/`.

## Files

| File | What it does |
|---|---|
| `data.py` | Table 1 isotherm data and Table 2 column/operating parameters (plus the one assumed value, particle voidage) |
| `isotherms.py` | Langmuir, Sips, Toth and dual-site Langmuir models; non-linear least-squares fitting (`scipy.optimize.least_squares`, trust-region reflective) and model ranking by R², RMSE, AARD and AICc |
| `breakthrough.py` | Finite-volume breakthrough model: axially dispersed plug flow, variable velocity from the total mass balance, LDF mass transfer, extended Langmuir equilibrium, Danckwerts inlet; van Leer TVD convection; `solve_ivp` (RK45). Also equilibrium theory and mass-balance checks |
| `main.py` | Runs everything: fits, selectivity, base case, sensitivity runs, writes results and the figure |
| `plotting.py` | The single 7 × 5 in figure for the technical note |
| `check_grid.py` | Repeats the base case with 50/100/200 cells |
| `results/` | `figure.png`, `isotherm_fits.csv`, `sensitivity.csv`, `breakthrough_*.csv`, `summary.json` |
| `project_brief.docx` | The problem statement and data |
| `report/technical_note.docx` | 2-page technical note (10 pt Arial, one figure) with a cover page |

## Model summary

* Component balance (N₂): ∂y/∂t = D_L ∂²y/∂z² − ∂(yv)/∂z − S ∂q₁/∂t,
  with S = (RT/P)·ρₛ(1−ε)/ε
* Total balance (Σyᵢ = 1, isobaric): ∂v/∂z = −S(∂q₁/∂t + ∂q₂/∂t)
* LDF: ∂qᵢ/∂t = kᵢ(qᵢ* − qᵢ), kᵢ = (cᵢ/ρₛqᵢ*)·15 εₚ Dₚ / rₚ², Dₚ = Dₘ/τ
* D_L = 0.7 Dₘ + v_in rₚ
* Inlet: Danckwerts; outlet: zero gradient
* Initial state: bed (gas and solid) saturated with pure O₂ at 100 kPa;
  feed: pure N₂.
* Breakthrough time t_b: outlet N₂ mole fraction reaches 0.05.

## Assumptions

Isothermal (303 K), isobaric (Ergun pressure drop ≈ 5 Pa ≪ 100 kPa),
ideal gas, extended Langmuir for the mixture, macropore-controlled LDF with
assumed particle voidage εₚ = 0.35 (not given in the brief; value from
Haghpanah et al., 2013), constant bed properties, no wall effects.

## Reference

Haghpanah, R., Majumder, A., Nilam, R., Rajendran, A., Farooq, S., Karimi,
I. A., & Amanullah, M. (2013). Multiobjective optimization of a four-step
adsorption process for postcombustion CO₂ capture via finite volume
simulation. *Ind. Eng. Chem. Res.*, 52(11), 4249–4265.
