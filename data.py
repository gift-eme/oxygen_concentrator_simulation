import numpy as np

# Table 1, isotherms at 303 K. P in Pa, q in mol/kg

ISOTHERMS = {
    "Romulus": {
        "N2": (
            np.array([0.000, 2026.500, 3039.750, 4053.000, 6079.500, 8106.000,
                      61909.575, 111356.175, 173265.750, 247435.650,
                      321706.875, 395978.100, 470148.000, 544419.225,
                      618690.450]),
            np.array([0.000, 0.025, 0.038, 0.053, 0.079, 0.108, 0.562, 0.835,
                      1.082, 1.295, 1.482, 1.597, 1.714, 1.795, 1.861]),
        ),
        "O2": (
            np.array([0.000, 8106.000, 20265.000, 247435.650, 371153.475,
                      494871.300, 618690.450]),
            np.array([0.000, 0.015, 0.035, 0.366, 0.521, 0.693, 0.815]),
        ),
    },
    "Remus": {
        "N2": (
            np.array([0, 63400, 88700, 117500, 162500, 208500, 303800,
                      427000, 609500, 776600, 919000], dtype=float),
            np.array([0.000, 0.1674, 0.2300, 0.2994, 0.3934, 0.4848, 0.6450,
                      0.8280, 1.0588, 1.2249, 1.3472]),
        ),
        "O2": (
            np.array([0, 33200, 61300, 96100, 143200, 214800, 355900,
                      548200, 747900, 859000, 931700], dtype=float),
            np.array([0.000, 0.0372, 0.0679, 0.1022, 0.1504, 0.2138, 0.3455,
                      0.5089, 0.6830, 0.7679, 0.8348]),
        ),
    },
}

# Table 2
PARAMS = {
    "L": 0.32,            # column length [m]
    "eps": 0.32,          # bed void fraction [-]
    "rho_s": 990.0,       # particle density [kg/m3]
    "r_p": 1.0e-3,        # particle radius [m]
    "tau": 3.0,           # tortuosity [-]
    "D_m": 7.481e-5,      # molecular diffusivity [m2/s]
    "R": 8.314,           # gas constant [m3 Pa / (mol K)]
    "T": 303.0,           # temperature [K]
    "P": 100.0e3,         # pressure [Pa]
    "v_in": 4.864e-3,     # inlet interstitial velocity [m/s]
    #assumption (not given in the brief)
    "eps_p": 0.35,        # particle (macropore) voidage [-], Haghpanah et al. 2013
}
