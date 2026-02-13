"""
Appendix 5: NONMEM Model for Anti-VEGF Therapeutic IgG Monoclonal Antibody — Python
=====================================================================================

Python translation of the NONMEM ADVAN15 model from Appendix 5.

This is the most complex model: a full PK model with FcRn-mediated recycling,
where FcRn binding at acidic and physiological pH is solved algebraically.

NONMEM structure:
  COMP 1 (CVEGF)  — central anti-VEGF drug         (ODE)
  COMP 2 (PVEGF)  — peripheral anti-VEGF drug       (ODE)
  COMP 3 (CENDO)  — central endogenous IgG          (ODE)
  COMP 4 (PENDO)  — peripheral endogenous IgG       (ODE)
  COMP 5 (VEGFRA) — VEGF–FcRn complex, acidic pH    (EQUIL)
  COMP 6 (VEGFRP) — VEGF–FcRn complex, physiological pH  (EQUIL)
  COMP 7 (ENDORA) — endo IgG–FcRn complex, acidic pH     (EQUIL)
  COMP 8 (ENDORP) — endo IgG–FcRn complex, physiological pH (EQUIL)

Key mechanism:
  - IgG binds FcRn at acidic pH (endosome) → rescued from degradation
  - IgG released from FcRn at physiological pH (cell surface)
  - Unbound IgG at acidic pH → degraded (cleared)
  - Bound IgG at physiological pH → also degraded (cleared)
  - FcRn recycling determines IgG half-life
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from typing import Dict, Optional, Tuple
import matplotlib.pyplot as plt
import warnings


# ──────────────────────────────────────────
#  Model Parameters (from NONMEM $THETA)
# ──────────────────────────────────────────

DEFAULT_PK_PARAMS = {
    'CL':   2.50,      # Clearance (mL/kg/hr)
    'V1':   34.8,      # Central volume (mL/kg)
    'Q':    18.7,      # Distribution clearance (mL/kg/day)
    'V2':   28.8,      # Peripheral volume (mL/kg)
    'RT':   36.8,      # Total FcRn concentration (nM)
    'KSYN': 27212.0,   # Endogenous IgG synthesis rate (nmole/kg/day)
    'COVF': 1000.0 / 149000.0,  # ug/mL → uM conversion (MW ≈ 149 kDa)
}

# KD values for FcRn binding (from dataset / literature)
DEFAULT_KD_PARAMS = {
    'KDA':  1.0,    # VEGF–FcRn KD at acidic pH (nM)
    'KDP':  1000.0, # VEGF–FcRn KD at physiological pH (nM)
    'EKDA': 1.0,    # Endo IgG–FcRn KD at acidic pH (nM)
    'EKDP': 1000.0, # Endo IgG–FcRn KD at physiological pH (nM)
}


# ──────────────────────────────────────────
#  Initial conditions (from $PK block)
# ──────────────────────────────────────────

def compute_initial_conditions(params: dict) -> np.ndarray:
    """
    Compute initial conditions for all 8 compartments.

    From NONMEM $PK:
      A_0(1) = 10           (anti-VEGF in central, ug/kg — will be replaced by dose)
      A_0(2) = 0            (anti-VEGF in peripheral)
      A_0(3) = 15630 * V1   (endogenous IgG in central, ug/kg)
      A_0(4) = 15630 * V2   (endogenous IgG in peripheral, ug/kg)
    """
    V1 = params['V1']
    V2 = params['V2']

    A0_diff = np.array([
        10.0,         # A(1): CVEGF (central anti-VEGF, ug/kg)
        0.0,          # A(2): PVEGF (peripheral anti-VEGF)
        15630 * V1,   # A(3): CENDO (central endogenous IgG, ug/kg)
        15630 * V2,   # A(4): PENDO (peripheral endogenous IgG, ug/kg)
    ])

    A0_equil = np.array([0.0, 0.0, 0.0, 0.0])  # INIT=0 from $AESINIT

    return A0_diff, A0_equil


# ────────────────────────────────────────────────────
#  $AES block: Algebraic equilibrium equations
# ────────────────────────────────────────────────────

def vegf_fcrn_aes(A_equil: np.ndarray,
                  A_diff: np.ndarray,
                  params: dict) -> np.ndarray:
    """
    Algebraic equilibrium equations — direct translation of NONMEM $AES.

    A_equil = [VEGFRA, VEGFRP, ENDORA, ENDORP]
    A_diff  = [CVEGF, PVEGF, CENDO, PENDO]

    Residuals E(5)–E(8) must equal 0.
    """
    VEGFRA, VEGFRP, ENDORA, ENDORP = A_equil

    V1   = params['V1']
    RT   = params['RT']
    COVF = params['COVF']

    # KD conversions (nM → uM, /1000)
    KDAF  = params['KDA']  / 1000.0
    KDPF  = params['KDP']  / 1000.0
    EKDAF = params['EKDA'] / 1000.0
    EKDPF = params['EKDP'] / 1000.0

    DDEL = 1.0e-08

    # Total concentrations in uM
    CTX = max((A_diff[0] / V1) * COVF, DDEL)   # total anti-VEGF (uM)
    CETX = (A_diff[2] / V1) * COVF              # total endogenous IgG (uM)

    # Free species at each pH
    CFAX  = CTX  - VEGFRA              # free VEGF at acidic pH
    CFPX  = VEGFRA - VEGFRP            # free VEGF at physiological pH
    CEFAX = CETX - ENDORA              # free endo IgG at acidic pH
    CEFPX = ENDORA - ENDORP            # free endo IgG at physiological pH

    # Free FcRn at each pH
    RFAX = RT - VEGFRA - ENDORA        # free FcRn at acidic pH
    RFPX = RT - VEGFRP - ENDORP        # free FcRn at physiological pH

    WSCALE = 1.0

    # E(5): VEGF–FcRn binding at acidic pH
    E5 = (CFAX * RFAX - KDAF * VEGFRA) * WSCALE

    # E(6): VEGF–FcRn binding at physiological pH
    E6 = (CFPX * RFPX - KDPF * VEGFRP) * WSCALE

    # E(7): Endo IgG–FcRn binding at acidic pH
    # Note: WSCALE*0.0001 in NONMEM to stabilize the DAE solver
    E7 = (CEFAX * RFAX - EKDAF * ENDORA) * WSCALE * 0.0001

    # E(8): Endo IgG–FcRn binding at physiological pH
    E8 = (CEFPX * RFPX - EKDPF * ENDORP) * WSCALE * 0.0001

    return np.array([E5, E6, E7, E8])


# ────────────────────────────────────────────────────
#  $DES block: Differential equations
# ────────────────────────────────────────────────────

def vegf_fcrn_des(t: float, A_diff: np.ndarray, A_equil: np.ndarray,
                  params: dict) -> np.ndarray:
    """
    Differential equations — direct translation of NONMEM $DES.

    Computes dA/dt for the 4 differential compartments.

    Key mechanism:
    - Fraction unbound at acidic pH (FUA) → degraded
    - Fraction unbound at physiological pH (1-FUPH) × fraction bound at acidic (1-FUA) → degraded
    - Combined: clearance * (FUA + (1-FUA)*(1-FUPH)) * A
    """
    CVEGF, PVEGF, CENDO, PENDO = A_diff
    VEGFRA, VEGFRP, ENDORA, ENDORP = A_equil

    CL   = params['CL']
    V1   = params['V1']
    Q    = params['Q']
    V2   = params['V2']
    KSYN = params['KSYN']
    COVF = params['COVF']

    DDEL = 1.0e-08

    # Total concentrations (uM)
    CT  = max((CVEGF / V1) * COVF, DDEL)
    CET = (CENDO / V1) * COVF

    # ── VEGF fractions ──
    CFA  = CT - VEGFRA                # free VEGF at acidic pH
    CFP  = VEGFRA - VEGFRP            # free VEGF at physiological pH

    if CT > 0:
        FUA  = CFA / CT               # fraction unbound at acidic pH
    else:
        FUA = 1.0

    if VEGFRA > 0:
        FUPH = CFP / VEGFRA           # fraction unbound at physiological pH
    else:
        FUPH = 1.0

    # ── Endogenous IgG fractions ──
    CEFA  = CET - ENDORA
    CEFP  = ENDORA - ENDORP

    if CET > 0:
        FEUA  = CEFA / CET
    else:
        FEUA = 1.0

    if ENDORA > 0:
        FEUPH = CEFP / ENDORA
    else:
        FEUPH = 1.0

    # Ensure CVEGF doesn't go below threshold
    TEMPA1 = max(CVEGF, 1e-8)

    # Distribution terms
    TEMP1 = (Q / V1) * TEMPA1 - (Q / V2) * PVEGF    # VEGF distribution
    TEMP2 = (Q / V1) * CENDO  - (Q / V2) * PENDO     # Endo IgG distribution

    # ── Differential equations ──
    # DADT(1): anti-VEGF central
    #   Clearance: degradation of unbound at acidic pH + bound-then-released at physiological pH
    dCVEGF = -(CL / V1) * (FUA + (1 - FUA) * (1 - FUPH)) * TEMPA1 - TEMP1

    # DADT(2): anti-VEGF peripheral
    dPVEGF = TEMP1

    # DADT(3): endogenous IgG central (with synthesis)
    dCENDO = -(CL / V1) * (FEUA + (1 - FEUA) * (1 - FEUPH)) * CENDO - TEMP2 + KSYN

    # DADT(4): endogenous IgG peripheral
    dPENDO = TEMP2

    return np.array([dCVEGF, dPVEGF, dCENDO, dPENDO])


# ────────────────────────────────────────────────────
#  Combined DAE solver
# ────────────────────────────────────────────────────

def solve_vegf_fcrn_aes_at_point(A_diff: np.ndarray, A_equil_guess: np.ndarray,
                                  params: dict) -> np.ndarray:
    """Solve algebraic equations at a single time point."""
    def residuals(A_eq):
        return vegf_fcrn_aes(A_eq, A_diff, params)

    sol = fsolve(residuals, A_equil_guess)
    return np.maximum(sol, 0.0)


def simulate_vegf_fcrn_pk(
    t_span: Tuple[float, float],
    t_eval: np.ndarray,
    params: dict,
    dose: float = 10.0,
    method: str = 'Radau',
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> Dict[str, np.ndarray]:
    """
    Simulate the anti-VEGF PK model with FcRn recycling.

    This is the full DAE solver: at each ODE step, the algebraic
    equilibrium equations are solved for FcRn binding states.

    Parameters
    ----------
    t_span : tuple   (t_start, t_end) in hours
    t_eval : array   Time points to report (hours)
    params : dict    All model parameters
    dose : float     Initial drug amount in central compartment (ug/kg)
    method : str     ODE method ('Radau' for stiff systems)

    Returns
    -------
    Dict with time course of all compartments and derived quantities
    """
    # Initial conditions
    A0_diff, A0_equil = compute_initial_conditions(params)
    A0_diff[0] = dose  # Set initial drug dose

    # Solve for consistent initial algebraic states
    A0_equil = solve_vegf_fcrn_aes_at_point(A0_diff, A0_equil, params)

    # Combined state: [4 diff + 4 equil]
    y0 = np.concatenate([A0_diff, A0_equil])

    def rhs(t, y):
        A_diff = y[:4]
        A_equil_guess = y[4:]

        # Solve algebraic equations
        A_equil = solve_vegf_fcrn_aes_at_point(A_diff, A_equil_guess, params)

        # Differential rates
        dadt = vegf_fcrn_des(t, A_diff, A_equil, params)

        # Relaxation for algebraic states
        tau = 1e-3
        dadt_equil = (A_equil - A_equil_guess) / tau

        return np.concatenate([dadt, dadt_equil])

    # Integrate
    sol = solve_ivp(rhs, t_span, y0, method=method, t_eval=t_eval,
                    rtol=rtol, atol=atol, max_step=1.0)

    if not sol.success:
        warnings.warn(f"ODE solver: {sol.message}")

    # Post-process
    V1 = params['V1']
    n = len(sol.t)

    result = {
        't': sol.t,
        'CVEGF': sol.y[0],       # central anti-VEGF (ug/kg)
        'PVEGF': sol.y[1],       # peripheral anti-VEGF
        'CENDO': sol.y[2],       # central endogenous IgG
        'PENDO': sol.y[3],       # peripheral endogenous IgG
        'VEGFRA': np.zeros(n),   # VEGF–FcRn at acidic pH
        'VEGFRP': np.zeros(n),   # VEGF–FcRn at physiological pH
        'ENDORA': np.zeros(n),   # endo–FcRn at acidic pH
        'ENDORP': np.zeros(n),   # endo–FcRn at physiological pH
        'CVEGF_conc': sol.y[0] / V1,     # anti-VEGF concentration (ug/mL)
        'CENDO_conc': sol.y[2] / V1,     # endo IgG concentration (ug/mL)
        'FUA': np.zeros(n),      # fraction unbound VEGF at acidic
        'FUPH': np.zeros(n),     # fraction unbound VEGF at physiological
        'FEUA': np.zeros(n),     # fraction unbound endo at acidic
        'FEUPH': np.zeros(n),    # fraction unbound endo at physiological
    }

    # Re-solve algebraic states at each output time for exact values
    COVF = params['COVF']
    for i in range(n):
        A_diff_i = sol.y[:4, i]
        A_equil_guess = sol.y[4:, i]
        A_equil = solve_vegf_fcrn_aes_at_point(A_diff_i, A_equil_guess, params)

        result['VEGFRA'][i] = A_equil[0]
        result['VEGFRP'][i] = A_equil[1]
        result['ENDORA'][i] = A_equil[2]
        result['ENDORP'][i] = A_equil[3]

        # Compute fractions
        CT_uM = max((A_diff_i[0] / V1) * COVF, 1e-8)
        CET_uM = (A_diff_i[2] / V1) * COVF

        CFA = CT_uM - A_equil[0]
        CFP = A_equil[0] - A_equil[1]
        result['FUA'][i] = CFA / CT_uM if CT_uM > 0 else 1.0
        result['FUPH'][i] = CFP / A_equil[0] if A_equil[0] > 0 else 1.0

        CEFA = CET_uM - A_equil[2]
        CEFP = A_equil[2] - A_equil[3]
        result['FEUA'][i] = CEFA / CET_uM if CET_uM > 0 else 1.0
        result['FEUPH'][i] = CEFP / A_equil[2] if A_equil[2] > 0 else 1.0

    return result


# ──────────────────────────────────────
#  Plotting
# ──────────────────────────────────────

def plot_vegf_fcrn_pk(result: Dict[str, np.ndarray]):
    """Plot PK profiles mirroring NONMEM output."""
    t_hr = result['t']
    t_day = t_hr / 24.0

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Anti-VEGF concentration
    axes[0, 0].semilogy(t_day, result['CVEGF_conc'], 'b-', lw=2)
    axes[0, 0].set_xlabel('Time (days)')
    axes[0, 0].set_ylabel('Anti-VEGF Conc (µg/mL)')
    axes[0, 0].set_title('Anti-VEGF PK')
    axes[0, 0].grid(True, alpha=0.3)

    # Endogenous IgG concentration
    axes[0, 1].plot(t_day, result['CENDO_conc'], 'r-', lw=2)
    axes[0, 1].set_xlabel('Time (days)')
    axes[0, 1].set_ylabel('Endo IgG Conc (µg/mL)')
    axes[0, 1].set_title('Endogenous IgG')
    axes[0, 1].grid(True, alpha=0.3)

    # FcRn binding — anti-VEGF
    axes[0, 2].plot(t_day, result['VEGFRA'], 'g-', lw=2, label='Acidic pH')
    axes[0, 2].plot(t_day, result['VEGFRP'], 'g--', lw=2, label='Physiological pH')
    axes[0, 2].set_xlabel('Time (days)')
    axes[0, 2].set_ylabel('VEGF–FcRn Complex (µM)')
    axes[0, 2].set_title('Anti-VEGF FcRn Binding')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # FcRn binding — endogenous IgG
    axes[1, 0].plot(t_day, result['ENDORA'], 'm-', lw=2, label='Acidic pH')
    axes[1, 0].plot(t_day, result['ENDORP'], 'm--', lw=2, label='Physiological pH')
    axes[1, 0].set_xlabel('Time (days)')
    axes[1, 0].set_ylabel('Endo–FcRn Complex (µM)')
    axes[1, 0].set_title('Endogenous IgG FcRn Binding')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Fraction unbound — anti-VEGF
    axes[1, 1].plot(t_day, result['FUA'], 'b-', lw=2, label='Acidic (FUA)')
    axes[1, 1].plot(t_day, result['FUPH'], 'b--', lw=2, label='Physiological (FUPH)')
    axes[1, 1].set_xlabel('Time (days)')
    axes[1, 1].set_ylabel('Fraction Unbound')
    axes[1, 1].set_title('Anti-VEGF Unbound Fractions')
    axes[1, 1].legend()
    axes[1, 1].set_ylim(0, 1.05)
    axes[1, 1].grid(True, alpha=0.3)

    # Fraction unbound — endogenous IgG
    axes[1, 2].plot(t_day, result['FEUA'], 'r-', lw=2, label='Acidic (FEUA)')
    axes[1, 2].plot(t_day, result['FEUPH'], 'r--', lw=2, label='Physiological (FEUPH)')
    axes[1, 2].set_xlabel('Time (days)')
    axes[1, 2].set_ylabel('Fraction Unbound')
    axes[1, 2].set_title('Endo IgG Unbound Fractions')
    axes[1, 2].legend()
    axes[1, 2].set_ylim(0, 1.05)
    axes[1, 2].grid(True, alpha=0.3)

    plt.suptitle('Anti-VEGF IgG PK with FcRn Recycling (DAE Model)', fontsize=14)
    plt.tight_layout()
    plt.show()


# ──────────────────────────
#  Example / self-test
# ──────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Appendix 5: Anti-VEGF IgG with FcRn Recycling")
    print("Python equivalent of NONMEM ADVAN15 model")
    print("=" * 60)

    # ── Combine parameters ──
    params = {**DEFAULT_PK_PARAMS, **DEFAULT_KD_PARAMS}

    print(f"\nPK Parameters:")
    print(f"  CL   = {params['CL']} mL/kg/hr")
    print(f"  V1   = {params['V1']} mL/kg")
    print(f"  Q    = {params['Q']} mL/kg/day")
    print(f"  V2   = {params['V2']} mL/kg")
    print(f"  RT   = {params['RT']} nM (total FcRn)")
    print(f"  KSYN = {params['KSYN']} nmole/kg/day")

    print(f"\nFcRn Binding Parameters:")
    print(f"  KD (VEGF, acidic)     = {params['KDA']} nM")
    print(f"  KD (VEGF, physiol.)   = {params['KDP']} nM")
    print(f"  KD (Endo, acidic)     = {params['EKDA']} nM")
    print(f"  KD (Endo, physiol.)   = {params['EKDP']} nM")

    # ── Run simulation ──
    t_end = 30 * 24   # 30 days in hours
    t_eval = np.linspace(0, t_end, 500)
    dose = 10.0  # ug/kg

    print(f"\nSimulating {t_end / 24:.0f} days with dose = {dose} ug/kg...")

    result = simulate_vegf_fcrn_pk(
        t_span=(0, t_end),
        t_eval=t_eval,
        params=params,
        dose=dose,
    )

    # Show a few time points
    for t_day in [0, 1, 7, 14, 28]:
        idx = np.argmin(np.abs(result['t'] - t_day * 24))
        print(f"\n  Day {t_day}:")
        print(f"    Anti-VEGF conc = {result['CVEGF_conc'][idx]:.4f} ug/mL")
        print(f"    Endo IgG conc  = {result['CENDO_conc'][idx]:.1f} ug/mL")
        print(f"    FUA  = {result['FUA'][idx]:.4f}")
        print(f"    FUPH = {result['FUPH'][idx]:.4f}")

    print(f"\n  To plot: plot_vegf_fcrn_pk(result)")
