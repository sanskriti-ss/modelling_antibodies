"""
Glofitamab PK/TMDD ODE Model
=============================

Full pharmacokinetic model with target-mediated drug disposition (TMDD)
for Glofitamab, implementing the Schropp et al. 2019 7-ODE framework.

Follows the ODE integration pattern from nonmem_appendix5.py (solve_ivp/Radau).

7 ODEs (Schropp 2019 TMDD for bispecific antibodies):
  d(C)/dt    — Free drug in central compartment
  d(RA)/dt   — Free CD20 (target receptor on B/tumor cells)
  d(RB)/dt   — Free CD3 (effector receptor on T cells)
  d(RCA)/dt  — Drug-CD20 dimer
  d(RCB)/dt  — Drug-CD3 dimer
  d(RCAB)/dt — Ternary complex (T cell-drug-B cell bridge, active species)
  d(AP)/dt   — Drug in peripheral compartment

Mapping to Ray et al. 2024 (Reference 2):
  Ray: Target receptor (IL-6R)  -> Glofitamab: CD20 (RA) — tumor marker
  Ray: Second arm (IL-8R)       -> Glofitamab: CD3 (RB)  — effector recruiter
  Ray: Ternary complex RCAB     -> Glofitamab: RCAB      — active species
  Ray: kint (internalization)   -> Glofitamab: kintAB    — complex turnover
  Ray: KD_target                -> Glofitamab: KD_CD20   — 4.8 nM

PK parameters from Glofitamab popPK / FDA label (Gibiansky 2023):
  CL = 0.602 L/day, Vc = 3.33 L, Vp = 2.18 L, Q = 0.459 L/day
  MW = 197,000 Da, t1/2 effective ~ 6.54 days

Sources:
  - Schropp et al., J Pharmacokinet Pharmacodyn 2019
  - Glofitamab FDA label / popPK (Gibiansky 2023)
  - Ray et al., PLoS Comput Biol 2024
  - Bacac et al., Clin Cancer Res 2018
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Dict, Optional, Tuple, List
import matplotlib.pyplot as plt
import warnings

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ──────────────────────────────────────────
#  PK and Binding Parameters
# ──────────────────────────────────────────

PK_PARAMS = {
    # PK parameters (Glofitamab popPK, Gibiansky 2023)
    'CL':     0.602,     # L/day — linear clearance
    'Vc':     3.33,      # L — central volume
    'Vp':     2.18,      # L — peripheral volume
    'Q':      0.459,     # L/day — intercompartmental clearance
    'MW':     197000.0,  # Da — molecular weight

    # Binding kinetics (Bacac 2018, Schropp 2019 framework)
    'kon_CD20':  0.1,     # 1/(nM*day) — CD20 on-rate
    'koff_CD20': 0.48,    # 1/day — CD20 off-rate (KD=4.8 nM)
    'kon_CD3':   0.01,    # 1/(nM*day) — CD3 on-rate
    'koff_CD3':  1.0,     # 1/day — CD3 off-rate (KD=100 nM)

    # Receptor turnover
    'ksyn_CD20': 5.0,     # nM/day — CD20 synthesis rate
    'kdeg_CD20': 0.1,     # 1/day — CD20 degradation rate (t1/2 ~ 7 days)
    'ksyn_CD3':  10.0,    # nM/day — CD3 synthesis rate
    'kdeg_CD3':  0.2,     # 1/day — CD3 degradation rate

    # Internalization rates
    'kint_CD20':  0.15,   # 1/day — drug-CD20 dimer internalization
    'kint_CD3':   0.1,    # 1/day — drug-CD3 dimer internalization
    'kint_RCAB':  0.3,    # 1/day — ternary complex internalization

    # TMDD clearance (CD20-mediated, time-varying)
    'CL_TMDD_0':  0.396,  # L/day — initial TMDD clearance
    'tau_TMDD':   1.56,   # days — TMDD clearance time constant
}


def mg_to_nM(dose_mg: float, Vc_L: float, MW: float) -> float:
    """Convert dose in mg to plasma concentration in nM.

    mg/L * 1e6 / MW(Da) = nM
    """
    conc_mg_per_L = dose_mg / Vc_L
    conc_nM = conc_mg_per_L * 1e6 / MW
    return conc_nM


def nM_to_ug_per_mL(conc_nM: float, MW: float) -> float:
    """Convert nM to ug/mL.

    nM * MW(Da) / 1e6 = ug/mL
    """
    return conc_nM * MW / 1e6


# ──────────────────────────────────────────
#  Steady-state initial receptor levels
# ──────────────────────────────────────────

def compute_steady_state_receptors(params: dict) -> Dict[str, float]:
    """Compute baseline receptor concentrations at steady state."""
    RA0 = params['ksyn_CD20'] / params['kdeg_CD20']  # CD20 baseline
    RB0 = params['ksyn_CD3'] / params['kdeg_CD3']    # CD3 baseline
    return {'RA0': RA0, 'RB0': RB0}


# ────────────────────────────────────────────────────
#  7-ODE System (Schropp 2019 TMDD for BsAbs)
# ────────────────────────────────────────────────────

def glofitamab_odes(t: float, y: np.ndarray, params: dict) -> np.ndarray:
    """
    Glofitamab TMDD ODE system — 7 differential equations.

    State vector y = [C, RA, RB, RCA, RCB, RCAB, AP]
      C    — free drug in central (nM)
      RA   — free CD20 (nM)
      RB   — free CD3 (nM)
      RCA  — drug-CD20 dimer (nM)
      RCB  — drug-CD3 dimer (nM)
      RCAB — ternary complex (nM)
      AP   — drug in peripheral (amount, nM*L)
    """
    C, RA, RB, RCA, RCB, RCAB, AP = np.maximum(y, 0.0)

    # Unpack parameters
    CL   = params['CL']
    Vc   = params['Vc']
    Vp   = params['Vp']
    Q    = params['Q']

    kon1  = params['kon_CD20']   # CD20 on-rate
    koff1 = params['koff_CD20']  # CD20 off-rate
    kon2  = params['kon_CD3']    # CD3 on-rate
    koff2 = params['koff_CD3']   # CD3 off-rate

    # For ternary complex formation, use same kon/koff
    # (cooperativity = 1.0 by default)
    kon3  = kon1    # RCA + RB -> RCAB (CD20-dimer binds CD3)
    koff3 = koff2   # RCAB -> RCA + RB
    kon4  = kon2    # RCB + RA -> RCAB (CD3-dimer binds CD20)
    koff4 = koff1   # RCAB -> RCB + RA

    ksynA  = params['ksyn_CD20']
    kdegA  = params['kdeg_CD20']
    ksynB  = params['ksyn_CD3']
    kdegB  = params['kdeg_CD3']
    kintA  = params['kint_CD20']
    kintB  = params['kint_CD3']
    kintAB = params['kint_RCAB']

    # Time-dependent TMDD clearance (exponential decay as CD20 is depleted)
    CL_TMDD = params['CL_TMDD_0'] * np.exp(-t / params['tau_TMDD'])

    # ── ODEs ──
    # d(C)/dt: free drug in central
    dC = (-(CL + CL_TMDD) * C / Vc
          - kon1 * C * RA + koff1 * RCA
          - kon2 * C * RB + koff2 * RCB
          - Q / Vc * C + Q / Vp * AP / Vc)

    # d(RA)/dt: free CD20
    dRA = (ksynA - kdegA * RA
           - kon1 * C * RA + koff1 * RCA
           - kon4 * RCB * RA + koff4 * RCAB)

    # d(RB)/dt: free CD3
    dRB = (ksynB - kdegB * RB
           - kon2 * C * RB + koff2 * RCB
           - kon3 * RCA * RB + koff3 * RCAB)

    # d(RCA)/dt: drug-CD20 dimer
    dRCA = (kon1 * C * RA - (koff1 + kintA) * RCA
            - kon3 * RCA * RB + koff3 * RCAB)

    # d(RCB)/dt: drug-CD3 dimer
    dRCB = (kon2 * C * RB - (koff2 + kintB) * RCB
            - kon4 * RCB * RA + koff4 * RCAB)

    # d(RCAB)/dt: ternary complex (pharmacologically active)
    dRCAB = (kon3 * RCA * RB + kon4 * RCB * RA
             - (koff3 + koff4 + kintAB) * RCAB)

    # d(AP)/dt: peripheral compartment
    dAP = Q * C - Q / Vp * AP

    return np.array([dC, dRA, dRB, dRCA, dRCB, dRCAB, dAP])


# ────────────────────────────────────────────────────
#  Single-dose simulation
# ────────────────────────────────────────────────────

def simulate_glofitamab_pk(
    dose_mg: float,
    t_end_days: float = 21.0,
    params: Optional[dict] = None,
    n_points: int = 500,
    obinutuzumab_pretreat: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Simulate Glofitamab PK after a single IV dose.

    Parameters
    ----------
    dose_mg : float        Dose in mg IV
    t_end_days : float     Simulation duration (days)
    params : dict          Model parameters (default: PK_PARAMS)
    n_points : int         Number of output time points
    obinutuzumab_pretreat : bool  If True, reduce CD20 baseline by 90%

    Returns
    -------
    Dict with time course of all compartments
    """
    if params is None:
        params = PK_PARAMS.copy()

    # Compute initial conditions
    ss = compute_steady_state_receptors(params)
    RA0 = ss['RA0']
    RB0 = ss['RB0']

    if obinutuzumab_pretreat:
        RA0 *= 0.1  # Obinutuzumab depletes ~90% of CD20+ B cells

    # Convert dose to central concentration (nM)
    C0 = mg_to_nM(dose_mg, params['Vc'], params['MW'])

    # y0 = [C, RA, RB, RCA, RCB, RCAB, AP]
    y0 = np.array([C0, RA0, RB0, 0.0, 0.0, 0.0, 0.0])

    t_eval = np.linspace(0, t_end_days, n_points)

    sol = solve_ivp(
        glofitamab_odes, (0, t_end_days), y0,
        args=(params,), method='Radau',
        t_eval=t_eval, rtol=1e-8, atol=1e-10,
        max_step=0.5,
    )

    if not sol.success:
        warnings.warn(f"ODE solver warning: {sol.message}")

    MW = params['MW']
    result = {
        't_days':     sol.t,
        'C_nM':       sol.y[0],   # free drug (nM)
        'RA_nM':      sol.y[1],   # free CD20 (nM)
        'RB_nM':      sol.y[2],   # free CD3 (nM)
        'RCA_nM':     sol.y[3],   # drug-CD20 dimer (nM)
        'RCB_nM':     sol.y[4],   # drug-CD3 dimer (nM)
        'RCAB_nM':    sol.y[5],   # ternary complex (nM)
        'AP_nM':      sol.y[6],   # peripheral (nM*L)
        'C_ug_mL':    np.array([nM_to_ug_per_mL(c, MW) for c in sol.y[0]]),
    }

    return result


# ────────────────────────────────────────────────────
#  Step-up dosing protocol simulation
# ────────────────────────────────────────────────────

def simulate_stepup_protocol(
    params: Optional[dict] = None,
    n_points_per_cycle: int = 200,
) -> Dict[str, np.ndarray]:
    """
    Simulate the clinical step-up dosing protocol for Glofitamab.

    Protocol (FDA-approved):
      Day -7:   Obinutuzumab 1000mg (modeled as CD20 depletion)
      C1D8:     Glofitamab 2.5 mg IV
      C1D15:    Glofitamab 10 mg IV
      C2D1+:    Glofitamab 30 mg IV Q3W x 10 more cycles

    Returns
    -------
    Dict with full time course across all cycles
    """
    if params is None:
        params = PK_PARAMS.copy()

    # Dosing schedule: (day, dose_mg)
    # Day 0 = obinutuzumab pretreatment
    # Day 7 = C1D8 (2.5 mg), Day 14 = C1D15 (10 mg)
    # Day 28 = C2D1 (30 mg), then Q3W
    doses = [
        (7,   2.5),   # C1D8
        (14,  10.0),  # C1D15
    ]
    # C2D1 through C12D1: 30 mg Q3W
    for cycle in range(10):
        doses.append((28 + cycle * 21, 30.0))

    t_end = doses[-1][0] + 21  # 3 weeks after last dose
    n_total = n_points_per_cycle * 12

    # Initial conditions with obinutuzumab pretreatment
    ss = compute_steady_state_receptors(params)
    RA0 = ss['RA0'] * 0.1   # 90% CD20 depletion from obinutuzumab
    RB0 = ss['RB0']

    # State: [C, RA, RB, RCA, RCB, RCAB, AP]
    y_current = np.array([0.0, RA0, RB0, 0.0, 0.0, 0.0, 0.0])

    all_t = []
    all_y = []
    t_current = 0.0

    for dose_day, dose_mg in doses:
        if dose_day > t_current:
            # Simulate from current time to dose time
            t_segment = np.linspace(t_current, dose_day, 50)
            sol = solve_ivp(
                glofitamab_odes, (t_current, dose_day), y_current,
                args=(params,), method='Radau',
                t_eval=t_segment, rtol=1e-8, atol=1e-10,
                max_step=0.5,
            )
            if sol.success:
                all_t.append(sol.t)
                all_y.append(sol.y)
                y_current = sol.y[:, -1].copy()

        # Add dose as bolus to central compartment
        C_add = mg_to_nM(dose_mg, params['Vc'], params['MW'])
        y_current[0] += C_add
        t_current = dose_day

    # Final segment after last dose
    t_segment = np.linspace(t_current, t_end, 100)
    sol = solve_ivp(
        glofitamab_odes, (t_current, t_end), y_current,
        args=(params,), method='Radau',
        t_eval=t_segment, rtol=1e-8, atol=1e-10,
        max_step=0.5,
    )
    if sol.success:
        all_t.append(sol.t)
        all_y.append(sol.y)

    # Concatenate
    t_full = np.concatenate(all_t)
    y_full = np.concatenate(all_y, axis=1)

    MW = params['MW']
    result = {
        't_days':     t_full,
        'C_nM':       y_full[0],
        'RA_nM':      y_full[1],
        'RB_nM':      y_full[2],
        'RCA_nM':     y_full[3],
        'RCB_nM':     y_full[4],
        'RCAB_nM':    y_full[5],
        'AP_nM':      y_full[6],
        'C_ug_mL':    np.array([nM_to_ug_per_mL(c, MW) for c in y_full[0]]),
        'dose_times': [d[0] for d in doses],
        'dose_amounts': [d[1] for d in doses],
    }

    return result


# ──────────────────────────────────────
#  Plotting
# ──────────────────────────────────────

def plot_pk_profile(result: Dict[str, np.ndarray],
                    savepath: Optional[str] = None):
    """Plot PK profile with drug concentration, receptors, and complexes."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    t = result['t_days']

    # Drug concentration (ug/mL)
    axes[0, 0].semilogy(t, np.maximum(result['C_ug_mL'], 1e-6), 'b-', lw=2)
    axes[0, 0].set_ylabel('Drug Conc (ug/mL)')
    axes[0, 0].set_title('Glofitamab PK')
    axes[0, 0].grid(True, alpha=0.3)
    # Mark doses if available
    if 'dose_times' in result:
        for dt, da in zip(result['dose_times'], result['dose_amounts']):
            axes[0, 0].axvline(dt, color='gray', ls='--', alpha=0.4)
            axes[0, 0].text(dt, axes[0, 0].get_ylim()[1] * 0.5,
                            f'{da}mg', rotation=90, fontsize=7, alpha=0.6)

    # Free CD20
    axes[0, 1].plot(t, result['RA_nM'], 'g-', lw=2)
    axes[0, 1].set_ylabel('Free CD20 (nM)')
    axes[0, 1].set_title('Free CD20 (Target)')
    axes[0, 1].grid(True, alpha=0.3)

    # Free CD3
    axes[0, 2].plot(t, result['RB_nM'], 'r-', lw=2)
    axes[0, 2].set_ylabel('Free CD3 (nM)')
    axes[0, 2].set_title('Free CD3 (Effector)')
    axes[0, 2].grid(True, alpha=0.3)

    # Drug-CD20 dimer
    axes[1, 0].plot(t, result['RCA_nM'], 'g--', lw=2)
    axes[1, 0].set_ylabel('Drug-CD20 Dimer (nM)')
    axes[1, 0].set_title('Drug-CD20 Dimer')
    axes[1, 0].set_xlabel('Time (days)')
    axes[1, 0].grid(True, alpha=0.3)

    # Ternary complex (RCAB) — the key PD driver
    axes[1, 1].plot(t, result['RCAB_nM'], 'm-', lw=2.5)
    axes[1, 1].set_ylabel('Ternary Complex (nM)')
    axes[1, 1].set_title('RCAB (Active Species)')
    axes[1, 1].set_xlabel('Time (days)')
    axes[1, 1].grid(True, alpha=0.3)

    # Drug-CD3 dimer
    axes[1, 2].plot(t, result['RCB_nM'], 'r--', lw=2)
    axes[1, 2].set_ylabel('Drug-CD3 Dimer (nM)')
    axes[1, 2].set_title('Drug-CD3 Dimer')
    axes[1, 2].set_xlabel('Time (days)')
    axes[1, 2].grid(True, alpha=0.3)

    plt.suptitle('Glofitamab PK/TMDD Model (Schropp 2019 Framework)',
                 fontsize=14)
    plt.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches='tight')
        print(f"Figure saved to {savepath}")
    else:
        plt.show()


# ──────────────────────────
#  Example / self-test
# ──────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Glofitamab PK/TMDD Model (7-ODE, Schropp 2019)")
    print("=" * 60)

    params = PK_PARAMS.copy()

    # ── Steady-state receptors ──
    ss = compute_steady_state_receptors(params)
    print(f"\nBaseline receptor levels:")
    print(f"  CD20 (RA0) = {ss['RA0']:.1f} nM")
    print(f"  CD3  (RB0) = {ss['RB0']:.1f} nM")

    # ── Single dose: 2.5 mg with obinutuzumab pretreatment ──
    print(f"\n--- Single dose: 2.5 mg (with obinutuzumab pretreatment) ---")
    result_2p5 = simulate_glofitamab_pk(
        dose_mg=2.5, t_end_days=21.0,
        obinutuzumab_pretreat=True,
    )
    cmax = np.max(result_2p5['C_ug_mL'])
    print(f"  Cmax = {cmax:.4f} ug/mL")
    print(f"  Clinical target Cmax ~ 0.674 ug/mL")

    # ── Single dose: 10 mg ──
    print(f"\n--- Single dose: 10 mg ---")
    result_10 = simulate_glofitamab_pk(
        dose_mg=10.0, t_end_days=21.0,
        obinutuzumab_pretreat=True,
    )
    cmax_10 = np.max(result_10['C_ug_mL'])
    print(f"  Cmax = {cmax_10:.4f} ug/mL")
    print(f"  Clinical target Cmax ~ 2.34 ug/mL")

    # ── Peak RCAB for each dose ──
    print(f"\n  Peak RCAB (2.5 mg)  = {np.max(result_2p5['RCAB_nM']):.4f} nM")
    print(f"  Peak RCAB (10 mg)   = {np.max(result_10['RCAB_nM']):.4f} nM")

    # ── Step-up protocol ──
    print(f"\n--- Step-up dosing protocol ---")
    result_stepup = simulate_stepup_protocol()
    print(f"  Total simulation time: {result_stepup['t_days'][-1]:.0f} days")
    print(f"  Number of doses: {len(result_stepup['dose_times'])}")
    print(f"  Peak drug conc: {np.max(result_stepup['C_ug_mL']):.4f} ug/mL")
    print(f"  Peak RCAB:      {np.max(result_stepup['RCAB_nM']):.4f} nM")

    print(f"\n  To plot: plot_pk_profile(result_stepup)")
