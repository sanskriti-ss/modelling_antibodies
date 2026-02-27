"""
Glofitamab Equilibrium Binding Model (2:1 CD20xCD3 Bispecific)
==============================================================

Equilibrium binding for Glofitamab's 2:1 format (two CD20-binding Fabs,
one CD3-binding Fab). Extends the bispecific framework from
ng_2024_paper/appendix1_equations.py and nonmem_appendix3.py.

Species:
  C     — Free drug (Glofitamab)
  CD3f  — Free CD3 on T cells
  CD20f — Free CD20 on B cells / tumor
  CRA   — Drug-CD3 dimer
  CRB   — Drug-CD20 dimer
  RCAB  — Ternary complex (T cell-drug-B cell bridge, pharmacologically active)

Key parameters (literature):
  KD_CD3  = 100 nM   (intentionally weak, Bacac 2018)
  KD_CD20 = 4.8 nM   (EC50 on human B cells, Bacac 2018)
  CD3/cell = 124,000  (from nonmem_appendix3.py)
  CD20/cell = 250,000 (literature standard for NHL)
  alpha = 1.0         (cooperativity, default)

Sources:
  - Bacac et al., Clin Cancer Res 2018;24:4785-4797
  - Ng et al., 2024 (existing codebase)
"""

import numpy as np
from scipy.optimize import fsolve
from typing import Dict, Optional
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────
#  Glofitamab-Specific Parameters
# ─────────────────────────────────────────────

GLOFITAMAB_PARAMS = {
    'KD_CD3':  100.0,    # nM — intentionally weak CD3 affinity (Bacac 2018)
    'KD_CD20': 4.8,      # nM — CD20 affinity on human B cells (Bacac 2018)
    'alpha':   1.0,      # cooperativity factor (tunable)
    'MW':      197000.0, # Da — molecular weight of Glofitamab
}

# Cell-surface receptor densities
RECEPTOR_PARAMS = {
    'CD3_per_cell':  124000,   # CD3 molecules per T cell (Haber 2021)
    'CD20_per_cell': 250000,   # CD20 molecules per B cell (NHL standard)
}


def compute_receptor_concentrations(
    T_cells_per_well: float = 50000,
    B_cells_per_well: float = 50000,
    well_volume_uL: float = 100.0,
    CD3_per_cell: int = 124000,
    CD20_per_cell: int = 250000,
) -> Dict[str, float]:
    """
    Compute receptor concentrations in nM from cell counts.

    Follows the pattern from nonmem_appendix3.py:60-82.

    Parameters
    ----------
    T_cells_per_well : float  T cells per well
    B_cells_per_well : float  B/tumor cells per well
    well_volume_uL : float    Well volume in microliters
    CD3_per_cell : int        CD3 molecules per T cell
    CD20_per_cell : int       CD20 molecules per B/tumor cell
    """
    cells_per_mL = 1.0 / (well_volume_uL * 1e-6)  # wells/mL

    # Total molecules per mL -> nM
    CD3_total_nM = (T_cells_per_well * cells_per_mL *
                    CD3_per_cell / 6.022e23) * 1e9
    CD20_total_nM = (B_cells_per_well * cells_per_mL *
                     CD20_per_cell / 6.022e23) * 1e9

    return {'CD3': CD3_total_nM, 'CD20': CD20_total_nM}


# ────────────────────────────────────────────────────
#  Core equilibrium solver (mirrors nonmem_appendix3)
# ────────────────────────────────────────────────────

def glofitamab_aes(A_equil: np.ndarray, params: dict) -> np.ndarray:
    """
    Algebraic equilibrium equations for Glofitamab binding.

    A_equil = [CRA, CRB, RCAB]
      CRA  = drug-CD3 dimer
      CRB  = drug-CD20 dimer
      RCAB = ternary complex (T cell-drug-B cell bridge)

    Residuals E4, E5, E6 must all equal 0 at equilibrium.

    Mapping to nonmem_appendix3:
      RA -> CD3 (effector receptor on T cells)
      RB -> CD20 (target receptor on B/tumor cells)
    """
    CRA, CRB, RCAB = A_equil

    CT  = params['CT']    # total drug
    RAT = params['RAT']   # total CD3
    RBT = params['RBT']   # total CD20
    KAC = params['KAC']   # KD for CD3 binding (100 nM)
    KBC = params['KBC']   # KD for CD20 binding (4.8 nM)
    ALP = params['ALP']   # cooperativity

    # Free species (mass balance)
    CF   = CT  - CRA - CRB - RCAB    # free drug
    CD3f = RAT - CRA - RCAB          # free CD3
    CD20f = RBT - CRB - RCAB         # free CD20

    # Equilibrium residuals (Eq A34, A36, A40)
    E4 = CF * CD3f - KAC * CRA                         # drug-CD3 dimer
    E5 = CF * CD20f - KBC * CRB                        # drug-CD20 dimer
    E6 = CF * CD3f * CD20f - ALP * KAC * KBC * RCAB    # ternary complex

    return np.array([E4, E5, E6])


def solve_equilibrium(
    CT: float, RAT: float, RBT: float,
    KAC: float = 100.0, KBC: float = 4.8, ALP: float = 1.0,
    initial_guess: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Solve the Glofitamab binding equilibrium.

    Parameters
    ----------
    CT : float   Total drug concentration (nM)
    RAT : float  Total CD3 concentration (nM)
    RBT : float  Total CD20 concentration (nM)
    KAC : float  KD for CD3 binding (nM), default 100
    KBC : float  KD for CD20 binding (nM), default 4.8
    ALP : float  Cooperativity factor, default 1.0

    Returns
    -------
    Dict with all species concentrations and mass-balance checks
    """
    params = {'CT': CT, 'RAT': RAT, 'RBT': RBT,
              'KAC': KAC, 'KBC': KBC, 'ALP': ALP}

    if initial_guess is None:
        initial_guess = np.array([CT * 0.01, CT * 0.01, CT * 0.001])

    solution = fsolve(glofitamab_aes, initial_guess, args=(params,),
                      full_output=False)
    CRA, CRB, RCAB = np.maximum(solution, 0.0)

    CF    = CT  - CRA - CRB - RCAB
    CD3f  = RAT - CRA - RCAB
    CD20f = RBT - CRB - RCAB

    return {
        'CF':    max(CF, 0.0),
        'CD3f':  max(CD3f, 0.0),
        'CD20f': max(CD20f, 0.0),
        'CRA':   CRA,     # drug-CD3 dimer
        'CRB':   CRB,     # drug-CD20 dimer
        'RCAB':  RCAB,    # ternary complex (active species)
        # Mass-balance checks
        'CT_check':  CF + CRA + CRB + RCAB,
        'RAT_check': CD3f + CRA + RCAB,
        'RBT_check': CD20f + CRB + RCAB,
    }


# ──────────────────────────────────────────────────
#  Dose-response simulation with warm-starting
# ──────────────────────────────────────────────────

def simulate_dose_response(
    CT_range: np.ndarray,
    RAT: float,
    RBT: float,
    KAC: float = 100.0,
    KBC: float = 4.8,
    ALP: float = 1.0,
) -> Dict[str, np.ndarray]:
    """
    Simulate Glofitamab dose-response across drug concentrations.

    Uses warm-starting (previous solution as next initial guess)
    following the pattern from nonmem_appendix3.py:191-238.

    Returns
    -------
    Dict of arrays for each species concentration
    """
    n = len(CT_range)
    result = {
        'CT':    CT_range,
        'CF':    np.zeros(n),
        'CD3f':  np.zeros(n),
        'CD20f': np.zeros(n),
        'CRA':   np.zeros(n),
        'CRB':   np.zeros(n),
        'RCAB':  np.zeros(n),
    }

    prev_guess = np.array([1e-6, 1e-6, 1e-8])

    for i, ct in enumerate(CT_range):
        sol = solve_equilibrium(ct, RAT, RBT, KAC, KBC, ALP,
                                initial_guess=prev_guess)
        for key in ['CF', 'CD3f', 'CD20f', 'CRA', 'CRB', 'RCAB']:
            result[key][i] = sol[key]

        # Warm-start next solve
        prev_guess = np.array([sol['CRA'], sol['CRB'], sol['RCAB']])
        prev_guess = np.maximum(prev_guess, 1e-12)

    return result


# ──────────────────────────────────────────────
#  Plotting
# ──────────────────────────────────────────────

def plot_dose_response(result: Dict[str, np.ndarray],
                       savepath: Optional[str] = None):
    """Plot Glofitamab dose-response: free species, dimers, and ternary complex."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # Free drug
    axes[0, 0].semilogx(result['CT'], result['CF'], 'b-', lw=2)
    axes[0, 0].set_xlabel('Total Glofitamab (nM)')
    axes[0, 0].set_ylabel('Free Drug (nM)')
    axes[0, 0].set_title('Free Drug')
    axes[0, 0].grid(True, alpha=0.3)

    # Drug-CD3 dimer
    axes[0, 1].semilogx(result['CT'], result['CRA'], 'r-', lw=2)
    axes[0, 1].set_xlabel('Total Glofitamab (nM)')
    axes[0, 1].set_ylabel('Drug-CD3 Dimer (nM)')
    axes[0, 1].set_title('Drug-CD3 Dimer (CRA)')
    axes[0, 1].grid(True, alpha=0.3)

    # Drug-CD20 dimer
    axes[1, 0].semilogx(result['CT'], result['CRB'], 'g-', lw=2)
    axes[1, 0].set_xlabel('Total Glofitamab (nM)')
    axes[1, 0].set_ylabel('Drug-CD20 Dimer (nM)')
    axes[1, 0].set_title('Drug-CD20 Dimer (CRB)')
    axes[1, 0].grid(True, alpha=0.3)

    # Ternary complex (RCAB) — the active species
    axes[1, 1].semilogx(result['CT'], result['RCAB'], 'm-', lw=2)
    axes[1, 1].set_xlabel('Total Glofitamab (nM)')
    axes[1, 1].set_ylabel('Ternary Complex (nM)')
    axes[1, 1].set_title('T cell-Drug-B cell Bridge (RCAB)')
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('Glofitamab (2:1 CD20xCD3) Equilibrium Dose-Response\n'
                 f'KD_CD3={GLOFITAMAB_PARAMS["KD_CD3"]} nM, '
                 f'KD_CD20={GLOFITAMAB_PARAMS["KD_CD20"]} nM',
                 fontsize=13)
    plt.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches='tight')
        print(f"Figure saved to {savepath}")
    else:
        plt.show()


def plot_receptor_occupancy(result: Dict[str, np.ndarray],
                            RAT: float, RBT: float,
                            savepath: Optional[str] = None):
    """Plot fractional receptor occupancy for CD3 and CD20."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # CD3 occupancy: fraction of CD3 engaged = (CRA + RCAB) / RAT
    cd3_occ = (result['CRA'] + result['RCAB']) / RAT
    axes[0].semilogx(result['CT'], cd3_occ * 100, 'r-', lw=2)
    axes[0].set_xlabel('Total Glofitamab (nM)')
    axes[0].set_ylabel('CD3 Occupancy (%)')
    axes[0].set_title('CD3 Receptor Occupancy (T cells)')
    axes[0].set_ylim(0, 105)
    axes[0].grid(True, alpha=0.3)

    # CD20 occupancy: fraction of CD20 engaged = (CRB + RCAB) / RBT
    cd20_occ = (result['CRB'] + result['RCAB']) / RBT
    axes[1].semilogx(result['CT'], cd20_occ * 100, 'g-', lw=2)
    axes[1].set_xlabel('Total Glofitamab (nM)')
    axes[1].set_ylabel('CD20 Occupancy (%)')
    axes[1].set_title('CD20 Receptor Occupancy (B/Tumor cells)')
    axes[1].set_ylim(0, 105)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Glofitamab Receptor Occupancy', fontsize=13)
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
    print("Glofitamab (2:1 CD20xCD3) Equilibrium Binding Model")
    print("=" * 60)

    # ── Receptor concentrations ──
    init = compute_receptor_concentrations(
        T_cells_per_well=50000,
        B_cells_per_well=50000,
    )
    RAT = init['CD3']
    RBT = init['CD20']
    print(f"\nReceptor concentrations:")
    print(f"  CD3  (RAT) = {RAT:.4f} nM  ({RECEPTOR_PARAMS['CD3_per_cell']:,} per T cell)")
    print(f"  CD20 (RBT) = {RBT:.4f} nM  ({RECEPTOR_PARAMS['CD20_per_cell']:,} per B cell)")

    # ── Glofitamab parameters ──
    KAC = GLOFITAMAB_PARAMS['KD_CD3']    # 100 nM
    KBC = GLOFITAMAB_PARAMS['KD_CD20']   # 4.8 nM
    ALP = GLOFITAMAB_PARAMS['alpha']     # 1.0

    print(f"\nGlofitamab binding parameters:")
    print(f"  KD_CD3  = {KAC} nM (intentionally weak)")
    print(f"  KD_CD20 = {KBC} nM")
    print(f"  alpha   = {ALP}")

    # ── Dose-response ──
    CT_range = np.logspace(-3, 4, 300)
    result = simulate_dose_response(CT_range, RAT, RBT, KAC, KBC, ALP)

    # Show key concentrations
    print(f"\nDose-response at selected concentrations:")
    for ct_test in [0.1, 1.0, 10.0, 100.0, 1000.0]:
        idx = np.argmin(np.abs(CT_range - ct_test))
        print(f"\n  CT = {CT_range[idx]:.2f} nM:")
        print(f"    Free drug  = {result['CF'][idx]:.4f} nM")
        print(f"    CRA (drug-CD3)  = {result['CRA'][idx]:.4f} nM")
        print(f"    CRB (drug-CD20) = {result['CRB'][idx]:.4f} nM")
        print(f"    RCAB (ternary)  = {result['RCAB'][idx]:.6f} nM")

    # ── Mass balance verification ──
    print(f"\nMass balance verification (at CT=10 nM):")
    sol = solve_equilibrium(10.0, RAT, RBT, KAC, KBC, ALP)
    print(f"  CT  = {10.0:.4f},  CT_check  = {sol['CT_check']:.4f}")
    print(f"  RAT = {RAT:.4f},  RAT_check = {sol['RAT_check']:.4f}")
    print(f"  RBT = {RBT:.4f},  RBT_check = {sol['RBT_check']:.4f}")

    # ── Verify bell-shaped RCAB curve ──
    rcab_max_idx = np.argmax(result['RCAB'])
    print(f"\nBell-shaped RCAB verification:")
    print(f"  Peak RCAB = {result['RCAB'][rcab_max_idx]:.6f} nM "
          f"at CT = {result['CT'][rcab_max_idx]:.2f} nM")
    print(f"  RCAB at low CT  = {result['RCAB'][0]:.2e} nM")
    print(f"  RCAB at high CT = {result['RCAB'][-1]:.2e} nM")
    is_bell = (result['RCAB'][rcab_max_idx] > result['RCAB'][0] and
               result['RCAB'][rcab_max_idx] > result['RCAB'][-1])
    print(f"  Bell-shaped: {is_bell}")

    print(f"\n  To plot: plot_dose_response(result)")
    print(f"  To plot occupancy: plot_receptor_occupancy(result, RAT, RBT)")
