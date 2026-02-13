"""
Appendix 3: NONMEM Model for T-Cell-Redirecting Bispecific Antibody — Python
=============================================================================

Python translation of the NONMEM ADVAN15 model from Appendix 3.

NONMEM structure:
  COMP 1 (CT)   — total drug concentration         (DADT = ~0, constant)
  COMP 2 (RAT)  — total effector receptor (CD3)     (DADT = ~0, constant)
  COMP 3 (RBT)  — total target receptor (MUC16)     (DADT = ~0, constant)
  COMP 4 (CRA)  — drug–effector dimer               (EQUIL, algebraic)
  COMP 5 (CRB)  — drug–target dimer                 (EQUIL, algebraic)
  COMP 6 (CRAB) — drug–effector–target trimer        (EQUIL, algebraic)

Algebraic equations (from $AES):
  E(4): CF*RAF - KAC*CRA  = 0        → Eq A34
  E(5): CF*RBF - KBC*CRB  = 0        → Eq A36
  E(6): CF*RAF*RBF - ALP*KAC*KBC*CRAB = 0  → Eq A40

where:
  CF  = CT  - CRA - CRB - CRAB       (free drug)
  RAF = RAT - CRA - CRAB             (free effector / CD3)
  RBF = RBT - CRB - CRAB             (free target / MUC16)

Source: Haber L et al. Sci Report 2021
Drug variants:
  1 = MUC16xCD3M   (KD_CD3 = 58.0 nM)
  2 = MUC16xCD3W   (KD_CD3 = 835 nM)
  3 = MUC16xCD3VW  (KD_CD3 = estimated)
"""

import numpy as np
from scipy.optimize import fsolve
from typing import Dict, Optional, List, Tuple
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────
#  Model Parameters (from NONMEM $THETA block)
# ─────────────────────────────────────────────
DRUG_PARAMS = {
    1: {  # MUC16xCD3M
        'name': 'MUC16xCD3M',
        'KAC': 58.0,     # KD for CD3 binding (nM)
        'ALP': None,     # estimated (THETA 10)
    },
    2: {  # MUC16xCD3W
        'name': 'MUC16xCD3W',
        'KAC': 835.0,    # KD for CD3 binding (nM)
        'ALP': 1.0,      # fixed
    },
    3: {  # MUC16xCD3VW
        'name': 'MUC16xCD3VW',
        'KAC': 2000.0,   # estimated (initial)
        'ALP': 1.0,      # fixed
    },
}


def compute_initial_concentrations(
    MUCF: float = 1.0,
) -> Dict[str, float]:
    """
    Compute initial receptor concentrations from $PK block.

    CD3 on Jurkat cells:
      50,000 cells / 100 µL = 500,000 cells/mL
      124 × 10³ CD3 molecules per T cell
      CD3 (nM) = (50000 * 10 * 124*1000 * 1000 / 6.022e23) * 1e9

    MUC16:
      500,000 cells/mL × MUCF × 1e-6

    Parameters
    ----------
    MUCF : float
        MUC16 scaling factor (estimated in NONMEM).
    """
    CD3 = (50000 * 10 * 124 * 1000 * 1000 / 6.022e23) * 1e9  # nM
    MUC16 = 500000 * (MUCF * 1e-6)  # nM

    return {'CD3': CD3, 'MUC16': MUC16}


# ────────────────────────────────────────────────────
#  Core equilibrium solver (equivalent to $AES block)
# ────────────────────────────────────────────────────

def bispecific_aes(A_equil: np.ndarray, params: dict) -> np.ndarray:
    """
    Algebraic equilibrium equations — direct translation of NONMEM $AES.

    A_equil = [CRA, CRB, CRAB]  (the three equilibrium compartments)

    Residuals E(4), E(5), E(6) must all equal 0.
    """
    CRA, CRB, CRAB = A_equil

    CT  = params['CT']
    RAT = params['RAT']
    RBT = params['RBT']
    KAC = params['KAC']
    KBC = params['KBC']
    ALP = params['ALP']

    # Free species (mass balance)
    CF  = CT  - CRA - CRB - CRAB     # free drug
    RAF = RAT - CRA - CRAB           # free effector (CD3)
    RBF = RBT - CRB - CRAB           # free target (MUC16)

    # Equilibrium residuals
    E4 = CF * RAF - KAC * CRA                  # Eq A34: drug–effector dimer
    E5 = CF * RBF - KBC * CRB                  # Eq A36: drug–target dimer
    E6 = CF * RAF * RBF - ALP * KAC * KBC * CRAB  # Eq A40: trimer

    return np.array([E4, E5, E6])


def solve_bispecific_equilibrium(
    CT: float, RAT: float, RBT: float,
    KAC: float, KBC: float, ALP: float = 1.0,
    initial_guess: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Solve the bispecific antibody binding equilibrium.

    Parameters
    ----------
    CT : float   Total drug concentration (nM)
    RAT : float  Total effector (CD3) concentration (nM)
    RBT : float  Total target (MUC16) concentration (nM)
    KAC : float  Dissociation constant for CD3 (nM)
    KBC : float  Dissociation constant for MUC16 (nM)
    ALP : float  Cooperativity factor (α). Default 1.0

    Returns
    -------
    Dict with: CF, RAF, RBF, CRA, CRB, CRAB and mass-balance checks
    """
    params = {'CT': CT, 'RAT': RAT, 'RBT': RBT,
              'KAC': KAC, 'KBC': KBC, 'ALP': ALP}

    if initial_guess is None:
        initial_guess = np.array([CT * 0.01, CT * 0.01, CT * 0.001])

    solution = fsolve(bispecific_aes, initial_guess, args=(params,))
    CRA, CRB, CRAB = np.maximum(solution, 0.0)

    CF  = CT  - CRA - CRB - CRAB
    RAF = RAT - CRA - CRAB
    RBF = RBT - CRB - CRAB

    return {
        'CF':   CF,
        'RAF':  RAF,
        'RBF':  RBF,
        'CRA':  CRA,   # drug–effector dimer
        'CRB':  CRB,   # drug–target dimer
        'CRAB': CRAB,   # trimer
        # Mass-balance checks
        'CT_check':  CF + CRA + CRB + CRAB,
        'RAT_check': RAF + CRA + CRAB,
        'RBT_check': RBF + CRB + CRAB,
    }


# ──────────────────────────────────────────────
#  $ERROR block: convert to observed signal
# ──────────────────────────────────────────────

def compute_fluorescence(CRAB: float, COVF: float, BASE: float) -> float:
    """
    Compute predicted fluorescence intensity from $ERROR block.

    IPRED = ABC * (COVF * 1E9) + BASE

    Parameters
    ----------
    CRAB : float  Ternary complex concentration (nM)
    COVF : float  Conversion factor to fluorescence
    BASE : float  Baseline fluorescence
    """
    return CRAB * (COVF * 1e9) + BASE


# ──────────────────────────────────────────────────
#  Dose–response simulation (equivalent to running
#  NONMEM across multiple drug concentrations)
# ──────────────────────────────────────────────────

def simulate_dose_response(
    CT_range: np.ndarray,
    RAT: float,
    RBT: float,
    KAC: float,
    KBC: float,
    ALP: float = 1.0,
) -> Dict[str, np.ndarray]:
    """
    Simulate bispecific antibody dose–response.

    Parameters
    ----------
    CT_range : array  Drug concentrations to evaluate (nM)
    RAT : float       Total effector (CD3) concentration (nM)
    RBT : float       Total target (MUC16) concentration (nM)
    KAC : float       KD for CD3 binding (nM)
    KBC : float       KD for MUC16 binding (nM)
    ALP : float       Cooperativity (default 1.0)

    Returns
    -------
    Dict of arrays for each species concentration
    """
    n = len(CT_range)
    result = {
        'CT': CT_range,
        'CF':   np.zeros(n),
        'RAF':  np.zeros(n),
        'RBF':  np.zeros(n),
        'CRA':  np.zeros(n),
        'CRB':  np.zeros(n),
        'CRAB': np.zeros(n),
    }

    prev_guess = np.array([1e-6, 1e-6, 1e-8])

    for i, ct in enumerate(CT_range):
        sol = solve_bispecific_equilibrium(ct, RAT, RBT, KAC, KBC, ALP,
                                          initial_guess=prev_guess)
        for key in ['CF', 'RAF', 'RBF', 'CRA', 'CRB', 'CRAB']:
            result[key][i] = sol[key]

        # Warm-start next solve
        prev_guess = np.array([sol['CRA'], sol['CRB'], sol['CRAB']])
        prev_guess = np.maximum(prev_guess, 1e-12)

    return result


def plot_bispecific_dose_response(result: Dict[str, np.ndarray],
                                  drug_name: str = "Bispecific Ab",
                                  KAC: float = None, KBC: float = None):
    """Plot dose–response curves mirroring NONMEM Figure A1."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # Free drug
    axes[0, 0].semilogx(result['CT'], result['CF'], 'b-', lw=2)
    axes[0, 0].set_xlabel('Total Drug (nM)')
    axes[0, 0].set_ylabel('Free Drug (nM)')
    axes[0, 0].set_title('Free Drug')
    axes[0, 0].grid(True, alpha=0.3)

    # Drug–CD3 dimer
    axes[0, 1].semilogx(result['CT'], result['CRA'], 'r-', lw=2)
    axes[0, 1].set_xlabel('Total Drug (nM)')
    axes[0, 1].set_ylabel('Drug–CD3 Dimer (nM)')
    axes[0, 1].set_title('Drug–Effector Dimer (CRA)')
    axes[0, 1].grid(True, alpha=0.3)

    # Drug–MUC16 dimer
    axes[1, 0].semilogx(result['CT'], result['CRB'], 'g-', lw=2)
    axes[1, 0].set_xlabel('Total Drug (nM)')
    axes[1, 0].set_ylabel('Drug–MUC16 Dimer (nM)')
    axes[1, 0].set_title('Drug–Target Dimer (CRB)')
    axes[1, 0].grid(True, alpha=0.3)

    # Trimer
    axes[1, 1].semilogx(result['CT'], result['CRAB'], 'm-', lw=2)
    axes[1, 1].set_xlabel('Total Drug (nM)')
    axes[1, 1].set_ylabel('Trimer (nM)')
    axes[1, 1].set_title('Drug–Effector–Target Trimer (CRAB)')
    axes[1, 1].grid(True, alpha=0.3)

    title = f'{drug_name} Dose–Response'
    if KAC is not None:
        title += f'  |  KD_CD3={KAC} nM'
    if KBC is not None:
        title += f',  KD_MUC16={KBC} nM'
    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.show()


# ──────────────────────────
#  Example / self-test
# ──────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Appendix 3: T-Cell-Redirecting Bispecific Antibody")
    print("Python equivalent of NONMEM ADVAN15 model")
    print("=" * 60)

    # ── Initial receptor concentrations (from $PK) ──
    init = compute_initial_concentrations(MUCF=0.5)
    RAT = init['CD3']
    RBT = init['MUC16']
    print(f"\nInitial concentrations:")
    print(f"  CD3 (RAT)   = {RAT:.4f} nM")
    print(f"  MUC16 (RBT) = {RBT:.4f} nM")

    # ── Solve for Drug 1: MUC16xCD3M ──
    KAC = 58.0   # nM  (CD3 KD)
    KBC = 0.1    # nM  (MUC16 KD, estimated)
    ALP = 1.0

    CT_range = np.logspace(-3, 3, 200)
    result = simulate_dose_response(CT_range, RAT, RBT, KAC, KBC, ALP)

    # Show a few values
    for ct_test in [0.1, 1.0, 10.0, 100.0]:
        idx = np.argmin(np.abs(CT_range - ct_test))
        print(f"\n  CT = {CT_range[idx]:.2f} nM:")
        print(f"    Free drug  = {result['CF'][idx]:.4f} nM")
        print(f"    CRA (dimer) = {result['CRA'][idx]:.4f} nM")
        print(f"    CRB (dimer) = {result['CRB'][idx]:.4f} nM")
        print(f"    CRAB (trimer) = {result['CRAB'][idx]:.6f} nM")

    print("\n  To plot: plot_bispecific_dose_response(result, 'MUC16xCD3M', KAC, KBC)")
