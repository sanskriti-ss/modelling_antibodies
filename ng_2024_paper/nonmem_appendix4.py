"""
Appendix 4: NONMEM Model for Multi-specific Multi-affinity Antibody — Python
=============================================================================

Python translation of the NONMEM ADVAN15 model from Appendix 4.

NONMEM structure:
  COMP 1  (CT)   — total drug concentration          (DADT = 0)
  COMP 2  (RT)   — total receptor concentration       (DADT = 0)
  COMP 3  (RC)   — drug with 1 receptor bound         (EQUIL)
  COMP 4  (R2C)  — drug with 2 receptors bound        (EQUIL)
  ...
  COMP 26 (R24C) — drug with 24 receptors bound       (EQUIL)

Handles two drug types:
  Drug 1: IgG           — only N=2 binding sites active
  Drug 2: Multimeric-24 — all N=24 binding sites active

Algebraic equations from $AES implement Eq B16:
  E(j): (N-j+1) * KD * RjC  -  (N-j+2) * R_{j-1}C * Rf  =  0
  (rearranged from the NONMEM code's form:
   j*KD*RjC - (N-j+1)*R_{j-1}C*Rf = 0)

Data source: Rujas et al.
"""

import numpy as np
from scipy.optimize import fsolve
from typing import Dict, Optional, List
import matplotlib.pyplot as plt


# ────────────────────────────────────────────────────
#  Core equilibrium solver (equivalent to $AES block)
# ────────────────────────────────────────────────────

def multivalent_aes_general(A_equil: np.ndarray, params: dict) -> np.ndarray:
    """
    Algebraic equilibrium equations for multivalent binding.
    Direct translation of NONMEM $AES block (Drug 2, multimeric case).

    A_equil = [RC, R2C, R3C, ..., R_N_C]   (N equilibrium compartments)

    Implements Eq B16 for each species:
      j*ALP_j*KD*RjC - (N-j+1)*R_{j-1}C*Rf = 0

    where R0C = CF (free drug) and Rf = free receptor.
    """
    CT  = params['CT']
    RT  = params['RT']
    KD  = params['KD']
    N   = params['N']       # number of active binding sites
    N_max = params['N_max'] # max compartments (always 24 in NONMEM model)
    ALP = params.get('ALP', 1.0)  # cooperativity (common for all sites)

    # Free drug: CT minus all bound species
    total_bound_drug = np.sum(A_equil[:N])  # each RjC has one drug molecule
    CF = CT - total_bound_drug

    # Free receptor: RT minus all bound receptors (j receptors per RjC)
    total_bound_receptor = sum((j + 1) * A_equil[j] for j in range(N))
    RF = RT - total_bound_receptor

    residuals = np.zeros(N_max)

    for j_idx in range(N):
        j = j_idx + 1  # species index (1-based, j=1 is RC, j=2 is R2C, ...)

        # R_{j-1}C concentration
        if j == 1:
            Rj_minus_1_C = CF  # R0C = free drug
        else:
            Rj_minus_1_C = A_equil[j_idx - 1]

        RjC = A_equil[j_idx]

        # From NONMEM $AES (Drug 2 branch):
        # E(j+2) = ((N - (j-1)) * ALP_j * KD * RjC  -  (N - (j-2)) * R_{j-1}C * RF) * WS1
        # Simplified with NONMEM indexing: factor_off = j, factor_on = (N - j + 1)
        # E = j * ALP * KD * RjC  -  (N - j + 1) * R_{j-1}C * RF
        residuals[j_idx] = j * ALP * KD * RjC - (N - j + 1) * Rj_minus_1_C * RF

    # For IgG (Drug 1): sites beyond N=2 are forced to 0
    for j_idx in range(N, N_max):
        residuals[j_idx] = A_equil[j_idx]  # RjC = 0 for j > N

    return residuals


def multivalent_aes_igg(A_equil: np.ndarray, params: dict) -> np.ndarray:
    """
    Algebraic equilibrium equations for IgG (Drug 1, N=2).
    Direct translation of the IF (DRUG==1) branch in NONMEM $AES.

    Only RC and R2C are active; all other species forced to 0.

    E(3): KD*RC  - 2*CF*RF  = 0          (N=2, j=1)
    E(4): 2*ALP*KD*R2C - RC*RF = 0       (N=2, j=2)
    E(5..26): RjC = 0
    """
    CT  = params['CT']
    RT  = params['RT']
    KD  = params['KD']
    ALP = params.get('ALP', 1.0)
    N_max = params['N_max']

    RC  = A_equil[0]
    R2C = A_equil[1]

    # Free drug and receptor (only RC and R2C contribute)
    CF = CT - RC - R2C
    RF = RT - RC - 2 * R2C

    residuals = np.zeros(N_max)

    # E(3): KD * RC - 2 * CF * RF = 0     (from NONMEM)
    residuals[0] = KD * RC - 2 * CF * RF

    # E(4): 2 * ALP * KD * R2C - RC * RF = 0  (from NONMEM)
    residuals[1] = 2 * ALP * KD * R2C - RC * RF

    # E(5..26): RjC = 0 for j >= 3
    for j_idx in range(2, N_max):
        residuals[j_idx] = A_equil[j_idx]

    return residuals


def solve_multivalent_equilibrium(
    CT: float, RT: float, KD: float,
    N: int = 24, drug_type: int = 2,
    ALP: float = 1.0,
    initial_guess: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Solve the multivalent binding equilibrium.

    Parameters
    ----------
    CT : float     Total drug concentration (nM)
    RT : float     Total receptor concentration (nM)
    KD : float     Dissociation constant (nM)
    N : int        Number of active binding sites (2 for IgG, 24 for multimeric)
    drug_type : int  1 = IgG, 2 = Multimeric
    ALP : float    Cooperativity factor (default 1.0)

    Returns
    -------
    Dict with CF, RF, and all RjC concentrations
    """
    N_max = 24
    params = {'CT': CT, 'RT': RT, 'KD': KD, 'N': N,
              'N_max': N_max, 'ALP': ALP}

    if initial_guess is None:
        initial_guess = np.full(N_max, 1e-8)
        if CT > 0:
            initial_guess[0] = min(CT, RT) * 0.01

    # Choose the right $AES branch
    if drug_type == 1:
        aes_func = multivalent_aes_igg
    else:
        aes_func = multivalent_aes_general

    solution = fsolve(aes_func, initial_guess, args=(params,))
    solution = np.maximum(solution, 0.0)

    # Compute derived quantities
    total_bound_drug = sum(solution[:N])
    CF = CT - total_bound_drug

    total_bound_receptor = sum((j + 1) * solution[j] for j in range(N))
    RF = RT - total_bound_receptor

    result = {
        'CF': CF,
        'RF': RF,
    }

    # Individual species
    for j in range(1, N_max + 1):
        result[f'R{j}C'] = solution[j - 1]

    # Total bound receptor (from $ERROR: TEMPRC)
    result['total_bound_receptor'] = total_bound_receptor

    # Total bound drug (from $ERROR: TEMPCC)
    result['total_bound_drug'] = total_bound_drug

    # Percent free receptor (from $ERROR: IPRED)
    if RT > 0:
        result['pct_free_receptor'] = (RT - total_bound_receptor) / RT * 100
    else:
        result['pct_free_receptor'] = 100.0

    return result


# ──────────────────────────────────────────────
#  Dose–response simulation
# ──────────────────────────────────────────────

def simulate_multivalent_dose_response(
    CT_range: np.ndarray,
    RT: float,
    KD: float,
    N: int = 24,
    drug_type: int = 2,
    ALP: float = 1.0,
) -> Dict[str, np.ndarray]:
    """
    Simulate dose–response for multivalent binding.
    Equivalent to running NONMEM across multiple CONC levels.
    """
    n_pts = len(CT_range)
    result = {
        'CT': CT_range,
        'CF': np.zeros(n_pts),
        'RF': np.zeros(n_pts),
        'pct_free_receptor': np.zeros(n_pts),
        'total_bound_receptor': np.zeros(n_pts),
        'total_bound_drug': np.zeros(n_pts),
    }

    # Store individual species
    for j in range(1, N + 1):
        result[f'R{j}C'] = np.zeros(n_pts)

    prev_guess = None

    for i, ct in enumerate(CT_range):
        sol = solve_multivalent_equilibrium(
            ct, RT, KD, N, drug_type, ALP,
            initial_guess=prev_guess
        )

        result['CF'][i] = sol['CF']
        result['RF'][i] = sol['RF']
        result['pct_free_receptor'][i] = sol['pct_free_receptor']
        result['total_bound_receptor'][i] = sol['total_bound_receptor']
        result['total_bound_drug'][i] = sol['total_bound_drug']

        for j in range(1, N + 1):
            result[f'R{j}C'][i] = sol[f'R{j}C']

        # Warm-start
        prev_guess = np.array([sol[f'R{j}C'] for j in range(1, 25)])
        prev_guess = np.maximum(prev_guess, 1e-15)

    return result


def plot_multivalent_dose_response(result: Dict[str, np.ndarray],
                                    N: int, drug_name: str = "Multivalent Ab"):
    """Plot dose–response curves mirroring NONMEM output."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # % Free receptor (this is IPRED in NONMEM $ERROR)
    axes[0, 0].semilogx(result['CT'], result['pct_free_receptor'], 'b-', lw=2)
    axes[0, 0].set_xlabel('Total Drug (nM)')
    axes[0, 0].set_ylabel('% Free Receptor')
    axes[0, 0].set_title('% Free Receptor (IPRED)')
    axes[0, 0].set_ylim(0, 105)
    axes[0, 0].grid(True, alpha=0.3)

    # Free drug and free receptor
    axes[0, 1].semilogx(result['CT'], result['CF'], 'k-', lw=2, label='Free Drug')
    axes[0, 1].semilogx(result['CT'], result['RF'], 'r--', lw=2, label='Free Receptor')
    axes[0, 1].set_xlabel('Total Drug (nM)')
    axes[0, 1].set_ylabel('Concentration (nM)')
    axes[0, 1].set_title('Free Species')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Individual species (show first few and last few)
    N_show = min(N, 6)
    colors = plt.cm.viridis(np.linspace(0, 1, N_show))
    for idx, j in enumerate(range(1, N_show + 1)):
        axes[1, 0].semilogx(result['CT'], result[f'R{j}C'],
                            color=colors[idx], lw=1.5, label=f'R{j}C')
    axes[1, 0].set_xlabel('Total Drug (nM)')
    axes[1, 0].set_ylabel('Complex Conc (nM)')
    axes[1, 0].set_title(f'Individual Species (first {N_show})')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    # Total bound receptor
    axes[1, 1].semilogx(result['CT'], result['total_bound_receptor'], 'g-', lw=2)
    axes[1, 1].set_xlabel('Total Drug (nM)')
    axes[1, 1].set_ylabel('Total Bound Receptor (nM)')
    axes[1, 1].set_title('Total Receptor Engagement')
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(f'{drug_name}  (N={N})', fontsize=14)
    plt.tight_layout()
    plt.show()


def compare_igg_vs_multimeric(
    CT_range: np.ndarray,
    RT: float,
    KD_igg: float,
    KD_multi: float,
    N_multi: int = 24,
    ALP: float = 1.0,
) -> None:
    """
    Compare IgG (N=2) vs multimeric (N=24) binding,
    reproducing the key comparison from the NONMEM analysis.
    """
    result_igg = simulate_multivalent_dose_response(
        CT_range, RT, KD_igg, N=2, drug_type=1, ALP=ALP)

    result_multi = simulate_multivalent_dose_response(
        CT_range, RT, KD_multi, N=N_multi, drug_type=2, ALP=ALP)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].semilogx(CT_range, result_igg['pct_free_receptor'],
                     'b-', lw=2, label=f'IgG (N=2, KD={KD_igg})')
    axes[0].semilogx(CT_range, result_multi['pct_free_receptor'],
                     'r-', lw=2, label=f'Multimeric (N={N_multi}, KD={KD_multi})')
    axes[0].set_xlabel('Total Drug (nM)')
    axes[0].set_ylabel('% Free Receptor')
    axes[0].set_title('Receptor Occupancy Comparison')
    axes[0].legend()
    axes[0].set_ylim(0, 105)
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogx(CT_range, result_igg['total_bound_receptor'],
                     'b-', lw=2, label='IgG (N=2)')
    axes[1].semilogx(CT_range, result_multi['total_bound_receptor'],
                     'r-', lw=2, label=f'Multimeric (N={N_multi})')
    axes[1].set_xlabel('Total Drug (nM)')
    axes[1].set_ylabel('Total Bound Receptor (nM)')
    axes[1].set_title('Total Receptor Engagement')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('IgG vs Multimeric Antibody Comparison', fontsize=14)
    plt.tight_layout()
    plt.show()


# ──────────────────────────
#  Example / self-test
# ──────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Appendix 4: Multi-specific Multi-affinity Antibody")
    print("Python equivalent of NONMEM ADVAN15 model")
    print("=" * 60)

    # ── Parameters from NONMEM $THETA ──
    RT = 0.417  # nM (FIXED in NONMEM)

    # Test IgG (Drug 1)
    KD_igg = 10.0   # nM (estimated)
    print(f"\n── IgG (Drug 1, N=2) ──")
    print(f"  RT = {RT} nM,  KD = {KD_igg} nM")

    CT_test = 1.0
    sol = solve_multivalent_equilibrium(CT_test, RT, KD_igg, N=2, drug_type=1)
    print(f"  CT = {CT_test} nM:")
    print(f"    CF = {sol['CF']:.6f} nM")
    print(f"    RF = {sol['RF']:.6f} nM")
    print(f"    R1C = {sol['R1C']:.6f} nM")
    print(f"    R2C = {sol['R2C']:.6f} nM")
    print(f"    % Free receptor = {sol['pct_free_receptor']:.2f}%")

    # Test Multimeric (Drug 2, N=24)
    KD_multi = 10.0   # nM (estimated)
    print(f"\n── Multimeric (Drug 2, N=24) ──")
    print(f"  RT = {RT} nM,  KD = {KD_multi} nM")

    sol24 = solve_multivalent_equilibrium(CT_test, RT, KD_multi, N=24, drug_type=2)
    print(f"  CT = {CT_test} nM:")
    print(f"    CF = {sol24['CF']:.6f} nM")
    print(f"    RF = {sol24['RF']:.6f} nM")
    print(f"    Total bound receptor = {sol24['total_bound_receptor']:.6f} nM")
    print(f"    % Free receptor = {sol24['pct_free_receptor']:.2f}%")
    # Show first few species
    for j in range(1, 7):
        print(f"    R{j}C = {sol24[f'R{j}C']:.8f} nM")
    print(f"    ...")

    # Dose–response
    print(f"\n  Generating dose–response...")
    CT_range = np.logspace(-3, 2, 150)

    print(f"  To compare IgG vs Multimeric:")
    print(f"    compare_igg_vs_multimeric(CT_range, RT, KD_igg, KD_multi)")
