"""
Appendix 1: Equilibrium Assessment for Bispecific Antibodies
Python implementation of equations (A1) through (A42) and beyond

This module contains the mathematical framework for microscopic reversibility
and equilibrium assessment in bispecific antibody drug-receptor binding.

Variables:
- CT: Total drug concentration
- RT: Total R_T receptor concentration  
- RE: Total R_E receptor concentration
- Cf: Free drug concentration
- RTf: Free R_T receptor concentration
- REf: Free R_E receptor concentration
- RECf: R_E-drug complex concentration
- RTCf: R_T-drug complex concentration  
- RETCf: R_E-R_T-drug ternary complex concentration
"""

import numpy as np
from typing import Dict, Tuple, Optional
from scipy.optimize import fsolve
import matplotlib.pyplot as plt


class BispecificAntibodyEquilibrium:
    """
    Implements the equilibrium equations for bispecific antibody binding.
    Based on assumptions of microscopic reversibility.
    
    This represents a system where:
    - Drug C can bind to receptor R_E forming R_E*C
    - Drug C can bind to receptor R_T forming R_T*C  
    - R_E*C can bind to R_T forming ternary complex R_E*T*C
    - R_T*C can bind to R_E forming ternary complex R_E*T*C
    """
    
    def __init__(self, CT: float, RT: float, RE: float):
        """
        Initialize the system with total concentrations.
        
        Parameters:
        - CT: Total drug concentration (constant)
        - RT: Total R_T receptor concentration (constant)
        - RE: Total R_E receptor concentration (constant)
        """
        self.CT = CT
        self.RT = RT
        self.RE = RE
    
    # ========== Mass Conservation Equations (A1-A3) ==========
    
    def mass_conservation_drug_A1(self, Cf: float, RECf: float, RTCf: float, RETCf: float) -> float:
        """
        Equation A1: CT = Cf + RECf + RTCf + RETCf
        Returns the mass balance error (should be 0)
        """
        return self.CT - (Cf + RECf + RTCf + RETCf)
    
    def mass_conservation_RT_A2(self, RTf: float, RTCf: float, RETCf: float) -> float:
        """
        Equation A2: RT = RTf + RTCf + RETCf  
        Returns the mass balance error (should be 0)
        """
        return self.RT - (RTf + RTCf + RETCf)
    
    def mass_conservation_RE_A3(self, REf: float, RECf: float, RETCf: float) -> float:
        """
        Equation A3: RE = REf + RECf + RETCf
        Returns the mass balance error (should be 0)
        """
        return self.RE - (REf + RECf + RETCf)
    
    # ========== Mass Balance Rate Equations (A4-A6) ==========
    
    def dRECf_dt_A4(self, Cf: float, REf: float, RECf: float, RTf: float, RETCf: float,
                    konE: float, koffE: float, konET: float, koffET: float) -> float:
        """
        Equation A4: dRECf/dt = (konE*Cf*REf - koffE*RECf) - (konET*RECf*RTf - koffET*RETCf)
        """
        term1 = konE * Cf * REf - koffE * RECf
        term2 = konET * RECf * RTf - koffET * RETCf
        return term1 - term2
    
    def dRTCf_dt_A5(self, Cf: float, RTf: float, RTCf: float, REf: float, RETCf: float,
                    konT: float, koffT: float, konTE: float, koffTE: float) -> float:
        """
        Equation A5: dRTCf/dt = (konT*Cf*RTf - koffT*RTCf) - (konTE*RTCf*REf - koffTE*RETCf)
        """
        term1 = konT * Cf * RTf - koffT * RTCf
        term2 = konTE * RTCf * REf - koffTE * RETCf
        return term1 - term2
    
    def dRETCf_dt_A6(self, RECf: float, RTf: float, RTCf: float, REf: float, RETCf: float,
                     konET: float, koffET: float, konTE: float, koffTE: float) -> float:
        """
        Equation A6: dRETCf/dt = (konET*RECf*RTf - koffET*RETCf) + (konTE*RTCf*REf - koffTE*RETCf)
        """
        term1 = konET * RECf * RTf - koffET * RETCf
        term2 = konTE * RTCf * REf - koffTE * RETCf
        return term1 + term2
    
    # ========== Individual Transition Terms (A7-A14) ==========
    
    def AonE_A7(self, konE: float, Cf: float, REf: float) -> float:
        """Equation A7: AonE = konE * Cf * REf"""
        return konE * Cf * REf
    
    def AoffE_A8(self, koffE: float, RECf: float) -> float:
        """Equation A8: AoffE = koffE * RECf"""
        return koffE * RECf
    
    def AonT_A9(self, konT: float, Cf: float, RTf: float) -> float:
        """Equation A9: AonT = konT * Cf * RTf"""
        return konT * Cf * RTf
    
    def AoffT_A10(self, koffT: float, RTCf: float) -> float:
        """Equation A10: AoffT = koffT * RTCf"""
        return koffT * RTCf
    
    def AonET_A11(self, konET: float, RECf: float, RTf: float) -> float:
        """Equation A11: AonET = konET * RECf * RTf"""
        return konET * RECf * RTf
    
    def AoffET_A12(self, koffET: float, RETCf: float) -> float:
        """Equation A12: AoffET = koffET * RETCf"""
        return koffET * RETCf
    
    def AonTE_A13(self, konTE: float, RTCf: float, REf: float) -> float:
        """Equation A13: AonTE = konTE * RTCf * REf"""
        return konTE * RTCf * REf
    
    def AoffTE_A14(self, koffTE: float, RETCf: float) -> float:
        """Equation A14: AoffTE = koffTE * RETCf"""
        return koffTE * RETCf
    
    # ========== Forward/Backward Transaction Pairs (A15-A18) ==========
    
    def AE_A15(self, AonE: float, AoffE: float) -> float:
        """Equation A15: AE = AonE - AoffE"""
        return AonE - AoffE
    
    def AT_A16(self, AonT: float, AoffT: float) -> float:
        """Equation A16: AT = AonT - AoffT"""
        return AonT - AoffT
    
    def AET_A17(self, AonET: float, AoffET: float) -> float:
        """Equation A17: AET = AonET - AoffET"""
        return AonET - AoffET
    
    def ATE_A18(self, AonTE: float, AoffTE: float) -> float:
        """Equation A18: ATE = AonTE - AoffTE"""
        return AonTE - AoffTE
    
    # ========== Rate Equations as Net Transitions (A19-A21) ==========
    
    def dRECf_dt_A19(self, AE: float, AET: float) -> float:
        """Equation A19: dRECf/dt = AE - AET"""
        return AE - AET
    
    def dRTCf_dt_A20(self, AT: float, ATE: float) -> float:
        """Equation A20: dRTCf/dt = AT - ATE"""
        return AT - ATE
    
    def dRETCf_dt_A21(self, AET: float, ATE: float) -> float:
        """Equation A21: dRETCf/dt = AET + ATE"""
        return AET + ATE
    
    # ========== Microscopic Reversibility Conditions (A22-A25) ==========
    
    def microscopic_reversibility_A22(self, Ax: float) -> bool:
        """
        Equation A22: Ax = 0 for all individual transition expressions
        Returns True if microscopic reversibility is satisfied
        """
        return np.abs(Ax) < 1e-12
    
    def equilibrium_dRECf_A23(self, AE: float, AET: float) -> float:
        """Equation A23: dRECf/dt = AE - AET = 0"""
        return AE - AET
    
    def equilibrium_dRTCf_A24(self, AT: float, ATE: float) -> float:
        """Equation A24: dRTCf/dt = AT - ATE = 0"""
        return AT - ATE
    
    def equilibrium_dRETCf_A25(self, AET: float, ATE: float) -> float:
        """Equation A25: dRETCf/dt = AET + ATE = 0"""
        return AET + ATE
    
    # ========== Rate Constant Constraints (A26-A28) ==========
    
    def rate_constant_constraint_A26(self, AonX: float, AoffX: float) -> float:
        """
        Equation A26: AonX = AoffX for all X
        Returns the constraint error (should be 0)
        """
        return AonX - AoffX
    
    def composite_rate_constraint_A27_error(self, AonE: float, AonET: float, AoffTE: float, koffT: float,
                                           AonT: float, AonTE: float, AoffET: float, AoffE: float) -> float:
        """
        Equation A27: AonE*AonET*AoffTE*koffT = AonT*AonTE*AoffET*AoffE
        Returns the constraint error (should be 0)
        """
        left_side = AonE * AonET * AoffTE * koffT
        right_side = AonT * AonTE * AoffET * AoffE
        return left_side - right_side
    
    def rate_constant_relationship_A28_error(self, konE: float, konET: float, koffTE: float, koffT: float,
                                            konT: float, konTE: float, koffET: float, koffE: float) -> float:
        """
        Equation A28: konE*konET*koffTE*koffT = konT*konTE*koffET*koffE
        Returns the constraint error (should be 0)
        """
        left_side = konE * konET * koffTE * koffT
        right_side = konT * konTE * koffET * koffE
        return left_side - right_side
    
    # ========== Dissociation Constants (A29) ==========
    
    def dissociation_constant_A29(self, koff: float, kon: float) -> float:
        """
        Equation A29: Ki = koffi / koni
        """
        return koff / kon
    
    # ========== Rate Constant Relationships (A30) ==========
    
    def rate_relationship_A30_error(self, KT: float, KTE: float, KE: float, KET: float) -> float:
        """
        Equation A30: KT * KTE = KE * KET
        Returns the constraint error (should be 0)
        """
        return KT * KTE - KE * KET
    
    # ========== Cooperativity Definition (A31-A33) ==========
    
    def cooperativity_alpha_A31(self, KTE: float, KE: float, KET: float, KT: float) -> Tuple[float, float]:
        """
        Equation A31: α = KTE/KE = KET/KT
        Returns both expressions for alpha (should be equal)
        """
        alpha1 = KTE / KE
        alpha2 = KET / KT
        return alpha1, alpha2
    
    def KET_from_cooperativity_A32(self, alpha: float, KT: float) -> float:
        """Equation A32: KET = α * KT"""
        return alpha * KT
    
    def KTE_from_cooperativity_A33(self, alpha: float, KE: float) -> float:
        """Equation A33: KTE = α * KE"""
        return alpha * KE
    
    # ========== Simplified Equilibrium Equations (A34-A40) ==========
    
    def equilibrium_constraint_A34(self, KE: float, RECf: float, Cf: float, REf: float) -> float:
        """
        Equation A34: KE * RECf - Cf * REf = 0
        Returns the constraint error (should be 0)
        """
        return KE * RECf - Cf * REf
    
    def equilibrium_RECf_A35(self, Cf: float, REf: float, KE: float) -> float:
        """
        Equation A35: RECf = Cf * REf / KE
        """
        return (Cf * REf) / KE
    
    def equilibrium_constraint_A36(self, KT: float, RTCf: float, Cf: float, RTf: float) -> float:
        """
        Equation A36: KT * RTCf - Cf * RTf = 0
        Returns the constraint error (should be 0)
        """
        return KT * RTCf - Cf * RTf
    
    def equilibrium_RTCf_A37(self, Cf: float, RTf: float, KT: float) -> float:
        """
        Equation A37: RTCf = Cf * RTf / KT
        """
        return (Cf * RTf) / KT
    
    def equilibrium_constraint_A38(self, RECf: float, RTf: float, KET: float, RETCf: float) -> float:
        """
        Equation A38: RECf * RTf - KET * RETCf = 0
        Returns the constraint error (should be 0)
        """
        return RECf * RTf - KET * RETCf
    
    def equilibrium_RETCf_A39(self, RECf: float, RTf: float, KET: float) -> float:
        """
        Equation A39: RETCf = RECf * RTf / KET = RECf * RTf / (α * KT) = Cf * REf * RTf / (α * KE * KT)
        """
        return (RECf * RTf) / KET
    
    def equilibrium_constraint_A40(self, alpha: float, KE: float, KT: float, RETCf: float, 
                                   Cf: float, REf: float, RTf: float) -> float:
        """
        Equation A40: α * KE * KT * RETCf - Cf * REf * RTf = 0
        Returns the constraint error (should be 0)
        """
        return alpha * KE * KT * RETCf - Cf * REf * RTf
    
    # ========== Special Case: No Cooperativity (A41-A42) ==========
    
    def no_cooperativity_A41(self, KTE: float, KE: float) -> bool:
        """
        Equation A41: KTE = KE (no cooperativity condition)
        Returns True if condition is satisfied within tolerance
        """
        return np.isclose(KTE, KE)
    
    def no_cooperativity_A42(self, KET: float, KT: float) -> bool:
        """
        Equation A42: KET = KT (no cooperativity condition)
        Returns True if condition is satisfied within tolerance
        """
        return np.isclose(KET, KT)
    
    def check_no_cooperativity(self, KTE: float, KE: float, KET: float, KT: float) -> bool:
        """
        Check if both no-cooperativity conditions are satisfied (α = 1)
        """
        return self.no_cooperativity_A41(KTE, KE) and self.no_cooperativity_A42(KET, KT)
    
    # ========== System Solver Methods ==========
    
    def solve_equilibrium_concentrations(self, KE: float, KT: float, KET: float, 
                                       initial_guess: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Solve for equilibrium concentrations given dissociation constants.
        
        Parameters:
        - KE: Dissociation constant for R_E + C ⇌ R_E*C
        - KT: Dissociation constant for R_T + C ⇌ R_T*C
        - KET: Dissociation constant for R_E*C + R_T ⇌ R_E*T*C
        
        Returns:
        Dictionary with equilibrium concentrations
        """
        def equations(vars):
            Cf, REf, RTf, RECf, RTCf, RETCf = vars
            
            # Mass conservation constraints
            eq1 = self.mass_conservation_drug_A1(Cf, RECf, RTCf, RETCf)
            eq2 = self.mass_conservation_RT_A2(RTf, RTCf, RETCf)
            eq3 = self.mass_conservation_RE_A3(REf, RECf, RETCf)
            
            # Equilibrium constraints
            eq4 = self.equilibrium_constraint_A34(KE, RECf, Cf, REf)
            eq5 = self.equilibrium_constraint_A36(KT, RTCf, Cf, RTf)
            eq6 = self.equilibrium_constraint_A38(RECf, RTf, KET, RETCf)
            
            return [eq1, eq2, eq3, eq4, eq5, eq6]
        
        if initial_guess is None:
            # Reasonable initial guess
            initial_guess = np.array([self.CT/2, self.RE/2, self.RT/2, self.CT/4, self.CT/4, self.CT/8])
        
        solution = fsolve(equations, initial_guess)
        Cf, REf, RTf, RECf, RTCf, RETCf = solution
        
        return {
            'Cf': Cf,
            'REf': REf, 
            'RTf': RTf,
            'RECf': RECf,
            'RTCf': RTCf,
            'RETCf': RETCf,
            'CT_check': Cf + RECf + RTCf + RETCf,
            'RT_check': RTf + RTCf + RETCf,
            'RE_check': REf + RECf + RETCf
        }
    
    def calculate_cooperativity(self, KE: float, KT: float, KET: float, KTE: float) -> Dict[str, float]:
        """
        Calculate cooperativity parameters and check consistency.
        
        Returns:
        Dictionary with cooperativity analysis
        """
        alpha1, alpha2 = self.cooperativity_alpha_A31(KTE, KE, KET, KT)
        
        # Check if the two expressions for alpha are consistent
        alpha_consistent = np.isclose(alpha1, alpha2)
        
        # Check rate constant relationship A30
        relationship_error = self.rate_relationship_A30_error(KT, KTE, KE, KET)
        
        return {
            'alpha_from_KTE_KE': alpha1,
            'alpha_from_KET_KT': alpha2,
            'alpha_consistent': alpha_consistent,
            'alpha_average': (alpha1 + alpha2) / 2,
            'rate_relationship_A30_error': relationship_error,
            'no_cooperativity': alpha1 == 1.0 and alpha2 == 1.0
        }


# ========== Utility Functions ==========

def create_parameter_table_A1() -> Dict[str, Dict[str, float]]:
    """
    Create parameter table from Table A1 in the document.
    
    Returns:
    Dictionary with original and DAE-QEMB model parameters
    """
    return {
        'Original_Model': {
            'kon_CD3': 2.85e-4,  # nM^-1 s^-1
            'koff_CD3': 2.06e-3,  # s^-1
            'KD_CD3': 7.23,      # nM
            'kon_BCMA': 6.65e-4, # nM^-1 s^-1
            'koff_BCMA': 1.59e-4, # s^-1
            'KD_BCMA': 0.239     # nM
        },
        'DAE_QEMB_Model': {
            'kon_CD3': None,     # Not estimated
            'koff_CD3': None,    # Not estimated
            'KD_CD3': 7.23,      # nM (identical)
            'kon_BCMA': None,    # Not estimated
            'koff_BCMA': None,   # Not estimated
            'KD_BCMA': 0.239     # nM (identical)
        }
    }


def simulate_dose_response(CT_range: np.ndarray, RT: float = 100.0, RE: float = 100.0,
                          KE: float = 7.23, KT: float = 0.239, alpha: float = 1.0) -> Dict[str, np.ndarray]:
    """
    Simulate dose-response relationship for bispecific antibody binding.
    
    Parameters:
    - CT_range: Array of total drug concentrations
    - RT: Total R_T concentration (default 100 nM from study)
    - RE: Total R_E concentration (default 100 nM from study)
    - KE: Dissociation constant for R_E binding (default 7.23 nM)
    - KT: Dissociation constant for R_T binding (default 0.239 nM)
    - alpha: Cooperativity factor (default 1.0)
    
    Returns:
    Dictionary with concentration arrays for each species
    """
    KET = alpha * KT  # From equation A32
    
    results = {
        'CT': CT_range,
        'Cf': np.zeros_like(CT_range),
        'RECf': np.zeros_like(CT_range),
        'RTCf': np.zeros_like(CT_range),
        'RETCf': np.zeros_like(CT_range),
        'REf': np.zeros_like(CT_range),
        'RTf': np.zeros_like(CT_range)
    }
    
    for i, CT_val in enumerate(CT_range):
        eq = BispecificAntibodyEquilibrium(CT_val, RT, RE)
        try:
            sol = eq.solve_equilibrium_concentrations(KE, KT, KET)
            results['Cf'][i] = sol['Cf']
            results['RECf'][i] = sol['RECf']
            results['RTCf'][i] = sol['RTCf']
            results['RETCf'][i] = sol['RETCf']
            results['REf'][i] = sol['REf']
            results['RTf'][i] = sol['RTf']
        except:
            # If solver fails, use NaN
            for key in ['Cf', 'RECf', 'RTCf', 'RETCf', 'REf', 'RTf']:
                results[key][i] = np.nan
    
    return results


def plot_dose_response(results: Dict[str, np.ndarray], title: str = "Bispecific Antibody Dose Response"):
    """
    Plot dose-response curves for all species.
    
    Parameters:
    - results: Output from simulate_dose_response()
    - title: Plot title
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot complexes
    ax1.semilogx(results['CT'], results['RECf'], 'b-', label='R_E*C')
    ax1.semilogx(results['CT'], results['RTCf'], 'r-', label='R_T*C')
    ax1.semilogx(results['CT'], results['RETCf'], 'g-', label='R_E*T*C (Ternary)')
    ax1.set_xlabel('Total Drug Concentration (nM)')
    ax1.set_ylabel('Complex Concentration (nM)')
    ax1.set_title('Drug-Receptor Complexes')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot free species
    ax2.semilogx(results['CT'], results['Cf'], 'k-', label='Free Drug')
    ax2.semilogx(results['CT'], results['REf'], 'b--', label='Free R_E')
    ax2.semilogx(results['CT'], results['RTf'], 'r--', label='Free R_T')
    ax2.set_xlabel('Total Drug Concentration (nM)')
    ax2.set_ylabel('Free Concentration (nM)')
    ax2.set_title('Free Species')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_figure_A1(results: Dict[str, np.ndarray],
                   simulated_data: Optional[Dict[str, np.ndarray]] = None,
                   savepath: Optional[str] = None):
    """
    Reproduce Figure A1 from the paper.

    2×2 log-log panel plot:
      Top-left:     Free Drug
      Top-right:    Drug-CD3 Dimer
      Bottom-left:  Drug-BCMA Dimer
      Bottom-right: Trimer

    Parameters
    ----------
    results : dict
        Output from simulate_dose_response() — model predictions (line).
    simulated_data : dict, optional
        Same structure as results but for "observed" data points (dots).
        If None, results is plotted as both dots and line.
    savepath : str, optional
        If given, save the figure to this path instead of showing.
    """
    # Panel configuration: (key in results dict, panel title)
    panels = [
        ('Cf',    'Free Drug'),
        ('RECf',  'Drug-CD3 Dimer'),
        ('RTCf',  'Drug-BCMA Dimer'),
        ('RETCf', 'Trimer'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(9, 7.5),
                              sharex=True, sharey=False)

    CT = results['CT']

    for ax, (key, panel_title) in zip(axes.flat, panels):
        y_pred = results[key]

        # ── Model prediction (solid line) ──
        ax.loglog(CT, y_pred, '-', color='black', linewidth=1.2, zorder=2)

        # ── Simulated / observed data (filled circles) ──
        if simulated_data is not None:
            y_obs = simulated_data[key]
            ct_obs = simulated_data['CT']
        else:
            # Use a subset of the prediction as "observed" dots (every ~3rd point)
            step = max(1, len(CT) // 25)
            ct_obs = CT[::step]
            y_obs = y_pred[::step]

        ax.loglog(ct_obs, y_obs, 'o', color='black', markersize=5,
                  markerfacecolor='black', markeredgecolor='black',
                  zorder=3)

        # ── Gray header strip (like ggplot facet label) ──
        ax.set_title(panel_title, fontsize=11, fontweight='normal',
                     bbox=dict(facecolor='#d9d9d9', edgecolor='#999999',
                               boxstyle='square,pad=0.4'),
                     pad=8)

        # ── Axis formatting ──
        ax.grid(False)
        ax.tick_params(which='both', direction='in', top=True, right=True)

        # Thin border
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
            spine.set_color('#333333')

    # ── Shared axis labels ──
    fig.text(0.5, 0.02, 'Concentrations (nM)', ha='center', fontsize=11)
    fig.text(0.02, 0.5, 'Concentrations (nM)', va='center',
             rotation='vertical', fontsize=11)

    # ── Title ──
    fig.suptitle(
        'Figure A1. Simulated vs model predicted free drug, drug-CD3 dimer,\n'
        'drug-BCMA and trimer concentrations',
        fontsize=11, y=0.99, va='top')

    fig.subplots_adjust(left=0.1, right=0.97, top=0.88, bottom=0.1,
                        hspace=0.35, wspace=0.35)

    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches='tight')
        print(f"Figure saved to {savepath}")
    else:
        plt.show()


if __name__ == "__main__":
    # Example usage matching the document parameters
    print("Bispecific Antibody Equilibrium Assessment")
    print("=" * 50)
    
    # Parameters from Table A1
    params = create_parameter_table_A1()
    KE = params['Original_Model']['KD_CD3']   # 7.23 nM
    KT = params['Original_Model']['KD_BCMA']  # 0.239 nM
    
    print(f"\nUsing parameters from Table A1:")
    print(f"K_D(CD3) = K_E = {KE} nM")
    print(f"K_D(BCMA) = K_T = {KT} nM")
    
    # Initialize system with concentrations from simulation study
    CT = 10.0    # Example drug concentration
    RT = 100.0   # Total CD3 receptor (100 nM from study)
    RE = 100.0   # Total BCMA receptor (100 nM from study)
    
    eq = BispecificAntibodyEquilibrium(CT, RT, RE)
    
    # Assume no cooperativity initially (α = 1)
    alpha = 1.0
    KET = alpha * KT  # From equation A32
    KTE = alpha * KE  # From equation A33
    
    print(f"\nAssuming no cooperativity (α = 1):")
    print(f"K_ET = {KET} nM")
    print(f"K_TE = {KTE} nM")
    
    # Solve equilibrium
    try:
        solution = eq.solve_equilibrium_concentrations(KE, KT, KET)
        print(f"\nEquilibrium concentrations at CT = {CT} nM:")
        print(f"Free drug (Cf) = {solution['Cf']:.4f} nM")
        print(f"R_E*C complex = {solution['RECf']:.4f} nM")
        print(f"R_T*C complex = {solution['RTCf']:.4f} nM") 
        print(f"R_E*T*C ternary complex = {solution['RETCf']:.4f} nM")
        print(f"Free R_E = {solution['REf']:.4f} nM")
        print(f"Free R_T = {solution['RTf']:.4f} nM")
        
        # Check mass balance
        print(f"\nMass balance checks:")
        print(f"Drug: {solution['CT_check']:.4f} vs {CT:.4f} (should match)")
        print(f"R_T: {solution['RT_check']:.4f} vs {RT:.4f} (should match)")
        print(f"R_E: {solution['RE_check']:.4f} vs {RE:.4f} (should match)")
        
    except Exception as e:
        print(f"Error solving equilibrium: {e}")
    
    # Analyze cooperativity
    coop_analysis = eq.calculate_cooperativity(KE, KT, KET, KTE)
    print(f"\nCooperativity analysis:")
    print(f"α from K_TE/K_E = {coop_analysis['alpha_from_KTE_KE']:.4f}")
    print(f"α from K_ET/K_T = {coop_analysis['alpha_from_KET_KT']:.4f}")
    print(f"Consistent α values: {coop_analysis['alpha_consistent']}")
    print(f"No cooperativity: {coop_analysis['no_cooperativity']}")
    
    # ── Dose-response simulation matching Figure A1 ──
    # Figure A1 x-axis spans ~1e-4 to ~1e4 nM
    print(f"\nGenerating dose-response curve (Figure A1 range)...")
    CT_range = np.logspace(-4, 4, 200)
    dose_response = simulate_dose_response(CT_range, RT, RE, KE, KT, alpha)

    # Show results at a few concentrations
    for ct_test in [0.001, 0.1, 10.0, 1000.0]:
        idx = np.argmin(np.abs(CT_range - ct_test))
        ct = dose_response['CT'][idx]
        print(f"  CT = {ct:.4f} nM  →  Free={dose_response['Cf'][idx]:.2e}  "
              f"CD3-dimer={dose_response['RECf'][idx]:.2e}  "
              f"BCMA-dimer={dose_response['RTCf'][idx]:.2e}  "
              f"Trimer={dose_response['RETCf'][idx]:.2e}")

    # ── Generate Figure A1 ──
    print(f"\nPlotting Figure A1...")
    savepath = "/Users/sanskriti/Documents/GitHub/modelling_antibodies/ng_2024_paper/figure_A1.png"
    plot_figure_A1(dose_response, savepath=savepath)