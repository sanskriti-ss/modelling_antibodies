"""
Appendix 2: Multivalent Binding, with Common Binding Affinity
Python implementation of equations (B1) through (B16)

This module contains the mathematical framework for multivalent drug binding
where a single drug molecule can bind multiple receptor molecules (R).

Key assumptions:
- Drug has N possible binding sites
- Common binding affinity (same kon/koff for all sites)
- No cooperativity (occupancy of one site doesn't affect affinity of another)
- Quasi-equilibrium conditions apply

Variables:
- C: Free drug concentration
- Rf: Free receptor concentration  
- RjC: Drug with j receptors bound (j = 0, 1, 2, ..., N)
- R0C = C (drug with no receptors bound = free drug)
- RN+1C = 0 (no species beyond N receptors)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.optimize import fsolve
import matplotlib.pyplot as plt


class MultivalentBindingSystem:
    """
    Implements multivalent binding with common binding affinity.
    
    A single drug molecule can bind up to N receptor molecules.
    All binding sites have identical kinetic parameters.
    """
    
    def __init__(self, C_total: float, R_total: float, N: int):
        """
        Initialize the multivalent binding system.
        
        Parameters:
        - C_total: Total drug concentration
        - R_total: Total receptor concentration
        - N: Maximum number of receptors that can bind to one drug molecule
        """
        self.C_total = C_total
        self.R_total = R_total
        self.N = N
        
        # Validate inputs
        if N < 1:
            raise ValueError("N must be at least 1")
        if C_total < 0 or R_total < 0:
            raise ValueError("Concentrations must be non-negative")
    
    # ========== Differential Rate Equations (B1-B6) ==========
    
    def dR1C_dt_B1(self, C: float, Rf: float, R1C: float, R2C: float, 
                   N: int, kon: float, koff: float) -> float:
        """
        Equation B1: dR₁C/dt = (N*kon*C*Rf - koff*R₁C) - ((N-1)*kon*R₁C*Rf - 2*koff*R₂C) = 0
        
        Parameters:
        - C: Free drug concentration
        - Rf: Free receptor concentration
        - R1C: Concentration of drug with 1 receptor bound
        - R2C: Concentration of drug with 2 receptors bound
        - N: Maximum binding sites
        - kon: Association rate constant
        - koff: Dissociation rate constant
        """
        forward_term = N * kon * C * Rf - koff * R1C
        backward_term = (N - 1) * kon * R1C * Rf - 2 * koff * R2C
        return forward_term - backward_term
    
    def dR2C_dt_B2(self, R1C: float, Rf: float, R2C: float, R3C: float,
                   N: int, kon: float, koff: float) -> float:
        """
        Equation B2: dR₂C/dt = ((N-1)*kon*R₁C*Rf - 2*koff*R₂C) - ((N-2)*kon*R₂C*Rf - 3*koff*R₃C) = 0
        """
        forward_term = (N - 1) * kon * R1C * Rf - 2 * koff * R2C
        backward_term = (N - 2) * kon * R2C * Rf - 3 * koff * R3C
        return forward_term - backward_term
    
    def R0C_definition_B3(self, C: float) -> float:
        """
        Equation B3: R₀C = C
        Free drug is equivalent to drug with zero receptors bound
        """
        return C
    
    def RN_plus_1_C_constraint_B4(self) -> float:
        """
        Equation B4: R_{N+1}C = 0
        No species can exist with more than N receptors bound
        """
        return 0.0
    
    def dRjC_dt_general_B5(self, RjC: float, Rf: float, Rj_minus_1_C: float, Rj_plus_1_C: float,
                           j: int, N: int, kon: float, koff: float) -> float:
        """
        Equation B5: General equation for any species RⱼC
        dRⱼC/dt = ((N-j+1)*kon*R_{j-1}C*Rf - j*koff*RⱼC) - ((N-j)*kon*RⱼC*Rf - (j+1)*koff*R_{j+1}C) = 0
        
        For j=1 to N
        """
        if j < 1 or j > N:
            raise ValueError(f"j must be between 1 and {N}")
        
        forward_term = (N - j + 1) * kon * Rj_minus_1_C * Rf - j * koff * RjC
        backward_term = (N - j) * kon * RjC * Rf - (j + 1) * koff * Rj_plus_1_C
        return forward_term - backward_term
    
    def dRNC_dt_B6(self, RN_minus_1_C: float, Rf: float, RNC: float,
                   N: int, kon: float, koff: float) -> float:
        """
        Equation B6: dR_NC/dt = (kon*R_{N-1}C*Rf - N*koff*R_NC) = 0
        For j=N (maximum occupancy), there's no forward reaction beyond this
        """
        return kon * RN_minus_1_C * Rf - N * koff * RNC
    
    # ========== Normalized Equations (B7) ==========
    
    def normalized_dRjC_dt_B7(self, RjC: float, Rf: float, Rj_minus_1_C: float, Rj_plus_1_C: float,
                              j: int, N: int, KD: float) -> float:
        """
        Equation B7: Normalized by kon, using KD = koff/kon
        (1/kon) * dRⱼC/dt = ((N-j+1)*R_{j-1}C*Rf - j*KD*RⱼC) - ((N-j)*RⱼC*Rf - (j+1)*KD*R_{j+1}C) = 0
        """
        forward_term = (N - j + 1) * Rj_minus_1_C * Rf - j * KD * RjC
        backward_term = (N - j) * RjC * Rf - (j + 1) * KD * Rj_plus_1_C
        return forward_term - backward_term
    
    # ========== Transition Terms (B8-B10) ==========
    
    def Aonj_B8(self, j: int, N: int, Rj_minus_1_C: float, Rf: float) -> float:
        """
        Equation B8: Aonj = (N - j + 1) * R_{j-1}C * Rf
        Forward transition rate for binding jth receptor
        """
        return (N - j + 1) * Rj_minus_1_C * Rf
    
    def Aoffj_B9(self, j: int, KD: float, RjC: float) -> float:
        """
        Equation B9: Aoffj = j * KD * RⱼC
        Reverse transition rate for unbinding from jth receptor
        """
        return j * KD * RjC
    
    def Aj_net_B10(self, Aonj: float, Aoffj: float) -> float:
        """
        Equation B10: Aj = Aonj - Aoffj
        Net transition rate
        """
        return Aonj - Aoffj
    
    # ========== Equilibrium Conditions (B11-B16) ==========
    
    def equilibrium_condition_B11(self, Aj: float, Aj_plus_1: float) -> float:
        """
        Equation B11: (1/kon) * dRⱼC/dt = Aj - A_{j+1} = 0
        Returns equilibrium constraint error
        """
        return Aj - Aj_plus_1
    
    def microscopic_reversibility_B12(self, Aj: float, Aj_plus_1: float) -> bool:
        """
        Equation B12: Aj = A_{j+1} for all j, j=1 to N
        Returns True if microscopic reversibility is satisfied
        """
        return np.abs(Aj - Aj_plus_1) < 1e-12
    
    def boundary_condition_B13(self, N: int, RNC: float, Rf: float) -> float:
        """
        Equation B13: A_{on(N+1)} = (N - N) * R_NC * Rf = 0
        Boundary condition for maximum occupancy
        """
        return (N - N) * RNC * Rf  # This equals 0
    
    def boundary_condition_B14(self, N: int, KD: float, RN_plus_1_C: float) -> float:
        """
        Equation B14: A_{off(N+1)} = (N + 1) * KD * R_{N+1}C = 0
        Since R_{N+1}C = 0, this is automatically 0
        """
        return (N + 1) * KD * RN_plus_1_C  # This equals 0 since RN_plus_1_C = 0
    
    def all_transitions_zero_B15(self) -> float:
        """
        Equation B15: A_{N+1} = 0
        """
        return 0.0
    
    def equilibrium_relationship_B16(self, j: int, N: int, Rj_minus_1_C: float, 
                                   Rf: float, KD: float, RjC: float) -> float:
        """
        Equation B16: Aj = Aonj - Aoffj = (N - j + 1) * R_{j-1}C * Rf - j * KD * RⱼC = 0
        Final equilibrium relationship for species j
        """
        return (N - j + 1) * Rj_minus_1_C * Rf - j * KD * RjC
    
    # ========== Mass Conservation ==========
    
    def mass_conservation_drug(self, species_concentrations: List[float]) -> float:
        """
        Mass conservation for drug: C_total = C + R₁C + R₂C + ... + R_NC
        
        Parameters:
        - species_concentrations: [C, R1C, R2C, ..., RNC]
        
        Returns error (should be 0)
        """
        return self.C_total - sum(species_concentrations)
    
    def mass_conservation_receptor(self, Rf: float, species_concentrations: List[float]) -> float:
        """
        Mass conservation for receptor: R_total = Rf + 1*R₁C + 2*R₂C + ... + N*R_NC
        
        Parameters:
        - Rf: Free receptor concentration
        - species_concentrations: [C, R1C, R2C, ..., RNC]
        
        Returns error (should be 0)
        """
        # Skip C (index 0), count bound receptors
        bound_receptors = sum(j * species_concentrations[j] for j in range(1, len(species_concentrations)))
        return self.R_total - (Rf + bound_receptors)
    
    # ========== System Solver ==========
    
    def solve_equilibrium_concentrations(self, KD: float, 
                                       initial_guess: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Solve for equilibrium concentrations of all species.
        
        Parameters:
        - KD: Dissociation constant (koff/kon)
        - initial_guess: Optional initial guess for solver
        
        Returns:
        Dictionary with equilibrium concentrations
        """
        def equations(vars):
            # vars = [C, R1C, R2C, ..., RNC, Rf]
            species_conc = vars[:-1]  # All RjC species (including C=R0C)
            Rf = vars[-1]  # Free receptor
            C = species_conc[0]  # Free drug
            
            eq_list = []
            
            # Mass conservation constraints
            eq_list.append(self.mass_conservation_drug(species_conc))
            eq_list.append(self.mass_conservation_receptor(Rf, species_conc))
            
            # Equilibrium constraints for each species (B16)
            for j in range(1, self.N + 1):
                if j == 1:
                    Rj_minus_1_C = C  # R0C = C
                else:
                    Rj_minus_1_C = species_conc[j-1]  # R_{j-1}C
                
                RjC = species_conc[j]  # RjC
                
                eq_error = self.equilibrium_relationship_B16(j, self.N, Rj_minus_1_C, Rf, KD, RjC)
                eq_list.append(eq_error)
            
            return eq_list
        
        # Create initial guess if not provided
        if initial_guess is None:
            # Reasonable initial guess: distribute drug and receptors
            initial_guess = np.zeros(self.N + 2)  # [C, R1C, ..., RNC, Rf]
            initial_guess[0] = self.C_total / 2  # Start with half drug free
            
            # Distribute remaining drug among bound species
            remaining_drug = self.C_total / 2
            for j in range(1, self.N + 1):
                initial_guess[j] = remaining_drug / self.N
            
            # Initial free receptor guess
            initial_guess[-1] = self.R_total / 2
        
        try:
            solution = fsolve(equations, initial_guess)
            
            # Extract results
            species_conc = solution[:-1]
            Rf = solution[-1]
            
            result = {'Rf': Rf, 'C': species_conc[0]}
            
            # Add all RjC species
            for j in range(1, self.N + 1):
                result[f'R{j}C'] = species_conc[j]
            
            # Calculate mass balance checks
            result['C_total_check'] = sum(species_conc)
            result['R_total_check'] = Rf + sum(j * species_conc[j] for j in range(1, len(species_conc)))
            
            return result
            
        except Exception as e:
            raise ValueError(f"Failed to solve equilibrium system: {e}")
    
    def calculate_fractional_occupancy(self, concentrations: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate fractional occupancy of binding sites.
        
        Parameters:
        - concentrations: Output from solve_equilibrium_concentrations()
        
        Returns:
        Dictionary with fractional occupancies
        """
        # Total number of occupied sites
        total_occupied = 0
        total_drug_molecules = 0
        
        for j in range(1, self.N + 1):
            RjC_conc = concentrations[f'R{j}C']
            total_occupied += j * RjC_conc  # j sites per RjC molecule
            total_drug_molecules += RjC_conc
        
        # Add free drug molecules
        total_drug_molecules += concentrations['C']
        
        # Maximum possible occupied sites
        max_sites = self.N * self.C_total
        
        # Fractional occupancy
        if max_sites > 0:
            fractional_occupancy = total_occupied / max_sites
        else:
            fractional_occupancy = 0.0
        
        # Average occupancy per drug molecule
        if total_drug_molecules > 0:
            avg_occupancy_per_drug = total_occupied / total_drug_molecules
        else:
            avg_occupancy_per_drug = 0.0
        
        return {
            'fractional_occupancy': fractional_occupancy,
            'avg_occupancy_per_drug': avg_occupancy_per_drug,
            'total_occupied_sites': total_occupied,
            'max_possible_sites': max_sites
        }


# ========== Utility Functions ==========

def simulate_titration_curve(C_total_range: np.ndarray, R_total: float, N: int, 
                           KD: float) -> Dict[str, np.ndarray]:
    """
    Simulate titration curve for multivalent binding.
    
    Parameters:
    - C_total_range: Array of total drug concentrations
    - R_total: Total receptor concentration
    - N: Maximum binding sites per drug
    - KD: Dissociation constant
    
    Returns:
    Dictionary with concentration arrays for each species
    """
    results = {'C_total': C_total_range}
    
    # Initialize arrays for each species
    results['C'] = np.zeros_like(C_total_range)
    results['Rf'] = np.zeros_like(C_total_range)
    for j in range(1, N + 1):
        results[f'R{j}C'] = np.zeros_like(C_total_range)
    
    results['fractional_occupancy'] = np.zeros_like(C_total_range)
    results['avg_occupancy'] = np.zeros_like(C_total_range)
    
    for i, C_total in enumerate(C_total_range):
        try:
            system = MultivalentBindingSystem(C_total, R_total, N)
            concentrations = system.solve_equilibrium_concentrations(KD)
            occupancy = system.calculate_fractional_occupancy(concentrations)
            
            results['C'][i] = concentrations['C']
            results['Rf'][i] = concentrations['Rf']
            
            for j in range(1, N + 1):
                results[f'R{j}C'][i] = concentrations[f'R{j}C']
            
            results['fractional_occupancy'][i] = occupancy['fractional_occupancy']
            results['avg_occupancy'][i] = occupancy['avg_occupancy_per_drug']
            
        except:
            # If solver fails, use NaN
            results['C'][i] = np.nan
            results['Rf'][i] = np.nan
            for j in range(1, N + 1):
                results[f'R{j}C'][i] = np.nan
            results['fractional_occupancy'][i] = np.nan
            results['avg_occupancy'][i] = np.nan
    
    return results


def plot_multivalent_binding(results: Dict[str, np.ndarray], N: int, 
                            title: str = "Multivalent Binding"):
    """
    Plot multivalent binding results.
    
    Parameters:
    - results: Output from simulate_titration_curve()
    - N: Maximum binding sites
    - title: Plot title
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Species concentrations
    ax1.semilogx(results['C_total'], results['C'], 'k-', label='Free Drug (C)', linewidth=2)
    colors = plt.cm.viridis(np.linspace(0, 1, N))
    for j in range(1, N + 1):
        ax1.semilogx(results['C_total'], results[f'R{j}C'], 
                     color=colors[j-1], label=f'R{j}C', linewidth=2)
    
    ax1.set_xlabel('Total Drug Concentration')
    ax1.set_ylabel('Species Concentration')
    ax1.set_title('Drug-Receptor Complex Species')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Free receptor
    ax2.semilogx(results['C_total'], results['Rf'], 'b-', linewidth=2)
    ax2.set_xlabel('Total Drug Concentration')
    ax2.set_ylabel('Free Receptor Concentration')
    ax2.set_title('Free Receptor')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Fractional occupancy
    ax3.semilogx(results['C_total'], results['fractional_occupancy'], 'r-', linewidth=2)
    ax3.set_xlabel('Total Drug Concentration')
    ax3.set_ylabel('Fractional Occupancy')
    ax3.set_title('Overall Site Occupancy')
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Average occupancy per drug molecule
    ax4.semilogx(results['C_total'], results['avg_occupancy'], 'g-', linewidth=2)
    ax4.set_xlabel('Total Drug Concentration')
    ax4.set_ylabel('Average Receptors per Drug')
    ax4.set_title('Average Occupancy per Drug Molecule')
    ax4.set_ylim(0, N)
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def compare_valencies(C_total_range: np.ndarray, R_total: float, 
                     N_values: List[int], KD: float) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Compare binding behavior for different valencies (N values).
    
    Parameters:
    - C_total_range: Array of total drug concentrations
    - R_total: Total receptor concentration
    - N_values: List of valencies to compare
    - KD: Dissociation constant
    
    Returns:
    Dictionary keyed by N with simulation results
    """
    results = {}
    
    for N in N_values:
        results[N] = simulate_titration_curve(C_total_range, R_total, N, KD)
    
    return results


if __name__ == "__main__":
    # Example usage
    print("Multivalent Binding with Common Binding Affinity")
    print("=" * 50)
    
    # System parameters
    C_total = 1.0   # nM
    R_total = 100.0 # nM
    N = 4           # Maximum 4 receptors per drug
    KD = 10.0       # nM
    
    print(f"\nSystem parameters:")
    print(f"Total drug concentration: {C_total} nM")
    print(f"Total receptor concentration: {R_total} nM")
    print(f"Maximum binding sites (N): {N}")
    print(f"Dissociation constant (KD): {KD} nM")
    
    # Solve equilibrium
    system = MultivalentBindingSystem(C_total, R_total, N)
    
    try:
        concentrations = system.solve_equilibrium_concentrations(KD)
        occupancy = system.calculate_fractional_occupancy(concentrations)
        
        print(f"\nEquilibrium concentrations:")
        print(f"Free drug (C): {concentrations['C']:.4f} nM")
        print(f"Free receptor (Rf): {concentrations['Rf']:.4f} nM")
        
        for j in range(1, N + 1):
            print(f"R{j}C: {concentrations[f'R{j}C']:.4f} nM")
        
        print(f"\nMass balance checks:")
        print(f"Drug: {concentrations['C_total_check']:.4f} vs {C_total:.4f}")
        print(f"Receptor: {concentrations['R_total_check']:.4f} vs {R_total:.4f}")
        
        print(f"\nOccupancy analysis:")
        print(f"Fractional occupancy: {occupancy['fractional_occupancy']:.4f}")
        print(f"Average receptors per drug: {occupancy['avg_occupancy_per_drug']:.4f}")
        
    except Exception as e:
        print(f"Error solving equilibrium: {e}")
    
    # Generate titration curve
    print(f"\nGenerating titration curve...")
    C_range = np.logspace(-2, 2, 50)  # 0.01 to 100 nM
    titration_results = simulate_titration_curve(C_range, R_total, N, KD)
    
    print(f"To plot results, run: plot_multivalent_binding(titration_results, {N})")
    
    # Compare different valencies
    print(f"\nComparing different valencies...")
    N_values = [1, 2, 4, 8]
    comparison = compare_valencies(C_range, R_total, N_values, KD)
    
    print(f"Comparison generated for N = {N_values}")
    print(f"To plot comparison:")
    print(f"for N in {N_values}:")
    print(f"    plot_multivalent_binding(comparison[N], N, f'N={N} Multivalent Binding')")