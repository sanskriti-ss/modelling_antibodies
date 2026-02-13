"""
NONMEM ADVAN15 DAE Solver — Python Implementation
===================================================

This module provides a Python equivalent of NONMEM's ADVAN15 subroutine,
which solves mixed Differential-Algebraic Equation (DAE) systems.

In NONMEM ADVAN15:
  - $DES / DADT()    → differential states (ODEs)
  - $AES / E()       → algebraic equilibrium states (solved via root-finding)
  - $AESINIT / INIT  → initial values for algebraic states

This Python implementation uses:
  - scipy.integrate.solve_ivp  → for the ODE integration
  - scipy.optimize.fsolve      → for root-finding of algebraic equations at each step

This mirrors the approach described in the paper: coding equilibrium equations
(=0) directly and having a DAE solver find the algebraic state variables.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from typing import Dict, List, Callable, Optional, Tuple
import warnings


class DAESolver:
    """
    A Python equivalent of NONMEM's ADVAN15 DAE solver.

    Solves a mixed system of:
      - Differential equations:  dA/dt = f(t, A_diff, A_equil)
      - Algebraic equations:     0     = g(t, A_diff, A_equil)

    At each ODE evaluation step, the algebraic states are solved to
    satisfy E(i) = 0 via root-finding, exactly as NONMEM does.
    """

    def __init__(self, n_diff: int, n_equil: int, name: str = "DAE Model"):
        """
        Parameters
        ----------
        n_diff : int
            Number of differential (ODE) compartments.
        n_equil : int
            Number of algebraic (equilibrium) compartments.
        name : str
            Descriptive name for the model.
        """
        self.n_diff = n_diff
        self.n_equil = n_equil
        self.n_total = n_diff + n_equil
        self.name = name

        # User must override these
        self._des_func = None    # dA/dt for differential states
        self._aes_func = None    # E() residuals for algebraic states
        self._error_func = None  # $ERROR block: compute predictions

    def set_des(self, func: Callable):
        """
        Set the differential equation function (equivalent to $DES / DADT).

        func(t, A_diff, A_equil, params) -> np.ndarray of length n_diff
        """
        self._des_func = func

    def set_aes(self, func: Callable):
        """
        Set the algebraic equation function (equivalent to $AES / E()).

        func(t, A_diff, A_equil, params) -> np.ndarray of length n_equil
        Residuals: each E(i) should be 0 at equilibrium.
        """
        self._aes_func = func

    def set_error(self, func: Callable):
        """
        Set the error/output function (equivalent to $ERROR).

        func(t, A_diff, A_equil, params) -> dict of output predictions
        """
        self._error_func = func

    def _solve_algebraic(self, t: float, A_diff: np.ndarray,
                         A_equil_guess: np.ndarray,
                         params: dict) -> np.ndarray:
        """
        Solve the algebraic equations E(i)=0 for the equilibrium states,
        given the current differential states. This is called at every
        ODE evaluation step.
        """
        def residuals(A_equil):
            return self._aes_func(t, A_diff, A_equil, params)

        solution, info, ier, msg = fsolve(residuals, A_equil_guess, full_output=True)

        if ier != 1:
            # Try with different initial guess if first attempt fails
            A_equil_guess2 = np.full_like(A_equil_guess, 1e-10)
            solution, info, ier, msg = fsolve(residuals, A_equil_guess2, full_output=True)

        # Ensure non-negative concentrations
        solution = np.maximum(solution, 0.0)
        return solution

    def _combined_rhs(self, t: float, y: np.ndarray, params: dict) -> np.ndarray:
        """
        Combined right-hand side for the ODE integrator.
        At each step:
          1. Extract differential states
          2. Solve algebraic equations for equilibrium states
          3. Store equilibrium states in the combined vector
          4. Return dA/dt for differential states (algebraic derivatives = 0)
        """
        A_diff = y[:self.n_diff]
        A_equil_guess = y[self.n_diff:]

        # Solve algebraic equations
        A_equil = self._solve_algebraic(t, A_diff, A_equil_guess, params)

        # Compute differential rates
        dadt_diff = self._des_func(t, A_diff, A_equil, params)

        # For the algebraic states, we use a "fast relaxation" approach:
        # push the algebraic states toward the root-found solution.
        # This is numerically more stable than returning exact 0.
        tau = 1e-3  # relaxation time constant
        dadt_equil = (A_equil - A_equil_guess) / tau

        return np.concatenate([dadt_diff, dadt_equil])

    def solve(self, t_span: Tuple[float, float], t_eval: np.ndarray,
              A0_diff: np.ndarray, A0_equil: np.ndarray,
              params: dict,
              method: str = 'Radau',
              rtol: float = 1e-8, atol: float = 1e-10,
              max_step: float = np.inf) -> Dict:
        """
        Solve the DAE system over a time span.

        Parameters
        ----------
        t_span : tuple
            (t_start, t_end)
        t_eval : np.ndarray
            Time points at which to report the solution.
        A0_diff : np.ndarray
            Initial values for differential states.
        A0_equil : np.ndarray
            Initial guesses for algebraic states (or 0 if INIT=0).
        params : dict
            Model parameters.
        method : str
            ODE solver method. 'Radau' (implicit) recommended for stiff DAEs.
        rtol, atol : float
            Tolerances (analogous to NONMEM TOL/ATOL).
        max_step : float
            Maximum step size.

        Returns
        -------
        dict with keys:
            't'       : time points
            'A_diff'  : differential states at each time (n_times x n_diff)
            'A_equil' : algebraic states at each time (n_times x n_equil)
            'A_all'   : all states combined (n_times x n_total)
            'outputs' : predictions from $ERROR block (if set)
        """
        # Solve algebraic equations for consistent initial conditions
        A0_equil = self._solve_algebraic(0.0, A0_diff, A0_equil, params)
        y0 = np.concatenate([A0_diff, A0_equil])

        # Integrate
        sol = solve_ivp(
            fun=lambda t, y: self._combined_rhs(t, y, params),
            t_span=t_span,
            y0=y0,
            method=method,
            t_eval=t_eval,
            rtol=rtol,
            atol=atol,
            max_step=max_step,
        )

        if not sol.success:
            warnings.warn(f"ODE solver warning: {sol.message}")

        # Post-process: re-solve algebraic equations at each output time
        # to ensure exact equilibrium (not the relaxed approximation)
        A_equil_exact = np.zeros((len(sol.t), self.n_equil))
        for i in range(len(sol.t)):
            A_diff_i = sol.y[:self.n_diff, i]
            A_equil_guess_i = sol.y[self.n_diff:, i]
            A_equil_exact[i, :] = self._solve_algebraic(
                sol.t[i], A_diff_i, A_equil_guess_i, params
            )

        result = {
            't': sol.t,
            'A_diff': sol.y[:self.n_diff, :].T,
            'A_equil': A_equil_exact,
            'A_all': np.column_stack([sol.y[:self.n_diff, :].T, A_equil_exact]),
        }

        # Compute $ERROR outputs
        if self._error_func is not None:
            outputs = []
            for i in range(len(sol.t)):
                out = self._error_func(
                    sol.t[i],
                    sol.y[:self.n_diff, i],
                    A_equil_exact[i, :],
                    params
                )
                outputs.append(out)
            result['outputs'] = outputs

        return result


class SteadyStateDAESolver:
    """
    Solver for purely algebraic (steady-state) DAE systems,
    where all DADT = 0 and the system is defined entirely by
    algebraic equilibrium equations.

    This is the simpler case (like Appendix 3 & 4 NONMEM models)
    where total concentrations are constant and only equilibrium
    complexes need to be solved.
    """

    def __init__(self, n_states: int, name: str = "Steady-State Model"):
        self.n_states = n_states
        self.name = name
        self._aes_func = None

    def set_aes(self, func: Callable):
        """
        func(A_equil, params) -> residuals (np.ndarray of length n_states)
        """
        self._aes_func = func

    def solve(self, A0_guess: np.ndarray, params: dict) -> np.ndarray:
        """Solve the algebraic system E(i) = 0."""
        def residuals(A_equil):
            return self._aes_func(A_equil, params)

        solution = fsolve(residuals, A0_guess)
        return np.maximum(solution, 0.0)

    def solve_over_concentrations(self, conc_range: np.ndarray,
                                  conc_param_name: str,
                                  base_params: dict) -> List[Dict]:
        """
        Solve the steady-state system across a range of total drug concentrations.
        Mimics running NONMEM with multiple dose levels.
        """
        results = []
        prev_guess = np.full(self.n_states, 1e-6)

        for conc in conc_range:
            params = base_params.copy()
            params[conc_param_name] = conc
            try:
                sol = self.solve(prev_guess, params)
                results.append({'conc': conc, 'states': sol, 'success': True})
                prev_guess = sol  # warm-start next solve
            except Exception as e:
                results.append({'conc': conc, 'states': None, 'success': False, 'error': str(e)})

        return results
