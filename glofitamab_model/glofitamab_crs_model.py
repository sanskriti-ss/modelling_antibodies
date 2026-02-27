"""
Glofitamab CRS (Cytokine Release Syndrome) Model
=================================================

Models CRS as a pharmacodynamic response driven by ternary complex (RCAB)
formation from the PK/TMDD model.

Three coupled layers:
  Layer 1 — T-cell activation (indirect response driven by RCAB)
  Layer 2 — Cytokine release (IL-6, TNF-alpha)
  Layer 3 — CRS grade prediction (logistic regression on RCAB peak)

Clinical validation targets (Dickinson et al., JCO 2021):
  - Overall CRS incidence: ~70%
  - CRS at dose 1: 56%, dropping to 2.8% at later doses
  - Median onset: 14 hours post-infusion
  - Median duration: 2 days
  - Grade 3-4: 4.1%

The key insight: step-up dosing (2.5 -> 10 -> 30 mg) reduces peak RCAB
formation rate, which is the driver of acute CRS.

Sources:
  - Dickinson et al., JCO 2021 (Phase I trial)
  - Lee et al., Blood 2019 (CRS grading)
  - Glofitamab FDA label
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Dict, Optional, List
import matplotlib.pyplot as plt
import warnings

try:
    from glofitamab_pk_tmdd import (
        simulate_glofitamab_pk, simulate_stepup_protocol,
        mg_to_nM, PK_PARAMS,
    )
except ImportError:
    from glofitamab_model.glofitamab_pk_tmdd import (
        simulate_glofitamab_pk, simulate_stepup_protocol,
        mg_to_nM, PK_PARAMS,
    )


# ──────────────────────────────────────────
#  CRS Model Parameters
# ──────────────────────────────────────────

CRS_PARAMS = {
    # Layer 1: T-cell activation
    'kact':      3.0,      # 1/day — activation rate constant
    'kdeact':    1.5,      # 1/day — deactivation rate constant
    'EC50_act':  0.3,      # nM — RCAB EC50 for T-cell activation
    'n_hill':    2.0,      # Hill coefficient for activation

    # Layer 2: Cytokine dynamics
    'ksyn_IL6':  10.0,     # pg/mL/day — baseline IL-6 synthesis
    'kdeg_IL6':  6.0,      # 1/day — IL-6 degradation (t1/2 ~ 2.8 hr)
    'Emax_IL6':  500.0,    # fold-increase in IL-6 at full activation
    'ksyn_TNF':  5.0,      # pg/mL/day — baseline TNF-alpha synthesis
    'kdeg_TNF':  8.0,      # 1/day — TNF-alpha degradation (t1/2 ~ 2 hr)
    'Emax_TNF':  200.0,    # fold-increase in TNF-alpha at full activation

    # Layer 3: CRS grade prediction (logistic)
    'beta0':    -0.5,      # intercept (tuned for ~56% at dose 1)
    'beta1':     1.8,      # coefficient for log(RCAB_peak)
    'beta2':    -1.5,      # coefficient for dose_number (reduces with repeat)
}


class CRSModel:
    """
    Cytokine Release Syndrome model driven by ternary complex (RCAB).

    Takes RCAB timecourse from PK model and predicts:
    - T-cell activation dynamics
    - Cytokine release (IL-6, TNF-alpha)
    - CRS probability and grading
    """

    def __init__(self, params: Optional[dict] = None):
        self.params = CRS_PARAMS.copy()
        if params is not None:
            self.params.update(params)

    def _crs_odes(self, t: float, y: np.ndarray,
                  rcab_func) -> np.ndarray:
        """
        CRS ODE system (Layers 1 + 2).

        State: y = [A, IL6, TNF]
          A   — T-cell activation level (0-1)
          IL6 — IL-6 concentration (pg/mL)
          TNF — TNF-alpha concentration (pg/mL)
        """
        A, IL6, TNF = np.maximum(y, 0.0)

        p = self.params
        RCAB = rcab_func(t)

        # Layer 1: T-cell activation (indirect response)
        stim = RCAB**p['n_hill'] / (RCAB**p['n_hill'] + p['EC50_act']**p['n_hill'])
        dA = p['kact'] * stim * (1.0 - A) - p['kdeact'] * A

        # Layer 2: Cytokine release
        dIL6 = p['ksyn_IL6'] * (1.0 + p['Emax_IL6'] * A) - p['kdeg_IL6'] * IL6
        dTNF = p['ksyn_TNF'] * (1.0 + p['Emax_TNF'] * A) - p['kdeg_TNF'] * TNF

        return np.array([dA, dIL6, dTNF])

    def predict_crs_probability(self, RCAB_peak: float,
                                dose_number: int = 1) -> Dict[str, float]:
        """
        Layer 3: Predict CRS probability using logistic model.

        P(CRS >= Grade k) = logistic(beta0 + beta1*log(RCAB_peak) + beta2*dose_number)
        """
        p = self.params
        log_rcab = np.log(max(RCAB_peak, 1e-12))

        # Any-grade CRS
        logit = p['beta0'] + p['beta1'] * log_rcab + p['beta2'] * (dose_number - 1)
        prob_any = 1.0 / (1.0 + np.exp(-logit))

        # Grade >= 2 (shift intercept down)
        logit_g2 = logit - 2.0
        prob_g2 = 1.0 / (1.0 + np.exp(-logit_g2))

        # Grade >= 3 (shift further)
        logit_g3 = logit - 4.0
        prob_g3 = 1.0 / (1.0 + np.exp(-logit_g3))

        return {
            'prob_any_grade': prob_any,
            'prob_grade_ge2': prob_g2,
            'prob_grade_ge3': prob_g3,
            'prob_grade1': prob_any - prob_g2,
        }


def simulate_crs_timecourse(
    pk_result: Dict[str, np.ndarray],
    crs_params: Optional[dict] = None,
    dose_number: int = 1,
) -> Dict[str, np.ndarray]:
    """
    Simulate CRS timecourse driven by RCAB from PK model.

    Parameters
    ----------
    pk_result : dict   Output from simulate_glofitamab_pk() or simulate_stepup_protocol()
    crs_params : dict  Optional CRS parameter overrides
    dose_number : int  Which dose number (affects CRS probability)

    Returns
    -------
    Dict with activation, cytokine, and CRS probability timecourses
    """
    model = CRSModel(crs_params)

    # Create RCAB interpolation function
    t_pk = pk_result['t_days']
    rcab_pk = pk_result['RCAB_nM']

    def rcab_func(t):
        return max(np.interp(t, t_pk, rcab_pk), 0.0)

    # Baseline cytokine levels at steady state
    p = model.params
    IL6_baseline = p['ksyn_IL6'] / p['kdeg_IL6']
    TNF_baseline = p['ksyn_TNF'] / p['kdeg_TNF']

    # Initial conditions: [A, IL6, TNF]
    y0 = np.array([0.0, IL6_baseline, TNF_baseline])

    t_end = t_pk[-1]
    t_eval = np.linspace(0, t_end, max(len(t_pk), 500))

    sol = solve_ivp(
        model._crs_odes, (0, t_end), y0,
        args=(rcab_func,), method='RK45',
        t_eval=t_eval, rtol=1e-6, atol=1e-8,
    )

    if not sol.success:
        warnings.warn(f"CRS ODE solver: {sol.message}")

    # Compute CRS probability over time
    RCAB_peak = np.max(rcab_pk)
    crs_prob = model.predict_crs_probability(RCAB_peak, dose_number)

    result = {
        't_days':      sol.t,
        't_hours':     sol.t * 24.0,
        'activation':  sol.y[0],   # T-cell activation (0-1)
        'IL6':         sol.y[1],   # IL-6 (pg/mL)
        'TNF':         sol.y[2],   # TNF-alpha (pg/mL)
        'RCAB':        np.array([rcab_func(t) for t in sol.t]),
        'IL6_baseline': IL6_baseline,
        'TNF_baseline': TNF_baseline,
        'RCAB_peak':   RCAB_peak,
        'crs_probability': crs_prob,
    }

    return result


def simulate_stepup_crs_benefit(
    pk_params: Optional[dict] = None,
    crs_params: Optional[dict] = None,
) -> Dict[str, Dict]:
    """
    Compare CRS with step-up dosing vs flat 30mg dosing.

    Demonstrates that step-up reduces peak RCAB formation rate,
    which is the driver of acute CRS.

    Returns
    -------
    Dict with 'stepup' and 'flat' keys, each containing PK and CRS results
    """
    if pk_params is None:
        pk_params = PK_PARAMS.copy()

    # ── Step-up first dose: 2.5 mg (apples-to-apples comparison) ──
    pk_stepup_d1 = simulate_glofitamab_pk(
        dose_mg=2.5, t_end_days=7.0,
        params=pk_params, obinutuzumab_pretreat=True,
    )
    crs_stepup_d1 = simulate_crs_timecourse(pk_stepup_d1, crs_params, dose_number=1)

    # ── Flat 30 mg first dose (no step-up) ──
    pk_flat_d1 = simulate_glofitamab_pk(
        dose_mg=30.0, t_end_days=7.0,
        params=pk_params, obinutuzumab_pretreat=True,
    )
    crs_flat_d1 = simulate_crs_timecourse(pk_flat_d1, crs_params, dose_number=1)

    # ── Comparison metrics (first dose only) ──
    comparison = {
        'stepup': {
            'pk': pk_stepup_d1,
            'crs': crs_stepup_d1,
            'RCAB_peak': crs_stepup_d1['RCAB_peak'],
            'IL6_peak': np.max(crs_stepup_d1['IL6']),
            'activation_peak': np.max(crs_stepup_d1['activation']),
            'crs_prob': crs_stepup_d1['crs_probability'],
        },
        'flat': {
            'pk': pk_flat_d1,
            'crs': crs_flat_d1,
            'RCAB_peak': crs_flat_d1['RCAB_peak'],
            'IL6_peak': np.max(crs_flat_d1['IL6']),
            'activation_peak': np.max(crs_flat_d1['activation']),
            'crs_prob': crs_flat_d1['crs_probability'],
        },
    }

    return comparison


# ──────────────────────────────────────
#  Plotting
# ──────────────────────────────────────

def plot_crs_timecourse(crs_result: Dict[str, np.ndarray],
                        title: str = "Glofitamab CRS Timecourse",
                        savepath: Optional[str] = None):
    """Plot CRS model outputs: activation, cytokines, RCAB."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    t_hr = crs_result['t_hours']
    # Limit to first 7 days for detail
    mask = t_hr <= 168

    # RCAB driving signal
    axes[0, 0].plot(t_hr[mask], crs_result['RCAB'][mask], 'm-', lw=2)
    axes[0, 0].set_xlabel('Time (hours)')
    axes[0, 0].set_ylabel('RCAB (nM)')
    axes[0, 0].set_title('Ternary Complex (RCAB)')
    axes[0, 0].grid(True, alpha=0.3)

    # T-cell activation
    axes[0, 1].plot(t_hr[mask], crs_result['activation'][mask], 'r-', lw=2)
    axes[0, 1].set_xlabel('Time (hours)')
    axes[0, 1].set_ylabel('Activation (0-1)')
    axes[0, 1].set_title('T-cell Activation')
    axes[0, 1].set_ylim(-0.05, 1.05)
    axes[0, 1].grid(True, alpha=0.3)

    # IL-6
    axes[1, 0].plot(t_hr[mask], crs_result['IL6'][mask], 'b-', lw=2)
    axes[1, 0].axhline(crs_result['IL6_baseline'], color='gray',
                        ls='--', alpha=0.5, label='Baseline')
    axes[1, 0].set_xlabel('Time (hours)')
    axes[1, 0].set_ylabel('IL-6 (pg/mL)')
    axes[1, 0].set_title('IL-6')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # TNF-alpha
    axes[1, 1].plot(t_hr[mask], crs_result['TNF'][mask], 'orange', lw=2)
    axes[1, 1].axhline(crs_result['TNF_baseline'], color='gray',
                        ls='--', alpha=0.5, label='Baseline')
    axes[1, 1].set_xlabel('Time (hours)')
    axes[1, 1].set_ylabel('TNF-alpha (pg/mL)')
    axes[1, 1].set_title('TNF-alpha')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Add CRS probability annotation
    prob = crs_result['crs_probability']
    fig.text(0.5, 0.01,
             f"CRS Probability: Any={prob['prob_any_grade']:.1%}, "
             f"Gr1={prob['prob_grade1']:.1%}, "
             f"Gr>=2={prob['prob_grade_ge2']:.1%}, "
             f"Gr>=3={prob['prob_grade_ge3']:.1%}",
             ha='center', fontsize=10,
             bbox=dict(facecolor='lightyellow', alpha=0.8))

    plt.suptitle(title, fontsize=13)
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])

    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches='tight')
        print(f"Figure saved to {savepath}")
    else:
        plt.show()


def plot_stepup_vs_flat(comparison: Dict,
                        savepath: Optional[str] = None):
    """Plot step-up vs flat dosing CRS comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    su = comparison['stepup']['crs']
    fl = comparison['flat']['crs']

    # Limit to first 7 days
    mask_su = su['t_hours'] <= 168
    mask_fl = fl['t_hours'] <= 168

    # RCAB
    axes[0].plot(su['t_hours'][mask_su], su['RCAB'][mask_su],
                 'b-', lw=2, label='Step-up (2.5mg)')
    axes[0].plot(fl['t_hours'][mask_fl], fl['RCAB'][mask_fl],
                 'r-', lw=2, label='Flat (30mg)')
    axes[0].set_xlabel('Time (hours)')
    axes[0].set_ylabel('RCAB (nM)')
    axes[0].set_title('Peak RCAB Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # IL-6
    axes[1].plot(su['t_hours'][mask_su], su['IL6'][mask_su],
                 'b-', lw=2, label='Step-up')
    axes[1].plot(fl['t_hours'][mask_fl], fl['IL6'][mask_fl],
                 'r-', lw=2, label='Flat 30mg')
    axes[1].set_xlabel('Time (hours)')
    axes[1].set_ylabel('IL-6 (pg/mL)')
    axes[1].set_title('IL-6 Release')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # CRS probability bar chart
    categories = ['Any CRS', 'Grade 1', 'Grade>=2', 'Grade>=3']
    su_probs = [
        comparison['stepup']['crs_prob']['prob_any_grade'],
        comparison['stepup']['crs_prob']['prob_grade1'],
        comparison['stepup']['crs_prob']['prob_grade_ge2'],
        comparison['stepup']['crs_prob']['prob_grade_ge3'],
    ]
    fl_probs = [
        comparison['flat']['crs_prob']['prob_any_grade'],
        comparison['flat']['crs_prob']['prob_grade1'],
        comparison['flat']['crs_prob']['prob_grade_ge2'],
        comparison['flat']['crs_prob']['prob_grade_ge3'],
    ]

    x = np.arange(len(categories))
    width = 0.35
    axes[2].bar(x - width/2, [p*100 for p in su_probs], width,
                label='Step-up', color='steelblue')
    axes[2].bar(x + width/2, [p*100 for p in fl_probs], width,
                label='Flat 30mg', color='indianred')
    axes[2].set_ylabel('Probability (%)')
    axes[2].set_title('CRS Risk')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(categories, rotation=15)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.suptitle('Step-up vs Flat Dosing: CRS Protection',
                 fontsize=13)
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
    print("Glofitamab CRS Model")
    print("=" * 60)

    # ── Single dose CRS (first dose = 2.5 mg) ──
    print(f"\n--- CRS after first dose (2.5 mg, with obinutuzumab) ---")
    pk_result = simulate_glofitamab_pk(
        dose_mg=2.5, t_end_days=7.0,
        obinutuzumab_pretreat=True,
    )
    crs_result = simulate_crs_timecourse(pk_result, dose_number=1)

    # Check onset timing
    activation = crs_result['activation']
    t_hours = crs_result['t_hours']
    # Find time to 50% of peak activation
    peak_act = np.max(activation)
    if peak_act > 0.01:
        half_peak_idx = np.argmax(activation >= peak_act * 0.5)
        onset_hours = t_hours[half_peak_idx]
        print(f"  Peak T-cell activation: {peak_act:.3f}")
        print(f"  Onset (50% peak): {onset_hours:.1f} hours")
        print(f"  Clinical target onset: ~14 hours")
    else:
        print(f"  Minimal activation: {peak_act:.3e}")

    print(f"  Peak IL-6:     {np.max(crs_result['IL6']):.1f} pg/mL "
          f"(baseline: {crs_result['IL6_baseline']:.1f})")
    print(f"  Peak TNF-alpha: {np.max(crs_result['TNF']):.1f} pg/mL "
          f"(baseline: {crs_result['TNF_baseline']:.1f})")

    prob = crs_result['crs_probability']
    print(f"\n  CRS Probability (dose 1):")
    print(f"    Any grade: {prob['prob_any_grade']:.1%}")
    print(f"    Grade 1:   {prob['prob_grade1']:.1%}")
    print(f"    Grade >=2: {prob['prob_grade_ge2']:.1%}")
    print(f"    Grade >=3: {prob['prob_grade_ge3']:.1%}")

    # ── Step-up vs flat dosing comparison ──
    print(f"\n--- Step-up vs Flat Dosing CRS Comparison ---")
    comparison = simulate_stepup_crs_benefit()

    for label in ['stepup', 'flat']:
        data = comparison[label]
        print(f"\n  {label.upper()}:")
        print(f"    Peak RCAB:       {data['RCAB_peak']:.4f} nM")
        print(f"    Peak IL-6:       {data['IL6_peak']:.1f} pg/mL")
        print(f"    Peak activation: {data['activation_peak']:.3f}")
        print(f"    CRS any-grade:   {data['crs_prob']['prob_any_grade']:.1%}")

    # ── CRS attenuation over doses ──
    print(f"\n--- CRS attenuation across doses ---")
    model = CRSModel()
    rcab_peak_example = 1.0  # nM
    for dose_num in [1, 2, 3, 5, 8]:
        prob = model.predict_crs_probability(rcab_peak_example, dose_num)
        print(f"  Dose {dose_num}: CRS any-grade = {prob['prob_any_grade']:.1%}")

    print(f"\n  Clinical target: dose 1 ~ 56%, later doses ~ 2.8%")
    print(f"\n  To plot: plot_crs_timecourse(crs_result)")
    print(f"  To compare: plot_stepup_vs_flat(comparison)")
