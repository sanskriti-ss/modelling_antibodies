"""
Glofitamab Hypothalamic Temperature Regulation Model
=====================================================

Models fever as a side effect of CRS-driven IL-6 elevation, via the
hypothalamic thermoregulatory setpoint mechanism.

Mechanism (from Lefevre et al. 2024, Stolwijk 1971):
  1. IL-6 released during CRS acts on the hypothalamus
  2. The hypothalamic setpoint T_hypo shifts upward logarithmically:
       T_hypo(t) = T_base + alpha * ln([IL6](t) / [IL6]_base)
  3. Body temperature T_body follows T_hypo via a first-order ODE
     with heat dissipation to the environment:
       dT_body/dt = k3 * (T_hypo - T_body) - kd3 * (T_body - T_room)

Parameters:
  T_base  = 37.0 C    — normal core body temperature
  T_room  = 22.0 C    — ambient/room temperature
  alpha   = 0.35 C    — IL-6 sensitivity of hypothalamic setpoint
                         (calibrated so a ~50-fold IL-6 rise gives ~1.4 C fever,
                          consistent with Grade 1-2 CRS fever of 38-39 C;
                          Gritti et al. 2024 report median fever ~38.3 C)
  k3      = 4.0 1/day — rate at which body temp approaches setpoint
                         (time constant ~6 hours; Stolwijk 1971)
  kd3     = 0.5 1/day — passive heat dissipation rate to environment
                         (slow relative to active thermoregulation)

Clinical validation targets:
  - Fever is the most common CRS symptom (~70% of patients)
  - Median peak temperature during CRS: 38.0-38.5 C (Dickinson et al. 2021)
  - Fever onset: ~10-24 hours post-infusion (parallels CRS onset)
  - Step-up dosing should reduce fever severity at first dose
  - Fever resolves within 48-72 hours for most Grade 1-2 CRS

Fever grading (Lee et al. 2019, ASTCT consensus):
  Grade 1: T >= 38.0 C
  Grade 2: T >= 38.0 C with hypotension/hypoxia (not modeled here;
           we use T >= 38.5 C as proxy for higher-grade fever)
  Grade 3: T >= 40.0 C (high fever)

Sources:
  - Lefevre, N. et al. (2024). Thermoregulatory modeling of cytokine-driven
    fever in bispecific antibody therapy. PMC11782561.
  - Gritti, A. et al. (2024). CRS characterization in bispecific T-cell
    engager therapy.
  - Stolwijk, J. A. J. (1971). A mathematical model of physiological
    temperature regulation in man. NASA CR-1855.
  - Dickinson, M. J. et al. (2021). Glofitamab Phase I trial. JCO.
  - Lee, D. W. et al. (2019). ASTCT CRS consensus grading. Biol Blood
    Marrow Transplant.
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Dict, Optional
import matplotlib.pyplot as plt
import warnings

try:
    from glofitamab_pk_tmdd import (
        simulate_glofitamab_pk, simulate_stepup_protocol,
        PK_PARAMS,
    )
    from glofitamab_crs_model import (
        simulate_crs_timecourse, simulate_stepup_crs_benefit,
        CRS_PARAMS,
    )
except ImportError:
    from glofitamab_model.glofitamab_pk_tmdd import (
        simulate_glofitamab_pk, simulate_stepup_protocol,
        PK_PARAMS,
    )
    from glofitamab_model.glofitamab_crs_model import (
        simulate_crs_timecourse, simulate_stepup_crs_benefit,
        CRS_PARAMS,
    )


# ──────────────────────────────────────────
#  Thermoregulation Parameters
# ──────────────────────────────────────────

THERMO_PARAMS = {
    # Baseline temperatures (Stolwijk 1971)
    'T_base':  37.0,    # C — normal hypothalamic setpoint
    'T_room':  22.0,    # C — ambient temperature

    # Hypothalamic IL-6 sensitivity (Lefevre et al. 2024)
    # alpha calibrated so that:
    #   - 10x IL-6 rise -> +0.81 C (mild fever)
    #   - 50x IL-6 rise -> +1.37 C (38.4 C, Grade 1 CRS fever)
    #   - 200x IL-6 rise -> +1.85 C (38.9 C, moderate CRS)
    #   - 1000x IL-6 rise -> +2.42 C (39.4 C, significant CRS)
    'alpha':   0.35,    # C — setpoint shift per ln-unit of IL-6 ratio

    # Body temperature dynamics (Stolwijk 1971)
    'k3':      4.0,     # 1/day — active thermoregulation rate
                        # (body temp approaches setpoint; tau ~ 6 hr)
    'kd3':     0.5,     # 1/day — passive heat loss to environment
                        # (slow; convection, radiation, evaporation)

    # Fever grade thresholds (Lee et al. 2019, ASTCT)
    'grade1_threshold': 38.0,   # C — any fever
    'grade2_threshold': 38.5,   # C — moderate fever (proxy for Gr2)
    'grade3_threshold': 40.0,   # C — high fever
}


class ThermoregulationModel:
    """
    Hypothalamic temperature regulation model.

    Takes IL-6 timecourse from CRS model and predicts body temperature
    via the hypothalamic setpoint mechanism.
    """

    def __init__(self, params: Optional[dict] = None):
        self.params = THERMO_PARAMS.copy()
        if params is not None:
            self.params.update(params)

    def hypothalamic_setpoint(self, IL6: float, IL6_baseline: float) -> float:
        """
        Compute hypothalamic temperature setpoint from IL-6 level.

        T_hypo = T_base + alpha * ln([IL6] / [IL6]_base)

        From Lefevre et al. (2024), Eq. 4.
        """
        p = self.params
        ratio = max(IL6 / max(IL6_baseline, 1e-6), 1.0)  # clamp >= 1
        return p['T_base'] + p['alpha'] * np.log(ratio)

    def _thermo_ode(self, t: float, y: np.ndarray,
                    il6_func, il6_baseline: float) -> np.ndarray:
        """
        Body temperature ODE.

        State: y = [T_body]

        dT_body/dt = k3 * (T_hypo - T_body) - kd3 * (T_body - T_base)

        The first term drives T_body toward the hypothalamic setpoint.
        The second term represents active thermoregulatory heat dissipation
        (sweating, vasodilation) when body temp exceeds the normal baseline,
        which acts as a negative feedback limiting fever magnitude.

        Note: We use T_base (not T_room) as the dissipation reference.
        The Stolwijk (1971) full model accounts for passive heat loss to
        ambient via a separate skin compartment; here we collapse that
        into the single-compartment formulation where the body actively
        maintains T_base via metabolic heat production. The kd3 term
        then captures the net *additional* heat loss during fever.

        From Lefevre et al. (2024), Eq. 5; Stolwijk (1971).
        """
        T_body = y[0]
        p = self.params

        IL6 = il6_func(t)
        T_hypo = self.hypothalamic_setpoint(IL6, il6_baseline)

        dT = (p['k3'] * (T_hypo - T_body)
              - p['kd3'] * (T_body - p['T_base']))

        return np.array([dT])

    def classify_fever_grade(self, T_body: float) -> int:
        """Classify fever grade from body temperature."""
        p = self.params
        if T_body >= p['grade3_threshold']:
            return 3
        elif T_body >= p['grade2_threshold']:
            return 2
        elif T_body >= p['grade1_threshold']:
            return 1
        return 0


# ──────────────────────────────────────────
#  Simulation Functions
# ──────────────────────────────────────────

def simulate_temperature_response(
    pk_result: Dict[str, np.ndarray],
    crs_params: Optional[dict] = None,
    thermo_params: Optional[dict] = None,
    dose_number: int = 1,
) -> Dict[str, np.ndarray]:
    """
    Simulate body temperature response to a glofitamab dose.

    Chain: PK (RCAB) -> CRS (IL-6) -> Thermoregulation (T_body)

    Parameters
    ----------
    pk_result : dict     Output from simulate_glofitamab_pk()
    crs_params : dict    Optional CRS parameter overrides
    thermo_params : dict Optional thermoregulation parameter overrides
    dose_number : int    Dose number (affects CRS dynamics)

    Returns
    -------
    Dict with temperature timecourse and fever metrics
    """
    # Step 1: Run CRS model to get IL-6 timecourse
    crs_result = simulate_crs_timecourse(pk_result, crs_params, dose_number)

    # Step 2: Run thermoregulation model driven by IL-6
    model = ThermoregulationModel(thermo_params)

    t_crs = crs_result['t_days']
    il6_data = crs_result['IL6']
    il6_baseline = crs_result['IL6_baseline']

    def il6_func(t):
        return max(np.interp(t, t_crs, il6_data), 0.0)

    p = model.params
    T0 = p['T_base']
    y0 = np.array([T0])

    t_end = t_crs[-1]
    t_eval = np.linspace(0, t_end, max(len(t_crs), 500))

    sol = solve_ivp(
        model._thermo_ode, (0, t_end), y0,
        args=(il6_func, il6_baseline),
        method='RK45', t_eval=t_eval, rtol=1e-6, atol=1e-8,
    )

    if not sol.success:
        warnings.warn(f"Thermo ODE solver: {sol.message}")

    T_body = sol.y[0]

    # Compute hypothalamic setpoint timecourse for plotting
    T_hypo = np.array([
        model.hypothalamic_setpoint(il6_func(t), il6_baseline)
        for t in sol.t
    ])

    # Fever metrics
    T_peak = np.max(T_body)
    peak_grade = model.classify_fever_grade(T_peak)

    # Fever duration: time with T >= 38.0 C
    fever_mask = T_body >= p['grade1_threshold']
    if np.any(fever_mask):
        fever_indices = np.where(fever_mask)[0]
        fever_duration_days = (sol.t[fever_indices[-1]] - sol.t[fever_indices[0]])
        fever_duration_hours = fever_duration_days * 24.0
    else:
        fever_duration_hours = 0.0

    # Fever onset: time to first T >= 38.0
    if np.any(fever_mask):
        onset_idx = fever_indices[0]
        fever_onset_hours = sol.t[onset_idx] * 24.0
    else:
        fever_onset_hours = None

    # Grade time fractions
    grades = np.array([model.classify_fever_grade(T) for T in T_body])

    result = {
        't_days':          sol.t,
        't_hours':         sol.t * 24.0,
        'T_body':          T_body,
        'T_hypo':          T_hypo,
        'IL6':             np.array([il6_func(t) for t in sol.t]),
        'IL6_baseline':    il6_baseline,
        'activation':      np.array([
            np.interp(t, t_crs, crs_result['activation']) for t in sol.t
        ]),
        'RCAB':            np.array([
            np.interp(t, t_crs, crs_result['RCAB']) for t in sol.t
        ]),
        'T_peak':          T_peak,
        'peak_fever_grade': peak_grade,
        'fever_onset_hours': fever_onset_hours,
        'fever_duration_hours': fever_duration_hours,
        'fever_grades':    grades,
        'grade_fractions': {
            'no_fever':  np.mean(grades == 0),
            'grade_1':   np.mean(grades == 1),
            'grade_2':   np.mean(grades == 2),
            'grade_3':   np.mean(grades >= 3),
        },
        'crs_result':      crs_result,
    }

    return result


def simulate_dose_dependent_temperature(
    doses_mg: list = None,
    thermo_params: Optional[dict] = None,
) -> Dict[str, Dict]:
    """
    Compare temperature response across different dose levels.

    Returns dict keyed by dose level with temperature simulation results.
    """
    if doses_mg is None:
        doses_mg = [1.0, 2.5, 10.0, 30.0]

    results = {}
    for dose in doses_mg:
        pk = simulate_glofitamab_pk(
            dose_mg=dose, t_end_days=5.0,
            obinutuzumab_pretreat=True,
        )
        temp_result = simulate_temperature_response(
            pk, thermo_params=thermo_params, dose_number=1,
        )
        results[dose] = temp_result

    return results


def simulate_stepup_vs_flat_temperature(
    thermo_params: Optional[dict] = None,
) -> Dict[str, Dict]:
    """
    Compare temperature response: step-up (2.5mg first) vs flat (30mg first).
    """
    # Step-up: first dose 2.5 mg
    pk_stepup = simulate_glofitamab_pk(
        dose_mg=2.5, t_end_days=5.0,
        obinutuzumab_pretreat=True,
    )
    temp_stepup = simulate_temperature_response(
        pk_stepup, thermo_params=thermo_params, dose_number=1,
    )

    # Flat: first dose 30 mg
    pk_flat = simulate_glofitamab_pk(
        dose_mg=30.0, t_end_days=5.0,
        obinutuzumab_pretreat=True,
    )
    temp_flat = simulate_temperature_response(
        pk_flat, thermo_params=thermo_params, dose_number=1,
    )

    return {
        'stepup': {
            'dose_mg': 2.5,
            'label': 'Step-up (2.5 mg first)',
            'result': temp_stepup,
        },
        'flat': {
            'dose_mg': 30.0,
            'label': 'Flat (30 mg first)',
            'result': temp_flat,
        },
    }


# ──────────────────────────────────────
#  Plotting
# ──────────────────────────────────────

def plot_temperature_timecourse(
    temp_result: Dict[str, np.ndarray],
    title: str = "Glofitamab — Hypothalamic Temperature Regulation",
    savepath: Optional[str] = None,
):
    """Plot temperature model outputs: T_body, T_hypo, IL-6 driver, and RCAB."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    t_hr = temp_result['t_hours']
    mask = t_hr <= 120  # first 5 days

    # Body temperature and hypothalamic setpoint
    ax = axes[0, 0]
    ax.plot(t_hr[mask], temp_result['T_body'][mask], 'r-', lw=2.5,
            label='T_body')
    ax.plot(t_hr[mask], temp_result['T_hypo'][mask], 'r--', lw=1.5,
            alpha=0.6, label='T_hypo (setpoint)')
    ax.axhline(37.0, color='gray', ls=':', alpha=0.4)
    ax.axhline(38.0, color='orange', ls='--', alpha=0.5, label='Fever (38C)')
    ax.axhline(40.0, color='red', ls='--', alpha=0.4, label='High fever (40C)')
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Temperature (C)')
    ax.set_title('Body Temperature Response')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(36.5, max(np.max(temp_result['T_body'][mask]) + 0.5, 39.0))

    # IL-6 driving the setpoint
    ax = axes[0, 1]
    il6_fold = temp_result['IL6'][mask] / temp_result['IL6_baseline']
    ax.plot(t_hr[mask], il6_fold, 'b-', lw=2)
    ax.axhline(1.0, color='gray', ls='--', alpha=0.5, label='Baseline')
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('IL-6 (fold over baseline)')
    ax.set_title('IL-6 (Fever Driver)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # RCAB (upstream driver)
    ax = axes[1, 0]
    ax.plot(t_hr[mask], temp_result['RCAB'][mask], 'm-', lw=2)
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('RCAB (nM)')
    ax.set_title('Ternary Complex (Upstream Signal)')
    ax.grid(True, alpha=0.3)

    # Fever grade timeline
    ax = axes[1, 1]
    grades = temp_result['fever_grades'][mask]
    grade_colors = {0: 'lightgreen', 1: '#FFCC80', 2: '#FF8A65', 3: '#EF5350'}
    grade_labels = {0: 'Normal', 1: 'Grade 1 (38-38.5C)',
                    2: 'Grade 2 (38.5-40C)', 3: 'Grade 3 (>40C)'}
    for g, c in grade_colors.items():
        g_mask = grades == g
        if np.any(g_mask):
            ax.fill_between(t_hr[mask], 0, 1, where=g_mask,
                            alpha=0.7, color=c,
                            label=grade_labels[g], step='mid')
    ax.set_xlabel('Time (hours)')
    ax.set_yticks([])
    ax.set_title('Fever Grade Over Time')
    ax.legend(fontsize=7, loc='upper right')

    # Annotation
    onset_str = (f"{temp_result['fever_onset_hours']:.0f}h"
                 if temp_result['fever_onset_hours'] is not None else "None")
    fig.text(0.5, 0.01,
             f"Peak T = {temp_result['T_peak']:.1f}C (Grade {temp_result['peak_fever_grade']}) | "
             f"Fever onset: {onset_str} | "
             f"Duration: {temp_result['fever_duration_hours']:.0f}h",
             ha='center', fontsize=10,
             bbox=dict(facecolor='mistyrose', edgecolor='red', alpha=0.8))

    plt.suptitle(title, fontsize=13)
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])

    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches='tight')
        print(f"Figure saved to {savepath}")
    else:
        plt.show()


def plot_dose_dependent_temperature(
    dose_results: Dict[float, Dict],
    savepath: Optional[str] = None,
):
    """Plot temperature response across different dose levels."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    colors = ['#4CAF50', '#2196F3', '#FF9800', '#E91E63']
    doses = sorted(dose_results.keys())

    # Panel 1: Temperature timecourse
    ax = axes[0]
    for dose, color in zip(doses, colors[:len(doses)]):
        r = dose_results[dose]
        mask = r['t_hours'] <= 96
        ax.plot(r['t_hours'][mask], r['T_body'][mask],
                color=color, lw=2, label=f'{dose} mg')
    ax.axhline(38.0, color='gray', ls='--', alpha=0.5)
    ax.axhline(37.0, color='gray', ls=':', alpha=0.3)
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Body Temperature (C)')
    ax.set_title('Temperature vs Dose Level')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(36.5, max(max(np.max(r['T_body']) for r in dose_results.values()) + 0.3, 38.5))

    # Panel 2: Peak temperature bar chart
    ax = axes[1]
    peak_temps = [dose_results[d]['T_peak'] for d in doses]
    bar_colors = []
    for T in peak_temps:
        if T >= 40.0:
            bar_colors.append('#EF5350')
        elif T >= 38.5:
            bar_colors.append('#FF8A65')
        elif T >= 38.0:
            bar_colors.append('#FFCC80')
        else:
            bar_colors.append('#81C784')

    bars = ax.bar(range(len(doses)), peak_temps, color=bar_colors, edgecolor='gray')
    ax.set_xticks(range(len(doses)))
    ax.set_xticklabels([f'{d} mg' for d in doses])
    ax.set_ylabel('Peak Temperature (C)')
    ax.set_title('Peak Fever by Dose')
    ax.axhline(38.0, color='orange', ls='--', alpha=0.6, label='Fever')
    ax.axhline(40.0, color='red', ls='--', alpha=0.4, label='High fever')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(36.5, max(max(peak_temps) + 0.5, 39.0))
    # Value labels
    for bar, T in zip(bars, peak_temps):
        ax.text(bar.get_x() + bar.get_width()/2., T + 0.05,
                f'{T:.1f}C', ha='center', va='bottom', fontsize=9)

    # Panel 3: Fever duration
    ax = axes[2]
    durations = [dose_results[d]['fever_duration_hours'] for d in doses]
    ax.bar(range(len(doses)), durations, color='salmon', edgecolor='gray')
    ax.set_xticks(range(len(doses)))
    ax.set_xticklabels([f'{d} mg' for d in doses])
    ax.set_ylabel('Fever Duration (hours)')
    ax.set_title('Fever Duration by Dose')
    ax.grid(True, alpha=0.3, axis='y')
    for i, dur in enumerate(durations):
        if dur > 0:
            ax.text(i, dur + 0.5, f'{dur:.0f}h', ha='center', fontsize=9)

    plt.suptitle('Dose-Dependent Hypothalamic Temperature Dysregulation\n'
                 '(Lefevre et al. 2024 / Stolwijk 1971 Model)',
                 fontsize=13)
    plt.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches='tight')
        print(f"Figure saved to {savepath}")
    else:
        plt.show()


def plot_stepup_vs_flat_temperature(
    comparison: Dict[str, Dict],
    savepath: Optional[str] = None,
):
    """Plot step-up vs flat dosing temperature comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    su = comparison['stepup']['result']
    fl = comparison['flat']['result']

    mask_su = su['t_hours'] <= 96
    mask_fl = fl['t_hours'] <= 96

    # Panel 1: Temperature timecourse
    ax = axes[0]
    ax.plot(su['t_hours'][mask_su], su['T_body'][mask_su],
            'b-', lw=2.5, label='Step-up (2.5 mg)')
    ax.plot(fl['t_hours'][mask_fl], fl['T_body'][mask_fl],
            'r-', lw=2.5, label='Flat (30 mg)')
    ax.axhline(38.0, color='gray', ls='--', alpha=0.5, label='Fever threshold')
    ax.axhline(37.0, color='gray', ls=':', alpha=0.3)
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Body Temperature (C)')
    ax.set_title('Temperature: Step-Up vs Flat')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(36.5, max(fl['T_peak'] + 0.5, 39.0))

    # Panel 2: IL-6 driving comparison
    ax = axes[1]
    su_il6_fold = su['IL6'][mask_su] / su['IL6_baseline']
    fl_il6_fold = fl['IL6'][mask_fl] / fl['IL6_baseline']
    ax.plot(su['t_hours'][mask_su], su_il6_fold,
            'b-', lw=2, label='Step-up')
    ax.plot(fl['t_hours'][mask_fl], fl_il6_fold,
            'r-', lw=2, label='Flat 30mg')
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('IL-6 (fold over baseline)')
    ax.set_title('IL-6 (Fever Driver)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 3: Fever metrics comparison
    ax = axes[2]
    categories = ['Peak Temp\n(C above 37)', 'Fever Duration\n(hours)', 'Fever Onset\n(hours)']
    su_vals = [
        su['T_peak'] - 37.0,
        su['fever_duration_hours'],
        su['fever_onset_hours'] if su['fever_onset_hours'] is not None else 0,
    ]
    fl_vals = [
        fl['T_peak'] - 37.0,
        fl['fever_duration_hours'],
        fl['fever_onset_hours'] if fl['fever_onset_hours'] is not None else 0,
    ]

    x = np.arange(len(categories))
    w = 0.35
    bars1 = ax.bar(x - w/2, su_vals, w, label='Step-up', color='steelblue')
    bars2 = ax.bar(x + w/2, fl_vals, w, label='Flat 30mg', color='indianred')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylabel('Value')
    ax.set_title('Fever Metrics Comparison')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # Value labels
    for bar in bars1:
        h = bar.get_height()
        if h > 0.1:
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.05,
                    f'{h:.1f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        if h > 0.1:
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.05,
                    f'{h:.1f}', ha='center', va='bottom', fontsize=8)

    plt.suptitle('Step-Up vs Flat Dosing: Fever Protection\n'
                 'Step-up reduces IL-6 peak, lowering hypothalamic setpoint shift',
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
    print("Glofitamab Hypothalamic Temperature Regulation Model")
    print("=" * 60)

    # ── Single dose temperature response (2.5 mg) ──
    print(f"\n--- Temperature after first dose (2.5 mg) ---")
    pk = simulate_glofitamab_pk(
        dose_mg=2.5, t_end_days=5.0,
        obinutuzumab_pretreat=True,
    )
    temp = simulate_temperature_response(pk, dose_number=1)

    print(f"  Peak body temperature: {temp['T_peak']:.2f} C")
    print(f"  Peak fever grade:      {temp['peak_fever_grade']}")
    if temp['fever_onset_hours'] is not None:
        print(f"  Fever onset:           {temp['fever_onset_hours']:.1f} hours")
    else:
        print(f"  No fever (T < 38.0 C)")
    print(f"  Fever duration:        {temp['fever_duration_hours']:.1f} hours")
    print(f"  Clinical target: median 38.0-38.5 C, onset ~14h")

    # ── Dose-dependent temperature ──
    print(f"\n--- Dose-dependent temperature ---")
    dose_results = simulate_dose_dependent_temperature()
    for dose in sorted(dose_results.keys()):
        r = dose_results[dose]
        print(f"  {dose:5.1f} mg: T_peak = {r['T_peak']:.2f} C  "
              f"(Grade {r['peak_fever_grade']})  "
              f"Duration = {r['fever_duration_hours']:.0f}h")

    # ── Step-up vs flat ──
    print(f"\n--- Step-up vs Flat first dose ---")
    comp = simulate_stepup_vs_flat_temperature()
    for key in ['stepup', 'flat']:
        r = comp[key]['result']
        label = comp[key]['label']
        print(f"  {label}:")
        print(f"    T_peak = {r['T_peak']:.2f} C (Grade {r['peak_fever_grade']})")
        print(f"    Fever duration = {r['fever_duration_hours']:.0f}h")

    # ── Verify parameter sensitivity ──
    print(f"\n--- Hypothalamic setpoint sensitivity ---")
    model = ThermoregulationModel()
    for fold in [1, 5, 10, 50, 200, 1000]:
        T_hypo = model.hypothalamic_setpoint(fold * 1.667, 1.667)  # IL6 baseline
        print(f"  IL-6 {fold:4d}x baseline -> T_hypo = {T_hypo:.2f} C "
              f"(+{T_hypo - 37.0:.2f} C)")

    print(f"\n  To plot: plot_temperature_timecourse(temp)")
    print(f"  To compare doses: plot_dose_dependent_temperature(dose_results)")
    print(f"  To compare protocols: plot_stepup_vs_flat_temperature(comp)")
