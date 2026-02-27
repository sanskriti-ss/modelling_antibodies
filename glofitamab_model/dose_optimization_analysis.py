"""
Glofitamab Dose Optimization Analysis
======================================

Goal: Explore dosing strategies that reduce CRS and fatigue while
maintaining therapeutic efficacy (RCAB exposure).

Key levers to explore:
  1. First dose level (step-up entry dose)
  2. Maintenance dose level
  3. Dosing interval (Q3W vs Q4W vs Q5W)
  4. Number of cycles
  5. Step-up ramp speed (1-step vs 2-step vs 3-step)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import warnings
import os

import sys
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from glofitamab_pk_tmdd import (
        glofitamab_odes, mg_to_nM, nM_to_ug_per_mL,
        PK_PARAMS, compute_steady_state_receptors,
    )
    from glofitamab_crs_model import CRSModel, CRS_PARAMS
    from glofitamab_fatigue_model import FatigueModel, FATIGUE_PARAMS
except ImportError:
    from glofitamab_model.glofitamab_pk_tmdd import (
        glofitamab_odes, mg_to_nM, nM_to_ug_per_mL,
        PK_PARAMS, compute_steady_state_receptors,
    )
    from glofitamab_model.glofitamab_crs_model import CRSModel, CRS_PARAMS
    from glofitamab_model.glofitamab_fatigue_model import FatigueModel, FATIGUE_PARAMS

OUTDIR = os.path.dirname(__file__)


# ──────────────────────────────────────────────────
#  Generic protocol simulator
# ──────────────────────────────────────────────────

def simulate_protocol(
    doses: list,
    params: dict = None,
    obinutuzumab: bool = True,
    t_end_extra: float = 21.0,
) -> dict:
    """
    Simulate an arbitrary dosing protocol.

    Parameters
    ----------
    doses : list of (day, dose_mg) tuples
    params : PK parameters
    obinutuzumab : whether obinutuzumab pretreatment is given
    t_end_extra : days of simulation after last dose

    Returns
    -------
    Dict with t_days, C_nM, RCAB_nM, C_ug_mL, RA_nM, RB_nM, etc.
    """
    if params is None:
        params = PK_PARAMS.copy()

    ss = compute_steady_state_receptors(params)
    RA0 = ss['RA0'] * (0.1 if obinutuzumab else 1.0)
    RB0 = ss['RB0']

    y_current = np.array([0.0, RA0, RB0, 0.0, 0.0, 0.0, 0.0])
    t_current = 0.0
    t_end = doses[-1][0] + t_end_extra

    all_t, all_y = [], []

    for dose_day, dose_mg in doses:
        if dose_day > t_current:
            t_seg = np.linspace(t_current, dose_day, max(int((dose_day - t_current) * 5), 20))
            sol = solve_ivp(
                glofitamab_odes, (t_current, dose_day), y_current,
                args=(params,), method='Radau', t_eval=t_seg,
                rtol=1e-8, atol=1e-10, max_step=0.5,
            )
            if sol.success:
                all_t.append(sol.t)
                all_y.append(sol.y)
                y_current = sol.y[:, -1].copy()

        y_current[0] += mg_to_nM(dose_mg, params['Vc'], params['MW'])
        t_current = dose_day

    # Final washout
    t_seg = np.linspace(t_current, t_end, 80)
    sol = solve_ivp(
        glofitamab_odes, (t_current, t_end), y_current,
        args=(params,), method='Radau', t_eval=t_seg,
        rtol=1e-8, atol=1e-10, max_step=0.5,
    )
    if sol.success:
        all_t.append(sol.t)
        all_y.append(sol.y)

    t_full = np.concatenate(all_t)
    y_full = np.concatenate(all_y, axis=1)
    MW = params['MW']

    return {
        't_days': t_full,
        'C_nM': y_full[0],
        'RA_nM': y_full[1],
        'RB_nM': y_full[2],
        'RCA_nM': y_full[3],
        'RCB_nM': y_full[4],
        'RCAB_nM': y_full[5],
        'AP_nM': y_full[6],
        'C_ug_mL': np.array([nM_to_ug_per_mL(c, MW) for c in y_full[0]]),
        'dose_times': [d[0] for d in doses],
        'dose_amounts': [d[1] for d in doses],
    }


def compute_efficacy_and_safety(pk_result: dict, crs_params=None, fatigue_params=None):
    """
    Compute efficacy and safety metrics from a PK simulation.

    Efficacy: cumulative RCAB exposure (AUC) — drives tumor killing
    Safety: peak RCAB (CRS driver), CRS probability, fatigue score
    """
    t = pk_result['t_days']
    rcab = pk_result['RCAB_nM']

    # Efficacy: RCAB AUC (trapezoidal)
    rcab_auc = np.trapezoid(rcab, t)

    # First-dose peak RCAB (CRS driver)
    if len(pk_result['dose_times']) >= 2:
        first_dose_end = pk_result['dose_times'][1]
    else:
        first_dose_end = t[-1]
    mask_d1 = t <= first_dose_end
    rcab_peak_d1 = np.max(rcab[mask_d1]) if np.any(mask_d1) else np.max(rcab)

    # Overall peak RCAB
    rcab_peak_overall = np.max(rcab)

    # CRS model
    crs_model = CRSModel(crs_params)

    def rcab_func(ti):
        return max(np.interp(ti, t, rcab), 0.0)

    p_crs = crs_model.params
    IL6_baseline = p_crs['ksyn_IL6'] / p_crs['kdeg_IL6']
    TNF_baseline = p_crs['ksyn_TNF'] / p_crs['kdeg_TNF']

    # Simulate CRS for first 7 days only (first-dose CRS)
    t_crs_end = min(first_dose_end, 7.0)
    t_crs_eval = np.linspace(0, t_crs_end, 200)
    y0_crs = np.array([0.0, IL6_baseline, TNF_baseline])

    sol_crs = solve_ivp(
        crs_model._crs_odes, (0, t_crs_end), y0_crs,
        args=(rcab_func,), method='RK45', t_eval=t_crs_eval,
        rtol=1e-6, atol=1e-8,
    )

    il6_peak = np.max(sol_crs.y[1]) if sol_crs.success else IL6_baseline
    activation_peak = np.max(sol_crs.y[0]) if sol_crs.success else 0.0

    # CRS probability
    crs_prob = crs_model.predict_crs_probability(rcab_peak_d1, dose_number=1)

    # Fatigue: simulate over full protocol
    fatigue_model = FatigueModel(fatigue_params)

    # Full CRS timecourse for fatigue driver
    t_full_end = t[-1]
    t_full_eval = np.linspace(0, t_full_end, 500)
    y0_full = np.array([0.0, IL6_baseline, TNF_baseline])

    sol_full = solve_ivp(
        crs_model._crs_odes, (0, t_full_end), y0_full,
        args=(rcab_func,), method='RK45', t_eval=t_full_eval,
        rtol=1e-6, atol=1e-8,
    )

    if sol_full.success:
        act_data = sol_full.y[0]
        il6_data = sol_full.y[1]
        t_crs_full = sol_full.t

        def act_func(ti):
            return max(np.interp(ti, t_crs_full, act_data), 0.0)
        def il6_func(ti):
            return max(np.interp(ti, t_crs_full, il6_data), 0.0)

        sol_fat = solve_ivp(
            fatigue_model._fatigue_ode, (0, t_full_end), np.array([0.0]),
            args=(act_func, il6_func, IL6_baseline),
            method='RK45', t_eval=t_full_eval, rtol=1e-6, atol=1e-8,
        )
        peak_fatigue = np.max(sol_fat.y[0]) if sol_fat.success else 0.0
        fatigue_grade = fatigue_model.classify_grade(peak_fatigue)
    else:
        peak_fatigue = 0.0
        fatigue_grade = 0

    return {
        'rcab_auc': rcab_auc,
        'rcab_peak_d1': rcab_peak_d1,
        'rcab_peak_overall': rcab_peak_overall,
        'il6_peak': il6_peak,
        'il6_fold': il6_peak / IL6_baseline,
        'activation_peak': activation_peak,
        'crs_any_grade': crs_prob['prob_any_grade'],
        'crs_grade_ge2': crs_prob['prob_grade_ge2'],
        'crs_grade_ge3': crs_prob['prob_grade_ge3'],
        'peak_fatigue': peak_fatigue,
        'fatigue_grade': fatigue_grade,
    }


# ──────────────────────────────────────────────────
#  ANALYSIS 1: First-dose level sweep
# ──────────────────────────────────────────────────

def analysis_first_dose_sweep():
    """What first dose minimizes CRS while preserving RCAB AUC?"""
    print("\n=== ANALYSIS 1: First Dose Level Sweep ===")

    first_doses = [0.5, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0]
    results = []

    for fd in first_doses:
        # Protocol: first dose -> 10mg -> 30mg Q3W x 10
        doses = [(7, fd), (14, 10.0)]
        for c in range(10):
            doses.append((28 + c * 21, 30.0))

        pk = simulate_protocol(doses)
        metrics = compute_efficacy_and_safety(pk)
        metrics['first_dose'] = fd
        results.append(metrics)

        print(f"  {fd:5.1f} mg: RCAB_AUC={metrics['rcab_auc']:8.1f}  "
              f"CRS={metrics['crs_any_grade']:5.1%}  "
              f"IL6_fold={metrics['il6_fold']:6.1f}x  "
              f"Fatigue={metrics['peak_fatigue']:.2f}")

    return results


# ──────────────────────────────────────────────────
#  ANALYSIS 2: Maintenance dose sweep
# ──────────────────────────────────────────────────

def analysis_maintenance_dose_sweep():
    """What maintenance dose balances efficacy vs fatigue?"""
    print("\n=== ANALYSIS 2: Maintenance Dose Sweep ===")

    maint_doses = [10.0, 15.0, 20.0, 25.0, 30.0, 40.0]
    results = []

    for md in maint_doses:
        doses = [(7, 2.5), (14, min(10.0, md))]
        for c in range(10):
            doses.append((28 + c * 21, md))

        pk = simulate_protocol(doses)
        metrics = compute_efficacy_and_safety(pk)
        metrics['maint_dose'] = md
        results.append(metrics)

        print(f"  {md:5.1f} mg: RCAB_AUC={metrics['rcab_auc']:8.1f}  "
              f"CRS={metrics['crs_any_grade']:5.1%}  "
              f"Fatigue={metrics['peak_fatigue']:.2f} (Gr{metrics['fatigue_grade']})")

    return results


# ──────────────────────────────────────────────────
#  ANALYSIS 3: Dosing interval sweep
# ──────────────────────────────────────────────────

def analysis_interval_sweep():
    """Does extending the interval (Q3W->Q4W->Q5W) reduce fatigue?"""
    print("\n=== ANALYSIS 3: Dosing Interval Sweep ===")

    intervals = [14, 21, 28, 35, 42]
    results = []

    for interval in intervals:
        doses = [(7, 2.5), (14, 10.0)]
        for c in range(10):
            doses.append((28 + c * interval, 30.0))

        pk = simulate_protocol(doses)
        metrics = compute_efficacy_and_safety(pk)
        metrics['interval'] = interval
        results.append(metrics)

        label = f"Q{interval//7}W"
        print(f"  {label}: RCAB_AUC={metrics['rcab_auc']:8.1f}  "
              f"CRS={metrics['crs_any_grade']:5.1%}  "
              f"Fatigue={metrics['peak_fatigue']:.2f} (Gr{metrics['fatigue_grade']})")

    return results


# ──────────────────────────────────────────────────
#  ANALYSIS 4: Number of cycles
# ──────────────────────────────────────────────────

def analysis_cycle_count():
    """What's the minimum number of cycles needed?"""
    print("\n=== ANALYSIS 4: Number of Treatment Cycles ===")

    n_cycles_list = [4, 6, 8, 10, 12]
    results = []

    for nc in n_cycles_list:
        doses = [(7, 2.5), (14, 10.0)]
        for c in range(nc - 2):
            doses.append((28 + c * 21, 30.0))

        pk = simulate_protocol(doses)
        metrics = compute_efficacy_and_safety(pk)
        metrics['n_cycles'] = nc
        results.append(metrics)

        print(f"  {nc:2d} cycles: RCAB_AUC={metrics['rcab_auc']:8.1f}  "
              f"Fatigue={metrics['peak_fatigue']:.2f} (Gr{metrics['fatigue_grade']})  "
              f"Duration={doses[-1][0]+21:.0f} days")

    return results


# ──────────────────────────────────────────────────
#  ANALYSIS 5: Step-up ramp design
# ──────────────────────────────────────────────────

def analysis_stepup_ramp():
    """Compare different step-up ramp strategies."""
    print("\n=== ANALYSIS 5: Step-Up Ramp Design ===")

    ramps = {
        'No step-up (flat 30mg)':      [(7, 30.0)],
        'Current (2.5->10->30)':       [(7, 2.5), (14, 10.0)],
        'Slower (1->2.5->10->30)':     [(7, 1.0), (14, 2.5), (21, 10.0)],
        'Gentler (2.5->5->10->30)':    [(7, 2.5), (14, 5.0), (21, 10.0)],
        'Aggressive (10->30)':         [(7, 10.0)],
    }

    results = {}
    for name, ramp in ramps.items():
        doses = list(ramp)
        # Add maintenance: 30mg Q3W x 10 starting at day 28
        maint_start = max(d[0] for d in ramp) + 14
        for c in range(10):
            doses.append((maint_start + c * 21, 30.0))

        pk = simulate_protocol(doses)
        metrics = compute_efficacy_and_safety(pk)
        metrics['name'] = name
        metrics['ramp_doses'] = ramp
        results[name] = metrics

        print(f"  {name:35s}: CRS={metrics['crs_any_grade']:5.1%}  "
              f"Gr>=2={metrics['crs_grade_ge2']:5.1%}  "
              f"IL6={metrics['il6_fold']:5.1f}x  "
              f"RCAB_AUC={metrics['rcab_auc']:8.1f}")

    return results


# ──────────────────────────────────────────────────
#  PLOTTING
# ──────────────────────────────────────────────────

def plot_all_analyses(r1, r2, r3, r4, r5):
    """Generate the main optimization figure."""

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # ── Panel A: First dose vs CRS and efficacy ──
    ax = axes[0, 0]
    fds = [r['first_dose'] for r in r1]
    crs_probs = [r['crs_any_grade'] * 100 for r in r1]
    il6_folds = [r['il6_fold'] for r in r1]
    rcab_aucs = [r['rcab_auc'] for r in r1]

    ax.semilogx(fds, crs_probs, 'ro-', lw=2, markersize=7, label='CRS prob (%)')
    ax.set_xlabel('First Dose (mg)')
    ax.set_ylabel('CRS Probability (%)', color='r')
    ax.tick_params(axis='y', labelcolor='r')
    ax.set_ylim(0, 100)

    ax2 = ax.twinx()
    # Normalize RCAB AUC to % of max
    max_auc = max(rcab_aucs)
    rcab_pct = [a / max_auc * 100 for a in rcab_aucs]
    ax2.semilogx(fds, rcab_pct, 'b^--', lw=1.5, markersize=6, label='Efficacy (% max)')
    ax2.set_ylabel('RCAB AUC (% of max)', color='b')
    ax2.tick_params(axis='y', labelcolor='b')
    ax2.set_ylim(50, 105)

    ax.set_title('A. First Dose: CRS vs Efficacy')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='center right')
    ax.grid(True, alpha=0.3)
    ax.axvline(2.5, color='green', ls=':', alpha=0.5)
    ax.annotate('Current\n(2.5mg)', xy=(2.5, 5), fontsize=8, color='green', ha='center')

    # ── Panel B: Maintenance dose tradeoff ──
    ax = axes[0, 1]
    mds = [r['maint_dose'] for r in r2]
    fat_scores = [r['peak_fatigue'] for r in r2]
    rcab_aucs_m = [r['rcab_auc'] for r in r2]
    max_auc_m = max(rcab_aucs_m)

    ax.plot(mds, fat_scores, 'purple', marker='s', lw=2, markersize=7, label='Peak Fatigue')
    ax.set_xlabel('Maintenance Dose (mg)')
    ax.set_ylabel('Peak Fatigue Score', color='purple')
    ax.tick_params(axis='y', labelcolor='purple')
    ax.axhline(FATIGUE_PARAMS['grade2_threshold'], color='orange', ls='--', alpha=0.5,
               label='Grade 2 threshold')

    ax2 = ax.twinx()
    rcab_pct_m = [a / max_auc_m * 100 for a in rcab_aucs_m]
    ax2.plot(mds, rcab_pct_m, 'b^--', lw=1.5, markersize=6, label='Efficacy')
    ax2.set_ylabel('RCAB AUC (% of max)', color='b')
    ax2.tick_params(axis='y', labelcolor='b')
    ax2.set_ylim(0, 105)

    ax.set_title('B. Maintenance Dose: Fatigue vs Efficacy')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='center left')
    ax.grid(True, alpha=0.3)
    ax.axvline(30, color='green', ls=':', alpha=0.5)
    ax.annotate('Current\n(30mg)', xy=(30, ax.get_ylim()[0]), fontsize=8,
                color='green', ha='center')

    # ── Panel C: Dosing interval ──
    ax = axes[0, 2]
    intervals = [r['interval'] for r in r3]
    week_labels = [f"Q{i//7}W" for i in intervals]
    fat_int = [r['peak_fatigue'] for r in r3]
    rcab_int = [r['rcab_auc'] for r in r3]
    max_auc_i = max(rcab_int)

    x = np.arange(len(intervals))
    w = 0.35
    bars_fat = ax.bar(x - w/2, fat_int, w, color='plum', label='Peak Fatigue')
    ax.set_ylabel('Peak Fatigue Score')

    ax2 = ax.twinx()
    rcab_pct_i = [a / max_auc_i * 100 for a in rcab_int]
    bars_eff = ax2.bar(x + w/2, rcab_pct_i, w, color='steelblue', alpha=0.7,
                        label='Efficacy (%)')
    ax2.set_ylabel('RCAB AUC (% of max)', color='steelblue')
    ax2.set_ylim(0, 120)

    ax.set_xticks(x)
    ax.set_xticklabels(week_labels)
    ax.set_xlabel('Dosing Interval')
    ax.set_title('C. Dosing Interval: Fatigue vs Efficacy')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # ── Panel D: Number of cycles ──
    ax = axes[1, 0]
    ncs = [r['n_cycles'] for r in r4]
    fat_nc = [r['peak_fatigue'] for r in r4]
    rcab_nc = [r['rcab_auc'] for r in r4]
    max_auc_nc = max(rcab_nc)

    ax.plot(ncs, fat_nc, 'purple', marker='s', lw=2, markersize=7, label='Peak Fatigue')
    ax.set_xlabel('Number of Cycles')
    ax.set_ylabel('Peak Fatigue Score', color='purple')
    ax.tick_params(axis='y', labelcolor='purple')

    ax2 = ax.twinx()
    rcab_pct_nc = [a / max_auc_nc * 100 for a in rcab_nc]
    ax2.plot(ncs, rcab_pct_nc, 'b^--', lw=1.5, markersize=6, label='Efficacy')
    ax2.set_ylabel('RCAB AUC (% of max)', color='b')
    ax2.tick_params(axis='y', labelcolor='b')

    ax.set_title('D. Number of Cycles: Fatigue vs Efficacy')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel E: Step-up ramp comparison ──
    ax = axes[1, 1]
    names = list(r5.keys())
    short_names = ['Flat 30', 'Current\n2.5→10→30', 'Slower\n1→2.5→10→30',
                   'Gentler\n2.5→5→10→30', 'Aggressive\n10→30']
    crs_vals = [r5[n]['crs_any_grade'] * 100 for n in names]
    g2_vals = [r5[n]['crs_grade_ge2'] * 100 for n in names]

    x = np.arange(len(names))
    w = 0.35
    ax.bar(x - w/2, crs_vals, w, color='indianred', alpha=0.8, label='Any CRS')
    ax.bar(x + w/2, g2_vals, w, color='darkred', alpha=0.8, label='Grade >=2')
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=8)
    ax.set_ylabel('CRS Probability (%)')
    ax.set_title('E. Step-Up Ramp: CRS Risk')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # ── Panel F: Optimal protocol summary ──
    ax = axes[1, 2]
    ax.axis('off')

    # Find optimal from each analysis
    best_fd = min(r1, key=lambda r: r['crs_any_grade']
                  if r['rcab_auc'] > max(rr['rcab_auc'] for rr in r1) * 0.9 else 999)
    best_md = min(r2, key=lambda r: r['peak_fatigue']
                  if r['rcab_auc'] > max(rr['rcab_auc'] for rr in r2) * 0.8 else 999)
    best_int = min(r3, key=lambda r: r['peak_fatigue']
                   if r['rcab_auc'] > max(rr['rcab_auc'] for rr in r3) * 0.75 else 999)
    best_nc = min(r4, key=lambda r: r['peak_fatigue']
                  if r['rcab_auc'] > max(rr['rcab_auc'] for rr in r4) * 0.7 else 999)
    best_ramp = min(r5.values(), key=lambda r: r['crs_any_grade']
                    if r['rcab_auc'] > max(v['rcab_auc'] for v in r5.values()) * 0.9 else 999)

    # Current protocol metrics
    curr = [r for r in r1 if r['first_dose'] == 2.5][0]

    text = (
        "DOSE OPTIMIZATION RECOMMENDATIONS\n"
        "══════════════════════════════════\n\n"
        f"CURRENT PROTOCOL:\n"
        f"  2.5→10→30mg Q3W x12\n"
        f"  CRS: {curr['crs_any_grade']:.0%} | Fatigue: {curr['peak_fatigue']:.2f}\n\n"
        f"OPTIMIZED FIRST DOSE:\n"
        f"  {best_fd['first_dose']:.1f} mg (was 2.5mg)\n"
        f"  CRS: {best_fd['crs_any_grade']:.0%}\n"
        f"  Efficacy retained: {best_fd['rcab_auc']/max(rr['rcab_auc'] for rr in r1)*100:.0f}%\n\n"
        f"OPTIMIZED MAINTENANCE:\n"
        f"  {best_md['maint_dose']:.0f} mg (was 30mg)\n"
        f"  Fatigue: {best_md['peak_fatigue']:.2f} (Gr{best_md['fatigue_grade']})\n\n"
        f"OPTIMIZED INTERVAL:\n"
        f"  Q{best_int['interval']//7}W (was Q3W)\n"
        f"  Fatigue: {best_int['peak_fatigue']:.2f}\n\n"
        f"BEST RAMP:\n"
        f"  {best_ramp['name']}\n"
        f"  CRS: {best_ramp['crs_any_grade']:.0%} | "
        f"Gr>=2: {best_ramp['crs_grade_ge2']:.0%}"
    )
    ax.text(0.05, 0.95, text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(facecolor='honeydew', edgecolor='darkgreen', alpha=0.9))

    plt.suptitle('Glofitamab Dose Optimization: Reducing CRS and Fatigue\n'
                 'While Maintaining Anti-Tumor Efficacy (RCAB Exposure)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'plot9_dose_optimization.png'),
                dpi=200, bbox_inches='tight')
    print("\n  Saved plot9_dose_optimization.png")


# ──────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Glofitamab Dose Optimization Analysis")
    print("Goal: Reduce CRS and fatigue while preserving efficacy")
    print("=" * 60)

    r1 = analysis_first_dose_sweep()
    r2 = analysis_maintenance_dose_sweep()
    r3 = analysis_interval_sweep()
    r4 = analysis_cycle_count()
    r5 = analysis_stepup_ramp()

    plot_all_analyses(r1, r2, r3, r4, r5)

    print("\n" + "=" * 60)
    print("Analysis complete.")
    print("=" * 60)
