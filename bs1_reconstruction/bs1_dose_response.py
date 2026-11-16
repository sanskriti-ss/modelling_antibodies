"""Reconstruct the BS1 dose sweep from project4.slx, not from digitized curves.
Run: python bs1_dose_response.py [--duration 14400] [--points 181]
Rates are copied from chart_32.xml. The chart calls them Glofitamab estimates;
they are not verified original BS1 parameters. Time is interpreted as seconds,
following the article's kinetic convention. Duration is an explicit assumption.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import argparse, csv, json
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from figure_theme import PALETTE as P, apply_theme, save_figure

@dataclass(frozen=True)
class Parameters:
    kon_6: float = 1e-4
    koff_6: float = 1.6e-4
    kon_8: float = 1e-4
    koff_8: float = 1e-2
    kon_8_then_6: float = 1e-2
    koff_8_then_6: float = 1.6e-4
    kon_6_then_8: float = 1e-2
    koff_6_then_8: float = 1e-2

def rhs(t, y, p):
    """States: free antibody, IL6R, IL8R, Ab-IL6R, Ab-IL8R, ternary.
    Reversible fluxes reproduce the source MATLAB Function block, alpha=1.
    """
    a,r6,r8,b6,b8,ternary=y
    v6=p.kon_6*a*r6-p.koff_6*b6
    v8=p.kon_8*a*r8-p.koff_8*b8
    v86=p.kon_8_then_6*b8*r6-p.koff_8_then_6*ternary
    v68=p.kon_6_then_8*b6*r8-p.koff_6_then_8*ternary
    return np.array([-v6-v8,-v6-v86,-v8-v68,v6-v68,v8-v86,v86+v68])

def totals(y):
    a,r6,r8,b6,b8,t=y
    return np.array([a+b6+b8+t, r6+b6+t, r8+b8+t])

def run_sweep(duration=14400, points=181):
    p=Parameters();doses=np.logspace(-3,4,points);rows=[];max_error=0.;min_state=0.
    for dose in doses:
        initial=np.array([dose,50.,50.,0.,0.,0.])
        sol=solve_ivp(rhs,(0,duration),initial,args=(p,),method='Radau',rtol=1e-8,atol=1e-10)
        if not sol.success: raise RuntimeError(sol.message)
        errors=np.max(np.abs(np.stack([totals(y) for y in sol.y.T])-totals(initial)),axis=0)
        max_error=max(max_error,float(max(errors)));min_state=min(min_state,float(sol.y.min()))
        if max(errors)>1e-5*max(1,dose) or sol.y.min() < -1e-7: raise AssertionError('Conservation or positivity failure')
        a,r6,r8,b6,b8,t=sol.y[:,-1]
        # Each ternary occupies two receptors; antibody-bound count is a different metric.
        rows.append([dose,a,r6,r8,b6,b8,t,b6+b8+2*t,b6+b8+t,float(np.linalg.norm(rhs(duration,sol.y[:,-1],p),ord=np.inf))])
    rows=np.array(rows); normalized=rows[:,6]/max(rows[:,6]);rows=np.column_stack([rows,normalized])
    # Independently verify selected doses with a second implicit solver.
    cross_error=0.
    for idx in [0, int(np.argmax(rows[:,6])),len(doses)-1]:
        initial=[doses[idx],50,50,0,0,0]
        other=solve_ivp(rhs,(0,duration),initial,args=(p,),method='BDF',rtol=1e-9,atol=1e-11)
        if not other.success:raise RuntimeError(other.message)
        cross_error=max(cross_error,float(np.max(np.abs(other.y[:,-1]-rows[idx,1:7]))))
    assert cross_error<1e-4
    return rows, {'parameters':asdict(p),'source':'simulink_model/project4.slx: simulink/stateflow/chart_32.xml','receptors':[50,50],'duration_seconds_assumed':duration,'points':points,'normalization':'ternary / maximum ternary across sweep','total_bound_definition':'b6 + b8 + 2*ternary (occupied receptors)','max_conservation_error':max_error,'minimum_state':min_state,'radau_bdf_max_absolute_difference':cross_error,'peak_dose_on_grid':float(doses[np.argmax(rows[:,6])]),'peak_ternary':float(max(rows[:,6])),'status':'Reconstruction; original sweep script and settings unavailable.'}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--duration',type=float,default=14400);parser.add_argument('--points',type=int,default=181);args=parser.parse_args()
    if args.duration<=0 or args.points<3:parser.error('Positive duration and at least three points required')
    out=Path(__file__).parent;rows,meta=run_sweep(args.duration,args.points)
    with (out/'bs1_dose_sweep.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['initial_bs1','free_bs1','free_il6r','free_il8r','bs1_il6r','bs1_il8r','ternary','total_bound_receptors','bound_antibody','final_derivative_inf_norm','normalized_ternary']);w.writerows(rows)
    (out/'calculation_details.json').write_text(json.dumps(meta,indent=2))
    apply_theme();fig,axes=plt.subplots(1,2,figsize=(11,4.2));fig.subplots_adjust(wspace=.3,bottom=.18,top=.97)
    axes[0].semilogx(rows[:,0],rows[:,10],color=P['PURPLE'],lw=2)
    axes[0].set_ylabel('Normalized final ternary complex')
    axes[0].set_ylim(0,1.05)
    axes[1].semilogx(rows[:,0],rows[:,7],color=P['BLUE'],lw=2,label='Total bound receptors')
    axes[1].semilogx(rows[:,0],rows[:,6],color=P['PURPLE'],lw=2,label='Ternary complex')
    axes[1].set_ylabel('Final amount (model units)');axes[1].set_ylim(0,105);axes[1].legend(frameon=False)
    for ax in axes:
        ax.set_xlabel('Initial BS1 dose (model units)');ax.grid(True,alpha=.5);ax.spines[['top','right']].set_visible(False)
    save_figure(fig,out/'bs1_dose_response_combined.png',dpi=200,bbox_inches='tight');plt.close(fig)
    print(json.dumps(meta,indent=2))
if __name__=='__main__':main()
