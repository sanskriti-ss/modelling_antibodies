"""Audit Figures 11 and 13 against appendix-driven mass-action simulations.
This is a reconstruction, not recovered original plotting code.
"""
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from figure_theme import PALETTE as P,apply_theme,save_figure
import json,csv
OUT=Path(__file__).parent
WASHOUT=7200.

def bs1_flux(y,drug):
    b6,b8,t=y;r6=3.16e5-b6-t;r8=6.18e5-b8-t
    return np.array([5.92e-6*drug*r6-5.61e-5*b6,9.03e-6*drug*r8-6.38e-5*b8,8.11e-8*b8*r6-5.61e-5*t,1.24e-7*b6*r8-6.38e-5*t])
def bs1_rhs(t,y,drug):
    v6,v8,v86,v68=bs1_flux(y,drug)
    return np.array([v6-v68,v8-v86,v86+v68])
def glo_flux(y,drug):
    b,bb,t,s,l=y;rb=2e5-b-2*bb-s-2*l;rt=1e5-t-s-l
    return np.array([1e-5*drug*rb-1e-5*b,8.3e-9*rb*b-1e-5*bb,1e-5*drug*rt-3e-4*t,8.3e-10*rt*b-3e-4*s,8.3e-9*rb*t-1e-5*s,8.3e-10*rt*bb-3e-4*l,8.3e-9*rb*s-1e-5*l])
def glo_rhs(t,y,drug):
    v1,v2,v3,v4,v5,v6,v7=glo_flux(y,drug)
    return np.array([v1-v2-v4,v2-v6,v3-v5,v4+v5-v7,v6+v7])
def simulate(rhs,n,method='Radau'):
    chunks=[];y=np.zeros(n)
    for start,end,drug in [(0,WASHOUT,100.),(WASHOUT,14400.,0.)]:
        times=np.linspace(start,end,501)
        sol=solve_ivp(rhs,(start,end),y,args=(drug,),method=method,t_eval=times,rtol=1e-9,atol=1e-7)
        if not sol.success:raise RuntimeError(sol.message)
        chunks.append((sol.t,sol.y.T));y=sol.y[:,-1]
    # Keep duplicate washout time: concentration continuous; flux has a jump.
    return np.concatenate([a for a,b in chunks]),np.concatenate([b for a,b in chunks])
def main():
    apply_theme();results={}
    specs=[('bs1',bs1_rhs,bs1_flux,3,['BS1–IL6R','BS1–IL8R','Ternary'],[P['ORANGE'],P['BLUE'],P['PURPLE']]),('glofitamab',glo_rhs,glo_flux,5,['CD20–Ab','CD20–Ab–CD20','CD3–Ab','CD20–Ab–CD3','Locked synapse'],[P['ORANGE'],P['BLUE'],P['RED'],P['TEAL'],P['PURPLE']])]
    for name,rhs,flux,n,labels,colors in specs:
        t,y=simulate(rhs,n);tb,yb=simulate(rhs,n,'BDF');assert np.max(abs(y-yb))<.05
        if name=='bs1':
            free=np.column_stack([3.16e5-y[:,0]-y[:,2],6.18e5-y[:,1]-y[:,2]])
        else:
            free=np.column_stack([2e5-y[:,0]-2*y[:,1]-y[:,3]-2*y[:,4],1e5-y[:,2]-y[:,3]-y[:,4]])
        assert np.min(y)>-1e-5 and np.min(free)>-1e-5
        # derivative evaluates the ODE directly, not a noisy numerical gradient
        derivative=np.array([rhs(ti,yi,100. if i<=500 else 0.) for i,(ti,yi) in enumerate(zip(t,y))])
        with (OUT/(name+'_timecourse.csv')).open('w') as f:
            w=csv.writer(f);w.writerow(['time_seconds',*labels,*['net_rate_'+l for l in labels]]);w.writerows(np.column_stack([t,y,derivative]))
        reaction_labels = (['IL6R binding', 'IL8R binding', 'Cross-link IL6R', 'Cross-link IL8R'] if name=='bs1' else ['CD20 binding','CD20 anchoring','CD3 binding','CD3 recruitment','CD20 recruitment','Anchor + CD3','Synapse locking'])
        reaction_colors = [P[k] for k in ['BLUE','ORANGE','TEAL','PURPLE','RED','GOLD','GRAY']]
        fluxes=np.array([flux(yi,100. if i<=500 else 0.) for i,yi in enumerate(y)])
        stoich = (np.array([[1,0,0,-1],[0,1,-1,0],[0,0,1,1]]) if name=='bs1' else np.array([[1,-1,0,-1,0,0,0],[0,1,0,0,0,-1,0],[0,0,1,0,-1,0,0],[0,0,0,1,1,0,-1],[0,0,0,0,0,1,1]]))
        assert np.allclose(fluxes @ stoich.T, derivative)
        with (OUT/(name+'_reaction_fluxes.csv')).open('w') as f:
            w=csv.writer(f);w.writerow(['time_seconds',*reaction_labels]);w.writerows(np.column_stack([t,fluxes]))
        # Export separate assets; LaTeX supplies the panel labels and captions.
        for panel in ['concentration','flux']:
            fig,ax=plt.subplots(figsize=(6,3.6));fig.subplots_adjust(left=.15,right=.98,bottom=.18,top=.97)
            if panel=='concentration':
                for j,(label,color) in enumerate(zip(labels,colors)):ax.plot(t/3600,y[:,j],color=color,label=label,lw=1.7)
                ax.set_ylabel('Complex amount (#/cell)')
            else:
                for j,(label,color) in enumerate(zip(reaction_labels,reaction_colors)):ax.plot(t/3600,fluxes[:,j],color=color,label=label,lw=1.5,ls='--' if j>=5 else '-')
                ax.set_yscale('symlog',linthresh=.1);ax.set_ylabel('Net reaction flux (#/cell/s)');ax.axhline(0,color=P['GRAY'],lw=.8)
            ax.axvline(2,color=P['GRAY'],ls='--',lw=1);ax.set_xlabel('Time (hours)');ax.grid(alpha=.5);ax.spines[['top','right']].set_visible(False)
            ax.legend(fontsize=7,frameon=False,loc='upper right',ncol=2 if panel=='flux' else 1)
            save_figure(fig,OUT/(name+'_'+panel+'.png'),dpi=200,bbox_inches='tight');plt.close(fig)
        results[name]={'at_washout':y[500].tolist(),'at_four_hours':y[-1].tolist(),'solver_difference':float(np.max(abs(y-yb)))}
        # Full-size comparison source available in report; no report replacement yet.
    results['provenance']={'source':'Article-linked Google document, Tables 1 and 2','drug_nM':100,'bs1_dose_status':'inferred assumption; screenshot comparison required','glofitamab_B_syn_on':'8.3e-9 (#/cell)^-1 s^-1; CD20 cross-linking rate from Table 2, assigned to B_syn' ,'glofitamab_B_syn_off':'1e-5 s^-1; inferred CD20 off-rate','washout':'clamped free bath changes 100 to 0 nM at 2 h; dissociated antibody removed','flux_panel':'individual net reaction flux = forward association minus reverse dissociation; not species dC/dt'}
    (OUT/'audit_results.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
if __name__=='__main__':main()
