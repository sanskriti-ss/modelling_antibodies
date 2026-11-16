"""Local Figures 9/10: mass-action avidity equilibrium and independent ODE check.
Source: https://pmc.ncbi.nlm.nih.gov/articles/PMC4900278/ Table 1, Figure 3.
A closed antibody pool is used; the assay cell density is an explicit assumption.
Run: python generate_avidity_figures.py
"""
from pathlib import Path
import json,csv
import numpy as np
from scipy.optimize import brentq
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from figure_theme import PALETTE as P,apply_theme,save_figure
NA=6.02214076e23
KON=1e5/1e9 # nM^-1 s^-1
KOFF=1e-4
KD=KOFF/KON
EGFR=1e6
CELL_DENSITY=6.02214076e5 # cells/mL, assumed; sets EGFR pool to 1 nM
CELL_RADIUS=8e-6
REACH=125e-10
OUT=Path(__file__).parent

def receptor_pool(n):return n*CELL_DENSITY*1000/NA*1e9

def effective_pool(n):
    # surface density times hemisphere footprint, divided by hemisphere volume
    surface=4*np.pi*CELL_RADIUS**2
    accessible=n*np.pi*REACH**2/surface
    hemisphere=2*np.pi*REACH**3/3
    return accessible/(NA*hemisphere*1000)*1e9

def rhs(t,y,q):
    a,r1,r2,b1,b2,d=y
    v1=KON*a*r1-KOFF*b1;v2=KON*a*r2-KOFF*b2
    v12=KON*q*b1*r2-KOFF*d;v21=KON*q*b2*r1-KOFF*d
    return [-v1-v2,-v1-v21,-v2-v12,v1-v12,v2-v21,v12+v21]

def totals(y):
    a,r1,r2,b1,b2,d=y
    return np.array([a+b1+b2+d,r1+b1+d,r2+b2+d])

def equilibrium(a0,r10,r20,q):
    # For a prescribed free antibody a, free receptor pools satisfy
    # b1=a*r1/KD, b2=a*r2/KD, d=a*q*r1*r2/KD^2.
    # Solve the shared double-bound pool first; then close antibody mass balance.
    def state(a):
        f=1+a/KD;c=a*q/KD**2/f**2
        # d=c*(r10-d)*(r20-d), stable quadratic root
        bb=1+c*(r10+r20)
        d=2*c*r10*r20/(bb+np.sqrt(max(0,bb*bb-4*c*c*r10*r20)))
        r1=(r10-d)/f;r2=(r20-d)/f
        return np.array([a,r1,r2,a*r1/KD,a*r2/KD,d])
    loga=brentq(lambda x: totals(state(np.exp(x)))[0]-a0,np.log(a0)-60,np.log(a0),xtol=1e-12)
    return state(np.exp(loga))

def main():
    apply_theme(); doses=np.logspace(-3,3,91);ratios=[1,10,100];data=[];validation=[]
    for ratio in ratios:
        n2=EGFR/ratio;r10=receptor_pool(EGFR);r20=receptor_pool(n2);q=effective_pool(EGFR)/r10
        assert np.isclose(q,effective_pool(n2)/r20)
        eqs=[];odes=[];mass_error=0.;agreement=0.;max_time=0.
        for a0 in doses:
            e=equilibrium(a0,r10,r20,q);y=np.array([a0,r10,r20,0.,0.,0.]);t=0.
            # Numerical relaxation to stationary state, not an assumed assay endpoint.
            for end in [1e4,1e5,1e6,1e7,1e8,1e9]:
                sol=solve_ivp(rhs,(t,end),y,args=(q,),method='Radau',rtol=2e-9,atol=1e-12)
                if not sol.success:raise RuntimeError(sol.message)
                y=sol.y[:,-1];t=end
                mass_error=max(mass_error,float(np.max(np.abs(np.array([totals(v) for v in sol.y.T])-np.array([a0,r10,r20])))))
                if np.max(np.abs(np.asarray(rhs(t,y,q))))<1e-11:break
            if np.min(y)<-1e-8:raise AssertionError('Negative state')
            delta=float(np.max(np.abs(y-e)));agreement=max(agreement,delta);max_time=max(max_time,t)
            assert delta<2e-5, (ratio,a0,delta)
            eqs.append(e);odes.append(y)
        eqs=np.array(eqs);odes=np.array(odes)
        # Antibody signal counts antibody molecules, not occupied receptors.
        for method,states in [('algebraic',eqs),('ode',odes)]:
            bound=states[:,3]+states[:,4]+states[:,5]
            normalized=bound/bound[-1] # highest sampled dose, as for a normalized binding signal
            for a0,y,b in zip(doses,states,normalized):data.append([ratio,method,a0*1e-9,*y,b,y[1]/r10,y[2]/r20])
        validation.append({'ratio':ratio,'max_ode_vs_equilibrium_nM':agreement,'max_conservation_error_nM':mass_error,'max_relaxation_seconds':max_time,'egfr_effective_nM':effective_pool(EGFR),'met_effective_nM':effective_pool(n2)})
        if ratio==1:figs=[plt.subplots(1,3,figsize=(13,3.6),sharey=True) for _ in range(2)]
        for j,states in enumerate([eqs,odes]):
            fig,axs=figs[j];ax=axs[ratios.index(ratio)];bound=states[:,3]+states[:,4]+states[:,5];b=bound/bound[-1]
            for vals,color,ls,label in [(b,P['BLUE'],'-','Bound antibody'),(states[:,1]/r10,P['ORANGE'],'--','Unbound EGFR'),(states[:,2]/r20,P['TEAL'],':','Unbound MET')]:
                ax.semilogx(doses*1e-9,vals,color=color,ls=ls,lw=2,label=label)
            ax.set_ylim(-.02,1.03);ax.set_xlabel('Initial antibody concentration (M)');ax.grid(True,alpha=.5);ax.spines[['top','right']].set_visible(False)
            if ratio==1:ax.set_ylabel('Normalized bound / free fraction');ax.legend(fontsize=8,frameon=False)
    for i,(fig,axs) in enumerate(figs):
        fig.subplots_adjust(left=.06,right=.99,bottom=.18,top=.97,wspace=.13)
        save_figure(fig,OUT/(['avidity_equilibrium.png','avidity_ode.png'][i]),dpi=200,bbox_inches='tight');plt.close(fig)
    with (OUT/'avidity_calculations.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['egfr_met_ratio','method','initial_antibody_M','free_antibody_nM','free_egfr_nM','free_met_nM','ab_egfr_nM','ab_met_nM','double_bound_nM','normalized_bound_antibody','unbound_egfr_fraction','unbound_met_fraction']);w.writerows(data)
    meta={'source':'https://pmc.ncbi.nlm.nih.gov/articles/PMC4900278/','published_inputs':{'kon_M_inverse_s':1e5,'koff_s_inverse':KOFF,'egfr_per_cell':EGFR,'met_per_cell':[1e6,1e5,1e4],'cell_radius_m':CELL_RADIUS,'igg_reach_m':REACH},'assumptions':{'cell_density_per_mL':CELL_DENSITY,'antibody_pool':'closed; depletion included','endpoint':'stationary state; ODE relaxation until derivative <1e-11 nM/s','antibody_normalization':'bound antibody divided by bound antibody at highest sampled dose'},'validation':validation,'status':'Local reimplementation; original scripts and complete original simulation settings unavailable.'}
    (OUT/'parameters_and_validation.json').write_text(json.dumps(meta,indent=2));print(json.dumps(validation,indent=2))
if __name__=='__main__':main()
