BS1 binding model and reconstructed dose-response figures

Run from this folder:
  python bs1_dose_response.py
Requires Python, NumPy, SciPy and Matplotlib.
Optional settings: --duration 14400 --points 181

Source: modelling_antibodies/simulink_model/project4.slx,
simulink/stateflow/chart_32.xml (binding_model). The rate values and alpha=1
are copied exactly. The source block calls these 'Glofitamab Estimates',
despite BS1/IL6R/IL8R state names. These are not verified paper BS1 rates.
The 50/50 initial receptor pools and dose range 1e-3 to 1e4 follow the article
and screenshots. The four-hour duration (14400 seconds) is an assumption,
informed by the article's time-course section, not recovered sweep metadata.
The screenshot's original script and parameter settings were unavailable.
Do not interpret this reconstruction as a quantitative reproduction.

Six states: free antibody A, free receptors R6 and R8, binary complexes B6 and
B8, and ternary complex T. Initially all complexes are zero. There is no
washout, clearance, receptor turnover, or replenishment in this dose sweep.
The four reversible binding fluxes are explicit in rhs().
Antibody total = A+B6+B8+T; R6 total = R6+B6+T; R8 total = R8+B8+T.
'Final' means the finite simulation endpoint, not necessarily equilibrium.
The final derivative norm is exported to avoid treating it as equilibrium.
Total bound receptors = B6+B8+2T, bounded by 100. Bound antibody = B6+B8+T.
Normalized ternary = T/max(T) over this dose grid; this is not an absolute
binding fraction and peak location depends on the grid and model settings.

Outputs:
  bs1_dose_response_combined.png / .svg: two panels replacing Figures 12/13
  bs1_dose_sweep.csv: all final states, derived quantities and derivative norm
  calculation_details.json: parameters, assumptions and verification results

Validation: all three conserved totals checked throughout every trajectory;
nonnegative states checked; independent Radau and BDF solutions compared at
low dose, peak dose, and high dose. Numerical agreement verifies implementation,
not the scientific validity of assumed parameters. No fitting to screenshots.

Style: same shared palette as the other figures; no figure/panel titles;
axis labels and legend retained. Reproduction details belong in the caption.
