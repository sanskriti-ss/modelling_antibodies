Local generators for report Figures 9 and 10

Run: python generate_avidity_figures.py
Dependencies: Python, NumPy, SciPy, Matplotlib.

Source: Rhoden et al. (2016), Table 1 and Figure 3:
https://pmc.ncbi.nlm.nih.gov/articles/PMC4900278/
Published inputs: association 1e5 M^-1 s^-1, dissociation 1e-4 s^-1,
EGFR 1e6/cell, MET 1e6/1e5/1e4 per cell, cell radius 8 micrometers,
and antibody reach 125 angstroms. There are no fitted parameters.

Assumption: 602214.076 cells/mL, yielding an EGFR solution-phase pool of
1 nM. This density was not recovered from the original simulation. Initial
antibody concentrations are 1e-12 to 1e-6 M. The antibody pool is closed,
so ligand depletion is included. Receptors do not turn over.

Effective receptor concentration is calculated from receptors under a
circular footprint divided by the accessible hemisphere volume. Surface
area=4*pi*cell_radius^2; footprint=pi*reach^2; volume=2*pi*reach^3/3.
All calculations use nM internally. Six states track free antibody,
free EGFR, free MET, the two binary complexes, and double-bound antibody.
Their reaction fluxes implement the report's avidity equations.

Figure 9: algebraic stationary mass-balance solution, solved with a bounded
scalar root. Figure 10: independent six-state ODE relaxation from entirely
unbound initial conditions with the same parameters. It is not a second
experiment or a digitization of either screenshot. Both are local model
reimplementations, not verified exact reproductions of the published plots.

Each figure has three panels, left to right EGFR:MET=1:1, 10:1, 100:1.
Bound antibody signal=(binary1+binary2+double), normalized to its value at the highest sampled antibody dose.
Free receptors are divided by their respective starting totals.
Normalization is explicit and can depend on the concentration range.

Validation: receptor and antibody conservation, nonnegative states, and
ODE-versus-independent-equilibrium agreement at every sampled dose.
Numerical relaxation is allowed to extend until stationary; it is not an
experimental incubation time. See JSON for stopping times and tolerances.

Outputs: PNG/SVG for both figures, calculations CSV, parameter/validation
JSON. The original images remain preserved in report/figures/original_23.png
and original_24.png, but the report now uses the locally generated outputs.
