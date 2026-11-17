Figures 11 and 13: time-course reconstruction audit

Run:
  python reconstruct_timecourses.py
  python compare_screenshots.py
Requires NumPy, SciPy, Matplotlib and Pillow.

Parameter source: article-linked Google document, Tables 1 and 2:
https://docs.google.com/document/d/1TBuxTPJg9zp_3K0JgFu_k3pjpyx64Na5BnBF_wPYApE/edit
The relevant parameter excerpt is preserved locally.

This is newly written Python code, not recovered original MATLAB plotting code.
BS1: receptor totals 316000 and 618000/cell; Table 1 on/off/cross-link rates.
Glofitamab: receptor totals 200000 CD20 and 100000 CD3/cell; Table 2 rates.
All complexes start at zero. Free antibody is held at 100 nM for two hours,
then clamped to zero for two hours. Dissociated antibody is removed after
washout. A constant bath is an open system; antibody mass is not conserved.
Receptor pools are conserved and free receptor amounts are calculated from
complex stoichiometry. There is no PK, turnover, or cytokine model here.

Remaining inference: the BS1 dose is taken as 100 nM from the close curve match;
it is not specified in the recovered table. For the Glofitamab CD3-first route,
B_syn is assigned the Table 2 CD20 cross-link rate (8.3e-9), with off-rate 1e-5.
Using the restricted CD3 rate for B_syn does not reproduce the original curves.
Thus the reproduction is quantitatively close but is not proof of original
implementation identity. No optimizer, curve fitting, or amplitude rescaling
was used. The saved project4.slx uses different initial values and a 500-unit
stop time; it is not the original model configuration behind these screenshots.

CSV files contain every complex concentration and its net time derivative.
Panel (a): concentration; panel (b): net accumulation rate, directly from the
ODE. The rate is not a forward association flux or an individual reaction
flux. Its sign distinguishes accumulation from depletion. The rate panel uses
a symmetric-log axis (linear between -0.1 and +0.1) to show both rapid initial
binding and slower post-washout behavior. Washout has left/right derivative
values at the same timestamp; concentrations remain continuous.

Validation: Radau and BDF agree to below 0.001 complex/cell; complexes and free
receptor pools remain nonnegative. Screenshot comparison uses colored-pixel
curve extraction with manually calibrated axes; legends are masked. Residuals
are approximate because of raster resolution, axis calibration and sparse
source sampling. Per-series RMS discrepancies are about 0.4–2.6 pixels for
BS1 and 0.6–1.1 pixels for Glofitamab. These are comparisons, not fitting targets.

The report now uses native LaTeX concentration/flux subfigures for Figures 11 and 13.

Updated report layout: Figures 11a and 13a show concentrations. Figures 11b and 13b show individual NET REACTION FLUXES (forward association minus reverse dissociation), not species accumulation rates. Four BS1 fluxes and seven Glofitamab fluxes are shown; the stoichiometric projection of these fluxes is checked against the ODE derivatives. Separate PNG/SVG assets and reaction-flux CSVs are exported. Earlier combined concentration/rate previews remain as audit artifacts only.
