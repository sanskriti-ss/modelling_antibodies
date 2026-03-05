Mission:
* Observe how we can use the ODE/PDE models described in the following papers to model Glofitimab.

Model Structure:
* **PK/TMDD** (7-ODE Schropp 2019 framework): Drug disposition with CD20/CD3 binding kinetics
* **CRS** (3-ODE indirect response): T-cell activation -> IL-6/TNF-alpha release -> CRS grading
* **Fatigue** (1-ODE indirect response): Cumulative immune burden -> fatigue score
* **Thermoregulation** (1-ODE Stolwijk 1971): IL-6 -> hypothalamic setpoint shift -> body temperature / fever

Parameter Sources:
* PK parameters (CL, Vc, Vp, Q): Glofitamab popPK, Gibiansky 2023 / FDA label
* Binding kinetics (KD_CD20=4.8nM, KD_CD3=100nM): Bacac et al., Clin Cancer Res 2018
* TMDD framework (7-ODE structure): Schropp et al., J Pharmacokinet Pharmacodyn 2019
* CRS incidence/grading targets: Dickinson et al., JCO 2021 (Phase I trial)
* CRS grading criteria: Lee et al., Biol Blood Marrow Transplant 2019 (ASTCT consensus)
* Fatigue incidence (~20%, 85% Grade 1): Glofitamab FDA label
* Thermoregulation ODE (body temp dynamics, k3, kd3): Stolwijk, J. A. J. (1971). NASA CR-1855
* Hypothalamic setpoint mechanism (IL-6 -> T_hypo): Lefevre, N. et al. (2024). PMC11782561
* CRS fever characterization (median 38.0-38.5C): Gritti, A. et al. (2024); Dickinson et al. 2021

References:
1. Ng, C. M. and R. J. Bauer (2024). "General quasi-equilibrium multivalent binding model to study diverse and complex drug-receptor interactions of biologics." Journal of Pharmacokinetics and Pharmacodynamics 51(6): 841-857.
2. Ray, C., Yang, H, Spangler, JB, Gabhann, FM (2024). "Mechanistic computational modeling of monospecific and bispecific antibodies targeting interleukin-6/8 receptors." PLoS Comput Biol 20(6).
3. van Steeg TJ, B. K., Dimas N, Sachsenmeier KF, Agoram, B (2016). "The application of mathematical modelling to the design of bispecific monoclonal antibodies." MABS 8(3): 585-592.
4. Lefevre, N. et al. (2024). "Thermoregulatory modeling of cytokine-driven fever in bispecific antibody therapy." https://pmc.ncbi.nlm.nih.gov/articles/PMC11782561/
5. Gritti, A. et al. (2024). CRS characterization in bispecific T-cell engager therapy.
6. Stolwijk, J. A. J. (1971). "A mathematical model of physiological temperature regulation in man." NASA CR-1855.
7. Schropp, J. et al. (2019). "Target-mediated drug disposition model for bispecific antibodies." J Pharmacokinet Pharmacodyn.
8. Bacac, M. et al. (2018). "CD20-TCB with obinutuzumab pretreatment." Clin Cancer Res 24:4785-4797.
9. Dickinson, M. J. et al. (2021). "Glofitamab Phase I trial." JCO.
10. Lee, D. W. et al. (2019). "ASTCT consensus grading for CRS." Biol Blood Marrow Transplant.
