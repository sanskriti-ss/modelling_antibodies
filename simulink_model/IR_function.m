function [d_Tact, d_IL6, d_IL8, FeverScore, RigorScore] = infusion_reaction_model(B, dB, Tact, IL6, IL8)
% B    = BS1_6R_8R (ternary complex level)
% dB   = d_BS1_6R_8R (ternary complex formation rate)
% Tact = immune activation proxy
% IL6  = cytokine proxy
% IL8  = cytokine proxy

% 1) Parameters
k_act   = 1.0;   % activation rate
k_deact = 0.5;   % deactivation rate

k_IL6   = 2.0;   % IL6 production per Tact
k_IL8   = 1.5;   % IL8 production per Tact
k_cl6   = 1.0;   % IL6 clearance
k_cl8   = 1.0;   % IL8 clearance

EC50    = 5.0;   % bridge level giving half-max activation
n       = 2;     % Hill coefficient

EC50_F  = 1.0;   % fever sensitivity
EC50_R  = 1.0;   % rigor sensitivity

drive = B;

% Saturating activation (bounded 0..1)
Act = (drive^n) / (EC50^n + drive^n);

% 3) ODEs 
d_Tact = (k_act * Act) - (k_deact * Tact);

d_IL6  = (k_IL6 * Tact) - (k_cl6 * IL6);
d_IL8  = (k_IL8 * Tact) - (k_cl8 * IL8);

% 4) Symptoms
FeverScore = IL6 / (EC50_F + IL6);
RigorScore = IL8 / (EC50_R + IL8);

end