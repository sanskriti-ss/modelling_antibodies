function [d_BS1, d_IL6R, d_IL8R, d_BS1_6R, d_BS1_8R, d_BS1_6R_8R] = binding_model(BS1, IL6R, IL8R, BS1_6R, BS1_8R, BS1_6R_8R)
    % =====================================================================
    % 1. Parameters (Glofitamab Estimates)
    % =====================================================================
    kon_BS1_6R     = 1e-4;     % On-rate for CD20
    koff_BS1_6R    = 1.6e-4;   % Off-rate for CD20
    
    kon_BS1_8R     = 1e-4;     % On-rate for CD3
    koff_BS1_8R    = 1.0e-2;   % Off-rate for CD3
    
    kon_BS1_6R_8R  = 1e-2;     % Cross-linking (CD20 arm grabbing CD3)
    koff_BS1_6R_8R = 1.0e-2;   % Cross-linking off-rate
    
    kon_BS1_8R_6R  = 1e-2;     % Cross-linking (CD3 arm grabbing CD20)
    koff_BS1_8R_6R = 1.6e-4;   % Cross-linking off-rate

    alpha = 1;                 % Volume scaling

    % =====================================================================
    % 2. Reaction Fluxes (Binding - Unbinding)
    % =====================================================================
    v_BS1_6R    = (kon_BS1_6R * BS1 * IL6R) - (koff_BS1_6R * BS1_6R);
    v_BS1_8R    = (kon_BS1_8R * BS1 * IL8R) - (koff_BS1_8R * BS1_8R);
    v_BS1_8R_6R = (kon_BS1_8R_6R * BS1_8R * IL6R) - (koff_BS1_8R_6R * BS1_6R_8R);
    v_BS1_6R_8R = (kon_BS1_6R_8R * BS1_6R * IL8R) - (koff_BS1_6R_8R * BS1_6R_8R);

    % =====================================================================
    % 3. Differential Equations (Outputs to Integrators)
    % =====================================================================
    d_BS1       = alpha * (- v_BS1_6R - v_BS1_8R);
    d_IL6R      = - v_BS1_6R - v_BS1_8R_6R;
    d_IL8R      = - v_BS1_8R - v_BS1_6R_8R;
    d_BS1_6R    = v_BS1_6R - v_BS1_6R_8R;
    d_BS1_8R    = v_BS1_8R - v_BS1_8R_6R;
    d_BS1_6R_8R = v_BS1_8R_6R + v_BS1_6R_8R;
end