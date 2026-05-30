% src/tumor_dynamics.m
%
% Centralized ODE function for the tumor dynamics model.
% =========================================================================
function dN = tumor_dynamics(N, u, dyn_params)
    % Ensure N is non-negative for stability
    N(N < 0) = 0;
    
    % Calculate average population for competition term
    eN = dyn_params.e' * N;
    
    % Calculate the components of the ODE
    growth = dyn_params.R * N;
    drug_effect = dyn_params.Phi * (u / (1 + u)) * N;
    competition = log(1 + eN) * (dyn_params.M * N);
    
    % Return the derivative
    dN = growth - drug_effect - competition;
end