% compute_metrics.m
% Shared post-processing function to calculate costs and evaluation
% metrics for the chemotherapy nominal benchmark comparisons.
%
% Inputs:
%   time   - [1 x n] array of time points
%   N_traj - [m x n] matrix of trait populations over time
%   u_vec  - [1 x n] array of control inputs over time
%   cfg    - Global configuration struct (from config())
%
% Outputs:
%   metrics - struct containing integrated costs and final states
%==========================================================================
function metrics = compute_metrics(time, N_traj, u_vec, cfg)
    
    m = cfg.model.m;
    e = cfg.model.e;
    
    % Reconstruct proper vector weights identically to PMP formulation
    alpha_vec = cfg.cost.alpha * m * e;
    beta_vec  = cfg.cost.beta  * m * e;
    gamma     = cfg.cost.gamma;
    
    %% Cost Functional Calculations (J)
    
    % Terminal Cost: alpha^T * N(T)
    metrics.TerminalCost = alpha_vec' * N_traj(:, end);
    
    % Running Tumor Cost: integral of beta^T * N(t) dt
    running_tumor_integrand = beta_vec' * N_traj;
    metrics.RunningTumorCost = trapz(time, running_tumor_integrand);
    
    % Drug Control Cost: integral of gamma * u(t) dt
    drug_integrand = gamma * u_vec;
    metrics.DrugCost = trapz(time, drug_integrand);
    
    % Total Objective Cost
    metrics.TotalCost = metrics.TerminalCost + metrics.RunningTumorCost + metrics.DrugCost;
    
    %% Biological Efficacy Metrics
    
    % Final Total Tumor Burden
    metrics.FinalTumor = sum(N_traj(:, end));
    
    % Initial Total Tumor Burden
    initialTumor = sum(N_traj(:, 1));
    
    % Percent Reduction
    metrics.TumorReduction = 100 * (1 - metrics.FinalTumor / initialTumor);
    
    % Total Drug Administered (Raw Integral)
    metrics.TotalDrug = trapz(time, u_vec);
    
end
