% solve_pmp.m
%
% This version now calculates and saves the total cost J.
% =========================================================================

function pmp_results = solve_pmp(cfg)
    %% 1. Initialization and Setup
    T = cfg.model.T;
    n_steps = cfg.model.n_steps;
    m_states = cfg.model.m;
    time = linspace(0, T, n_steps + 1);

    % Model Parameters
    params = struct();
    params.x = cfg.model.x;
    params.e = cfg.model.e;
    params.R = cfg.model.R_vec;
    params.C = cfg.model.Phi_vec;
    params.M = cfg.model.M_vec;
    params.m = m_states;
    params.T = T;

    % Cost Function Weights
    params.alpha = cfg.cost.alpha * m_states * params.e;
    params.beta = cfg.cost.beta * m_states * params.e;
    params.gamma = cfg.cost.gamma;
    params.Umax = cfg.cost.Umax;

    % Initial Conditions
    N0 = cfg.model.N0;

    %% 2. Optimal Control Solution
    l0_guess = params.beta;
    [l0_optimal, ~, exitflag] = fsolve(@(l0) shootingError(l0, N0, params, time, cfg.pmp.ode_options), l0_guess, cfg.pmp.fsolve_options);
    if exitflag <= 0, error('PMP solver failed to converge.'); end

    %% 3. Final Integration
    initial_conditions_opt = [N0; l0_optimal];
    [~, NL_opt] = ode15s(@(t, y) stateCostateODE(t, y, params), time, initial_conditions_opt, cfg.pmp.ode_options);
    N_opt = NL_opt(:, 1:params.m);
    L_opt = NL_opt(:, (params.m+1):end);
    u_opt = arrayfun(@(k) calculateOptimalControl(N_opt(k, :)', L_opt(k, :)', params), 1:length(time))';

    %% 4. Package Results
    pmp_results.time = time;
    pmp_results.u = u_opt;
    pmp_results.N = N_opt;
    pmp_results.L = L_opt;
    pmp_results.params = params;
    % SHARED POST-PROCESSING
    % Ensure N_opt is transposed to [m x n] format required by compute_metrics
    eval_metrics = compute_metrics(time, N_opt', u_opt', cfg);
    pmp_results.metrics = eval_metrics;
end

%% --- LOCAL FUNCTION DEFINITIONS ---
function error = shootingError(l0, N0, params, time, ode_options)
    [~, y_out] = ode15s(@(t, y) stateCostateODE(t, y, params), time, [N0; l0], ode_options);
    error = y_out(end, (params.m+1):end)' - params.alpha;
end

function dy = stateCostateODE(~, y, p)
    m = p.m; e = p.e; R_vec = p.R; C_vec = p.C; M_vec = p.M; beta = p.beta;
    N = y(1:m); L = y(m+1:2*m);
    N(N < 0) = 0;
    u = calculateOptimalControl(N, L, p);
    N_total = e' * N;
    growth = R_vec .* N;
    drug_effect = (u / (1 + u)) * (C_vec .* N);
    competition = M_vec .* log(1 + N_total) .* N;
    dN = growth - drug_effect - competition;
    term_running_cost = beta;
    term_growth = R_vec .* L;
    term_drug_effect = (u / (1 + u)) * (C_vec .* L);
    term_competition_direct = M_vec .* L .* log(1 + N_total);
    term_competition_indirect = (M_vec .* (L' * N) / (1 + N_total)) .* e;
    term_competition = term_competition_direct + term_competition_indirect;
    dHdN = term_running_cost + term_growth - term_drug_effect - term_competition;
    dL = -dHdN;
    dy = [dN; dL];
end

function u = calculateOptimalControl(N, L, p)
    phi = (L' * (p.C .* N));
    phi = max(0, phi);
    u = sqrt(phi / p.gamma) - 1;
    u = min(max(0, u), p.Umax);
end