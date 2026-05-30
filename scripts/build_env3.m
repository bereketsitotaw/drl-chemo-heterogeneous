% build_env.m
%
% Creates a tumor-control RL environment.
% Supports fixed initial state (default) and random initialization for training.
% =========================================================================

function [env, params, obsInfo, actInfo] = build_env3(agentType, patientParams, useRandomInit)
    % --- Handle optional input arguments ---
    if nargin < 2, patientParams = []; end
    if nargin < 3, useRandomInit = false; end % default: fixed initial state

    % --- Load Shared Nominal Benchmark ---
    cfg = config();

    % --- Environment & Model Parameters ---
    params.n = cfg.model.m;
    params.dt = 0.1;
    params.max_steps = round(cfg.model.T / params.dt);
    params.u_max = cfg.cost.Umax;
    params.T = cfg.model.T;
    params.x = cfg.model.x;
    
    % Force single drug, no mutation benchmark flags
    % If patientParams is provided, we disable strict benchmark mode to allow robustness variation
    params.benchmark_mode = isempty(patientParams);
    params.single_drug_mode = true;
    params.enable_mutation = false;
    params.enable_randomization = false;

    % Core Shared Dynamics Matrices
    params.e = cfg.model.e;
    params.R = cfg.model.R_vec;       % Use vector forms for simplicity, diagonal matrix equivalent
    params.Phi = cfg.model.Phi_vec;
    params.M = cfg.model.M_vec;

    % Shared Cost Weights (exactly mapped to PMP scaling)
    params.alpha = cfg.cost.alpha * params.n * params.e;
    params.beta  = cfg.cost.beta  * params.n * params.e;
    params.gamma = cfg.cost.gamma;
    
    % Normalization for observation space only! Modifies NO biological terms.
    params.N_norm = 10;
    params.N_min = 1e-6;
    params.N_max = 1e3;
    
    % Default Initial Condition Matches PMP
    params.N0_nominal = cfg.model.N0;

    % Optional parameter overrides (disabled in benchmark mode)
    if ~params.benchmark_mode && ~isempty(patientParams)
        if isfield(patientParams, 'R'), params.R = patientParams.R; end
        if isfield(patientParams, 'Phi'), params.Phi = patientParams.Phi; end
        if isfield(patientParams, 'N0_nominal'), params.N0_nominal = patientParams.N0_nominal; end
    end

    % --- Store local flags for robust nested function capturing ---
    useRandomInitFlag = useRandomInit;
    benchmarkModeFlag = params.benchmark_mode;
    N0_nominal = params.N0_nominal;
    nStates = params.n;
    N_norm = params.N_norm;
    rewardScale = 1e3; % match the 1e3 optimization factor

    % --- Observation & Action Spaces ---
    obsInfo = rlNumericSpec([params.n, 1], 'Name', 'Observation');

    if strcmpi(agentType, 'DQN')
        numDiscreteActions = 7;
        actionSet = linspace(0, params.u_max, numDiscreteActions);
        actInfo = rlFiniteSetSpec(actionSet);
    else % TD3/DDPG
        actInfo = rlNumericSpec([1 1], 'LowerLimit', 0, 'UpperLimit', params.u_max);
    end

    env = rlFunctionEnv(obsInfo, actInfo, @stepFunction, @resetNested);

    %% --- Nested Functions ---

    function [initialObs, loggedSignals] = resetNested()
        if useRandomInitFlag
            N0 = 5 + 10 * rand(nStates,1);   % random start for training
        else
            N0 = N0_nominal;                 % strict nominal start (match PMP)
        end
        loggedSignals = struct('N', N0, 'step_count', 0, 'clipped_action', 0);
        initialObs = N0 ./ N_norm;
    end

    function [nextObs, reward, isDone, loggedSignals] = stepFunction(action, loggedSignals)
        N = loggedSignals.N(:);
        tstep = loggedSignals.step_count;
        action_val = min(max(double(action), 0), params.u_max);

        % Map to shared dynamics function parameter structure
        dyn_params.e = params.e;
        dyn_params.R = diag(params.R);     % tumor_dynamics expects diagonal/matrix form
        dyn_params.Phi = diag(params.Phi);
        dyn_params.M = diag(params.M);

        % --- State integration over [0, dt] ---
        % Using variable-step ode45 explicitly mapped onto shared benchmark physics
        ode_options = odeset('RelTol', 1e-6, 'AbsTol', 1e-8);
        [~, Nt] = ode45(@(t,nc) tumor_dynamics(nc, action_val, dyn_params), [0 params.dt], N, ode_options);
        N_next_raw = Nt(end,:)';
        
        % Clipping/Saturation
        N_next = min(max(N_next_raw, params.N_min), params.N_max);

        % --- Rigid Reward functional mapping (Match Negative Exact PMP Cost) ---
        r_running_tumor = -params.dt * (params.beta' * N_next);
        r_drug_usage = -params.dt * (params.gamma * action_val);

        r_terminal = 0;
        isTerminalStep = (tstep >= params.max_steps - 1);
        if isTerminalStep
            r_terminal = -(params.alpha' * N_next);
        end

        % Apply Reward Scaling for Numerical Conditioning
        % This improves gradients during training but preserves policy ordering.
        reward_physical = r_running_tumor + r_drug_usage + r_terminal;
        reward = reward_physical / rewardScale;

        loggedSignals.N = N_next;
        loggedSignals.step_count = tstep + 1;
        loggedSignals.clipped_action = action_val;

        nextObs = N_next ./ N_norm;
        isDone = isTerminalStep || all(N_next < params.N_min) || any(N_next > params.N_max);
    end
end
