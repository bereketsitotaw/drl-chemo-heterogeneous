function cfg = config()
%CONFIG Central parameters for the heterogeneous chemotherapy DRL model.
%   Matches Kidane, Motayed & Wang — adaptive chemotherapy via DRL paper.

    this_dir = fileparts(mfilename('fullpath'));
    cfg.repo_dir = fileparts(this_dir);
    addpath(fullfile(cfg.repo_dir, 'configs'));

    parent_dir = fileparts(cfg.repo_dir);
    % When nested under DRL_Chemotherapy, share agents/data with the parent workspace.
    if isfolder(fullfile(parent_dir, 'savedAgents'))
        cfg.workspace_dir = parent_dir;
    else
        cfg.workspace_dir = cfg.repo_dir;
    end

    cfg.agent_root = fullfile(cfg.workspace_dir, 'savedAgents');
    % Subfolders under savedAgents/DQN and savedAgents/TD3 (train or copy runs here)
    cfg.dqn_run_subdir = resolve_agent_subdir(cfg.agent_root, 'DQN');
    cfg.td3_run_subdir = resolve_agent_subdir(cfg.agent_root, 'TD3');

    cfg.data_dir = fullfile(cfg.workspace_dir, 'data');
    cfg.results_dir = fullfile(cfg.repo_dir, 'results');
    cfg.figures_dir = fullfile(cfg.results_dir, 'figures');
    cfg.generated_dir = fullfile(cfg.results_dir, 'generated');
    cfg.reference_dir = fullfile(cfg.results_dir, 'reference');

    cfg.pmp_data_path = fullfile(cfg.data_dir, 'pmp_data.mat');
    cfg.rl_data_path = fullfile(cfg.data_dir, 'rl_data.mat');
    cfg.pmp_generated_path = fullfile(cfg.generated_dir, 'pmp_data.mat');
    cfg.rl_generated_path = fullfile(cfg.generated_dir, 'rl_data.mat');

    cfg.num_patients = 100;
    cfg.cohort_variation = 0.10;   % ±10% on R and Phi (paper Sec. IV)

    %% Model (n = 21 subpopulations, Eq. 1)
    cfg.model.T = 10;
    cfg.model.n_steps = 100;
    cfg.model.m = 21;
    cfg.model.x = linspace(0, 1, cfg.model.m)';
    cfg.model.e = ones(cfg.model.m, 1) / cfg.model.m;
    cfg.model.R_vec = 2 ./ (1 + 3 * cfg.model.x.^4);
    cfg.model.Phi_vec = -sin(cfg.model.x - 1) + 1.5;
    cfg.model.M_vec = 0.5 * ones(cfg.model.m, 1);
    cfg.model.N0 = 10 * ones(cfg.model.m, 1);
    cfg.model.R = diag(cfg.model.R_vec);
    cfg.model.Phi = diag(cfg.model.Phi_vec);
    cfg.model.M = diag(cfg.model.M_vec);

    %% Cost (Eq. 2): alpha, beta, gamma
    cfg.cost.alpha = 500;
    cfg.cost.beta = 400;
    cfg.cost.gamma = 4000;
    cfg.cost.Umax = 3.0;

    cfg.pmp.ode_options = odeset('RelTol', 1e-7, 'AbsTol', 1e-8);
    cfg.pmp.fsolve_options = optimoptions('fsolve', 'Display', 'off', ...
        'TolFun', 1e-7, 'MaxIterations', 100, 'Algorithm', 'levenberg-marquardt');

    %% Training (Table II defaults)
    cfg.train.max_episodes = 3000;
    cfg.train.stop_reward = -286;
    cfg.train.dqn_actions = 7;
    cfg.train.rng_base = 42;          % DQN=42, TD3=43
    cfg.train.pool_workers = 8;
    cfg.train.run_tag = 'Clean';

    for d = {cfg.data_dir, cfg.results_dir, cfg.figures_dir, cfg.generated_dir, cfg.reference_dir}
        if ~exist(d{1}, 'dir'), mkdir(d{1}); end
    end
end

function sub = resolve_agent_subdir(agent_root, agentType)
    % Prefer PaperStyle (evaluation default), then newest *_Clean, else PaperStyle name.
    base = fullfile(agent_root, agentType);
    if isfile(fullfile(base, 'PaperStyle', 'agent_final.mat'))
        sub = 'PaperStyle';
    else
        d = dir(fullfile(base, '*_Clean'));
        d = d([d.isdir]);
        if isempty(d)
            sub = 'PaperStyle';
        else
            [~, ix] = max([d.datenum]);
            sub = d(ix).name;
        end
    end
end
