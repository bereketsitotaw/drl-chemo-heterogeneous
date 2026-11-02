function out = drl_pipeline(repo_root, action, arg2, arg3, varargin)
%DRL_PIPELINE Run PMP, training, cohort simulation, and paper figures.

    addpath(fullfile(repo_root, 'scripts'));
    addpath(fullfile(repo_root, 'configs'));
    addpath(fullfile(repo_root, 'tests'));
    cfg = config();
    assert_publication_defaults(cfg);

    switch lower(char(action))
        case 'help'
            print_help();
            out = [];
        case {'eval', 'evaluate'}
            opts = struct();
            if nargin >= 3 && ~isempty(arg2)
                if isstruct(arg2), opts = arg2;
                elseif isnumeric(arg2), opts.cohort_size = arg2; end
            end
            if nargin >= 4 && ~isempty(arg3) && islogical(arg3)
                opts.run_cohort = arg3;
            end
            for ii = 1:2:numel(varargin)
                if isstruct(varargin{ii}), opts = merge_struct(opts, varargin{ii}); end
            end
            out = run_evaluation(cfg, opts);
        case 'pmp'
            out = run_pmp(cfg);
        case 'smoke'
            out = run_smoke(cfg);
        case 'validate'
            out = validate_reference_pin(cfg);
        case 'train'
            if nargin >= 3 && isstruct(arg2)
                out = run_training(cfg, arg2);
            else
                if nargin < 3 || isempty(arg2), arg2 = 'both'; end
                opts = struct('agents', arg2, 'promote', true);
                if ~isempty(varargin) && isnumeric(varargin{1})
                    opts.max_episodes = varargin{1};
                end
                out = run_training(cfg, opts);
            end
        case 'simulate'
            if nargin < 3 || isempty(arg2), arg2 = 'nominal'; end
            if nargin < 4 || isempty(arg3), arg3 = 'both'; end
            out = run_simulate(cfg, arg2, arg3, varargin{:});
        case 'collect'
            n = cfg.num_patients;
            if nargin >= 3 && ~isempty(arg2) && isnumeric(arg2), n = arg2; end
            out = run_collect(cfg, n);
        case 'figures'
            figOpts = struct('display', true, 'save_png', false);
            which = [1, 2, 3, 4];
            if nargin >= 3 && ~isempty(arg2)
                if isstruct(arg2)
                    figOpts = merge_struct(figOpts, arg2);
                    if isfield(arg2, 'which'), which = arg2.which; end
                else
                    which = arg2;
                end
            end
            out = run_figures(cfg, which, figOpts);
        case 'ablation'
            out = run_ablation(cfg, varargin{:});
        otherwise
            error('Unknown action "%s".', action);
    end
end

%==========================================================================
function print_help()
    fprintf(['\nAdaptive Chemotherapy DRL — simulation / evaluation\n', ...
        '  main(''eval'')                     FULL eval: PMP + DQN/TD3 nominal + cohort\n', ...
        '  main(''eval'', struct(''run_cohort'',false))   nominal only (quick test)\n', ...
        '  main(''simulate'',''nominal'')     one-off nominal rollout\n', ...
        '  main(''simulate'',''cohort'',100)  cohort only\n', ...
        '  main(''pmp'')                      PMP benchmark only\n', ...
        '  main(''figures'')                  display Figs 1–4 (on screen)\n', ...
        '  main(''figures'', struct(''save_png'',true))  also save PNGs\n\n', ...
        '  main(''train'')                  clean train DQN+TD3, promote to PaperStyle\n', ...
        '  main(''train'', struct(''agents'',''DQN''))   one agent only\n\n', ...
        '  main(''test'')                   lightweight smoke unit tests\n', ...
        '  main(''smoke'')                  PMP plumbing + Table III soft pin\n', ...
        '  main(''validate'')               publication config / pin self-check\n\n']);
end

function s = merge_struct(s, t)
    fn = fieldnames(t);
    for k = 1:numel(fn), s.(fn{k}) = t.(fn{k}); end
end

%==========================================================================
function pmp_data = run_pmp(cfg)
    fprintf('Solving PMP benchmark (nominal patient)...\n');
    fprintf('  Publication pins: (alpha,beta,gamma)=(%g,%g,%g), n=%d, T=%g, Umax=%g\n', ...
        cfg.cost.alpha, cfg.cost.beta, cfg.cost.gamma, cfg.model.m, cfg.model.T, cfg.cost.Umax);
    pmp_raw = solve_pmp(cfg);
    pmp_data = package_pmp_for_figures(pmp_raw, cfg);
    save(cfg.pmp_data_path, 'pmp_data', '-v7.3');
    if isfield(cfg, 'pmp_generated_path')
        save(cfg.pmp_generated_path, 'pmp_data', '-v7.3');
        fprintf('Saved %s and %s\n', cfg.pmp_data_path, cfg.pmp_generated_path);
    else
        fprintf('Saved %s\n', cfg.pmp_data_path);
    end
    fprintf('  PMP: reduction=%.2f%%, drug=%.2f, J=%.3e, avg_final=%.3f\n', ...
        pmp_data.metrics.tumor_reduction, pmp_data.metrics.total_drug, ...
        pmp_data.metrics.TotalCost, pmp_data.metrics.avg_final_tumor);
end

function out = run_smoke(cfg)
    opts = apply_smoke_overrides(struct());
    fprintf('\n%s\n', opts.smoke_label);
    fprintf('Publication (alpha,beta,gamma)=(%g,%g,%g); n=%d; T=%g; Umax=%g; dqn_actions=%d\n', ...
        cfg.cost.alpha, cfg.cost.beta, cfg.cost.gamma, cfg.model.m, cfg.model.T, ...
        cfg.cost.Umax, cfg.train.dqn_actions);
    pmp_data = run_pmp(cfg);
    validate_reference_pin(cfg, pmp_data);
    out = struct('pmp_data', pmp_data, 'opts', opts);
    fprintf('Smoke complete (PMP only; no agent retrain).\n');
end

%==========================================================================
function results = run_simulate(cfg, mode, agents, varargin)
    nPatients = cfg.num_patients;
    if strcmpi(mode, 'cohort') && ~isempty(varargin) && isnumeric(varargin{1})
        nPatients = varargin{1};
    end

    agentList = resolve_agents(agents);
    results = struct();
    for k = 1:numel(agentList)
        ag = agentList{k};
        agent = load_agent(cfg, ag);
        results.(lower(ag)) = simulate_mode(cfg, agent, ag, mode, nPatients);
    end
end

function results = simulate_mode(cfg, agent, agentType, mode, nPatients)
    envType = agentType;
    if strcmpi(agentType, 'TD3'), envType = 'DDPG'; end

    if strcmpi(mode, 'nominal')
        [env, params, ~, ~] = build_env3(envType);
        simData = simulate_agent(agent, env, params);
        m = metrics_for_rl(compute_metrics(simData.time, simData.N, simData.u, cfg));
        results = struct('simData', simData, 'metrics', m);
        fprintf('%s nominal: reduction=%.1f%%, J=%.3e\n', agentType, m.TumorReductionPercent, m.TotalCost);
        return;
    end

    if ~strcmpi(mode, 'cohort')
        error('Simulation mode must be ''nominal'' or ''cohort''.');
    end

    [~, base_params, ~, ~] = build_env3(envType);
    simCell = cell(nPatients, 1);
    metCell = cell(nPatients, 1);
    parfor p = 1:nPatients
        pp = cohort_params(base_params, cfg.cohort_variation);
        [env, params, ~, ~] = build_env3(envType, pp, false);
        sd = simulate_agent(agent, env, params);
        simCell{p} = sd;
        metCell{p} = metrics_for_rl(compute_metrics(sd.time, sd.N, sd.u, cfg));
    end
    results.simData = simCell;
    results.metrics = metCell;
    red = cellfun(@(m) m.TumorReductionPercent, metCell);
    fprintf('%s cohort (n=%d): mean reduction=%.1f%% (std %.1f)\n', ...
        agentType, nPatients, mean(red), std(red));
end

%==========================================================================
function rl_results = run_collect(cfg, nPatients)
    fprintf('Collecting RL results for paper figures (n=%d cohort)...\n', nPatients);
    if ~isfile(cfg.pmp_data_path)
        run_pmp(cfg);
    end

    rl_results = struct();
    for ag = {'DQN', 'TD3'}
        name = ag{1};
        agent = load_agent(cfg, name);
        rl_results.([lower(name) '_nominal']) = simulate_mode(cfg, agent, name, 'nominal', 1);
        rl_results.([lower(name) '_multi']) = simulate_mode(cfg, agent, name, 'cohort', nPatients);
    end

    save(cfg.rl_data_path, 'rl_results', '-v7.3');
    if isfield(cfg, 'rl_generated_path')
        save(cfg.rl_generated_path, 'rl_results', '-v7.3');
        fprintf('Saved %s and %s\n', cfg.rl_data_path, cfg.rl_generated_path);
    else
        fprintf('Saved %s\n', cfg.rl_data_path);
    end
end

%==========================================================================
function out = run_figures(cfg, which, figOpts)
    if nargin < 3 || isempty(figOpts), figOpts = struct(); end
    if ~isfield(figOpts, 'display'), figOpts.display = true; end
    if ~isfield(figOpts, 'save_png'), figOpts.save_png = false; end

    if ~isfile(cfg.pmp_data_path)
        error('Missing %s. Run main(''eval'') first.', cfg.pmp_data_path);
    end
    if ~isfile(cfg.rl_data_path)
        error('Missing %s. Run main(''eval'') first.', cfg.rl_data_path);
    end
    load(cfg.pmp_data_path, 'pmp_data');
    load(cfg.rl_data_path, 'rl_results');
    mpc_results = [];

    if nargin < 2 || isempty(which), which = [1, 2, 3, 4]; end
    out = generate_figures('all', pmp_data, rl_results, mpc_results, cfg, which, figOpts);
end

%==========================================================================
function tbl = run_ablation(cfg, actionCounts)
    if nargin < 2 || isempty(actionCounts)
        actionCounts = [4, 7, 13];
    end
    fprintf('DQN action-space ablation (nominal patient)...\n');
    rows = {};
    for nAct = actionCounts(:)'
        [env, params, obsInfo, ~] = build_env3('DQN');
        actInfo = rlFiniteSetSpec(linspace(0, params.u_max, nAct));
        agent = build_agent('DQN', obsInfo, actInfo, params);
        simData = simulate_agent(agent, env, params);
        m = compute_metrics(simData.time, simData.N, simData.u, cfg);
        rows(end+1, :) = {nAct, m.TumorReduction, m.TotalCost, m.TotalDrug}; %#ok<AGROW>
    end
    tbl = cell2table(rows, 'VariableNames', {'NumActions', 'TumorReductionPct', 'TotalCost', 'TotalDrug'});
    disp(tbl);
    ablation_path = fullfile(cfg.results_dir, 'dqn_ablation_nominal_untrained.mat');
    save(ablation_path, 'tbl', 'actionCounts');
    fprintf('Note: table uses freshly initialized networks. Train per action count for paper Table I.\n');
    fprintf('Saved %s\n', ablation_path);
end

%==========================================================================
function agent = load_agent(cfg, agentType)
    sub = cfg.dqn_run_subdir;
    if strcmpi(agentType, 'TD3'), sub = cfg.td3_run_subdir; end
    base = fullfile(cfg.agent_root, agentType);
    run_dir = fullfile(base, sub);

    if ~isfolder(run_dir)
        dirs = dir(fullfile(base, '*_PaperStyle'));
        if isempty(dirs)
            error(['No pretrained agent in %s. Copy a run to PaperStyle/ or set ', ...
                'cfg.%s_run_subdir.'], base, lower(agentType));
        end
        [~, ix] = max([dirs.datenum]);
        run_dir = fullfile(dirs(ix).folder, dirs(ix).name);
    end

    f = fullfile(run_dir, 'agent_final.mat');
    if ~isfile(f)
        error('agent_final.mat not found in %s', run_dir);
    end
    d = load(f, 'agent');
    agent = d.agent;
    fprintf('Loaded %s from %s\n', agentType, run_dir);
end

function list = resolve_agents(spec)
    if nargin < 1 || isempty(spec) || strcmpi(spec, 'both')
        list = {'DQN', 'TD3'};
    else
        list = {upper(char(spec))};
    end
end

function pp = cohort_params(base_params, var_level)
    pp.R = base_params.R .* (1 + var_level * (2 * rand(size(base_params.R)) - 1));
    pp.Phi = base_params.Phi .* (1 + var_level * (2 * rand(size(base_params.Phi)) - 1));
    pp.N0_nominal = base_params.N0_nominal;
end

function pmp_data = package_pmp_for_figures(pmp_raw, cfg)
    m = pmp_raw.metrics;
    pmp_data = pmp_raw;
    pmp_data.params.x = pmp_raw.params.x;
    pmp_data.params.e = pmp_raw.params.e;
    pmp_data.params.C = pmp_raw.params.C;
    pmp_data.params.gamma = pmp_raw.params.gamma;
    pmp_data.metrics.avg_final_tumor = m.FinalTumor / cfg.model.m;
    pmp_data.metrics.total_drug = m.TotalDrug;
    pmp_data.metrics.tumor_reduction = m.TumorReduction;
    pmp_data.metrics.TotalCost = m.TotalCost;
end

function m = metrics_for_rl(raw)
    m.AverageFinalTumor = raw.FinalTumor;
    m.TumorReductionPercent = raw.TumorReduction;
    m.TotalDrugDosage = raw.TotalDrug;
    m.TotalCost = raw.TotalCost;
    m.TerminalCost = raw.TerminalCost;
    m.RunningTumorCost = raw.RunningTumorCost;
    m.DrugCost = raw.DrugCost;
end
