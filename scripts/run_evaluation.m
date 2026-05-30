function summary = run_evaluation(cfg, opts)
%RUN_EVALUATION Simulation-only evaluation with pretrained DQN/TD3 agents.
%
%   summary = run_evaluation()              % defaults from config()
%   summary = run_evaluation(cfg, opts)   % opts.cohort_size, .run_cohort, .save_figures
%
%   Does NOT train agents. Requires savedAgents/*/PaperStyle/agent_final.mat
%   (or latest *_PaperStyle folder).

    if nargin < 1 || isempty(cfg), cfg = config(); end
    if nargin < 2, opts = struct(); end
    opts = fill_eval_opts(opts, cfg);

    fprintf('\n%s\n', repmat('=', 1, 60));
    fprintf('  SIMULATION EVALUATION (pretrained agents only)\n');
    fprintf('%s\n\n', repmat('=', 1, 60));

    verify_agents(cfg);
    tic;

    %% PMP benchmark
    fprintf('[1/3] PMP benchmark (nominal patient)...\n');
    pmp_data = run_pmp_eval(cfg);

    %% RL — nominal
    fprintf('\n[2/3] RL nominal patient (DQN + TD3)...\n');
    rl_results = struct();
    for k = 1:numel(opts.agents)
        ag = opts.agents{k};
        agent = load_agent_eval(cfg, ag);
        rl_results.([lower(ag) '_nominal']) = rollout_nominal(cfg, agent, ag, opts.verbose);
    end

    %% RL — cohort
    if opts.run_cohort
        fprintf('\n[3/3] RL cohort (n=%d, +/-%.0f%% R, Phi)...\n', ...
            opts.cohort_size, 100 * cfg.cohort_variation);
        fprintf('  (Parallel rollouts — often 10–30+ min for 100 patients per agent.)\n');
        cohort_seeds = setup_cohort_rng(opts.cohort_size);
        for k = 1:numel(opts.agents)
            ag = opts.agents{k};
            agent = load_agent_eval(cfg, ag);
            rl_results.([lower(ag) '_multi']) = rollout_cohort(cfg, agent, ag, opts.cohort_size, cohort_seeds);
        end
    else
        fprintf('\n[3/3] Cohort skipped (opts.run_cohort = false).\n');
    end

  %% Summary tables
    summary = build_summary(pmp_data, rl_results, cfg, opts);
    print_summary(summary);

    %% Save
    stamp = datestr(now, 'yyyymmdd_HHMMSS');
    eval_path = fullfile(cfg.results_dir, ['eval_' stamp '.mat']);
    save(eval_path, 'pmp_data', 'rl_results', 'summary', 'opts', '-v7.3');
    fprintf('\nSaved evaluation bundle: %s\n', eval_path);

    save(cfg.pmp_data_path, 'pmp_data', '-v7.3');
    save(cfg.rl_data_path, 'rl_results', '-v7.3');
    fprintf('Updated %s and %s (for main(''figures'')).\n', cfg.pmp_data_path, cfg.rl_data_path);

    if opts.save_figures
        fprintf('\nDisplaying figures (use main(''figures'') after eval)...\n');
        generate_figures('all', pmp_data, rl_results, [], cfg, opts.figure_list, ...
            struct('display', true, 'save_png', false));
    end

    fprintf('\nEvaluation finished in %.1f s.\n', toc);
end

%==========================================================================
function opts = fill_eval_opts(opts, cfg)
    if ~isfield(opts, 'agents') || isempty(opts.agents)
        opts.agents = {'DQN', 'TD3'};
    end
    if ~isfield(opts, 'cohort_size') || isempty(opts.cohort_size)
        opts.cohort_size = cfg.num_patients;
    end
    if ~isfield(opts, 'run_cohort'), opts.run_cohort = true; end
    if ~isfield(opts, 'save_figures'), opts.save_figures = false; end  % legacy: use main('figures') instead
    if ~isfield(opts, 'figure_list') || isempty(opts.figure_list)
        opts.figure_list = [1, 2, 3, 4];
    end
    if ~isfield(opts, 'verbose'), opts.verbose = true; end
end

function verify_agents(cfg)
    for ag = {'DQN', 'TD3'}
        base = fullfile(cfg.agent_root, ag{1});
        sub = fullfile(base, cfg.dqn_run_subdir);
        if strcmpi(ag{1}, 'TD3')
            sub = fullfile(base, cfg.td3_run_subdir);
        end
        if ~isfile(fullfile(sub, 'agent_final.mat'))
            dirs = dir(fullfile(base, '*_PaperStyle'));
            if isempty(dirs)
                error('No pretrained agent under %s. Expected PaperStyle/agent_final.mat', base);
            end
        end
    end
end

function pmp_data = run_pmp_eval(cfg)
    pmp_raw = solve_pmp(cfg);
    pmp_data = package_pmp(pmp_raw, cfg);
    fprintf('  PMP: reduction=%.1f%%, J=%.3e, drug=%.2f\n', ...
        pmp_data.metrics.tumor_reduction, pmp_data.metrics.TotalCost, pmp_data.metrics.total_drug);
end

function out = rollout_nominal(cfg, agent, agentType, verbose)
    if nargin < 4, verbose = true; end
    envType = agentType;
    if strcmpi(agentType, 'TD3'), envType = 'DDPG'; end
    if verbose
        fprintf('  Building %s environment...\n', agentType);
    end
    [env, params, ~, ~] = build_env3(envType);
    if verbose
        fprintf('  Simulating %s nominal (%d steps, ode45 each — typically 1–3 min)...\n', ...
            agentType, params.max_steps);
    end
    tSim = tic;
    simData = simulate_agent(agent, env, params, verbose);
    if verbose
        fprintf('  %s rollout done in %.1f s.\n', agentType, toc(tSim));
    end
    m = rl_metrics(compute_metrics(simData.time, simData.N, simData.u, cfg));
    fprintf('  %s nominal: reduction=%.1f%%, J=%.3e, drug=%.2f\n', ...
        agentType, m.TumorReductionPercent, m.TotalCost, m.TotalDrugDosage);
    out = struct('simData', simData, 'metrics', m);
end

function seeds = setup_cohort_rng(n)
    rng(42, 'twister');
    seeds = randi(2^31 - 1, n, 1);
end

function out = rollout_cohort(cfg, agent, agentType, nPatients, seeds)
    envType = agentType;
    if strcmpi(agentType, 'TD3'), envType = 'DDPG'; end
    [~, base_params, ~, ~] = build_env3(envType);

    simCell = cell(nPatients, 1);
    metCell = cell(nPatients, 1);
    parfor p = 1:nPatients
        rng(seeds(p), 'twister');
        pp = cohort_params(base_params, cfg.cohort_variation);
        [env, params, ~, ~] = build_env3(envType, pp, false);
        sd = simulate_agent(agent, env, params);
        simCell{p} = sd;
        metCell{p} = rl_metrics(compute_metrics(sd.time, sd.N, sd.u, cfg));
    end

    red = cellfun(@(m) m.TumorReductionPercent, metCell);
    cost = cellfun(@(m) m.TotalCost, metCell);
    fprintf('  %s cohort: reduction=%.1f%% +/- %.1f,  J=%.2e +/- %.1e\n', ...
        agentType, mean(red), std(red), mean(cost), std(cost));

    out.simData = simCell;
    out.metrics = metCell;
end

function summary = build_summary(pmp_data, rl_results, cfg, opts)
    summary.timestamp = datestr(now);
    summary.cohort_size = opts.cohort_size;
    summary.pmp = pmp_data.metrics;

    for ag = {'dqn', 'td3'}
        key = ag{1};
        if isfield(rl_results, [key '_nominal'])
            summary.([key '_nominal']) = rl_results.([key '_nominal']).metrics;
        end
        if isfield(rl_results, [key '_multi'])
            m = rl_results.([key '_multi']).metrics;
            summary.([key '_cohort']).reduction_mean = mean(cellfun(@(x) x.TumorReductionPercent, m));
            summary.([key '_cohort']).reduction_std = std(cellfun(@(x) x.TumorReductionPercent, m));
            summary.([key '_cohort']).cost_mean = mean(cellfun(@(x) x.TotalCost, m));
            summary.([key '_cohort']).cost_std = std(cellfun(@(x) x.TotalCost, m));
        end
    end
    summary.cfg_snapshot = struct('num_patients', cfg.num_patients, ...
        'cohort_variation', cfg.cohort_variation, 'rng', '42 (cohort per-patient seeds)');
end

function print_summary(summary)
    fprintf('\n%s\n', repmat('-', 1, 60));
    fprintf('  NOMINAL COMPARISON\n');
    fprintf('%s\n', repmat('-', 1, 60));
    fprintf('| Controller | Reduction (%%) | Total drug | Total cost J |\n');
    fprintf('|------------|---------------|------------|-------------|\n');
    fprintf('| PMP        | %11.2f   | %9.2f   | %10.3e |\n', ...
        summary.pmp.tumor_reduction, summary.pmp.total_drug, summary.pmp.TotalCost);
    if isfield(summary, 'dqn_nominal')
        m = summary.dqn_nominal;
        fprintf('| DQN        | %11.2f   | %9.2f   | %10.3e |\n', ...
            m.TumorReductionPercent, m.TotalDrugDosage, m.TotalCost);
    end
    if isfield(summary, 'td3_nominal')
        m = summary.td3_nominal;
        fprintf('| TD3        | %11.2f   | %9.2f   | %10.3e |\n', ...
            m.TumorReductionPercent, m.TotalDrugDosage, m.TotalCost);
    end

    if isfield(summary, 'dqn_cohort')
        fprintf('\n%s\n', repmat('-', 1, 60));
        fprintf('  COHORT (mean +/- std)\n');
        fprintf('%s\n', repmat('-', 1, 60));
        fprintf('| Agent | Tumor reduction (%%) | Total cost J |\n');
        c = summary.dqn_cohort;
        fprintf('| DQN   | %6.2f +/- %5.2f      | %.2e +/- %.1e |\n', ...
            c.reduction_mean, c.reduction_std, c.cost_mean, c.cost_std);
        c = summary.td3_cohort;
        fprintf('| TD3   | %6.2f +/- %5.2f      | %.2e +/- %.1e |\n', ...
            c.reduction_mean, c.reduction_std, c.cost_mean, c.cost_std);
    end
    fprintf('%s\n\n', repmat('-', 1, 60));
end

function agent = load_agent_eval(cfg, agentType)
    sub = cfg.dqn_run_subdir;
    if strcmpi(agentType, 'TD3'), sub = cfg.td3_run_subdir; end
    base = fullfile(cfg.agent_root, agentType);
    run_dir = fullfile(base, sub);
    if ~isfolder(run_dir)
        dirs = dir(fullfile(base, '*_PaperStyle'));
        [~, ix] = max([dirs.datenum]);
        run_dir = fullfile(dirs(ix).folder, dirs(ix).name);
    end
    f = fullfile(run_dir, 'agent_final.mat');
    fprintf('  Loading %s from:\n    %s\n', agentType, run_dir);
    fprintf('  Reading agent_final.mat ...');
    drawnow;
    tLoad = tic;
    S = load(f, 'agent');
    agent = S.agent;
    fprintf(' done (%.1f s).\n', toc(tLoad));
    drawnow;
end

function pp = cohort_params(base_params, var_level)
    pp.R = base_params.R .* (1 + var_level * (2 * rand(size(base_params.R)) - 1));
    pp.Phi = base_params.Phi .* (1 + var_level * (2 * rand(size(base_params.Phi)) - 1));
    pp.N0_nominal = base_params.N0_nominal;
end

function pmp_data = package_pmp(pmp_raw, cfg)
    m = pmp_raw.metrics;
    pmp_data = pmp_raw;
    pmp_data.params.C = pmp_raw.params.C;
    pmp_data.metrics.avg_final_tumor = m.FinalTumor / cfg.model.m;
    pmp_data.metrics.total_drug = m.TotalDrug;
    pmp_data.metrics.tumor_reduction = m.TumorReduction;
    pmp_data.metrics.TotalCost = m.TotalCost;
end

function m = rl_metrics(raw)
    m.AverageFinalTumor = raw.FinalTumor;
    m.TumorReductionPercent = raw.TumorReduction;
    m.TotalDrugDosage = raw.TotalDrug;
    m.TotalCost = raw.TotalCost;
end
