function manifest = run_training(cfg, opts)
%RUN_TRAINING Clean DQN + TD3 training from scratch (paper hyperparameters).
%
%   manifest = run_training()
%   manifest = run_training(cfg, opts)
%
%   opts.agents       'both' | 'DQN' | 'TD3'  (default 'both')
%   opts.max_episodes  default 3000
%   opts.run_tag       folder suffix, default 'Clean'
%   opts.promote       if true, point savedAgents/*/PaperStyle to new runs
%   opts.pool_workers  parallel pool size (default 8)
%   opts.rng_base      base seed (DQN=base, TD3=base+1)

    if nargin < 1 || isempty(cfg), cfg = config(); end
    if nargin < 2, opts = struct(); end
    opts = fill_train_opts(opts, cfg);

    fprintf('\n%s\n', repmat('=', 1, 60));
    fprintf('  CLEAN TRAINING — DQN & TD3 (fresh agents)\n');
    fprintf('%s\n\n', repmat('=', 1, 60));

    pool = setup_pool(opts.pool_workers);

    manifest = struct();
    manifest.started = datestr(now);
    manifest.opts = opts;
    manifest.cfg_train = cfg.train;
    manifest.runs = struct();

    agents = resolve_train_agents(opts.agents);
    for k = 1:numel(agents)
        ag = agents{k};
        seed = opts.rng_base + (k - 1);
        fprintf('\n########## %s ##########\n', ag);
        manifest.runs.(lower(ag)) = train_one_clean(cfg, ag, opts, seed);
    end

    if opts.promote
        fprintf('\n--- Promoting new runs to PaperStyle (evaluation default) ---\n');
        for k = 1:numel(agents)
            ag = agents{k};
            rd = manifest.runs.(lower(ag)).run_dir;
            promote_agent_run(cfg, ag, rd);
        end
        fprintf('Evaluation will load: savedAgents/<DQN|TD3>/PaperStyle\n');
    else
        fprintf('\nNot promoted. Set opts.promote=true or update cfg.dqn_run_subdir / cfg.td3_run_subdir.\n');
        for k = 1:numel(agents)
            ag = agents{k};
            fprintf('  %s: %s\n', ag, manifest.runs.(lower(ag)).run_dir);
        end
    end

    manifest.finished = datestr(now);
    manifest.pool_workers = pool.NumWorkers;
    mpath = fullfile(cfg.results_dir, ['training_manifest_' datestr(now, 'yyyymmdd_HHMMSS') '.mat']);
    save(mpath, 'manifest', '-v7.3');
    fprintf('\nManifest saved: %s\n', mpath);
end

%==========================================================================
function opts = fill_train_opts(opts, cfg)
    if ~isfield(opts, 'agents') || isempty(opts.agents), opts.agents = 'both'; end
    if ~isfield(opts, 'max_episodes') || isempty(opts.max_episodes)
        opts.max_episodes = cfg.train.max_episodes;
    end
    if ~isfield(opts, 'run_tag') || isempty(opts.run_tag), opts.run_tag = 'Clean'; end
    if ~isfield(opts, 'promote'), opts.promote = true; end
    if ~isfield(opts, 'pool_workers') || isempty(opts.pool_workers)
        opts.pool_workers = cfg.train.pool_workers;
    end
    if ~isfield(opts, 'rng_base') || isempty(opts.rng_base)
        opts.rng_base = cfg.train.rng_base;
    end
end

function agents = resolve_train_agents(spec)
    switch lower(char(spec))
        case 'both', agents = {'DQN', 'TD3'};
        case 'dqn',  agents = {'DQN'};
        case 'td3',  agents = {'TD3'};
        otherwise, error('opts.agents must be both, DQN, or TD3.');
    end
end

function pool = setup_pool(nWorkers)
    pool = gcp('nocreate');
    if isempty(pool)
        pool = parpool('local', nWorkers);
        fprintf('Started parallel pool (%d workers).\n', pool.NumWorkers);
    else
        fprintf('Using parallel pool (%d workers).\n', pool.NumWorkers);
    end
end

function info = train_one_clean(cfg, agentType, opts, seed)
    envType = agentType;
    if strcmpi(agentType, 'TD3'), envType = 'DDPG'; end

    rng(seed, 'twister');
    fprintf('RNG seed = %d\n', seed);

    [env, params, obsInfo, actInfo] = build_env3(envType, [], true);
    if strcmpi(agentType, 'DQN')
        actionSet = linspace(0, params.u_max, cfg.train.dqn_actions);
        actInfo = rlFiniteSetSpec(actionSet);
        fprintf('DQN: %d discrete actions on [0, %.1f]\n', numel(actionSet), params.u_max);
    end

    agent = build_agent(agentType, obsInfo, actInfo, params);

    stamp = datestr(now, 'yyyymmdd_HHMMSS');
    run_name = sprintf('%s_%s', stamp, opts.run_tag);
    run_dir = fullfile(cfg.agent_root, agentType, run_name);
    mkdir(run_dir);

    trainOpts = rlTrainingOptions(...
        'MaxEpisodes', opts.max_episodes, ...
        'MaxStepsPerEpisode', params.max_steps, ...
        'Verbose', false, ...
        'Plots', 'training-progress', ...
        'ScoreAveragingWindowLength', 20, ...
        'StopTrainingCriteria', 'AverageReward', ...
        'StopTrainingValue', cfg.train.stop_reward, ...
        'UseParallel', true, ...
        'ParallelizationOptions', rl.option.ParallelTraining('Mode', 'async'));

    manifest_local = struct();
    manifest_local.agent = agentType;
    manifest_local.run_dir = run_dir;
    manifest_local.run_name = run_name;
    manifest_local.seed = seed;
    manifest_local.max_episodes = opts.max_episodes;
    manifest_local.max_steps = params.max_steps;
    manifest_local.stop_reward = cfg.train.stop_reward;
    manifest_local.dqn_actions = cfg.train.dqn_actions;
    manifest_local.random_init_training = true;
    manifest_local.started = datestr(now);
    save(fullfile(run_dir, 'training_manifest.mat'), 'manifest_local', 'cfg', 'opts', '-v7.3');

    fprintf('Training %s -> %s\n', agentType, run_dir);
    fprintf('(%d episodes; stop avg reward <= %.1f; may take 1–3 hours)\n', ...
        opts.max_episodes, cfg.train.stop_reward);

    tTrain = tic;
    trainingStats = train(agent, env, trainOpts);
    elapsed = toc(tTrain);

    save(fullfile(run_dir, 'agent_final.mat'), 'agent', '-v7.3');
    save(fullfile(run_dir, 'stats_final.mat'), 'trainingStats', '-v7.3');

    save_training_plot(run_dir, trainingStats, agentType);

    info = manifest_local;
    info.elapsed_sec = elapsed;
    info.finished = datestr(now);
    info.final_avg_reward = trainingStats.AverageReward(end);
    info.episodes_completed = numel(trainingStats.EpisodeIndex);

    fprintf('Done %s in %.1f min | episodes=%d | final avg reward=%.2f\n', ...
        agentType, elapsed/60, info.episodes_completed, info.final_avg_reward);
    fprintf('Saved: %s\n', run_dir);
end

function save_training_plot(run_dir, trainingStats, agentType)
    try
        fig = figure('Visible', 'off', 'Color', 'w');
        plot(trainingStats.EpisodeIndex, trainingStats.AverageReward, 'k-', 'LineWidth', 1.5);
        grid on;
        xlabel('Episode');
        ylabel('Average reward');
        title(sprintf('%s training curve', agentType));
        exportgraphics(fig, fullfile(run_dir, 'training_curve.png'), 'Resolution', 200);
        close(fig);
    catch
        % plotting optional
    end
end

function promote_agent_run(cfg, agentType, run_dir)
    base = fullfile(cfg.agent_root, agentType);
    linkPath = fullfile(base, 'PaperStyle');
    [~, folderName] = fileparts(run_dir);

    if exist(linkPath, 'dir') || exist(linkPath, 'file')
        backup = fullfile(base, ['PaperStyle_backup_' datestr(now, 'yyyymmdd_HHMMSS')]);
        fprintf('  Backing up old PaperStyle -> %s\n', backup);
        movefile(linkPath, backup);
    end

    if ismac || isunix
        oldpwd = pwd;
        cd(base);
        status = system(sprintf('ln -s "%s" PaperStyle', folderName));
        cd(oldpwd);
        if status ~= 0
            copyfile(run_dir, linkPath);
        end
    else
        copyfile(run_dir, linkPath);
    end
    fprintf('  %s PaperStyle -> %s\n', agentType, run_dir);
end
