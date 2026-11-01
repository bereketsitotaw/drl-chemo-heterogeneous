function s = reproducibility_settings()
%REPRODUCIBILITY_SETTINGS RNG / train / smoke plumbing (non-paper overrides).
%
% Paper-scale scientific defaults remain in config.m. Smoke fields only shrink
% plumbing for main('smoke') and never redefine (alpha,beta,gamma) or dynamics.

    s.rng_seed = 42;
    s.rng_algorithm = 'twister';

    s.train.max_episodes = 3000;
    s.train.stop_reward = -286;
    s.train.dqn_actions = 7;
    s.train.pool_workers = 8;
    s.train.rng_base = 42;   % DQN=42, TD3=43

    % Smoke / plumbing only (explicitly non-paper)
    s.smoke.label = 'SMOKE (non-paper plumbing)';
    s.smoke.run_cohort = false;
    s.smoke.cohort_size = 2;
    s.smoke.max_episodes = 2;          % never used unless explicitly opted in
    s.smoke.pool_workers = 0;          % avoid spinning a pool for smoke
    s.smoke.run_rl_nominal = false;    % PMP-only by default (no retrain; RL optional)
end
