function assert_publication_defaults(cfg)
%ASSERT_PUBLICATION_DEFAULTS Lock paper model / cost / MDP pins.
%
% Fails fast if cost weights, horizon, action bounds, cohort settings, or
% DQN action count drift from the manuscript defaults. Does not invent
% alternate scientific profiles.

    if nargin < 1 || isempty(cfg)
        cfg = config();
    end

    expect_close(cfg.cost.alpha, 500, 'cost.alpha');
    expect_close(cfg.cost.beta, 400, 'cost.beta');
    expect_close(cfg.cost.gamma, 4000, 'cost.gamma');
    expect_close(cfg.cost.Umax, 3.0, 'cost.Umax');

    expect_close(cfg.model.T, 10, 'model.T');
    expect_close(cfg.model.n_steps, 100, 'model.n_steps');
    expect_close(cfg.model.m, 21, 'model.m');
    expect_close(cfg.model.N0(1), 10, 'model.N0');
    if numel(cfg.model.N0) ~= 21 || any(abs(cfg.model.N0 - 10) > 1e-12)
        error('assert_publication_defaults:N0Drift', ...
            'N0 must be 10 * ones(21,1) for the paper nominal patient.');
    end
    if abs(cfg.model.M_vec(1) - 0.5) > 1e-12 || any(abs(cfg.model.M_vec - 0.5) > 1e-12)
        error('assert_publication_defaults:MDrift', 'M_vec must be 0.5 on all traits.');
    end

    expect_close(cfg.num_patients, 100, 'num_patients');
    expect_close(cfg.cohort_variation, 0.10, 'cohort_variation');

    expect_close(cfg.train.max_episodes, 3000, 'train.max_episodes');
    expect_close(cfg.train.stop_reward, -286, 'train.stop_reward');
    expect_close(cfg.train.dqn_actions, 7, 'train.dqn_actions');
    expect_close(cfg.train.rng_base, 42, 'train.rng_base');

    % Trait grids / kinetics used in Eq. (1)
    x = linspace(0, 1, 21)';
    R_expect = 2 ./ (1 + 3 * x.^4);
    Phi_expect = -sin(x - 1) + 1.5;
    if max(abs(cfg.model.x - x)) > 1e-12
        error('assert_publication_defaults:TraitGridDrift', 'Trait grid x drifted.');
    end
    if max(abs(cfg.model.R_vec - R_expect)) > 1e-12
        error('assert_publication_defaults:RDrift', 'R_vec drifted from paper map.');
    end
    if max(abs(cfg.model.Phi_vec - Phi_expect)) > 1e-12
        error('assert_publication_defaults:PhiDrift', 'Phi_vec drifted from paper map.');
    end
end

function expect_close(got, expected, name)
    if abs(double(got) - double(expected)) > 1e-12
        error('assert_publication_defaults:Drift', ...
            '%s drifted: got %g, expected %g.', name, got, expected);
    end
end
