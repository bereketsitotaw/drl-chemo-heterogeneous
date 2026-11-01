function results = run_smoke_tests()
%RUN_SMOKE_TESTS Lightweight MATLAB checks (no full train / cohort).
%
%   From repo root:
%     main('test')
%     run('tests/run_smoke_tests.m')

    here = fileparts(mfilename('fullpath'));
    root = fileparts(here);
    addpath(fullfile(root, 'scripts'));
    addpath(fullfile(root, 'configs'));
    addpath(here);

    tests = { ...
        @test_publication_cost_weights, ...
        @test_model_maps, ...
        @test_reference_metrics_pin, ...
        @test_dynamics_finite, ...
        @test_publication_assert, ...
        @test_smoke_overrides, ...
        @test_metrics_cost_identity ...
        };

    names = { ...
        'publication_cost_weights', ...
        'model_maps', ...
        'reference_metrics_pin', ...
        'dynamics_finite', ...
        'publication_assert', ...
        'smoke_overrides', ...
        'metrics_cost_identity' ...
        };

    n = numel(tests);
    passed = false(n, 1);
    messages = strings(n, 1);

    fprintf('=== drl-chemo-heterogeneous smoke tests ===\n');
    for i = 1:n
        try
            tests{i}();
            passed(i) = true;
            messages(i) = "ok";
            fprintf('  PASS  %s\n', names{i});
        catch ME
            passed(i) = false;
            messages(i) = string(ME.message);
            fprintf('  FAIL  %s — %s\n', names{i}, ME.message);
        end
    end

    results = struct('names', {names}, 'passed', passed, 'messages', messages);
    fprintf('=== %d/%d passed ===\n', nnz(passed), n);
    assert(all(passed), '%d smoke test(s) failed', nnz(~passed));
end

function test_publication_cost_weights()
    cfg = config();
    assert(abs(cfg.cost.alpha - 500) < 1e-12);
    assert(abs(cfg.cost.beta - 400) < 1e-12);
    assert(abs(cfg.cost.gamma - 4000) < 1e-12);
    assert(abs(cfg.cost.Umax - 3) < 1e-12);
    assert(cfg.train.dqn_actions == 7);
    assert(cfg.num_patients == 100);
    assert(abs(cfg.cohort_variation - 0.10) < 1e-12);
end

function test_model_maps()
    cfg = config();
    assert(cfg.model.m == 21);
    assert(abs(cfg.model.T - 10) < 1e-12);
    assert(cfg.model.n_steps == 100);
    x = linspace(0, 1, 21)';
    assert(max(abs(cfg.model.R_vec - (2 ./ (1 + 3 * x.^4)))) < 1e-12);
    assert(max(abs(cfg.model.Phi_vec - (-sin(x - 1) + 1.5))) < 1e-12);
end

function test_reference_metrics_pin()
    ref = reference_metrics();
    assert(abs(ref.table_iii.pmp_tumor_reduction_pct - 87.72) < 1e-9);
    assert(abs(ref.table_iii.dqn_tumor_reduction_pct - 86.93) < 1e-9);
    assert(abs(ref.table_iii.td3_tumor_reduction_pct - 89.11) < 1e-9);
    assert(abs(ref.table_iv.dqn_mean_reduction_pct - 82.78) < 1e-9);
    assert(abs(ref.table_iv.td3_mean_reduction_pct - 84.07) < 1e-9);
end

function test_dynamics_finite()
    cfg = config();
    dyn.e = cfg.model.e;
    dyn.R = cfg.model.R;
    dyn.Phi = cfg.model.Phi;
    dyn.M = cfg.model.M;
    N = cfg.model.N0;
    dN = tumor_dynamics(N, 1.5, dyn);
    assert(all(isfinite(dN)));
    assert(isequal(size(dN), size(N)));
end

function test_publication_assert()
    cfg = config();
    assert_publication_defaults(cfg);
end

function test_smoke_overrides()
    opts = apply_smoke_overrides(struct());
    assert(opts.run_cohort == false);
    assert(opts.smoke_mode == true);
    assert(contains(string(opts.smoke_label), "SMOKE"));
end

function test_metrics_cost_identity()
    cfg = config();
    t = linspace(0, cfg.model.T, cfg.model.n_steps + 1);
    N = repmat(cfg.model.N0, 1, numel(t));
    u = 0.5 * ones(1, numel(t));
    m = compute_metrics(t, N, u, cfg);
    assert(isfinite(m.TotalCost));
    assert(m.TotalCost > 0);
    % Constant N=10*ones, u=0.5 → terminal alpha*sum(N)=500*21*10=105000
    assert(abs(m.TerminalCost - 500 * 21 * 10) < 1e-6);
end
