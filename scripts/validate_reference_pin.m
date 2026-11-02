function ok = validate_reference_pin(cfg, pmp_data)
%VALIDATE_REFERENCE_PIN Compare config locks and optional PMP metrics to paper pins.

    if nargin < 1 || isempty(cfg), cfg = config(); end
    assert_publication_defaults(cfg);
    ref = reference_metrics();

    fprintf('Publication config pin: OK (alpha=%g, beta=%g, gamma=%g, n=%d, T=%g, Umax=%g, dqn_actions=%d)\n', ...
        cfg.cost.alpha, cfg.cost.beta, cfg.cost.gamma, cfg.model.m, cfg.model.T, ...
        cfg.cost.Umax, cfg.train.dqn_actions);

    fprintf('Table III paper pins: PMP red=%.2f%% drug=%.2f | DQN red=%.2f%% | TD3 red=%.2f%%\n', ...
        ref.table_iii.pmp_tumor_reduction_pct, ref.table_iii.pmp_total_drug, ...
        ref.table_iii.dqn_tumor_reduction_pct, ref.table_iii.td3_tumor_reduction_pct);

    ok = true;
    if nargin >= 2 && ~isempty(pmp_data)
        red = pmp_data.metrics.tumor_reduction;
        drug = pmp_data.metrics.total_drug;
        avg = pmp_data.metrics.avg_final_tumor;
        d_red = abs(red - ref.table_iii.pmp_tumor_reduction_pct);
        d_drug = abs(drug - ref.table_iii.pmp_total_drug);
        d_avg = abs(avg - ref.table_iii.pmp_final_tumor_avg);
        fprintf('Local PMP: red=%.2f%% (Δ=%.2f), drug=%.2f (Δ=%.2f), avg_final=%.3f (Δ=%.3f)\n', ...
            red, d_red, drug, d_drug, avg, d_avg);
        if d_red > ref.tol.pmp_reduction_abs || d_drug > ref.tol.pmp_drug_abs || d_avg > ref.tol.pmp_avg_tumor_abs
            ok = false;
            error('validate_reference_pin:PmpMismatch', ...
                'Local PMP outside soft Table III tolerance.');
        end
        fprintf('PMP soft pin: OK\n');
    else
        fprintf('PMP soft pin: skipped (no pmp_data). Run main(''smoke'') to compare.\n');
    end
end
