function opts = apply_smoke_overrides(opts)
%APPLY_SMOKE_OVERRIDES Tiny non-paper plumbing overrides for main('smoke').
%
% Does not change cost weights, dynamics maps, Umax, T, or paper train defaults
% stored in config.m — only evaluation plumbing flags.

    if nargin < 1 || isempty(opts), opts = struct(); end
    s = reproducibility_settings();

    opts.run_cohort = s.smoke.run_cohort;
    opts.cohort_size = s.smoke.cohort_size;
    opts.save_figures = false;
    opts.verbose = true;
    opts.smoke_mode = true;
    opts.smoke_label = s.smoke.label;
    opts.run_rl_nominal = s.smoke.run_rl_nominal;
end
