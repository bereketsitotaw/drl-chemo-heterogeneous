function figHandles = generate_figures(plotType, pmp_data, rl_results, mpc_results, cfg, figNum, figOpts)
%GENERATE_FIGURES Display paper figures (save PNG only if requested).
%
%   figOpts.display   (default true)  — show figure windows
%   figOpts.save_png  (default false) — write results/figures/*.png
%   figOpts.close_after_save (default false)

    if nargin < 7 || isempty(figOpts), figOpts = struct(); end
    figOpts = default_fig_opts(figOpts);

    styles = default_styles();
    hasMpc = ~isempty(mpc_results) && isstruct(mpc_results) && isfield(mpc_results, 'simData');

    if nargin < 6 || isempty(figNum)
        figNum = 0;
    end
    if figNum > 0
        which = figNum;
    elseif strcmpi(plotType, 'all')
        which = [1, 2, 3, 4];
    else
        which = sscanf(plotType, 'fig%d');
        if isempty(which), which = [1, 2, 3, 4]; end
    end

    figHandles = gobjects(numel(which), 1);
    for i = 1:numel(which)
        f = which(i);
        switch f
            case 1
                fig = new_fig('Fig 1: PMP summary');
                plot_pmp_summary(fig, pmp_data, styles);
                finish_fig(fig, cfg, 'fig1_pmp_summary', figOpts);
            case 2
                fig = new_fig('Fig 2: Nominal trajectories');
                plot_nominal_trajectories(fig, pmp_data, rl_results.td3_nominal, rl_results.dqn_nominal, styles);
                finish_fig(fig, cfg, 'fig2_nominal_trajectories', figOpts);
            case 3
                fig = new_fig('Fig 3: Cohort Pareto');
                plot_rl_analysis_summary(fig, rl_results, pmp_data, styles);
                finish_fig(fig, cfg, 'fig3_rl_analysis_pareto', figOpts);
            case 4
                fig = new_fig('Fig 4: Robustness');
                plot_robustness_trajectories(fig, pmp_data, rl_results, mpc_results, styles, cfg, hasMpc);
                finish_fig(fig, cfg, 'fig4_robustness_comparison', figOpts);
        end
        figHandles(i) = fig;
    end

    if figOpts.display
        fprintf('Figures displayed. Close windows manually or run savefig from each.\n');
    end
    if figOpts.save_png
        fprintf('PNG copies saved under %s\n', cfg.figures_dir);
    end
end

%==========================================================================
function opts = default_fig_opts(opts)
    if ~isfield(opts, 'display'), opts.display = true; end
    if ~isfield(opts, 'save_png'), opts.save_png = false; end
    if ~isfield(opts, 'close_after_save'), opts.close_after_save = false; end
end

function styles = default_styles()
    styles.PMPColor = [0 0 0];
    styles.TD3Color = [0.85 0.33 0.10];
    styles.DQNColor = [0 0.45 0.74];
    styles.MPCColor = [0.5 0 0.5];
    styles.TitleSize = 12;
    styles.LabelSize = 11;
    styles.LineWidth = 2.0;
    styles.RobustnessLineWidth = 2.5;
end

function fig = new_fig(figName)
    fig = figure( ...
        'Name', figName, ...
        'NumberTitle', 'off', ...
        'Color', 'w', ...
        'Units', 'normalized', ...
        'Position', [0.04 0.04 0.92 0.88], ...
        'Visible', 'on');
    figure(fig);
end

function finish_fig(fig, cfg, baseName, figOpts)
    drawnow;
    figure(fig);
    if figOpts.save_png
        save_fig_png(fig, cfg, baseName);
        if figOpts.close_after_save && isgraphics(fig)
            close(fig);
        end
    end
end

function save_fig_png(fig, cfg, name)
    if ~exist(cfg.figures_dir, 'dir'), mkdir(cfg.figures_dir); end
    png = fullfile(cfg.figures_dir, [name '.png']);
    figure(fig);
    drawnow;
    try
        exportgraphics(fig, png, 'Resolution', 300);
    catch
        frame = getframe(fig);
        imwrite(frame.cdata, png);
    end
    fprintf('  Saved %s\n', png);
end

%==========================================================================
function plot_pmp_summary(fig, pmp_data, styles)
    figure(fig); clf(fig);

    [N_state, N_time, L_time] = pmp_trajectory_mats(pmp_data);
    time = pmp_data.time(:);
    u_opt = pmp_data.u(:);
    n = min(numel(time), numel(u_opt));
    time = time(1:n);
    u_opt = u_opt(1:n);
    x = pmp_data.params.x;
    C = pmp_data.params.C(:)';

    subplot(2, 2, 1);
    plot(time, u_opt, 'Color', styles.PMPColor, 'LineWidth', styles.LineWidth);
    title('(a) Optimal control u* (t)', 'FontSize', styles.TitleSize);
    xlabel('Time (years)'); ylabel('Dosage'); grid on;

    subplot(2, 2, 2);
    plot(time, pmp_data.params.e' * N_state(:, 1:n), 'Color', styles.PMPColor, 'LineWidth', styles.LineWidth);
    title('(b) Average tumor burden', 'FontSize', styles.TitleSize);
    xlabel('Time (years)'); ylabel('Average population'); grid on;

    subplot(2, 2, 3);
    hold on;
    plot(x, N_state(:, 1), 'k--', 'LineWidth', styles.LineWidth, 'DisplayName', 'Initial');
    plot(x, N_state(:, end), 'r-', 'LineWidth', styles.LineWidth, 'DisplayName', 'Final');
    legend('Location', 'best'); title('(c) Trait distribution', 'FontSize', styles.TitleSize);
    xlabel('Trait x'); ylabel('Population'); grid on;

    subplot(2, 2, 4);
    hold on;
    if ~isempty(L_time)
        phi = sum((L_time(1:n, :) .* N_time(1:n, :)) .* C, 2);
        Hu = pmp_data.params.gamma - max(0, phi) ./ (1 + u_opt).^2;
        Huu = 2 * max(0, phi) ./ (1 + u_opt).^3;
        plot(time, Hu, 'Color', styles.DQNColor, 'LineWidth', styles.LineWidth, 'DisplayName', 'H_u');
        plot(time, Huu, 'Color', styles.TD3Color, 'LineStyle', '-.', 'LineWidth', styles.LineWidth, 'DisplayName', 'H_{uu}');
        legend('Location', 'best');
    end
    title('(d) Optimality check', 'FontSize', styles.TitleSize);
    xlabel('Time (years)'); grid on;

    sgtitle(fig, 'PMP Optimal Control (Nominal Patient)', 'FontSize', styles.TitleSize + 1, 'FontWeight', 'bold');
end

function plot_nominal_trajectories(fig, pmp, td3, dqn, styles)
    figure(fig); clf(fig);
    x = pmp.params.x;
    e = pmp.params.e;
    [Np, ~, ~] = pmp_trajectory_mats(pmp);

    subplot(2, 2, [1, 2]);
    hold on;
    plot(pmp.time(:), pmp.u(:), 'Color', styles.PMPColor, 'LineWidth', styles.LineWidth, 'DisplayName', 'PMP');
    plot(td3.simData.time(:), td3.simData.u(:), '--', 'Color', styles.TD3Color, 'LineWidth', styles.LineWidth, 'DisplayName', 'TD3');
    stairs(dqn.simData.time(:), dqn.simData.u(:), '-.', 'Color', styles.DQNColor, 'LineWidth', styles.LineWidth, 'DisplayName', 'DQN');
    legend('Location', 'best'); title('(a) Drug dosage', 'FontSize', styles.TitleSize);
    xlabel('Time (years)'); ylabel('Dosage'); grid on; ylim([-0.1 3.3]);

    subplot(2, 2, 3);
    hold on;
    plot(pmp.time(:), e' * Np, 'Color', styles.PMPColor, 'LineWidth', styles.LineWidth, 'DisplayName', 'PMP');
    plot(td3.simData.time(:), e' * td3.simData.N, '--', 'Color', styles.TD3Color, 'LineWidth', styles.LineWidth, 'DisplayName', 'TD3');
    plot(dqn.simData.time(:), e' * dqn.simData.N, '-.', 'Color', styles.DQNColor, 'LineWidth', styles.LineWidth, 'DisplayName', 'DQN');
    legend('Location', 'best'); title('(b) Average tumor burden', 'FontSize', styles.TitleSize);
    xlabel('Time (years)'); ylabel('Average population'); grid on;

    subplot(2, 2, 4);
    hold on;
    plot(x, Np(:, end), 'Color', styles.PMPColor, 'LineWidth', styles.LineWidth, 'DisplayName', 'PMP');
    plot(x, td3.simData.N(:, end), '--', 'Color', styles.TD3Color, 'LineWidth', styles.LineWidth, 'DisplayName', 'TD3');
    plot(x, dqn.simData.N(:, end), '-.', 'Color', styles.DQNColor, 'LineWidth', styles.LineWidth, 'DisplayName', 'DQN');
    legend('Location', 'best'); title('(c) Final tumor profile', 'FontSize', styles.TitleSize);
    xlabel('Phenotypic trait x'); ylabel('Population'); grid on;

    sgtitle(fig, 'Controller performance — nominal patient', 'FontSize', styles.TitleSize + 1, 'FontWeight', 'bold');
end

function plot_rl_analysis_summary(fig, rl_results, pmp_data, styles)
    figure(fig); clf(fig);

    td3_drug = cellfun(@(m) m.TotalDrugDosage, rl_results.td3_multi.metrics);
    td3_tumor = cellfun(@(m) m.AverageFinalTumor, rl_results.td3_multi.metrics);
    dqn_drug = cellfun(@(m) m.TotalDrugDosage, rl_results.dqn_multi.metrics);
    dqn_tumor = cellfun(@(m) m.AverageFinalTumor, rl_results.dqn_multi.metrics);

    hold on;
    scatter(dqn_drug, dqn_tumor, 45, 's', 'MarkerFaceColor', styles.DQNColor, ...
        'MarkerFaceAlpha', 0.65, 'MarkerEdgeColor', 'k', 'DisplayName', 'DQN cohort');
    scatter(td3_drug, td3_tumor, 45, 'o', 'MarkerFaceColor', styles.TD3Color, ...
        'MarkerFaceAlpha', 0.65, 'MarkerEdgeColor', 'k', 'DisplayName', 'TD3 cohort');
    scatter(pmp_data.metrics.total_drug, pmp_data.metrics.avg_final_tumor, 120, 'p', ...
        'MarkerFaceColor', [1 0.85 0], 'MarkerEdgeColor', 'k', 'LineWidth', 1.2, 'DisplayName', 'PMP nominal');
    xlabel('Total drug dosage', 'FontSize', styles.LabelSize);
    ylabel('Final tumor burden', 'FontSize', styles.LabelSize);
    title('Cohort outcomes: efficacy vs drug (Fig. 3)', 'FontSize', styles.TitleSize);
    legend('Location', 'best'); grid on;
end

function plot_robustness_trajectories(fig, pmp_data, rl_results, mpc_results, styles, cfg, hasMpc)
    figure(fig); clf(fig);

    median_dqn = find_median_trajectory(rl_results.dqn_multi);
    median_td3 = find_median_trajectory(rl_results.td3_multi);
    e_vec = pmp_data.params.e;

    [t_dqn, b_dqn] = avg_burden_series(median_dqn, e_vec);
    [t_td3, b_td3] = avg_burden_series(median_td3, e_vec);

    subplot(2, 1, 1);
    hold on;
    plot(t_dqn, b_dqn, '-.', 'Color', styles.DQNColor, 'LineWidth', styles.RobustnessLineWidth, 'DisplayName', 'DQN');
    plot(t_td3, b_td3, '--', 'Color', styles.TD3Color, 'LineWidth', styles.RobustnessLineWidth, 'DisplayName', 'TD3');
    if hasMpc
        median_mpc = find_median_trajectory(mpc_results);
        [t_mpc, b_mpc] = avg_burden_series(median_mpc, e_vec);
        plot(t_mpc, b_mpc, ':', 'Color', styles.MPCColor, 'LineWidth', styles.RobustnessLineWidth, 'DisplayName', 'MPC');
    end
    legend('Location', 'best'); title('Median tumor burden', 'FontSize', styles.TitleSize);
    xlabel('Time (years)'); ylabel('Average population'); grid on;

    [t_dqn, u_dqn] = align_tu(median_dqn.time, median_dqn.u);
    [t_td3, u_td3] = align_tu(median_td3.time, median_td3.u);

    subplot(2, 1, 2);
    hold on;
    plot(t_dqn, u_dqn, '-.', 'Color', styles.DQNColor, 'LineWidth', styles.RobustnessLineWidth, 'DisplayName', 'DQN');
    plot(t_td3, u_td3, '--', 'Color', styles.TD3Color, 'LineWidth', styles.RobustnessLineWidth, 'DisplayName', 'TD3');
    if hasMpc
        [t_mpc, u_mpc] = align_tu(median_mpc.time, median_mpc.u);
        plot(t_mpc, u_mpc, ':', 'Color', styles.MPCColor, 'LineWidth', styles.RobustnessLineWidth, 'DisplayName', 'MPC');
    end
    legend('Location', 'best');
    title('Median drug dosage', 'FontSize', styles.TitleSize);
    xlabel('Time (years)'); ylabel('Dosage'); ylim([-0.1, cfg.cost.Umax + 0.2]); grid on;

    sgtitle(fig, 'Median patient trajectories — 100-patient cohort (Fig. 4)', 'FontSize', styles.TitleSize + 1, 'FontWeight', 'bold');
end

function [t, burden] = avg_burden_series(sim, e_vec)
    t = sim.time(:)';
    burden = (e_vec' * sim.N)';
    n = min(numel(t), numel(burden));
    t = t(1:n);
    burden = burden(1:n);
end

function [t, u] = align_tu(t, u)
    t = t(:)';
    u = u(:)';
    n = min(numel(t), numel(u));
    t = t(1:n);
    u = u(1:n);
end

function median_traj = find_median_trajectory(results_data)
    reduction = cellfun(@(m) m.TumorReductionPercent, results_data.metrics);
    [~, median_idx] = min(abs(reduction - median(reduction)));
    median_traj = results_data.simData{median_idx};
end

function [N_state, N_time, L_time] = pmp_trajectory_mats(pmp_data)
    m = numel(pmp_data.params.x);
    N_time = pmp_data.N;
    if isfield(pmp_data, 'L'), L_time = pmp_data.L; else, L_time = []; end
    if size(N_time, 2) ~= m
        N_time = N_time';
        if ~isempty(L_time), L_time = L_time'; end
    end
    N_state = N_time';
end
