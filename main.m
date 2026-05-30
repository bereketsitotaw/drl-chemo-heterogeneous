function main(arg1, arg2, arg3, varargin)
%MAIN Adaptive chemotherapy DRL — simulation / evaluation (no training by default).
%
%   main('eval')                         PMP + DQN/TD3 nominal + 100-patient cohort
%   main('eval', struct('run_cohort',false))   quick nominal-only test
%   main('simulate', 'nominal')          single rollout
%   main('simulate', 'cohort', 100)      cohort rollout only
%   main('pmp')                          PMP benchmark only
%   main('figures')                      display Figs 1–4 (run eval first)
%   main('figures', struct('save_png', true))  also export PNGs
%
%   main('train')                      clean DQN + TD3 (~hours), auto PaperStyle

    root = fileparts(mfilename('fullpath'));
    addpath(fullfile(root, 'scripts'));
    clc; close all;
    rng(42, 'twister');

    if nargin < 1, arg1 = 'eval'; end

    switch lower(char(arg1))
        case {'help', 'all'}
            drl_pipeline(root, 'help');
        case {'eval', 'evaluate', 'test'}
            if nargin >= 2
                drl_pipeline(root, 'eval', arg2);
            else
                drl_pipeline(root, 'eval');
            end
        case 'pmp'
            drl_pipeline(root, 'pmp');
        case 'train'
            if nargin >= 2 && isstruct(arg2)
                drl_pipeline(root, 'train', arg2);
            elseif nargin < 2
                drl_pipeline(root, 'train', struct('agents', 'both', 'promote', true));
            else
                drl_pipeline(root, 'train', arg2, varargin{:});
            end
        case 'simulate'
            if nargin < 2, arg2 = 'nominal'; end
            if nargin < 3, arg3 = 'both'; end
            drl_pipeline(root, 'simulate', arg2, arg3, varargin{:});
        case 'collect'
            if nargin >= 2
                drl_pipeline(root, 'collect', arg2, varargin{:});
            else
                drl_pipeline(root, 'collect');
            end
        case {'figures', 'fig'}
            if nargin < 2, arg2 = [1, 2, 3, 4]; end
            drl_pipeline(root, 'figures', arg2);
        case 'ablation'
            drl_pipeline(root, 'ablation', varargin{:});
        otherwise
            error('Unknown command "%s". Run main(''help'').', arg1);
    end
end
