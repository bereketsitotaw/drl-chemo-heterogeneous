function simData = simulate_agent(agent, env, params, verbose)
%SIMULATE_AGENT Roll out a trained RL agent in the tumor environment.

    if nargin < 4, verbose = false; end

    obs = env.reset();
    n = params.n;
    maxT = params.max_steps + 1;
    simData.N = zeros(n, maxT);
    simData.u = zeros(1, maxT);
    simData.N(:, 1) = obs(1:n) .* params.N_norm;
    simData.u(1) = 0;
    isDone = false;
    t = 0;
    reportEvery = max(1, floor(params.max_steps / 5));

    while ~isDone && t < params.max_steps
        t = t + 1;
        if isa(agent, 'rl.agent.rlDQNAgent')
            actionCell = getAction(agent, {obs});
        else
            actionCell = getAction(getActor(agent), {obs});
        end
        [nextObs, ~, isDone, loggedSignals] = env.step(actionCell{1});
        simData.N(:, t + 1) = nextObs(1:n) .* params.N_norm;
        simData.u(t + 1) = loggedSignals.clipped_action;
        obs = nextObs;
        if verbose && (mod(t, reportEvery) == 0 || isDone)
            fprintf('    step %d / %d\n', t, params.max_steps);
            drawnow;
        end
    end
    simData.N = simData.N(:, 1:t+1);
    simData.u = simData.u(1:t+1);
    simData.time = (0:t) * params.dt;
end
