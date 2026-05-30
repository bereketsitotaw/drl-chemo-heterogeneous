% build_agent.m
%
% Creates a reinforcement learning agent with specified hyperparameters.
% =========================================================================
function agent = build_agent(agentType, obsInfo, actInfo, params)
    switch agentType
        case 'DQN'
            agent = create_dqn_agent(obsInfo, actInfo, params);
        case 'TD3'
            agent = create_td3_agent(obsInfo, actInfo, params);
        case 'SAC'
            agent = create_sac_agent(obsInfo, actInfo, params);
        otherwise
            error('Invalid agent type specified: %s', agentType);
    end
end

%% --- DQN Agent Definition ---
function agent = create_dqn_agent(obsInfo, actInfo, params)
    criticNetwork = [
        featureInputLayer(obsInfo.Dimension(1), 'Name', 'state')
        fullyConnectedLayer(400)
        reluLayer
        fullyConnectedLayer(300)
        reluLayer
        fullyConnectedLayer(numel(actInfo.Elements), 'Name', 'q_values')
    ];
    criticOptions = rlRepresentationOptions('LearnRate', 5e-5, 'GradientThreshold', 1);
    critic = rlQValueRepresentation(criticNetwork, obsInfo, actInfo, 'Observation', {'state'}, 'Options', criticOptions);
    agentOptions = rlDQNAgentOptions(...
        'SampleTime', params.dt, ...
        'UseDoubleDQN', true, ...
        'TargetSmoothFactor', 1e-3, ...
        'ExperienceBufferLength', 1e6, ...
        'DiscountFactor', 0.995, ...
        'MiniBatchSize', 256);
    agentOptions.EpsilonGreedyExploration.Epsilon = 1.0;
    agentOptions.EpsilonGreedyExploration.EpsilonDecay = 1e-4;
    agentOptions.EpsilonGreedyExploration.EpsilonMin = 0.1;
    agent = rlDQNAgent(critic, agentOptions);
end

%% --- TD3 Agent Definition ---
function agent = create_td3_agent(obsInfo, actInfo, params)
    actorLayers = [
        featureInputLayer(obsInfo.Dimension(1), 'Name', 'state')
        fullyConnectedLayer(400, 'Name', 'fc1')
        batchNormalizationLayer('Name', 'bn1')
        reluLayer('Name', 'relu1')
        fullyConnectedLayer(300, 'Name', 'fc2')
        reluLayer('Name', 'relu2')
        fullyConnectedLayer(actInfo.Dimension(1), 'Name', 'action_output')
        sigmoidLayer('Name', 'sigmoid')
        scalingLayer('Name', 'scale', 'Scale', actInfo.UpperLimit)
    ];
    actorNet = dlnetwork(actorLayers);
    actorOptions = rlOptimizerOptions('LearnRate', 1.5e-4, 'GradientThreshold', 1);
    actor = rlContinuousDeterministicActor(actorNet, obsInfo, actInfo);
    statePath = [featureInputLayer(obsInfo.Dimension(1),'Name','state')];
    actionPath = [featureInputLayer(actInfo.Dimension(1),'Name','action')];
    commonPath = [
        concatenationLayer(1,2,'Name','concat')
        fullyConnectedLayer(400)
        reluLayer
        fullyConnectedLayer(300)
        reluLayer
        fullyConnectedLayer(1)];
    criticGraph = layerGraph(statePath);
    criticGraph = addLayers(criticGraph, actionPath);
    criticGraph = addLayers(criticGraph, commonPath);
    criticGraph = connectLayers(criticGraph, 'state', 'concat/in1');
    criticGraph = connectLayers(criticGraph, 'action', 'concat/in2');
    critic1Net = dlnetwork(criticGraph);
    critic2Net = dlnetwork(criticGraph);
    criticOptions = rlOptimizerOptions('LearnRate', 5e-5, 'GradientThreshold', 1);
    critic1 = rlQValueFunction(critic1Net, obsInfo, actInfo, 'ObservationInputNames', 'state', 'ActionInputNames', 'action');
    critic2 = rlQValueFunction(critic2Net, obsInfo, actInfo, 'ObservationInputNames', 'state', 'ActionInputNames', 'action');
    agentOptions = rlTD3AgentOptions(...
        'SampleTime', params.dt, ...
        'ActorOptimizerOptions', actorOptions, ...
        'CriticOptimizerOptions', criticOptions, ...
        'ExperienceBufferLength', 1e6, ...
        'DiscountFactor', 0.995, ...
        'MiniBatchSize', 256, ...
        'TargetSmoothFactor', 1e-3, ...
        'PolicyUpdateFrequency', 2, ...
        'TargetPolicySmoothModel', rl.option.GaussianActionNoise('StandardDeviation', 0.3), ...
        'ExplorationModel', rl.option.GaussianActionNoise(...
            'StandardDeviation', 0.5, ...
            'StandardDeviationDecayRate', 1e-6));
    agent = rlTD3Agent(actor, [critic1; critic2], agentOptions);
end

%% --- SAC Agent Definition ---
function agent = create_sac_agent(obsInfo, actInfo, params)
    statePath = [
        featureInputLayer(obsInfo.Dimension(1), 'Name', 'state')
        fullyConnectedLayer(256)
        reluLayer
        fullyConnectedLayer(256)
        reluLayer];
    
    meanPath = [
        fullyConnectedLayer(actInfo.Dimension(1), 'Name', 'mean')
        tanhLayer('Name','tanh')
        scalingLayer('Name','action', 'Scale', actInfo.UpperLimit)];
    stdPath = [
        fullyConnectedLayer(actInfo.Dimension(1), 'Name', 'std')
        softplusLayer('Name','softplus')];
    
    actorNetwork = layerGraph(statePath);
    actorNetwork = addLayers(actorNetwork, meanPath);
    actorNetwork = addLayers(actorNetwork, stdPath);
    actorNetwork = connectLayers(actorNetwork,'relu_2','mean/in');
    actorNetwork = connectLayers(actorNetwork,'relu_2','std/in');
    
    actorOptions = rlOptimizerOptions('LearnRate', 1e-4, 'GradientThreshold', 1);
    actor = rlContinuousGaussianActor(dlnetwork(actorNetwork), obsInfo, actInfo, ...
        'ActionMeanOutputNames','action',...
        'ActionStandardDeviationOutputNames','softplus');
    
    criticNetwork = [
        featureInputLayer(obsInfo.Dimension(1), 'Normalization', 'none', 'Name', 'state')
        concatenationLayer(1, 2, 'Name', 'concat')
        fullyConnectedLayer(256, 'Name', 'fc1')
        reluLayer('Name', 'relu1')
        fullyConnectedLayer(256, 'Name', 'fc2')
        reluLayer('Name', 'relu2')
        fullyConnectedLayer(1, 'Name', 'qvalue')];
    actionPath = [
        featureInputLayer(actInfo.Dimension(1), 'Normalization', 'none', 'Name', 'action')
        fullyConnectedLayer(256, 'Name', 'fc_action')];
    criticGraph = layerGraph(criticNetwork);
    criticGraph = addLayers(criticGraph, actionPath);
    criticGraph = connectLayers(criticGraph, 'fc_action', 'concat/in2');
    
    criticOptions = rlOptimizerOptions('LearnRate', 1e-4, 'GradientThreshold', 1);
    critic1 = rlQValueFunction(dlnetwork(criticGraph), obsInfo, actInfo,...
        'ObservationInputNames', 'state', 'ActionInputNames', 'action');
    critic2 = rlQValueFunction(dlnetwork(criticGraph), obsInfo, actInfo,...
        'ObservationInputNames', 'state', 'ActionInputNames', 'action');
    
    agentOptions = rlSACAgentOptions(...
        'SampleTime', params.dt, ...
        'ActorOptimizerOptions', actorOptions, ...
        'CriticOptimizerOptions', criticOptions, ...
        'ExperienceBufferLength', 1e6, ...
        'DiscountFactor', 0.99, ...
        'MiniBatchSize', 256, ...
        'TargetSmoothFactor', 5e-3);
    
    % MODIFIED: In R2024a, automatic entropy tuning is the default.
    % The 'EntropyWeight' and 'EntropyWeightOptions' properties have been removed.
    % Therefore, this line is no longer needed.
    
    agent = rlSACAgent(actor, [critic1; critic2], agentOptions);
end