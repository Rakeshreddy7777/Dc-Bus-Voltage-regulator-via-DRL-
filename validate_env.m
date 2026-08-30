%% =========================================================================
%  VALIDATION SCRIPT — Tests DCBusEnv before full training
%  Run this FIRST to verify everything works correctly.
%  =========================================================================
clear classes; clear; clc; close all;
fprintf('====== DCBusEnv v3 Validation Script ======\n\n');

%% TEST 1: Environment instantiation
fprintf('TEST 1: Creating environment...\n');
try
    env = DCBusEnv();
    obsInfo = getObservationInfo(env);
    actInfo = getActionInfo(env);
    fprintf('  PASS: Environment created successfully.\n');
    fprintf('  Obs dim: %d, Action dim: %d\n', obsInfo.Dimension(1), actInfo.Dimension(1));
    fprintf('  Action range: [%.1f, %.1f]\n', actInfo.LowerLimit, actInfo.UpperLimit);
catch ME
    fprintf('  FAIL: %s\n', ME.message);
    return;
end

%% TEST 2: Reset returns correct initial state
fprintf('\nTEST 2: Reset function...\n');
obs = reset(env);
fprintf('  Initial state: [%.4f, %.4f, %.4f]\n', obs(1), obs(2), obs(3));
if all(obs == 0)
    fprintf('  PASS: Starts at exactly 300V (all zeros = no error).\n');
else
    fprintf('  WARNING: Initial state is not [0,0,0]. Check reset logic.\n');
end

%% TEST 3: Step function with zero action
fprintf('\nTEST 3: Step with action=0 (should track load ripple)...\n');
reset(env);
voltages = zeros(20, 1);
errors   = zeros(20, 1);
for i = 1:20
    [obs, rew, done, ~] = step(env, 0.0);
    voltages(i) = env.V_ref - obs(1) * env.ErrScale;  % Reconstruct voltage
    errors(i)   = obs(1) * env.ErrScale;
end
fprintf('  After 20 steps with action=0:\n');
fprintf('    Voltage range: [%.4f, %.4f] V\n', min(voltages), max(voltages));
fprintf('    Max error: %.4f V\n', max(abs(errors)));
fprintf('    IsDone: %d (should be 0)\n', done);

if ~done && max(abs(errors)) < 5.0
    fprintf('  PASS: Agent survives 20 steps, voltage stays near 300V.\n');
else
    fprintf('  FAIL: Agent died or voltage diverged!\n');
end

%% TEST 4: Full episode with zero action
fprintf('\nTEST 4: Full episode (2000 steps) with action=0...\n');
reset(env);
total_reward = 0;
for i = 1:2000
    [obs, rew, done, ~] = step(env, 0.0);
    total_reward = total_reward + rew;
    if done && i < 2000
        fprintf('  FAIL: Episode terminated early at step %d!\n', i);
        break;
    end
end
fprintf('  Total reward (zero action): %.2f\n', total_reward);
fprintf('  Final voltage: %.4f V\n', env.V_ref - obs(1)*env.ErrScale);
if done && i == 2000
    fprintf('  PASS: Episode completed all 2000 steps.\n');
end

%% TEST 5: Full episode with random actions
fprintf('\nTEST 5: Full episode (2000 steps) with RANDOM actions in [-1,1]...\n');
reset(env);
total_reward = 0;
max_err = 0;
for i = 1:2000
    rand_action = (rand()*2 - 1);  % Random in [-1, 1]
    [obs, rew, done, ~] = step(env, rand_action);
    total_reward = total_reward + rew;
    max_err = max(max_err, abs(obs(1) * env.ErrScale));
    if done && i < 2000
        fprintf('  FAIL: Episode terminated early at step %d!\n', i);
        break;
    end
end
fprintf('  Total reward (random): %.2f\n', total_reward);
fprintf('  Max error seen: %.2f V\n', max_err);
fprintf('  Final voltage: %.4f V\n', env.V_ref - obs(1)*env.ErrScale);
if i == 2000
    fprintf('  PASS: Agent survived full episode even with random actions!\n');
else
    fprintf('  FAIL: Agent died during random exploration.\n');
end

%% TEST 6: Observation is NOT clamped
fprintf('\nTEST 6: Observation unboundedness check...\n');
reset(env);
% Push voltage hard for many steps to create large error
for i = 1:100
    step(env, 1.0);  % Max positive action
end
obs_after = env.State;
err_scaled = obs_after(1);
err_raw = err_scaled * env.ErrScale;
fprintf('  After 100 steps of max action: scaled_err = %.4f, raw_err = %.2f V\n', err_scaled, err_raw);
if abs(err_scaled) > 1.0
    fprintf('  PASS: Observation exceeds [-1,1] — NOT clamped. Agent sees true error.\n');
else
    fprintf('  INFO: Error stayed within [-1,1] (physics may limit it). OK.\n');
end

%% TEST 7: Actor/Critic creation
fprintf('\nTEST 7: Creating Actor and Critic networks...\n');
try
    actorNet = [
        featureInputLayer(obsInfo.Dimension(1), 'Normalization', 'none', 'Name', 'StateIn')
        fullyConnectedLayer(128, 'Name', 'ActorFC1')
        reluLayer('Name', 'ActorRelu1')
        fullyConnectedLayer(128, 'Name', 'ActorFC2')
        reluLayer('Name', 'ActorRelu2')
        fullyConnectedLayer(actInfo.Dimension(1), 'Name', 'ActorOut')
        tanhLayer('Name', 'ActorTanh')
    ];
    actorNet = dlnetwork(actorNet);
    actor = rlContinuousDeterministicActor(actorNet, obsInfo, actInfo);
    fprintf('  PASS: Actor created (3 -> 128 -> 128 -> 1 -> tanh).\n');
catch ME
    fprintf('  FAIL: Actor creation error: %s\n', ME.message);
    return;
end

try
    statePath = [
        featureInputLayer(obsInfo.Dimension(1), 'Normalization', 'none', 'Name', 'StateIn')
        fullyConnectedLayer(128, 'Name', 'CritStateFC')
    ];
    actionPath = [
        featureInputLayer(actInfo.Dimension(1), 'Normalization', 'none', 'Name', 'ActionIn')
        fullyConnectedLayer(128, 'Name', 'CritActionFC')
    ];
    commonPath = [
        concatenationLayer(1, 2, 'Name', 'ConcatStreams')
        reluLayer('Name', 'CritRelu1')
        fullyConnectedLayer(128, 'Name', 'CritFC2')
        reluLayer('Name', 'CritRelu2')
        fullyConnectedLayer(1, 'Name', 'QValue')
    ];
    criticLG = layerGraph();
    criticLG = addLayers(criticLG, statePath);
    criticLG = addLayers(criticLG, actionPath);
    criticLG = addLayers(criticLG, commonPath);
    criticLG = connectLayers(criticLG, 'CritStateFC',  'ConcatStreams/in1');
    criticLG = connectLayers(criticLG, 'CritActionFC', 'ConcatStreams/in2');
    criticNet = dlnetwork(criticLG);
    critic = rlQValueFunction(criticNet, obsInfo, actInfo);
    fprintf('  PASS: Critic created (concatenation-based, 128-128-1).\n');
catch ME
    fprintf('  FAIL: Critic creation error: %s\n', ME.message);
    return;
end

%% TEST 8: DDPG Agent creation
fprintf('\nTEST 8: Creating DDPG Agent...\n');
try
    agentOpts = rlDDPGAgentOptions(...
        'SampleTime', env.dt, ...
        'TargetSmoothFactor', 1e-3, ...
        'DiscountFactor', 0.99, ...
        'MiniBatchSize', 128, ...
        'ExperienceBufferLength', 1e6);
    agentOpts.ActorOptimizerOptions.LearnRate  = 1e-4;
    agentOpts.CriticOptimizerOptions.LearnRate = 1e-3;
    agentOpts.NoiseOptions.Variance         = 0.3;
    agentOpts.NoiseOptions.VarianceDecayRate = 1e-4;
    agent = rlDDPGAgent(actor, critic, agentOpts);
    fprintf('  PASS: DDPG Agent created successfully.\n');
catch ME
    fprintf('  FAIL: Agent creation error: %s\n', ME.message);
    return;
end

%% TEST 9: Quick 5-episode training sanity check
fprintf('\nTEST 9: Running 5-episode mini training (sanity check)...\n');
try
    miniTrainOpts = rlTrainingOptions(...
        'MaxEpisodes', 5, ...
        'MaxStepsPerEpisode', env.MaxSteps, ...
        'Plots', 'none', ...
        'Verbose', true);
    miniStats = train(agent, env, miniTrainOpts);
    
    % Check episode lengths
    ep_steps = miniStats.EpisodeSteps;
    avg_steps = mean(ep_steps);
    fprintf('\n  Episode steps: %s\n', mat2str(ep_steps'));
    fprintf('  Average steps: %.1f\n', avg_steps);
    
    if avg_steps >= env.MaxSteps * 0.9
        fprintf('  PASS: All episodes ran near full length (%d steps). No death loop!\n', env.MaxSteps);
    elseif avg_steps >= 100
        fprintf('  WARNING: Episodes shorter than expected (%.0f vs %d). Check reward.\n', avg_steps, env.MaxSteps);
    else
        fprintf('  FAIL: Episodes still dying early (avg %.0f steps). Something is wrong.\n', avg_steps);
    end
catch ME
    fprintf('  FAIL: Training error: %s\n', ME.message);
    return;
end

fprintf('\n====== VALIDATION COMPLETE ======\n');
fprintf('If all tests passed, run train_ddpg_dcbus.m for full training.\n');
