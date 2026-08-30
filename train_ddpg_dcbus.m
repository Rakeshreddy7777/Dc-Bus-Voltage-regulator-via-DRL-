%% =========================================================================
%  DEEP REINFORCEMENT LEARNING DC BUS CONTROLLER 
%  =========================================================================
%  Matches DCBusEnv :
%    - Action space [-1, 1] (env scales internally to [-10, 10])
%    - No scaling layer on actor (tanh output = action)
%    - Concatenation-based critic
%    - Moderate exploration noise (variance 0.3 on [-1,1] range)
%    - Proper reconstruction of raw voltage from scaled observations
%  =========================================================================
clear classes;   % Clear cached class definitions
clear; clc; close all;

%% 1. Load Dataset for Benchmark Comparison
dataset_file = 'Case Study DCbusData.csv.xlsx';
has_dataset = false;

if exist(dataset_file, 'file')
    opts = detectImportOptions(dataset_file);
    opts.VariableNamingRule = 'preserve';
    raw_data = readtable(dataset_file, opts);
    has_dataset = true;
    fprintf('Loaded historical dataset with %d samples.\n', height(raw_data));
    
    pi_vref    = raw_data{:, 1};
    pi_vsensed = raw_data{:, 2};
    pi_verr    = raw_data{:, 3};
    pi_output  = raw_data{:, 4};
    pi_error   = pi_vref - pi_vsensed;
    
    fprintf('  PI Controller Baseline:\n');
    fprintf('    MAE  = %.4f V\n', mean(abs(pi_error)));
    fprintf('    RMS  = %.4f V\n', rms(pi_error));
    fprintf('    Max  = %.4f V\n', max(abs(pi_error)));
    fprintf('    within +/-0.5V = %.1f%%\n\n', 100*mean(abs(pi_error) < 0.5));
else
    warning('Dataset file not found. Running standalone.');
end

%% 2. Create Environment
env = DCBusEnv();
obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);

fprintf('Environment v3 Configuration:\n');
fprintf('  Obs: scaled (NOT clamped) [err/10, derr/1000, prev_act/10]\n');
fprintf('  Action: [-1, 1] -> internally [-10, 10]\n');
fprintf('  Episode: %d steps (%.1f sec)\n', env.MaxSteps, env.MaxSteps * env.dt);
fprintf('  Init: exactly 300V (no perturbation)\n');
fprintf('  Termination: NONE (always runs full episode)\n\n');

%% 3. Actor Network
%  tanh output -> [-1, 1] = action space directly
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

%% 4. Critic Network (Concatenation-based)
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

%% 5. DDPG Agent
agentOpts = rlDDPGAgentOptions(...
    'SampleTime', env.dt, ...
    'TargetSmoothFactor', 1e-3, ...
    'DiscountFactor', 0.99, ...
    'MiniBatchSize', 128, ...
    'ExperienceBufferLength', 1e6);

agentOpts.ActorOptimizerOptions.LearnRate  = 1e-4;
agentOpts.CriticOptimizerOptions.LearnRate = 1e-3;

% Noise on [-1, 1] action range: 0.3 variance = 30% exploration
agentOpts.NoiseOptions.Variance         = 0.3;
agentOpts.NoiseOptions.VarianceDecayRate = 1e-4;

agent = rlDDPGAgent(actor, critic, agentOpts);

%% 6. Training
trainOpts = rlTrainingOptions(...
    'MaxEpisodes', 1000, ...
    'MaxStepsPerEpisode', env.MaxSteps, ...
    'ScoreAveragingWindowLength', 30, ...
    'Plots', 'training-progress', ...
    'StopTrainingCriteria', 'AverageReward', ...
    'StopTrainingValue', -500, ...
    'SaveAgentCriteria', 'EpisodeReward', ...
    'SaveAgentValue', -300);

fprintf('=== Starting DRL Training v3 ===\n');
trainingStats = train(agent, env, trainOpts);

save('Trained_DRL_DCBus_Agent_v3.mat', 'agent', 'trainingStats');
fprintf('Agent and TrainingStats saved to Trained_DRL_DCBus_Agent_v3.mat\n');

%% 7. Validation
fprintf('\nRunning validation...\n');
simOptions = rlSimulationOptions('MaxSteps', env.MaxSteps);
experience = sim(env, agent, simOptions);

% Extract data
obs_data = squeeze(experience.Observation.DC_Bus_Observations.Data);
act_data = squeeze(experience.Action.Converter_Control_Effort.Data);

% Handle dimension ordering from squeeze
if size(obs_data, 1) == 3
    err_scaled_vec = obs_data(1, :)';
    num_steps = min(length(err_scaled_vec), length(act_data));
else
    err_scaled_vec = obs_data(:, 1);
    num_steps = min(length(err_scaled_vec), length(act_data));
end

act_norm_vec = reshape(act_data(1:num_steps), [], 1);
err_scaled_vec = err_scaled_vec(1:num_steps);

t_sim = (0:num_steps-1)' * env.dt;

% Reconstruct raw physical values
err_raw_vec = err_scaled_vec * env.ErrScale;          % Raw error (V)
v_drl       = env.V_ref - err_raw_vec;                 % Voltage (V)
act_raw_vec = act_norm_vec * env.ActScale;              % Raw action [-10, 10]

%% 8. Performance Metrics
mae_drl   = mean(abs(err_raw_vec));
rms_drl   = rms(err_raw_vec);
max_drl   = max(abs(err_raw_vec));
pct_tight = 100 * mean(abs(err_raw_vec) < 0.5);
mean_act  = mean(abs(act_raw_vec));

fprintf('\n========================================\n');
fprintf('     PERFORMANCE COMPARISON: DRL vs PI\n');
fprintf('========================================\n');
fprintf('%-20s | %-10s | %-10s | %-6s\n', 'Metric', 'PI (Data)', 'DRL', 'Better');
fprintf('%-20s-+-%-10s-+-%-10s-+-%-6s\n', repmat('-',1,20), repmat('-',1,10), repmat('-',1,10), repmat('-',1,6));

if has_dataset
    pi_mae = mean(abs(pi_error));
    pi_rms = rms(pi_error);
    pi_max = max(abs(pi_error));
    pi_pct = 100*mean(abs(pi_error) < 0.5);
    
    fprintf('%-20s | %-10.4f | %-10.4f | %-6s\n', 'MAE (V)',          pi_mae, mae_drl,   iff(mae_drl < pi_mae, 'YES','NO'));
    fprintf('%-20s | %-10.4f | %-10.4f | %-6s\n', 'RMS Error (V)',    pi_rms, rms_drl,   iff(rms_drl < pi_rms, 'YES','NO'));
    fprintf('%-20s | %-10.4f | %-10.4f | %-6s\n', 'Max |Error| (V)',  pi_max, max_drl,   iff(max_drl < pi_max, 'YES','NO'));
    fprintf('%-20s | %-10.1f | %-10.1f | %-6s\n', '%% within +/-0.5V', pi_pct, pct_tight, iff(pct_tight > pi_pct,'YES','NO'));
else
    fprintf('%-20s | %-10s | %-10.4f |\n', 'MAE (V)',          'N/A', mae_drl);
    fprintf('%-20s | %-10s | %-10.4f |\n', 'RMS Error (V)',    'N/A', rms_drl);
    fprintf('%-20s | %-10s | %-10.4f |\n', 'Max |Error| (V)',  'N/A', max_drl);
    fprintf('%-20s | %-10s | %-10.1f |\n', '%% within +/-0.5V', 'N/A', pct_tight);
end

fprintf('========================================\n');
fprintf('Voltage Range: [%.2f, %.2f] V\n', min(v_drl), max(v_drl));
fprintf('Mean |Control Action|: %.4f (raw), %.4f (normalized)\n', mean_act, mean(abs(act_norm_vec)));
fprintf('Episode Length: %d / %d steps\n', num_steps, env.MaxSteps);

%% 9. Plots
figure('Color', [1 1 1], 'Position', [80 80 1050 850]);

% Panel 1: Voltage
subplot(3,1,1);
plot(t_sim, v_drl, 'b-', 'LineWidth', 1.8); hold on;
yline(300, 'r--', 'V_{ref} = 300V', 'LineWidth', 1.5);
if has_dataset
    pi_len = min(num_steps, height(raw_data));
    t_pi = (0:pi_len-1)' * env.dt;
    plot(t_pi, pi_vsensed(1:pi_len), 'Color', [0.5 0.5 0.5], 'LineStyle', ':', 'LineWidth', 1.2);
    legend('DRL Agent', 'V_{ref} (300V)', 'PI Controller', 'Location', 'Best');
else
    legend('DRL Agent', 'V_{ref} (300V)', 'Location', 'Best');
end
grid on; title('DC Bus Voltage');
ylabel('Voltage (V)'); ylim([280 320]);

% Panel 2: Error
subplot(3,1,2);
plot(t_sim, err_raw_vec, 'r-', 'LineWidth', 1.5); hold on;
yline(0, 'k--', 'LineWidth', 1.0);
fill([t_sim(1) t_sim(end) t_sim(end) t_sim(1)], [0.5 0.5 -0.5 -0.5], ...
     'g', 'FaceAlpha', 0.1, 'EdgeColor', 'g', 'LineStyle', ':');
if has_dataset
    plot(t_pi, pi_error(1:pi_len), 'Color', [0.5 0.5 0.5], 'LineStyle', ':', 'LineWidth', 1.2);
    legend('DRL Error', 'Zero', '+/-0.5V Band', 'PI Error', 'Location', 'Best');
else
    legend('DRL Error', 'Zero', '+/-0.5V Band', 'Location', 'Best');
end
grid on; title('Voltage Error (V_{err}) — Goal: Minimize');
ylabel('Error (V)');

% Panel 3: Control Action
subplot(3,1,3);
plot(t_sim, act_raw_vec, 'Color', [0 0.6 0], 'LineWidth', 1.5); hold on;
if has_dataset
    plot(t_pi, pi_output(1:pi_len), 'm:', 'LineWidth', 1.2);
    legend('DRL Control', 'PI Output', 'Location', 'Best');
else
    legend('DRL Control', 'Location', 'Best');
end
grid on; title('Control Action');
xlabel('Time (seconds)'); ylabel('Control Output');
ylim([-10.5 10.5]);

sgtitle('DRL DC Bus Controller v3 — Validation', 'FontSize', 14, 'FontWeight', 'bold');

%% Helper
function result = iff(cond, t, f)
    if cond, result = t; else, result = f; end
end
