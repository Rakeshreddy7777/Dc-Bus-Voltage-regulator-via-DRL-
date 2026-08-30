%% Post-Training Analysis and Plot Generator
clear; clc; close all;
cd('C:\Users\Rakesh T\OneDrive\Documents\MATLAB');

% Load dataset
raw_data = readtable('Case Study DCbusData.csv.xlsx', detectImportOptions('Case Study DCbusData.csv.xlsx', 'VariableNamingRule', 'preserve'));
pi_vref    = raw_data{:, 1};
pi_vsensed = raw_data{:, 2};
pi_verr    = raw_data{:, 3};
pi_output  = raw_data{:, 4};
pi_error   = pi_vref - pi_vsensed;

% Load trained agent
load('Trained_DRL_DCBus_Agent_v3.mat', 'agent');
env = DCBusEnv();

% Run validation simulation
simOptions = rlSimulationOptions('MaxSteps', env.MaxSteps);
experience = sim(env, agent, simOptions);

% Extract simulation data
obs_data = squeeze(experience.Observation.DC_Bus_Observations.Data);
act_data = squeeze(experience.Action.Converter_Control_Effort.Data);

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
err_raw_vec = err_scaled_vec * env.ErrScale;
v_drl       = env.V_ref - err_raw_vec;
act_raw_vec = act_norm_vec * env.ActScale;

% Generate figure invisibly
f = figure('Visible', 'off', 'Color', [1 1 1], 'Position', [100 100 1000 850]);

subplot(3,1,1);
plot(t_sim, v_drl, 'b-', 'LineWidth', 1.8); hold on;
yline(300, 'r--', 'V_{ref} = 300V', 'LineWidth', 1.5);
pi_len = min(num_steps, height(raw_data));
t_pi = (0:pi_len-1)' * env.dt;
plot(t_pi, pi_vsensed(1:pi_len), 'Color', [0.6 0.6 0.6], 'LineStyle', ':', 'LineWidth', 1.2);
legend('DRL Agent v3', 'V_{ref} (300V)', 'PI Controller (Data)', 'Location', 'Best');
grid on; title('DC Bus Voltage (V)');
ylabel('Voltage (V)'); ylim([285 315]);

subplot(3,1,2);
plot(t_sim, err_raw_vec, 'r-', 'LineWidth', 1.5); hold on;
yline(0, 'k--', 'LineWidth', 1.0);
yline(0.5, 'g:', 'LineWidth', 1.0);
yline(-0.5, 'g:', 'LineWidth', 1.0);
plot(t_pi, pi_error(1:pi_len), 'Color', [0.6 0.6 0.6], 'LineStyle', ':', 'LineWidth', 1.2);
legend('DRL Error', 'Zero Error', '±0.5V Band', '', 'PI Error (Data)', 'Location', 'Best');
grid on; title('Voltage Error (V_{err} = V_{ref} - V) [Aim: Minimize]');
ylabel('Error (V)'); ylim([-10 10]);

subplot(3,1,3);
plot(t_sim, act_raw_vec, 'Color', [0 0.6 0], 'LineWidth', 1.5); hold on;
plot(t_pi, pi_output(1:pi_len), 'm:', 'LineWidth', 1.2);
legend('DRL Control Effort', 'PI Output (Data)', 'Location', 'Best');
grid on; title('Control Action Signal (u)');
xlabel('Time (seconds)'); ylabel('Control Action');
ylim([-10.5 10.5]);

% Export figure to artifact directory
out_img = 'C:\Users\Rakesh T\.gemini\antigravity\brain\dc15fb40-6114-4039-9548-f8dcd49e57c3\validation_results_v3.png';
exportgraphics(f, out_img, 'Resolution', 150);
fprintf('Plot exported successfully to %s\n', out_img);
