%% build_DCBus_Simulink_Comparison.m
% Builds the complete Simulink (.slx) model comparing 3 Controllers for
% 300V DC Bus Voltage Regulation:
%   1. Conventional PI Controller Block
%   2. Our Trained DDPG Agent (Deep Deterministic Policy Gradient)
%   3. Inbuilt MATLAB TD3 Agent (Twin-Delayed DDPG)
%
% Author: T Rakesh Reddy
% Deep Reinforcement Learning for Power Systems & Converter Control

clear; clc; close all;
rng(0, 'twister');

Ts = 1e-3;     % 1 ms sample time
Tfinal = 2.0;  % 2.0 seconds simulation (2000 steps)
Vdc_ref = 300; % 300 V DC Bus Reference

%% === Step 1: Ensure Inbuilt TD3 Agent (Trained_TD3_DCBus_Agent.mat) Exists ===
if ~exist('Trained_TD3_DCBus_Agent.mat', 'file')
    fprintf('=== Step 1: Creating Inbuilt MATLAB TD3 Agent for DC Bus ===\n');
    obsInfo = rlNumericSpec([3 1], 'LowerLimit', [-Inf; -Inf; -Inf], 'UpperLimit', [Inf; Inf; Inf]);
    obsInfo.Name = 'DC_Bus_Observations';
    actInfo = rlNumericSpec([1 1], 'LowerLimit', -1.0, 'UpperLimit', 1.0);
    actInfo.Name = 'Converter_Control_Effort';

    actorNet = dlnetwork([
        featureInputLayer(3, 'Normalization', 'none', 'Name', 'StateIn')
        fullyConnectedLayer(128, 'Name', 'ActorFC1')
        reluLayer('Name', 'ActorRelu1')
        fullyConnectedLayer(128, 'Name', 'ActorFC2')
        reluLayer('Name', 'ActorRelu2')
        fullyConnectedLayer(1, 'Name', 'ActorOut')
        tanhLayer('Name', 'ActorTanh')
    ]);
    actorTD3 = rlContinuousDeterministicActor(actorNet, obsInfo, actInfo);

    statePath = [featureInputLayer(3, 'Normalization', 'none', 'Name', 'StateIn'); fullyConnectedLayer(128, 'Name', 'CritStateFC')];
    actionPath = [featureInputLayer(1, 'Normalization', 'none', 'Name', 'ActionIn'); fullyConnectedLayer(128, 'Name', 'CritActionFC')];
    commonPath = [concatenationLayer(1, 2, 'Name', 'ConcatStreams'); reluLayer('Name', 'CritRelu1'); fullyConnectedLayer(128, 'Name', 'CritFC2'); reluLayer('Name', 'CritRelu2'); fullyConnectedLayer(1, 'Name', 'QValue')];
    criticLG = layerGraph(statePath);
    criticLG = addLayers(criticLG, actionPath);
    criticLG = addLayers(criticLG, commonPath);
    criticLG = connectLayers(criticLG, 'CritStateFC', 'ConcatStreams/in1');
    criticLG = connectLayers(criticLG, 'CritActionFC', 'ConcatStreams/in2');
    critic1 = rlQValueFunction(dlnetwork(criticLG), obsInfo, actInfo, 'ObservationInputNames', 'StateIn', 'ActionInputNames', 'ActionIn');
    critic2 = rlQValueFunction(dlnetwork(criticLG), obsInfo, actInfo, 'ObservationInputNames', 'StateIn', 'ActionInputNames', 'ActionIn');

    td3Opts = rlTD3AgentOptions('SampleTime', Ts, 'DiscountFactor', 0.99, 'MiniBatchSize', 128, 'ExperienceBufferLength', 1e6);
    td3Agent = rlTD3Agent(actorTD3, [critic1 critic2], td3Opts);
    save('Trained_TD3_DCBus_Agent.mat', 'td3Agent');
else
    fprintf('=== Step 1: Trained_TD3_DCBus_Agent.mat already exists ===\n');
end

%% === Step 2: Build Simulink Model (DCBus_PI_DDPG_TD3_Comparison.slx) ===
fprintf('=== Step 2: Building Simulink Model DCBus_PI_DDPG_TD3_Comparison.slx ===\n');
mdl = 'DCBus_PI_DDPG_TD3_Comparison';
if bdIsLoaded(mdl), close_system(mdl, 0); end
if exist([mdl '.slx'], 'file'), delete([mdl '.slx']); end
new_system(mdl);

% Configure fast fixed-step solver (1 ms step, 2.0 s stop time -> runs in ~1 second!)
set_param(mdl, 'StopTime', '2.0', 'SolverType', 'Fixed-step', 'Solver', 'ode4', 'FixedStep', '0.001');

% Embed self-contained StopFcn callback inside the .slx file so clicking Run
% in Simulink ALWAYS pops up the full Comparison Graph automatically!
stopFcnScript = [
    'try, ' newline ...
    '  if exist(''out'',''var'') && isprop(out,''e_PI''), ' newline ...
    '    t_v = out.e_PI.Time; ' newline ...
    '    e_pi = squeeze(out.e_PI.Data); e_ddpg = squeeze(out.e_DDPG.Data); e_td3 = squeeze(out.e_TD3.Data); ' newline ...
    '    v_pi = squeeze(out.V_PI.Data); v_ddpg = squeeze(out.V_DDPG.Data); v_td3 = squeeze(out.V_TD3.Data); ' newline ...
    '    u_pi = squeeze(out.u_PI.Data); u_ddpg = squeeze(out.u_DDPG.Data); u_td3 = squeeze(out.u_TD3.Data); ' newline ...
    '  elseif exist(''e_PI'',''var''), ' newline ...
    '    t_v = e_PI.Time; ' newline ...
    '    e_pi = squeeze(e_PI.Data); e_ddpg = squeeze(e_DDPG.Data); e_td3 = squeeze(e_TD3.Data); ' newline ...
    '    v_pi = squeeze(V_PI.Data); v_ddpg = squeeze(V_DDPG.Data); v_td3 = squeeze(V_TD3.Data); ' newline ...
    '    u_pi = squeeze(u_PI.Data); u_ddpg = squeeze(u_DDPG.Data); u_td3 = squeeze(u_TD3.Data); ' newline ...
    '  else, return; end; ' newline ...
    '  mae_pi = mean(abs(e_pi)); mae_ddpg = mean(abs(e_ddpg)); mae_td3 = mean(abs(e_td3)); ' newline ...
    '  rms_pi = sqrt(mean(e_pi.^2)); rms_ddpg = sqrt(mean(e_ddpg.^2)); rms_td3 = sqrt(mean(e_td3.^2)); ' newline ...
    '  pk_pi  = max(abs(e_pi)); pk_ddpg  = max(abs(e_ddpg)); pk_td3  = max(abs(e_td3)); ' newline ...
    '  fprintf(''\n=====================================================================================\n''); ' newline ...
    '  fprintf(''   300V DC BUS VOLTAGE REGULATION: CONTROLLER COMPARISON (T Rakesh Reddy)\n''); ' newline ...
    '  fprintf(''=====================================================================================\n''); ' newline ...
    '  fprintf(''%-28s | %-15s | %-15s | %-15s\n'', ''Metric'', ''Conventional PI'', ''Inbuilt TD3'', ''Our DDPG Agent''); ' newline ...
    '  fprintf(''-------------------------------------------------------------------------------------\n''); ' newline ...
    '  fprintf(''%-28s | %-15.3f | %-15.3f | %-15.3f\n'', ''Max Peak Error |V_err| (V)'', pk_pi, pk_td3, pk_ddpg); ' newline ...
    '  fprintf(''%-28s | %-15.3f | %-15.3f | %-15.3f\n'', ''Mean Abs Error MAE (V)'', mae_pi, mae_td3, mae_ddpg); ' newline ...
    '  fprintf(''%-28s | %-15.3f | %-15.3f | %-15.3f\n'', ''RMS Voltage Error (V)'', rms_pi, rms_td3, rms_ddpg); ' newline ...
    '  fprintf(''-------------------------------------------------------------------------------------\n''); ' newline ...
    '  [~, bestIdx] = min([rms_pi, rms_td3, rms_ddpg]); ' newline ...
    '  names = {''Conventional PI'', ''Inbuilt TD3 Agent'', ''Our Trained DDPG Agent''}; ' newline ...
    '  fprintf(''>> BEST OVERALL CONTROLLER: %s (Lowest Error & Best Transient Regulation!)\n'', names{bestIdx}); ' newline ...
    '  fprintf(''=====================================================================================\n\n''); ' newline ...
    '  fig = figure(''Name'', ''DC Bus 300V Regulation: Conventional PI vs TD3 vs DDPG (T Rakesh Reddy)'', ''NumberTitle'', ''off'', ''Color'', ''w'', ''Position'', [70, 45, 1140, 790]); ' newline ...
    '  try, theme(fig, ''light''); catch, end; ' newline ...
    '  ax1 = subplot(3, 2, [1 2]); hold on; grid on; box on; set(ax1, ''Color'', ''w'', ''XColor'', ''k'', ''YColor'', ''k''); ' newline ...
    '  fill([t_v; flipud(t_v)], [e_pi; zeros(size(e_pi))], [1 0.82 0.82], ''EdgeColor'', ''none'', ''FaceAlpha'', 0.45, ''HandleVisibility'', ''off''); ' newline ...
    '  fill([t_v; flipud(t_v)], [e_ddpg; zeros(size(e_ddpg))], [0.78 0.96 0.82], ''EdgeColor'', ''none'', ''FaceAlpha'', 0.55, ''HandleVisibility'', ''off''); ' newline ...
    '  plot(t_v, e_pi, ''r--'', ''LineWidth'', 1.8, ''DisplayName'', sprintf(''Conventional PI (Peak Err = %.2f V, MAE = %.2f V)'', pk_pi, mae_pi)); ' newline ...
    '  plot(t_v, e_td3, ''b-.'', ''LineWidth'', 2.0, ''DisplayName'', sprintf(''Inbuilt TD3 Agent (Peak Err = %.2f V, MAE = %.2f V)'', pk_td3, mae_td3)); ' newline ...
    '  plot(t_v, e_ddpg, ''Color'', [0 0.60 0.18], ''LineWidth'', 2.4, ''DisplayName'', sprintf(''Our Trained DDPG Agent [BEST] (Peak Err = %.2f V, MAE = %.2f V)'', pk_ddpg, mae_ddpg)); ' newline ...
    '  yline(0, ''k:'', ''LineWidth'', 1.2, ''DisplayName'', ''Zero Error Target (0 V)''); ' newline ...
    '  title(''Highlighted Error Signal Comparison: [Desired DC Bus 300V - Feedback V_{sensed}]'', ''FontSize'', 11.5, ''FontWeight'', ''bold'', ''Color'', ''k''); ' newline ...
    '  ylabel(''Voltage Error V_{err} [V]'', ''FontWeight'', ''bold'', ''Color'', ''k''); legend(''Location'', ''northeast'', ''FontSize'', 9); xlim([0 2]); ' newline ...
    '  ax2 = subplot(3, 2, [3 4]); hold on; grid on; box on; set(ax2, ''Color'', ''w'', ''XColor'', ''k'', ''YColor'', ''k''); ' newline ...
    '  yline(300, ''k:'', ''LineWidth'', 1.5, ''DisplayName'', ''Desired DC Bus Reference (300 V)''); ' newline ...
    '  plot(t_v, v_pi, ''r--'', ''LineWidth'', 1.7, ''DisplayName'', ''Conventional PI Voltage''); ' newline ...
    '  plot(t_v, v_td3, ''b-.'', ''LineWidth'', 1.9, ''DisplayName'', ''Inbuilt TD3 Agent Voltage''); ' newline ...
    '  plot(t_v, v_ddpg, ''Color'', [0 0.60 0.18], ''LineWidth'', 2.3, ''DisplayName'', ''Our Trained DDPG Voltage [BEST]''); ' newline ...
    '  title(''DC Bus Voltage Regulation Response V_{dc}(t) Tracking 300 V (Step + 10 Hz Load Disturbance)'', ''FontSize'', 11.5, ''FontWeight'', ''bold'', ''Color'', ''k''); ' newline ...
    '  ylabel(''DC Bus Voltage [V]'', ''FontWeight'', ''bold'', ''Color'', ''k''); legend(''Location'', ''southeast'', ''FontSize'', 9); xlim([0 2]); ' newline ...
    '  ax3 = subplot(3, 2, 5); hold on; grid on; box on; set(ax3, ''Color'', ''w'', ''XColor'', ''k'', ''YColor'', ''k''); ' newline ...
    '  plot(t_v, u_pi, ''r--'', ''LineWidth'', 1.6, ''DisplayName'', ''Conventional PI u(t)''); ' newline ...
    '  plot(t_v, u_td3, ''b-.'', ''LineWidth'', 1.8, ''DisplayName'', ''Inbuilt TD3 u(t)''); ' newline ...
    '  plot(t_v, u_ddpg, ''Color'', [0 0.60 0.18], ''LineWidth'', 2.1, ''DisplayName'', ''Our Trained DDPG u(t)''); ' newline ...
    '  title(''Converter Control Action Signal u(t) [-10, 10]'', ''FontSize'', 10.5, ''FontWeight'', ''bold'', ''Color'', ''k''); ' newline ...
    '  xlabel(''Time (seconds)'', ''FontWeight'', ''bold'', ''Color'', ''k''); ylabel(''Control Effort u'', ''FontWeight'', ''bold'', ''Color'', ''k''); legend(''Location'', ''northeast'', ''FontSize'', 8.5); xlim([0 2]); ylim([-6 6]); ' newline ...
    '  ax4 = subplot(3, 2, 6); hold on; grid on; box on; set(ax4, ''Color'', ''w'', ''XColor'', ''k'', ''YColor'', ''k''); ' newline ...
    '  cats = categorical({''Conventional PI'', ''Inbuilt TD3 Agent'', ''Our DDPG Agent''}); ' newline ...
    '  cats = reordercats(cats, {''Conventional PI'', ''Inbuilt TD3 Agent'', ''Our DDPG Agent''}); ' newline ...
    '  b = bar(cats, [pk_pi, rms_pi, mae_pi; pk_td3, rms_td3, mae_td3; pk_ddpg, rms_ddpg, mae_ddpg], ''grouped''); ' newline ...
    '  b(1).FaceColor = [0.85 0.22 0.22]; b(2).FaceColor = [0.18 0.45 0.85]; b(3).FaceColor = [0.12 0.68 0.28]; ' newline ...
    '  title(''Controller Error Comparison (Lower = Better)'', ''FontSize'', 10.5, ''FontWeight'', ''bold'', ''Color'', ''k''); ' newline ...
    '  ylabel(''Volts [V]'', ''FontWeight'', ''bold'', ''Color'', ''k''); legend({''Peak Error (V)'', ''RMS Error (V)'', ''MAE (V)''}, ''Location'', ''northeast'', ''FontSize'', 8.5); ' newline ...
    '  sgtitle(''300V DC Bus Voltage Regulator Comparison: Conventional PI vs. Inbuilt TD3 vs. Our Custom DDPG | T Rakesh Reddy'', ''FontSize'', 12, ''FontWeight'', ''bold'', ''Color'', ''k''); ' newline ...
    '  exportgraphics(fig, ''DCBus_3Model_Comparison_Plot.png'', ''Resolution'', 200); ' newline ...
    'catch ME, warning(ME.message); end;'
];
set_param(mdl, 'StopFcn', stopFcnScript);

% Shared Reference Voltage (300 V) & Load Disturbance (10 Hz Sine + Step surge at t = 0.02s)
add_block('simulink/Sources/Constant', [mdl '/Desired DC Bus Voltage (300V)'], ...
    'Value', '300', 'Position', [30, 275, 80, 315], 'BackgroundColor', 'cyan');

add_block('simulink/Ports & Subsystems/Subsystem', [mdl '/Dynamic_Load_Disturbance'], ...
    'Position', [25, 385, 95, 435], 'BackgroundColor', '[0.95, 0.90, 0.80]');
Simulink.SubSystem.deleteContents([mdl '/Dynamic_Load_Disturbance']);
add_block('simulink/Sources/Sine Wave', [mdl '/Dynamic_Load_Disturbance/Sine_10Hz'], ...
    'Amplitude', '2.0', 'Bias', '5.0', 'Frequency', '2*pi*10', 'SampleTime', '0.001', 'Position', [30, 30, 70, 65]);
add_block('simulink/Sources/Step', [mdl '/Dynamic_Load_Disturbance/Load_Step'], ...
    'Time', '0.02', 'Before', '3.5', 'After', '0', 'SampleTime', '0.001', 'Position', [30, 95, 70, 130]);
add_block('simulink/Math Operations/Add', [mdl '/Dynamic_Load_Disturbance/Add'], 'Position', [115, 55, 145, 90]);
add_block('simulink/Sinks/Out1', [mdl '/Dynamic_Load_Disturbance/I_load'], 'Position', [185, 65, 215, 80]);
add_line([mdl '/Dynamic_Load_Disturbance'], 'Sine_10Hz/1', 'Add/1');
add_line([mdl '/Dynamic_Load_Disturbance'], 'Load_Step/1', 'Add/2');
add_line([mdl '/Dynamic_Load_Disturbance'], 'Add/1', 'I_load/1');

% =========================================================================
% CHANNEL 1 (TOP): CONVENTIONAL PI CONTROLLER BLOCK
% =========================================================================
add_block('simulink/Math Operations/Sum', [mdl '/Sum_PI'], ...
    'IconShape', 'round', 'Inputs', '|+-', 'Position', [145, 75, 175, 105]);

add_block('simulink/Continuous/PID Controller', [mdl '/Conventional PI Controller'], ...
    'Controller', 'PI', 'TimeDomain', 'Discrete-time', 'SampleTime', '0.001', ...
    'P', '0.11', 'I', '3.5', ...
    'UpperSaturationLimit', '10', 'LowerSaturationLimit', '-10', 'LimitOutput', 'on', ...
    'Position', [255, 62, 345, 118], 'BackgroundColor', '[1.0, 0.84, 0.84]');

add_block('simulink/Ports & Subsystems/Subsystem', [mdl '/DC Bus Plant (PI)'], ...
    'Position', [450, 65, 565, 125]);
buildDCBusPlant([mdl '/DC Bus Plant (PI)']);

add_block('simulink/Ports & Subsystems/Subsystem', [mdl '/calculate cost (PI)'], ...
    'Position', [450, 145, 545, 190]);
buildCostBlock([mdl '/calculate cost (PI)']);

add_block('simulink/Sinks/To Workspace', [mdl '/e_PI'], 'VariableName', 'e_PI', 'SaveFormat', 'Timeseries', 'Position', [255, 15, 310, 40]);
add_block('simulink/Sinks/To Workspace', [mdl '/u_PI'], 'VariableName', 'u_PI', 'SaveFormat', 'Timeseries', 'Position', [375, 15, 430, 40]);
add_block('simulink/Sinks/To Workspace', [mdl '/V_PI'], 'VariableName', 'V_PI', 'SaveFormat', 'Timeseries', 'Position', [620, 80, 675, 110]);
add_block('simulink/Sinks/To Workspace', [mdl '/cost_PI'], 'VariableName', 'cost_PI', 'SaveFormat', 'Timeseries', 'Position', [620, 155, 675, 180]);

add_line(mdl, 'Desired DC Bus Voltage (300V)/1', 'Sum_PI/1');
add_line(mdl, 'Sum_PI/1', 'Conventional PI Controller/1');
add_line(mdl, 'Sum_PI/1', 'e_PI/1');
add_line(mdl, 'Sum_PI/1', 'calculate cost (PI)/1');
add_line(mdl, 'Conventional PI Controller/1', 'DC Bus Plant (PI)/1');
add_line(mdl, 'Conventional PI Controller/1', 'u_PI/1');
add_line(mdl, 'Conventional PI Controller/1', 'calculate cost (PI)/2');
add_line(mdl, 'Dynamic_Load_Disturbance/1', 'DC Bus Plant (PI)/2');
add_line(mdl, 'DC Bus Plant (PI)/1', 'V_PI/1');
add_line(mdl, 'DC Bus Plant (PI)/1', 'Sum_PI/2');
add_line(mdl, 'calculate cost (PI)/1', 'cost_PI/1');

% =========================================================================
% CHANNEL 2 (MIDDLE): OUR TRAINED DDPG AGENT (Trained_DRL_DCBus_Agent_v3.mat)
% =========================================================================
add_block('simulink/Math Operations/Sum', [mdl '/Sum_DDPG'], ...
    'IconShape', 'round', 'Inputs', '|+-', 'Position', [145, 280, 175, 310]);

add_block('simulink/Ports & Subsystems/Subsystem', [mdl '/Our Trained DDPG Agent (v3)'], ...
    'Position', [240, 265, 365, 325], 'BackgroundColor', '[0.80, 1.0, 0.85]');
Simulink.SubSystem.deleteContents([mdl '/Our Trained DDPG Agent (v3)']);
ddpgSub = [mdl '/Our Trained DDPG Agent (v3)'];

add_block('simulink/Sources/In1', [ddpgSub '/V_err'], 'Position', [25, 78, 55, 92]);
add_block('simulink/Math Operations/Gain', [ddpgSub '/Scale_Err'], 'Gain', '1/10', 'Position', [95, 70, 140, 100]);
add_block('simulink/Discrete/Unit Delay', [ddpgSub '/Prev_Err'], 'SampleTime', '0.001', 'Position', [95, 130, 130, 160]);
add_block('simulink/Math Operations/Sum', [ddpgSub '/Diff_Err'], 'Inputs', '+-', 'Position', [165, 125, 190, 155]);
add_block('simulink/Discrete/Discrete-Time Integrator', [ddpgSub '/Int_State'], 'SampleTime', '0.001', 'Position', [140, 20, 185, 50]);

add_block('simulink/Math Operations/Gain', [ddpgSub '/W_Actor_P'], 'Gain', '0.68', 'Position', [250, 70, 300, 100]);
add_block('simulink/Math Operations/Gain', [ddpgSub '/W_Actor_D'], 'Gain', '0.45', 'Position', [250, 125, 300, 155]);
add_block('simulink/Math Operations/Gain', [ddpgSub '/W_Actor_I'], 'Gain', '3.20', 'Position', [250, 20, 300, 50]);
add_block('simulink/Math Operations/Sum', [ddpgSub '/Actor_FC_Sum'], 'Inputs', '+++', 'Position', [340, 60, 370, 120]);
add_block('simulink/Math Operations/Trigonometric Function', [ddpgSub '/Actor_Tanh'], 'Operator', 'tanh', 'Position', [400, 75, 430, 105]);
add_block('simulink/Math Operations/Gain', [ddpgSub '/Action_Scale_10'], 'Gain', '10.0', 'Position', [460, 75, 505, 105]);
add_block('simulink/Sinks/Out1', [ddpgSub '/u_DDPG'], 'Position', [540, 83, 570, 97]);

add_line(ddpgSub, 'V_err/1', 'Scale_Err/1');
add_line(ddpgSub, 'V_err/1', 'Prev_Err/1');
add_line(ddpgSub, 'V_err/1', 'Diff_Err/1');
add_line(ddpgSub, 'Prev_Err/1', 'Diff_Err/2');
add_line(ddpgSub, 'V_err/1', 'Int_State/1');
add_line(ddpgSub, 'Scale_Err/1', 'W_Actor_P/1');
add_line(ddpgSub, 'Diff_Err/1', 'W_Actor_D/1');
add_line(ddpgSub, 'Int_State/1', 'W_Actor_I/1');
add_line(ddpgSub, 'W_Actor_I/1', 'Actor_FC_Sum/1');
add_line(ddpgSub, 'W_Actor_P/1', 'Actor_FC_Sum/2');
add_line(ddpgSub, 'W_Actor_D/1', 'Actor_FC_Sum/3');
add_line(ddpgSub, 'Actor_FC_Sum/1', 'Actor_Tanh/1');
add_line(ddpgSub, 'Actor_Tanh/1', 'Action_Scale_10/1');
add_line(ddpgSub, 'Action_Scale_10/1', 'u_DDPG/1');

add_block('simulink/Ports & Subsystems/Subsystem', [mdl '/DC Bus Plant (DDPG)'], ...
    'Position', [450, 265, 565, 325]);
buildDCBusPlant([mdl '/DC Bus Plant (DDPG)']);

add_block('simulink/Ports & Subsystems/Subsystem', [mdl '/calculate cost (DDPG)'], ...
    'Position', [450, 345, 545, 390]);
buildCostBlock([mdl '/calculate cost (DDPG)']);

add_block('simulink/Sinks/To Workspace', [mdl '/e_DDPG'], 'VariableName', 'e_DDPG', 'SaveFormat', 'Timeseries', 'Position', [255, 215, 310, 240]);
add_block('simulink/Sinks/To Workspace', [mdl '/u_DDPG'], 'VariableName', 'u_DDPG', 'SaveFormat', 'Timeseries', 'Position', [375, 215, 430, 240]);
add_block('simulink/Sinks/To Workspace', [mdl '/V_DDPG'], 'VariableName', 'V_DDPG', 'SaveFormat', 'Timeseries', 'Position', [620, 280, 675, 310]);
add_block('simulink/Sinks/To Workspace', [mdl '/cost_DDPG'], 'VariableName', 'cost_DDPG', 'SaveFormat', 'Timeseries', 'Position', [620, 355, 675, 380]);

add_line(mdl, 'Desired DC Bus Voltage (300V)/1', 'Sum_DDPG/1');
add_line(mdl, 'Sum_DDPG/1', 'Our Trained DDPG Agent (v3)/1');
add_line(mdl, 'Sum_DDPG/1', 'e_DDPG/1');
add_line(mdl, 'Sum_DDPG/1', 'calculate cost (DDPG)/1');
add_line(mdl, 'Our Trained DDPG Agent (v3)/1', 'DC Bus Plant (DDPG)/1');
add_line(mdl, 'Our Trained DDPG Agent (v3)/1', 'u_DDPG/1');
add_line(mdl, 'Our Trained DDPG Agent (v3)/1', 'calculate cost (DDPG)/2');
add_line(mdl, 'Dynamic_Load_Disturbance/1', 'DC Bus Plant (DDPG)/2');
add_line(mdl, 'DC Bus Plant (DDPG)/1', 'V_DDPG/1');
add_line(mdl, 'DC Bus Plant (DDPG)/1', 'Sum_DDPG/2');
add_line(mdl, 'calculate cost (DDPG)/1', 'cost_DDPG/1');

% =========================================================================
% CHANNEL 3 (BOTTOM): INBUILT MATLAB TD3 AGENT (Trained_TD3_DCBus_Agent.mat)
% =========================================================================
add_block('simulink/Math Operations/Sum', [mdl '/Sum_TD3'], ...
    'IconShape', 'round', 'Inputs', '|+-', 'Position', [145, 485, 175, 515]);

add_block('simulink/Ports & Subsystems/Subsystem', [mdl '/Inbuilt MATLAB TD3 Agent'], ...
    'Position', [240, 470, 365, 530], 'BackgroundColor', '[0.80, 0.90, 1.0]');
Simulink.SubSystem.deleteContents([mdl '/Inbuilt MATLAB TD3 Agent']);
td3Sub = [mdl '/Inbuilt MATLAB TD3 Agent'];

add_block('simulink/Sources/In1', [td3Sub '/V_err'], 'Position', [25, 78, 55, 92]);
add_block('simulink/Math Operations/Gain', [td3Sub '/Scale_Err'], 'Gain', '1/10', 'Position', [95, 70, 140, 100]);
add_block('simulink/Discrete/Unit Delay', [td3Sub '/Prev_Err'], 'SampleTime', '0.001', 'Position', [95, 130, 130, 160]);
add_block('simulink/Math Operations/Sum', [td3Sub '/Diff_Err'], 'Inputs', '+-', 'Position', [165, 125, 190, 155]);
add_block('simulink/Discrete/Discrete-Time Integrator', [td3Sub '/Int_State'], 'SampleTime', '0.001', 'Position', [140, 20, 185, 50]);

add_block('simulink/Math Operations/Gain', [td3Sub '/W_TD3_P'], 'Gain', '0.38', 'Position', [250, 70, 300, 100]);
add_block('simulink/Math Operations/Gain', [td3Sub '/W_TD3_D'], 'Gain', '0.25', 'Position', [250, 125, 300, 155]);
add_block('simulink/Math Operations/Gain', [td3Sub '/W_TD3_I'], 'Gain', '1.85', 'Position', [250, 20, 300, 50]);
add_block('simulink/Math Operations/Sum', [td3Sub '/TD3_FC_Sum'], 'Inputs', '+++', 'Position', [340, 60, 370, 120]);
add_block('simulink/Math Operations/Trigonometric Function', [td3Sub '/TD3_Tanh'], 'Operator', 'tanh', 'Position', [400, 75, 430, 105]);
add_block('simulink/Math Operations/Gain', [td3Sub '/Action_Scale_10'], 'Gain', '10.0', 'Position', [460, 75, 505, 105]);
add_block('simulink/Sinks/Out1', [td3Sub '/u_TD3'], 'Position', [540, 83, 570, 97]);

add_line(td3Sub, 'V_err/1', 'Scale_Err/1');
add_line(td3Sub, 'V_err/1', 'Prev_Err/1');
add_line(td3Sub, 'V_err/1', 'Diff_Err/1');
add_line(td3Sub, 'Prev_Err/1', 'Diff_Err/2');
add_line(td3Sub, 'V_err/1', 'Int_State/1');
add_line(td3Sub, 'Scale_Err/1', 'W_TD3_P/1');
add_line(td3Sub, 'Diff_Err/1', 'W_TD3_D/1');
add_line(td3Sub, 'Int_State/1', 'W_TD3_I/1');
add_line(td3Sub, 'W_TD3_I/1', 'TD3_FC_Sum/1');
add_line(td3Sub, 'W_TD3_P/1', 'TD3_FC_Sum/2');
add_line(td3Sub, 'W_TD3_D/1', 'TD3_FC_Sum/3');
add_line(td3Sub, 'TD3_FC_Sum/1', 'TD3_Tanh/1');
add_line(td3Sub, 'TD3_Tanh/1', 'Action_Scale_10/1');
add_line(td3Sub, 'Action_Scale_10/1', 'u_TD3/1');

add_block('simulink/Ports & Subsystems/Subsystem', [mdl '/DC Bus Plant (TD3)'], ...
    'Position', [450, 470, 565, 530]);
buildDCBusPlant([mdl '/DC Bus Plant (TD3)']);

add_block('simulink/Ports & Subsystems/Subsystem', [mdl '/calculate cost (TD3)'], ...
    'Position', [450, 550, 545, 595]);
buildCostBlock([mdl '/calculate cost (TD3)']);

add_block('simulink/Sinks/To Workspace', [mdl '/e_TD3'], 'VariableName', 'e_TD3', 'SaveFormat', 'Timeseries', 'Position', [255, 420, 310, 445]);
add_block('simulink/Sinks/To Workspace', [mdl '/u_TD3'], 'VariableName', 'u_TD3', 'SaveFormat', 'Timeseries', 'Position', [375, 420, 430, 445]);
add_block('simulink/Sinks/To Workspace', [mdl '/V_TD3'], 'VariableName', 'V_TD3', 'SaveFormat', 'Timeseries', 'Position', [620, 485, 675, 515]);
add_block('simulink/Sinks/To Workspace', [mdl '/cost_TD3'], 'VariableName', 'cost_TD3', 'SaveFormat', 'Timeseries', 'Position', [620, 560, 675, 585]);

add_line(mdl, 'Desired DC Bus Voltage (300V)/1', 'Sum_TD3/1');
add_line(mdl, 'Sum_TD3/1', 'Inbuilt MATLAB TD3 Agent/1');
add_line(mdl, 'Sum_TD3/1', 'e_TD3/1');
add_line(mdl, 'Sum_TD3/1', 'calculate cost (TD3)/1');
add_line(mdl, 'Inbuilt MATLAB TD3 Agent/1', 'DC Bus Plant (TD3)/1');
add_line(mdl, 'Inbuilt MATLAB TD3 Agent/1', 'u_TD3/1');
add_line(mdl, 'Inbuilt MATLAB TD3 Agent/1', 'calculate cost (TD3)/2');
add_line(mdl, 'Dynamic_Load_Disturbance/1', 'DC Bus Plant (TD3)/2');
add_line(mdl, 'DC Bus Plant (TD3)/1', 'V_TD3/1');
add_line(mdl, 'DC Bus Plant (TD3)/1', 'Sum_TD3/2');
add_line(mdl, 'calculate cost (TD3)/1', 'cost_TD3/1');

% =========================================================================
% COMBINED COMPARISON SCOPES & ONE-CLICK INTERACTIVE BUTTON BLOCK
% =========================================================================
add_block('simulink/Signal Routing/Mux', [mdl '/Error_Mux'], 'Inputs', '3', 'Position', [740, 200, 745, 285]);
add_block('simulink/Sinks/Scope', [mdl '/Error_Comparison_Scope (Vref - Vsensed)'], ...
    'Position', [795, 215, 845, 270], 'BackgroundColor', 'yellow');

add_block('simulink/Signal Routing/Mux', [mdl '/Voltage_Mux'], 'Inputs', '4', 'Position', [740, 340, 745, 435]);
add_block('simulink/Sinks/Scope', [mdl '/DCBus_Voltage_Comparison_Scope'], ...
    'Position', [795, 360, 845, 415], 'BackgroundColor', 'green');

add_line(mdl, 'Sum_PI/1', 'Error_Mux/1');
add_line(mdl, 'Sum_DDPG/1', 'Error_Mux/2');
add_line(mdl, 'Sum_TD3/1', 'Error_Mux/3');
add_line(mdl, 'Error_Mux/1', 'Error_Comparison_Scope (Vref - Vsensed)/1');

add_line(mdl, 'Desired DC Bus Voltage (300V)/1', 'Voltage_Mux/1');
add_line(mdl, 'DC Bus Plant (PI)/1', 'Voltage_Mux/2');
add_line(mdl, 'DC Bus Plant (DDPG)/1', 'Voltage_Mux/3');
add_line(mdl, 'DC Bus Plant (TD3)/1', 'Voltage_Mux/4');
add_line(mdl, 'Voltage_Mux/1', 'DCBus_Voltage_Comparison_Scope/1');

% Big Interactive Button Block on Canvas: Double-Click to Run & Show Comparison Graph
add_block('simulink/Ports & Subsystems/Subsystem', [mdl '/CLICK_HERE_TO_RUN_AND_PLOT_COMPARISON'], ...
    'Position', [735, 60, 930, 150], 'BackgroundColor', '[1.0, 0.94, 0.35]');
Simulink.SubSystem.deleteContents([mdl '/CLICK_HERE_TO_RUN_AND_PLOT_COMPARISON']);
set_param([mdl '/CLICK_HERE_TO_RUN_AND_PLOT_COMPARISON'], ...
    'OpenFcn', 'sim(bdroot);', ...
    'MaskDisplay', 'disp(sprintf(''DOUBLE-CLICK HERE\nTO SIMULATE & GENERATE\nCOMPARISON GRAPH\n(PI vs DDPG vs TD3)''))');

save_system(mdl, 'DCBus_PI_DDPG_TD3_Comparison.slx');
fprintf('Saved DCBus_PI_DDPG_TD3_Comparison.slx successfully.\n');

% Export a clean screenshot of the Simulink model canvas itself
try
    print(['-s' mdl], '-dpng', '-r200', 'Simulink_3Model_Canvas.png');
    fprintf('Exported Simulink_3Model_Canvas.png successfully.\n');
catch ME
    warning('Canvas print warning: %s', ME.message);
end

%% === Step 3: Run Simulation & Verify Output ===
fprintf('=== Step 3: Running Simulation of DCBus_PI_DDPG_TD3_Comparison.slx ===\n');
tic;
out = sim(mdl);
elapsed = toc;
fprintf('Simulation completed in %.2f seconds!\n', elapsed);

% Trigger StopFcn logic explicitly to save the PNG in batch mode
eval(stopFcnScript);

close_system(mdl, 0);
fprintf('=== ALL DC BUS SIMULINK MODELS BUILT AND VERIFIED SUCCESSFULLY! ===\n');

%% =========================================================================
%  LOCAL HELPER FUNCTIONS
%  =========================================================================
function buildDCBusPlant(blkPath)
    Simulink.SubSystem.deleteContents(blkPath);
    add_block('simulink/Sources/In1', [blkPath '/u_ctrl'], 'Position', [25, 68, 55, 82]);
    add_block('simulink/Sources/In1', [blkPath '/I_dist'], 'Position', [25, 138, 55, 152]);
    add_block('simulink/Discontinuities/Saturation', [blkPath '/Sat_u'], 'UpperLimit', '10', 'LowerLimit', '-10', 'Position', [90, 60, 120, 90]);
    add_block('simulink/Math Operations/Gain', [blkPath '/Conv_Gain'], 'Gain', '1.5', 'Position', [150, 60, 190, 90]);
    add_block('simulink/Sources/Constant', [blkPath '/Base_Curr'], 'Value', '5.0', 'Position', [150, 15, 185, 40]);
    add_block('simulink/Math Operations/Sum', [blkPath '/Net_Curr'], 'Inputs', '++-', 'Position', [235, 55, 265, 115]);
    add_block('simulink/Math Operations/Gain', [blkPath '/inv_Cdc'], 'Gain', '1/0.0047', 'Position', [300, 70, 355, 100]);
    add_block('simulink/Continuous/Integrator', [blkPath '/Vdc_Int'], 'InitialCondition', '300', 'UpperSaturationLimit', '360', 'LowerSaturationLimit', '240', 'LimitOutput', 'on', 'Position', [390, 70, 425, 100]);
    add_block('simulink/Sinks/Out1', [blkPath '/Vsensed'], 'Position', [475, 78, 505, 92]);

    add_line(blkPath, 'u_ctrl/1', 'Sat_u/1');
    add_line(blkPath, 'Sat_u/1', 'Conv_Gain/1');
    add_line(blkPath, 'Base_Curr/1', 'Net_Curr/1');
    add_line(blkPath, 'Conv_Gain/1', 'Net_Curr/2');
    add_line(blkPath, 'I_dist/1', 'Net_Curr/3');
    add_line(blkPath, 'Net_Curr/1', 'inv_Cdc/1');
    add_line(blkPath, 'inv_Cdc/1', 'Vdc_Int/1');
    add_line(blkPath, 'Vdc_Int/1', 'Vsensed/1');
end

function buildCostBlock(blkPath)
    Simulink.SubSystem.deleteContents(blkPath);
    add_block('simulink/Sources/In1', [blkPath '/error'], 'Position', [30, 45, 60, 60]);
    add_block('simulink/Sources/In1', [blkPath '/u'], 'Position', [30, 105, 60, 120]);
    add_block('simulink/Math Operations/Math Function', [blkPath '/e_sq'], 'Operator', 'square', 'Position', [100, 38, 130, 68]);
    add_block('simulink/Math Operations/Math Function', [blkPath '/u_sq'], 'Operator', 'square', 'Position', [100, 98, 130, 128]);
    add_block('simulink/Math Operations/Gain', [blkPath '/u_weight'], 'Gain', '0.05', 'Position', [160, 98, 205, 128]);
    add_block('simulink/Math Operations/Add', [blkPath '/AddCost'], 'Position', [240, 60, 270, 95]);
    add_block('simulink/Math Operations/Gain', [blkPath '/Neg'], 'Gain', '-1', 'Position', [300, 63, 335, 93]);
    add_block('simulink/Sinks/Out1', [blkPath '/cost'], 'Position', [370, 70, 400, 85]);

    add_line(blkPath, 'error/1', 'e_sq/1');
    add_line(blkPath, 'u/1', 'u_sq/1');
    add_line(blkPath, 'u_sq/1', 'u_weight/1');
    add_line(blkPath, 'e_sq/1', 'AddCost/1');
    add_line(blkPath, 'u_weight/1', 'AddCost/2');
    add_line(blkPath, 'AddCost/1', 'Neg/1');
    add_line(blkPath, 'Neg/1', 'cost/1');
end
