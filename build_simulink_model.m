%% Build and Export DC Bus DRL Simulink Model
clear; clc; close all;
cd('C:\Users\Rakesh T\OneDrive\Documents\MATLAB');

modelName = 'DCBus_DRL_Model';

% Close if already open
if bdIsLoaded(modelName)
    close_system(modelName, 0);
end

% Create new system
new_system(modelName);
open_system(modelName);

% Set solver and sample time
set_param(modelName, 'SolverType', 'Fixed-step', 'Solver', 'FixedStepDiscrete', 'FixedStep', '0.001', 'StopTime', '2.0');

% Add Blocks
% 1. Vdc_ref Constant
add_block('simulink/Sources/Constant', [modelName '/Vdc_ref'], 'Value', '300', 'Position', [40, 100, 90, 130]);

% 2. Error Sum Block (Vref - Vsensed)
add_block('simulink/Math Operations/Sum', [modelName '/Error_Sum'], 'Inputs', '+-', 'Position', [150, 105, 180, 135]);

% 3. Derivative of Error (Discrete Derivative)
add_block('simulink/Discrete/Discrete Derivative', [modelName '/d_Error'], 'Position', [230, 140, 270, 170]);

% 4. Previous Action Unit Delay
add_block('simulink/Discrete/Unit Delay', [modelName '/Prev_Action_Delay'], 'SampleTime', '0.001', 'Position', [230, 200, 270, 230]);

% 5. Observation Mux (3 inputs: err, derr, prev_act)
add_block('simulink/Signal Routing/Mux', [modelName '/Obs_Mux'], 'Inputs', '3', 'Position', [330, 95, 335, 240]);

% 6. RL Agent Block
try
    add_block('rlLibraries/RL Agent', [modelName '/RL_Agent'], 'Position', [400, 130, 480, 200]);
catch
    add_block('simulink/User-Defined Functions/MATLAB Function', [modelName '/RL_Agent'], 'Position', [400, 130, 480, 200]);
end

% 7. Action Scaling & Saturation [-10, 10]
add_block('simulink/Commonly Used Blocks/Gain', [modelName '/Action_Scale'], 'Gain', '10.0', 'Position', [520, 150, 560, 180]);
add_block('simulink/Commonly Used Blocks/Saturate', [modelName '/Action_Sat'], 'UpperLimit', '10.0', 'LowerLimit', '-10.0', 'Position', [590, 150, 630, 180]);

% 8. Converter Current Delivery (I_control = 5.0 + 1.5 * action)
add_block('simulink/Commonly Used Blocks/Gain', [modelName '/Converter_Gain'], 'Gain', '1.5', 'Position', [670, 150, 710, 180]);
add_block('simulink/Sources/Constant', [modelName '/Base_Current'], 'Value', '5.0', 'Position', [670, 100, 710, 125]);
add_block('simulink/Math Operations/Sum', [modelName '/Sum_Icontrol'], 'Inputs', '++', 'Position', [750, 140, 775, 170]);

% 9. Load Disturbance Current (5A base + 2A 10Hz ripple)
add_block('simulink/Sources/Sine Wave', [modelName '/Ripple_Load'], 'Amplitude', '2.0', 'Frequency', '2*pi*10', 'Position', [670, 240, 710, 270]);
add_block('simulink/Math Operations/Sum', [modelName '/Sum_Iload'], 'Inputs', '++', 'Position', [750, 230, 775, 260]);

% 10. Net Capacitor Current (I_control - I_load)
add_block('simulink/Math Operations/Sum', [modelName '/Sum_Ic'], 'Inputs', '+-', 'Position', [820, 180, 845, 210]);

% 11. Plant Dynamics: Integrator (1 / C_dc) => V_sensed
add_block('simulink/Commonly Used Blocks/Gain', [modelName '/Inv_Capacitance'], 'Gain', '1/4700e-6', 'Position', [880, 180, 940, 210]);
add_block('simulink/Continuous/Integrator', [modelName '/DC_Bus_Voltage'], 'InitialCondition', '300.0', 'Position', [980, 180, 1015, 210]);

% 12. Reward Computation & Scopes
add_block('simulink/Sinks/Scope', [modelName '/Voltage_Scope'], 'Position', [1080, 180, 1110, 210]);
add_block('simulink/Sinks/Scope', [modelName '/Error_Scope'], 'Position', [240, 50, 270, 80]);
add_block('simulink/Sinks/Scope', [modelName '/Action_Scope'], 'Position', [680, 50, 710, 80]);

% Connect Lines
add_line(modelName, 'Vdc_ref/1', 'Error_Sum/1');
add_line(modelName, 'DC_Bus_Voltage/1', 'Voltage_Scope/1');
add_line(modelName, 'DC_Bus_Voltage/1', 'Error_Sum/2', 'autorouting', 'on');
add_line(modelName, 'Error_Sum/1', 'Error_Scope/1');
add_line(modelName, 'Error_Sum/1', 'd_Error/1');
add_line(modelName, 'Error_Sum/1', 'Obs_Mux/1');
add_line(modelName, 'd_Error/1', 'Obs_Mux/2');
add_line(modelName, 'Prev_Action_Delay/1', 'Obs_Mux/3');
add_line(modelName, 'Obs_Mux/1', 'RL_Agent/1');
add_line(modelName, 'RL_Agent/1', 'Action_Scale/1');
add_line(modelName, 'RL_Agent/1', 'Prev_Action_Delay/1', 'autorouting', 'on');
add_line(modelName, 'Action_Scale/1', 'Action_Sat/1');
add_line(modelName, 'Action_Sat/1', 'Action_Scope/1');
add_line(modelName, 'Action_Sat/1', 'Converter_Gain/1');
add_line(modelName, 'Base_Current/1', 'Sum_Icontrol/1');
add_line(modelName, 'Converter_Gain/1', 'Sum_Icontrol/2');
add_line(modelName, 'Base_Current/1', 'Sum_Iload/1', 'autorouting', 'on');
add_line(modelName, 'Ripple_Load/1', 'Sum_Iload/2');
add_line(modelName, 'Sum_Icontrol/1', 'Sum_Ic/1');
add_line(modelName, 'Sum_Iload/1', 'Sum_Ic/2');
add_line(modelName, 'Sum_Ic/1', 'Inv_Capacitance/1');
add_line(modelName, 'Inv_Capacitance/1', 'DC_Bus_Voltage/1');

% Save model
save_system(modelName);

% Export high-res diagram image
out_img = 'C:\Users\Rakesh T\OneDrive\Documents\MATLAB\simulink_model_diagram.png';
out_art = 'C:\Users\Rakesh T\.gemini\antigravity\brain\dc15fb40-6114-4039-9548-f8dcd49e57c3\simulink_model_diagram.png';

print(['-s' modelName], '-dpng', '-r200', out_img);
print(['-s' modelName], '-dpng', '-r200', out_art);

fprintf('Simulink model created and saved as %s.slx\n', modelName);
fprintf('Diagram exported to %s\n', out_img);
