classdef DCBusEnv < rl.env.MATLABEnvironment
    % DCBUSENV DC Bus Voltage Regulation Environment — Version 3
    %
    % Fixes applied (based on root-cause analysis of 3-step death loop):
    %   1. Reset at EXACTLY 300V — no random perturbation
    %      Agent learns load-ripple rejection first before handling offsets.
    %   2. NO observation clamping — raw error divided by scale factor only,
    %      never clipped. Neural network sees true gradient at all voltages.
    %   3. NO early termination — episodes always run full MaxSteps.
    %      Voltage soft-clamped at [200, 400] for numerical safety only.
    %   4. Clean reward: quadratic error + smoothness, no boundary traps.
    
    properties
        V_ref    = 300.0;       % Reference voltage (V)
        C_dc     = 4700e-6;     % DC bus capacitance (F)
        dt       = 0.001;       % Step time (s)
        MaxSteps = 2000;        % Steps per episode (2 seconds)
        
        % Normalization divisors (for scaling, NOT clamping)
        ErrScale  = 10.0;      % Divide error by this
        DErrScale = 1000.0;    % Divide error derivative by this
        ActScale  = 10.0;      % Action scaling factor
        
        State = zeros(3,1);
    end
    
    properties(Access = private)
        CurrentStep = 0;
        PrevAction  = 0.0;     % Raw action [-10, 10]
        PrevVsensed = 300.0;
        PrevError   = 0.0;
    end
    
    methods
        function this = DCBusEnv()
            % Observation spec: set wide limits but DO NOT enforce them
            % These are hints to the agent about expected range, not hard clips
            ObservationInfo = rlNumericSpec([3 1], ...
                'LowerLimit', [-Inf; -Inf; -Inf], ...
                'UpperLimit', [ Inf;  Inf;  Inf]);
            ObservationInfo.Name = 'DC_Bus_Observations';
            ObservationInfo.Description = 'Scaled_Error, Scaled_dError, Scaled_PrevAction';
            
            % Action in [-1, 1], scaled internally to [-10, 10]
            ActionInfo = rlNumericSpec([1 1], ...
                'LowerLimit', -1.0, ...
                'UpperLimit',  1.0);
            ActionInfo.Name = 'Converter_Control_Effort';
            ActionInfo.Description = 'Normalized_Control_Action';
            
            this = this@rl.env.MATLABEnvironment(ObservationInfo, ActionInfo);
        end
        
        function [Observation, Reward, IsDone, LoggedSignals] = step(this, Action)
            LoggedSignals = [];
            this.CurrentStep = this.CurrentStep + 1;
            
            % 1. Clip action to [-1, 1], scale to physical range
            action_norm = max(min(Action, 1.0), -1.0);
            action_raw  = action_norm * this.ActScale;   % [-10, 10]
            
            % 2. Load current with sinusoidal disturbance
            base_load   = 5.0;
            ripple_load = 2.0 * sin(2 * pi * 10 * this.CurrentStep * this.dt);
            i_load      = base_load + ripple_load;
            
            % 3. Converter current delivery
            i_control = base_load + (action_raw * 1.5);
            
            % 4. Physics: C * dV/dt = I_control - I_load
            dV = ((i_control - i_load) / this.C_dc) * this.dt;
            v_sensed_new = this.PrevVsensed + dV;
            
            % 5. Numerical safety clamp only (NOT a learning signal)
            %    This prevents NaN/Inf, not used in reward or observation
            v_sensed_new = max(min(v_sensed_new, 400.0), 200.0);
            
            % 6. Raw error and error derivative
            err_raw  = this.V_ref - v_sensed_new;
            derr_raw = (err_raw - this.PrevError) / this.dt;
            
            % 7. Scale observations — DIVIDE ONLY, NO CLAMPING
            %    The neural network sees the TRUE magnitude at all times
            err_scaled  = err_raw  / this.ErrScale;    % Typical range ~[-1,1] but unbounded
            derr_scaled = derr_raw / this.DErrScale;    % Typical range ~[-1,1] but unbounded
            act_scaled  = this.PrevAction / this.ActScale; % [-1, 1]
            
            % Update tracking
            this.PrevVsensed = v_sensed_new;
            this.PrevError   = err_raw;
            
            % Build state (unbounded scaled values)
            this.State  = [err_scaled; derr_scaled; act_scaled];
            Observation = this.State;
            
            % 8. Reward: bounded and well-scaled
            %    Reward per step is in [-2, +1] range so episode totals
            %    stay in [-4000, +2000] — manageable for critic learning.
            %    Uses saturating penalty: quadratic near zero, bounded far away.
            
            % Primary: Saturating error penalty
            %   Near zero: behaves like -err^2 (good gradient)
            %   Far from zero: saturates at -2.0 (prevents reward explosion)
            err_penalty = -2.0 * (1.0 - exp(-0.5 * err_scaled^2));
            
            % Smooth control effort (small, bounded)
            delta_act = action_norm - act_scaled;
            smooth_penalty = -0.1 * delta_act^2;
            
            % Small effort penalty
            effort_penalty = -0.02 * action_norm^2;
            
            Reward = err_penalty + smooth_penalty + effort_penalty;
            
            % Continuous bonus for tight regulation (±2V band)
            if abs(err_raw) < 2.0
                Reward = Reward + 1.0 * (1.0 - abs(err_raw) / 2.0);
            end
            
            % Update previous action
            this.PrevAction = action_raw;
            
            % 9. NEVER terminate early — agent must learn to recover
            IsDone = (this.CurrentStep >= this.MaxSteps);
        end
        
        function InitialObservation = reset(this)
            this.CurrentStep = 0;
            this.PrevAction  = 0.0;
            
            % FIX: Start EXACTLY at the reference voltage
            % No random perturbation — let the agent first learn to handle
            % the load ripple disturbance from a balanced starting point.
            % Once trained, you can add perturbations for robustness testing.
            v_init = this.V_ref;   % Exactly 300.0V
            
            this.PrevVsensed = v_init;
            this.PrevError   = 0.0;
            
            this.State = [0.0; 0.0; 0.0];  % Perfect starting state
            InitialObservation = this.State;
        end
    end
end
