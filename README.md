
# Deep Reinforcement Learning for DC Bus Voltage Regulation

**Author:** T Rakesh Reddy
**Project:** AI/ML Voltage Controller for DC Microgrids & Power Converters
**Method:** Deep Deterministic Policy Gradient (DDPG) in MATLAB

---

An AI/ML-driven continuous voltage controller designed to replace traditional Proportional-Integral (PI) control in DC microgrid / DC bus systems using **Deep Deterministic Policy Gradient (DDPG)**. This project uses real operational data comprising **120,001 readings** collected from DC microgrid hardware.

---

## Project Objectives

1. **Replace Traditional PI Control:** Formulate DC bus voltage stabilization as a continuous Deep Reinforcement Learning (DRL) control problem.
2. **Minimize Error (Verr):** Regulate sensed bus voltage V to target reference setpoint V* = 300 V under dynamic, non-linear disturbance loads.
3. **Prevent Overshoot & Protect DC Bus:** Strictly clamp voltage excursions during sudden load shifts and transient spikes.

---

## System Physics & Architecture

```
V* (300V Ref) ——(+)——┐
                      ├——> Verr ——> [ DDPG Actor Network ] ——> Control Effort (u) ——> [ DC Bus Converter ]
V (Sensed)    ——(-)——┘
```

Capacitor Voltage Dynamics:

```
C * (dV/dt) = I_control - I_load
```

### Physical Parameters

- **Reference Voltage (V*):** 300.0 V
- **DC Bus Capacitance (C_dc):** 4700 µF (4.7 mF)
- **Simulation Time Step (Δt):** 1 ms (0.001 s)
- **Episode Duration:** 2,000 steps (2.0 s)
- **Disturbance Model:** I_load(t) = 5.0 A + 2.0 × sin(2π × 10t) A

---

## DRL Formulation (DDPG)

### Observation Space (State)

Continuous state vector St — 3-element vector:

- **Scaled Voltage Error** = (V* - V) / 10.0
- **Scaled Error Derivative** = (1 / 1000.0) × d(V* - V)/dt
- **Scaled Previous Action** = u_(t-1) / 10.0

### Action Space (Control Effort)

Continuous control effort a_t in the range [-1.0, 1.0], internally scaled to real converter duty action u_t in [-10.0, 10.0]:

```
I_control = I_base + 1.5 × u_t
```

### Reward Function

Designed with smooth, saturating penalties to avoid numerical explosions while maintaining steep gradient descent near setpoint:

```
R_t = -2.0 × (1 - exp(-0.5 × (Verr / 10)²)) - 0.1 × (Δu)² - 0.02 × u² + R_bonus
```

Where R_bonus = 1.0 × (1 - |Verr| / 2.0)  for  |Verr| < 2.0 V

---

## Comparative Performance Results

![DRL vs Historical PI Controller Performance](validation_results_v3.png)

### Quantitative Benchmark Table

| Performance Metric | Historical PI Controller | Trained DRL Controller | Winner |
|---|:---:|:---:|:---:|
| **Episode Survival** | N/A | **2,000 / 2,000 steps (100%)** | **Stable** |
| **Max Peak Error (\|Verr\|)** | **44.00 V** | **6.35 V** | **DRL (85.6% lower peak spike)** |
| **Voltage Operating Range** | 256.0 V to 344.0 V | 294.04 V to 306.35 V | **DRL (Strict safety bounds)** |
| **Mean Absolute Error (MAE)** | **2.15 V** | **3.10 V** | PI Baseline |
| **RMS Voltage Error** | **3.36 V** | **3.70 V** | PI Baseline |
| **Regulation within ±0.5 V** | **17.1%** | **9.4%** | PI Baseline |
| **Mean Control Effort \|u\|** | **5.50** | **0.56** | DRL uses <10% control power |

---

## Key Takeaways & Analysis

1. **Superior Overshoot Rejection:** The PI controller exhibited severe voltage spikes of up to **44 V** during transient events. The DRL agent clamped maximum error to **6.35 V**, protecting sensitive downstream DC bus loads.
2. **Stable Non-Chattering Control:** The actor policy produces smooth, non-oscillatory control signals using less than 10% of available power limits.
3. **Stability & Convergence:** Solved initial environment termination traps, achieving 100% survival rate across 2,000,000 training steps.

---

## Repository Structure

```
├── DCBusEnv.m                        # Custom MATLAB Reinforcement Learning Environment
├── train_ddpg_dcbus.m                # DDPG Agent Architecture & Hyperparameters
├── plot_results.m                    # 1-Click Validation & Comparison Waveform Generator
├── validate_env.m                    # 9-Step Environment Sanity Test Script
├── Trained_DRL_DCBus_Agent_v3.mat    # Pre-trained DDPG Neural Network Weights (1000 episodes)
├── Case Study DCbusData.csv.xlsx     # Benchmark PI Controller Dataset
├── training_monitor_screenshot.png   # MATLAB Training Progress GUI Screenshot
├── validation_results_v3.png         # DRL vs PI Performance Comparison Plot
└── README.md                         # Full Project Documentation
```

---

## How to Run & Reproduce

### 1. Evaluate Pre-Trained Model (Instant)

To run the validation simulation using the pre-trained weights and view the comparison plots:

```matlab
% In MATLAB Command Window:
run plot_results
```

### 2. Retrain from Scratch

```matlab
clear classes;
run train_ddpg_dcbus
```