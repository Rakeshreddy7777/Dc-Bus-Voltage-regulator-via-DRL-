
# Deep Reinforcement Learning for DC Bus Voltage Regulation

**Author:** T Rakesh Reddy

**Project:** AI/ML Voltage Controller for DC Microgrids & Power Converters

**Method:** Deep Deterministic Policy Gradient (DDPG) in MATLAB

---

## Overview

This project presents an AI/ML-driven continuous voltage controller designed to replace traditional Proportional-Integral (PI) control in DC microgrid and DC bus systems. The controller is built using **Deep Deterministic Policy Gradient (DDPG)** â€” a model-free, off-policy reinforcement learning algorithm suited for continuous action spaces.

The system is trained and validated on **real operational data comprising 120,001 readings**, capturing the full range of voltage dynamics, load disturbances, and transient conditions observed in actual DC microgrid deployments.

---

## Project Objectives

- Design a DDPG-based agent capable of regulating DC bus voltage under varying load conditions
- Replace conventional PI controllers with a learned policy that adapts to system dynamics without manual tuning
- Train the agent using real measurement data (120,001 time-series readings) from DC microgrid hardware
- Evaluate steady-state error, overshoot, and settling time against baseline PI performance
- Demonstrate robust voltage regulation across load steps and disturbance scenarios

---

## Why DDPG?

Traditional PI controllers require careful manual tuning and struggle with nonlinear, time-varying plant dynamics. DDPG addresses this by:

- Operating directly in **continuous action spaces** â€” the duty cycle or reference voltage is adjusted as a real-valued signal, not a discrete command
- Learning an **actor-critic architecture** where the actor proposes control actions and the critic evaluates their long-term reward
- Using an **experience replay buffer** to decorrelate training samples and improve sample efficiency
- Applying **target networks** for stable, convergent learning under noisy plant feedback

---

## Dataset

| Property | Detail |
|---|---|
| Source | Real DC microgrid hardware measurements |
| Total Readings | 120,001 time-series samples |
| Signals Captured | Bus voltage, load current, converter duty cycle, reference setpoints |
| Sampling Conditions | Variable load steps, steady-state periods, transient disturbances |
| Platform | Imported and processed in MATLAB |

The dataset reflects genuine operating conditions rather than simulated or synthetic environments, ensuring that the trained agent is directly applicable to real-world deployments.

---

## System Architecture

**Environment**
The DC bus system is modelled as a Markov Decision Process (MDP). At each time step, the agent observes the voltage error and rate of change, then outputs a control action (duty cycle adjustment) to drive the bus voltage toward its reference.

**Reward Function**
The reward is shaped to penalise voltage deviation from the setpoint, excessive control effort, and oscillatory behaviour â€” encouraging fast settling with minimal overshoot.

**Actor Network**
A fully connected neural network maps observed states to continuous control actions. Batch normalisation is applied between layers to stabilise training across varying input scales.

**Critic Network**
A separate network estimates the Q-value (expected cumulative reward) for a given state-action pair, providing the gradient signal used to update the actor.

**Training**
- Algorithm: DDPG with experience replay and soft target updates
- Framework: MATLAB Reinforcement Learning Toolbox
- Episodes trained over real measurement sequences from the 120,001-reading dataset
- Exploration noise: Ornstein-Uhlenbeck process for temporally correlated action perturbation

---

## Expected Outcomes

- Voltage regulation within tight error bands under both steady-state and transient conditions
- Faster disturbance rejection compared to a well-tuned PI baseline
- Adaptive response to load variations without re-tuning
- A trained MATLAB policy object deployable to hardware-in-the-loop or embedded targets

---

## Tools and Environment

| Component | Detail |
|---|---|
| Language / Platform | MATLAB R2023a or later |
| RL Toolbox | MATLAB Reinforcement Learning Toolbox |
| Neural Network | Deep Learning Toolbox |
| Data Source | Real hardware â€” 120,001 readings |
| Algorithm | Deep Deterministic Policy Gradient (DDPG) |

---

*Project by T Rakesh Reddy â€” AI/ML Voltage Controller for DC Microgrids & Power Converters*