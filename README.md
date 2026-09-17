# CoolRL — Data-Driven Adaptive Data Center Cooling Optimization using Reinforcement Learning

## Project Overview
CoolRL is a simulation-based, data-grounded reinforcement learning system for adaptive data-center cooling control. The system uses real-world public datasets to ground workload and environmental conditions, while the actual cooling-control environment operates as a controlled physical simulation based on ASHRAE TC 9.9 thermal guidelines.

## Core RL Framework
- **Algorithm**: Tabular Q-learning.
- **State Representation**: A compact discretized state consisting of:
  - Data-center internal temperature
  - Server workload
  - Ambient outdoor temperature
  - Current cooling level
  - Temperature trend (rate of change)
- **Action Space**: Five discrete cooling adjustments (bounded between 0% and 100% capacity):
  1. Decrease cooling by 20%
  2. Decrease cooling by 10%
  3. Maintain cooling (0% change)
  4. Increase cooling by 10%
  5. Increase cooling by 20%
- **Reward Function**: A comprehensive function that balances:
  - Thermal safety (maintaining temperatures in the 18°C–27°C recommended zone)
  - Cooling energy consumption minimization
  - Overheating penalties (exponential scaling for severe breaches >32°C)
  - Overcooling penalties (wasting energy when T < 18°C)
  - Action churn penalties (preventing excessive actuator wear)

## Interactive Dashboard
The system includes a rich, multi-page interactive Streamlit dashboard allowing users to visualize the RL agent's behavior, compare it against baselines, and analyze the learned policy.

### Running the Dashboard
1. Activate your virtual environment: `.\.venv\Scripts\Activate.ps1`
2. Run the Streamlit application: `streamlit run app/main.py`
3. Open your browser to the local URL provided by Streamlit (usually http://localhost:8501).

### Dashboard Pages
- **Live RL Simulation**: Watch the trained Q-learning agent make cooling decisions in real-time under standard and stress scenarios.
- **Controller Comparison**: Compare the Q-learning agent's performance (energy efficiency and thermal safety) against Fixed-Cooling and Rule-Based baseline controllers.
- **What If Simulator**: Subject the agent to extreme, synthetic anomaly scenarios (e.g., cooling degradation, sudden workload spikes, flash heatwaves) and see how it reacts.
- **Learned Policy Inspector**: Dive into the underlying Q-table to understand exactly *why* the agent makes certain decisions based on state variables.
- **Experimental Results**: Review statistical robustness (seed variation), hyperparameter sensitivity, and out-of-distribution generalization results (e.g., zero-shot transfer to real Google cluster traces).

## Evaluation Scenarios
The system is rigorously evaluated under the following profiles:
1. **Normal Workload**: Standard diurnal patterns.
2. **High Workload**: Sustained peak utilization.
3. **Workload Spikes**: Sudden, extreme surges in demand.
4. **High Ambient**: Extreme summer heatwaves impacting cooling efficiency.
5. **Combined Stress**: Worst-case scenario (high workload + heatwave).

## Academic Constraints & Environment
- **Supported Python Version**: Python 3.11 is the primary supported version.
- **Academic Integrity**: Experimental results are never fabricated. Real-world dataset observations, simulation assumptions, and experimental results are strictly distinguished.
- **Simulation Modeling**: Simulation parameters are explicitly modeled abstractions and not claimed to be exact representations of specific hardware.
