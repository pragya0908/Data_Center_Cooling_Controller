# CoolRL — Data-Driven Adaptive Data Center Cooling Optimization using Reinforcement Learning

## Project Overview
CoolRL is a simulation-based, data-grounded reinforcement learning system for adaptive data-center cooling control. The system uses real-world public datasets to ground workload and environmental conditions where appropriate, while the actual cooling-control environment operates as a controlled simulation.

## Core RL Framework
- **Algorithm**: Tabular Q-learning.
- **State Representation**: A compact discretized state consisting of:
  - Data-center temperature
  - Workload
  - Ambient temperature
  - Cooling level
  - Temperature trend
- **Action Space**: Five discrete cooling adjustments (bounded between 0% and 100% cooling):
  1. Decrease cooling by 20%
  2. Decrease cooling by 10%
  3. Maintain cooling (0%)
  4. Increase cooling by 10%
  5. Increase cooling by 20%
- **Reward Function**: Balances:
  - Thermal safety
  - Cooling energy consumption
  - Overheating penalties
  - Overcooling penalties

## Controllers Compared
1. Fixed cooling
2. Rule-based cooling
3. Q-learning adaptive cooling

## Evaluation Scenarios
The system will be evaluated under:
1. Normal workload
2. High workload
3. Sudden workload spikes
4. High ambient temperature
5. Combined stress conditions

## Interactive Interface
The final system will include an interactive Streamlit application where the trained Q-learning agent interacts with the simulated environment and makes cooling decisions dynamically.

## Academic Constraints & Environment
- **Supported Python Version**: Python 3.11 is the only supported Python version for this project.
- **Academic Integrity**: Experimental results are never fabricated. Real-world dataset observations, simulation assumptions, and experimental results are strictly distinguished.
- **Simulation Modeling**: Simulation parameters are explicitly modeled abstractions and not claimed to be real hardware specifications.
- **Methodological Discipline**: Dimensionality reduction (such as PCA) and external datasets are only introduced if strictly and technically justified for the RL task.
