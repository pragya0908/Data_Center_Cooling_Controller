# CoolRL

> Reinforcement Learning based adaptive cooling controller for energy-efficient data center management.

## Research Question

**Can a reinforcement-learning agent dynamically control data-center cooling to reduce cooling energy consumption while maintaining safe operating temperatures under changing workloads and environmental conditions?**

## Objectives

1. Develop a simulated data-center environment representing workload, temperature, ambient conditions, cooling level, and energy consumption.
2. Implement a Q-learning agent capable of dynamically adjusting cooling intensity.
3. Design a reward function balancing thermal safety and cooling energy consumption.
4. Compare the RL controller against fixed and rule-based cooling strategies.
5. Develop an interactive GUI for visualizing the data center and testing the learned controller.

## Project Structure

- `environment/`: Data center thermal simulation and workload generation
- `agent/`: Q-learning agent
- `baselines/`: Fixed and rule-based controllers
- `training/`: Training scripts
- `evaluation/`: Evaluation scripts to compare controllers
- `models/`: Saved Q-tables
- `outputs/`: Evaluation plots and metrics
- `app/`: Streamlit dashboard
- `notebooks/`: Exploratory analysis

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt
```
