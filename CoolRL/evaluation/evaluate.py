import os
import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add the parent directory to sys.path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.datacenter_env import DataCenterEnv
from agent.q_learning import QLearningAgent
from baselines.fixed_controller import FixedController
from baselines.rule_controller import RuleBasedController

def evaluate_controller(env, controller_func, name):
    """
    Evaluates a controller on the environment.
    controller_func should take (state, env) and return an action index.
    """
    state = env.reset()
    done = False
    
    total_energy = 0
    total_reward = 0
    temperatures = []
    energy_history = []
    workloads = []
    overheat_events = 0
    safe_time = 0
    
    steps = 0
    
    while not done:
        action = controller_func(state, env)
        next_state, reward, done, info = env.step(action)
        
        temp = next_state[0]
        workload = next_state[1]
        
        temperatures.append(temp)
        workloads.append(workload)
        energy_history.append(info['energy'])
        
        total_energy += info['energy']
        total_reward += reward
        
        if info['overheating']:
            overheat_events += 1
            
        if info['safe_range']:
            safe_time += 1
            
        state = next_state
        steps += 1
        
    metrics = {
        'Controller': name,
        'Energy (kWh)': round(total_energy, 2),
        'Avg Temp (°C)': round(np.mean(temperatures), 2),
        'Max Temp (°C)': round(np.max(temperatures), 2),
        'Overheating Events': overheat_events,
        'Safe Time (%)': round((safe_time / steps) * 100, 2),
        'Total Reward': round(total_reward, 2)
    }
    
    history = {
        'temperatures': temperatures,
        'energy': energy_history,
        'workloads': workloads
    }
    
    return metrics, history

def run_evaluation():
    print("Evaluating Controllers...")
    
    # 1. RL Agent (Load Trained Model)
    rl_agent = QLearningAgent(action_space=[0, 1, 2, 3, 4])
    try:
        rl_agent.q_table = joblib.load('models/q_table.pkl')
        print("Successfully loaded trained Q-table.")
    except Exception as e:
        print(f"Warning: Could not load trained Q-table ({e}). Using untrained agent.")
        
    # Wrapper function for RL agent
    def rl_action(state, env):
        # We pass evaluate=True to always pick the best known action
        return rl_agent.choose_action(state, evaluate=True)
        
    # 2. Fixed Controller
    fixed_ctrl = FixedController(fixed_level=70.0)
    def fixed_action(state, env):
        return fixed_ctrl.choose_action(state, env.cooling_level)
        
    # 3. Rule-Based Controller
    rule_ctrl = RuleBasedController()
    def rule_action(state, env):
        return rule_ctrl.choose_action(state)
        
    controllers = [
        ('Fixed (70%)', fixed_action),
        ('Rule-Based', rule_action),
        ('Q-Learning', rl_action)
    ]
    
    all_metrics = []
    all_histories = {}
    
    # Evaluate each controller using the same seed for identical workloads
    for name, func in controllers:
        env = DataCenterEnv(seed=123)  # Use fixed seed for evaluation test set
        metrics, history = evaluate_controller(env, func, name)
        all_metrics.append(metrics)
        all_histories[name] = history
        
    # Format results as a DataFrame
    df = pd.DataFrame(all_metrics)
    print("\n--- Evaluation Results ---")
    print(df.to_string(index=False))
    
    # Save results
    os.makedirs('outputs', exist_ok=True)
    df.to_csv('outputs/evaluation_metrics.csv', index=False)
    
    # Generate Comparison Plots
    generate_plots(all_histories)
    
def generate_plots(histories):
    names = list(histories.keys())
    
    # Plot 1: Temperature over time
    plt.figure(figsize=(12, 6))
    for name in names:
        plt.plot(histories[name]['temperatures'], label=name, alpha=0.8)
    
    # Add safe zone background
    plt.axhspan(22, 27, color='green', alpha=0.1, label='Safe Zone')
    plt.axhline(y=30, color='red', linestyle='--', label='Critical Threshold')
    
    plt.xlabel('Time Step')
    plt.ylabel('Temperature (°C)')
    plt.title('Temperature Comparison Across Controllers')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/temperature_comparison.png')
    
    # Plot 2: Workload over time (Same for all, so just plot one)
    plt.figure(figsize=(12, 4))
    plt.plot(histories[names[0]]['workloads'], color='purple')
    plt.xlabel('Time Step')
    plt.ylabel('Workload (%)')
    plt.title('Data Center Workload Profile (Evaluation Episode)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/workload_profile.png')
    
    # Plot 3: Energy Consumption (Cumulative)
    plt.figure(figsize=(10, 5))
    for name in names:
        cumulative_energy = np.cumsum(histories[name]['energy'])
        plt.plot(cumulative_energy, label=name)
        
    plt.xlabel('Time Step')
    plt.ylabel('Cumulative Energy (kWh)')
    plt.title('Energy Consumption Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/energy_comparison.png')
    
    print("\nPlots saved to 'outputs/' directory.")

if __name__ == "__main__":
    run_evaluation()
