import random
from environment.datacenter_env import DataCenterEnv

def run_test():
    print("Testing Data Center Simulation Environment...")
    env = DataCenterEnv(seed=42)
    
    state = env.reset()
    print(f"Initial State: Temp={state[0]}°C, Workload={state[1]}%, Ambient={state[2]}°C, Cooling={state[3]}%")
    print("-" * 50)
    
    total_reward = 0
    
    for step in range(1, 51):
        # Randomly choose an action
        # 0: -20%, 1: -10%, 2: 0%, 3: +10%, 4: +20%
        action = random.choice(env.action_space)
        
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        
        print(f"Step {step}")
        print(f"Action taken: {action} (Cooling -> {next_state[3]}%)")
        print(f"Workload: {next_state[1]}%")
        print(f"Temperature: {next_state[0]}°C")
        print(f"Energy: {info['energy']} kWh")
        print(f"Reward: {reward}")
        print("-" * 30)
        
        if done:
            print("Episode finished.")
            break
            
    print(f"Test completed. Total reward over 50 steps: {round(total_reward, 2)}")

if __name__ == "__main__":
    run_test()
