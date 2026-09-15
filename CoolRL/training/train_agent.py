import os
import sys
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Add the parent directory to sys.path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.datacenter_env import DataCenterEnv
from agent.q_learning import QLearningAgent

def train(episodes=2000):
    print(f"Starting training for {episodes} episodes...")
    env = DataCenterEnv(seed=42)
    agent = QLearningAgent(action_space=env.action_space, 
                           alpha=0.1, gamma=0.95, 
                           epsilon=1.0, min_epsilon=0.05, epsilon_decay=0.995)
                           
    episode_rewards = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            
            agent.learn(state, action, reward, next_state)
            
            state = next_state
            total_reward += reward
            
        agent.decay_epsilon()
        episode_rewards.append(total_reward)
        
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode + 1}/{episodes} - Avg Reward (last 100): {avg_reward:.2f} - Epsilon: {agent.epsilon:.3f}")
            
    print("Training finished.")
    
    # Save the Q-table model
    os.makedirs('models', exist_ok=True)
    model_path = 'models/q_table.pkl'
    joblib.dump(agent.q_table, model_path)
    print(f"Model saved to {model_path}")
    
    # Plot and save training curve
    os.makedirs('outputs', exist_ok=True)
    plot_path = 'outputs/training_curve.png'
    
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards, alpha=0.3, color='blue', label='Episode Reward')
    
    # Moving average
    if len(episode_rewards) >= 100:
        moving_avg = np.convolve(episode_rewards, np.ones(100)/100, mode='valid')
        plt.plot(np.arange(99, len(episode_rewards)), moving_avg, color='red', label='100-Episode Moving Avg')
        
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Q-Learning Agent Training Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Training curve saved to {plot_path}")

if __name__ == "__main__":
    train(episodes=2000)
