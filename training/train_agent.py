"""CoolRL training loop: trains a tabular Q-learning agent on DataCenterEnv."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless rendering
import matplotlib.pyplot as plt

from agent.q_learning import QLearningAgent
from environment.datacenter_env import DataCenterEnv

# ── Hyperparameters ──────────────────────────────────────────────────────
NUM_EPISODES: int = 2000
MAX_STEPS: int = 200
MODEL_PATH: Path = Path("models/q_table.pkl")
PLOT_PATH: Path = Path("outputs/training_curve.png")


def train() -> None:
    """Run the full training loop and persist artefacts."""
    env = DataCenterEnv(workload_mode="normal", seed=42)
    agent = QLearningAgent(num_actions=DataCenterEnv.NUM_ACTIONS, seed=42)

    episode_rewards: list[float] = []

    for episode in range(NUM_EPISODES):
        # Reset environment to initial conditions
        env = DataCenterEnv(workload_mode="normal", seed=42 + episode)
        state = env._get_discrete_state()
        total_reward = 0.0

        for _ in range(MAX_STEPS):
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state)

            state = next_state
            total_reward += reward

            if done:
                break

        episode_rewards.append(total_reward)

        # Progress logging every 100 episodes
        if (episode + 1) % 100 == 0:
            avg_reward = sum(episode_rewards[-100:]) / 100
            print(
                f"Episode {episode + 1:>5}/{NUM_EPISODES}  |  "
                f"Avg Reward (last 100): {avg_reward:>8.2f}  |  "
                f"Epsilon: {agent.epsilon:.4f}  |  "
                f"Q-table size: {len(agent.q_table)}"
            )

    # ── Save model ───────────────────────────────────────────────────────
    agent.save(MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

    # ── Plot training curve ──────────────────────────────────────────────
    _plot_training_curve(episode_rewards)
    print(f"Training curve saved to {PLOT_PATH}")


def _plot_training_curve(rewards: list[float]) -> None:
    """Plot episode vs. total reward and save to disk."""
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(rewards, linewidth=0.5, alpha=0.4, label="Per-episode reward")

    # Smoothed curve (100-episode rolling average)
    window = 100
    if len(rewards) >= window:
        smoothed = [
            sum(rewards[i : i + window]) / window
            for i in range(len(rewards) - window + 1)
        ]
        ax.plot(
            range(window - 1, len(rewards)),
            smoothed,
            linewidth=2,
            color="crimson",
            label=f"{window}-episode rolling avg",
        )

    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("CoolRL - Q-Learning Training Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    train()
