"""Evaluate Q-Learning, Fixed, and Rule controllers on identical workloads."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path so the script works when invoked
# directly (e.g. `python evaluation/evaluate.py`) as well as via
# `python -m evaluation.evaluate`.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agent.q_learning import QLearningAgent
from baselines.fixed_controller import FixedController
from baselines.rule_controller import RuleController
from environment.datacenter_env import DataCenterEnv

# ── Configuration ────────────────────────────────────────────────────────
MODEL_PATH: Path = Path("models/q_table.pkl")
NUM_STEPS: int = 300
ENV_SEED: int = 42
WORKLOAD_MODE: str = "normal"
TEMP_PLOT_PATH: Path = Path("outputs/temp_comparison.png")
ENERGY_PLOT_PATH: Path = Path("outputs/energy_comparison.png")


def run_episode(
    controller,
    controller_name: str,
    is_q_agent: bool = False,
) -> dict[str, list[float]]:
    """Run a single episode and return temperature / energy traces.

    The environment is always created with the **same** seed so every
    controller faces identical workload sequences and noise.
    """
    env = DataCenterEnv(workload_mode=WORKLOAD_MODE, seed=ENV_SEED)
    state = env._get_discrete_state()

    temps: list[float] = [env.current_temp]
    cooling_levels: list[float] = [env.current_cooling]

    for _ in range(NUM_STEPS):
        if is_q_agent:
            action = controller.choose_action(state)
        else:
            action = controller.choose_action(state, env.current_cooling)

        state, reward, done = env.step(action)

        temps.append(env.current_temp)
        cooling_levels.append(env.current_cooling)

        if done:
            break

    return {"temps": temps, "cooling": cooling_levels}


def evaluate() -> dict[str, dict[str, list[float]]]:
    """Evaluate all three controllers and print a summary table."""

    # ── Load trained Q-learning agent (pure exploitation) ────────────
    q_agent = QLearningAgent(num_actions=DataCenterEnv.NUM_ACTIONS)
    q_agent.load(MODEL_PATH)
    q_agent.epsilon = 0.0  # greedy — no exploration

    fixed = FixedController(target_cooling=0.7)
    rule = RuleController()

    controllers = [
        ("Q-Learning", q_agent, True),
        ("Fixed (0.7)", fixed, False),
        ("Rule-Based", rule, False),
    ]

    results: dict[str, dict[str, list[float]]] = {}

    for name, ctrl, is_q in controllers:
        results[name] = run_episode(ctrl, name, is_q_agent=is_q)

    # ── Summary table ────────────────────────────────────────────────
    header = f"{'Controller':<16} {'Avg Temp (C)':>13} {'Max Temp (C)':>13} {'Min Temp (C)':>13} {'Total Energy':>13}"
    print()
    print("=" * len(header))
    print("  CoolRL Evaluation Summary")
    print(f"  {NUM_STEPS} steps  |  seed={ENV_SEED}  |  mode='{WORKLOAD_MODE}'")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for name, data in results.items():
        temps = data["temps"]
        cooling = data["cooling"]
        avg_temp = sum(temps) / len(temps)
        max_temp = max(temps)
        min_temp = min(temps)
        total_energy = sum(cooling)

        print(
            f"{name:<16} "
            f"{avg_temp:>13.2f} "
            f"{max_temp:>13.2f} "
            f"{min_temp:>13.2f} "
            f"{total_energy:>13.2f}"
        )

    print("=" * len(header))
    print()

    return results


# ── Plotting ─────────────────────────────────────────────────────────────

def plot_temperature_comparison(results: dict[str, dict[str, list[float]]]) -> None:
    """Line chart: Temperature vs Time for all controllers."""
    TEMP_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 6))

    colors = {"Q-Learning": "#2563eb", "Fixed (0.7)": "#f59e0b", "Rule-Based": "#10b981"}

    for name, data in results.items():
        ax.plot(
            data["temps"],
            label=name,
            linewidth=1.5,
            color=colors.get(name),
        )

    # Safe-zone band (22 C - 27 C)
    ax.axhspan(22, 27, alpha=0.12, color="green", label="Safe Zone (22-27 C)")
    ax.axhline(22, color="green", linewidth=1, linestyle="--", alpha=0.6)
    ax.axhline(27, color="green", linewidth=1, linestyle="--", alpha=0.6)

    # Critical line
    ax.axhline(30, color="red", linewidth=1, linestyle=":", alpha=0.7, label="Critical (30 C)")

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Temperature (C)")
    ax.set_title("CoolRL - Temperature Comparison Across Controllers")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(TEMP_PLOT_PATH, dpi=150)
    plt.close(fig)
    print(f"Temperature plot saved to {TEMP_PLOT_PATH}")


def plot_energy_comparison(results: dict[str, dict[str, list[float]]]) -> None:
    """Bar chart: Total Energy consumption for each controller."""
    ENERGY_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)

    names: list[str] = []
    energies: list[float] = []
    colors = ["#2563eb", "#f59e0b", "#10b981"]

    for name, data in results.items():
        names.append(name)
        energies.append(sum(data["cooling"]))

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, energies, color=colors[:len(names)], edgecolor="white", width=0.5)

    # Value labels on top of each bar
    for bar, energy in zip(bars, energies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{energy:.1f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    ax.set_ylabel("Total Cooling Energy (sum of cooling levels)")
    ax.set_title("CoolRL - Energy Consumption Comparison")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(ENERGY_PLOT_PATH, dpi=150)
    plt.close(fig)
    print(f"Energy plot saved to {ENERGY_PLOT_PATH}")


if __name__ == "__main__":
    results = evaluate()
    plot_temperature_comparison(results)
    plot_energy_comparison(results)
