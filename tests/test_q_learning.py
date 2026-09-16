"""Unit tests for Tabular QLearningAgent in CoolRL.

Validates:
1. Q-table dimensions, zero initialization, and finiteness.
2. Epsilon-greedy action selection and pure greedy evaluation behavior.
3. Exact Bellman TD updates (non-terminal bootstrapping and terminal non-bootstrapping).
4. Epsilon exponential decay and minimum floor enforcement.
5. Deterministic policy extraction.
6. Model persistence (save / load roundtrip, Q-table equality, parameter equality).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np
import pytest

from src.agents.q_learning import QLearningAgent


class TestQTableInitializationAndSpecs:
    """Verify Q-table dimensions, memory allocation, and default parameters."""

    def test_default_initialization(self):
        """Agent initializes with 960 states, 5 actions, and all zero Q-values."""
        agent = QLearningAgent()
        assert agent.n_states == 960
        assert agent.n_actions == 5
        assert agent.alpha == 0.10
        assert agent.gamma == 0.95
        assert agent.epsilon == 1.00
        assert agent.epsilon_min == 0.05
        assert agent.epsilon_decay == 0.995

        assert agent.q_table.shape == (960, 5)
        assert agent.q_table.dtype == np.float64
        assert np.all(agent.q_table == 0.0)
        assert np.all(np.isfinite(agent.q_table))

    def test_custom_parameters_and_bounds_validation(self):
        """Constructor validates parameter ranges and raises ValueError on invalid inputs."""
        agent = QLearningAgent(n_states=100, n_actions=3, alpha=0.2, gamma=0.9, epsilon=0.5)
        assert agent.n_states == 100
        assert agent.n_actions == 3
        assert agent.q_table.shape == (100, 3)

        with pytest.raises(ValueError):
            QLearningAgent(alpha=-0.1)
        with pytest.raises(ValueError):
            QLearningAgent(alpha=1.5)
        with pytest.raises(ValueError):
            QLearningAgent(gamma=1.1)
        with pytest.raises(ValueError):
            QLearningAgent(epsilon=1.5)
        with pytest.raises(ValueError):
            QLearningAgent(epsilon_decay=0.0)


class TestActionSelection:
    """Verify exploration vs exploitation action selection mechanisms."""

    def test_action_bounds_and_validity(self):
        """choose_action must always return an action ID in [0, n_actions - 1]."""
        agent = QLearningAgent(seed=42)
        for state in [0, 42, 500, 959]:
            act_train = agent.choose_action(state, training=True)
            act_eval = agent.choose_action(state, training=False)
            assert 0 <= act_train < 5
            assert 0 <= act_eval < 5

    def test_evaluation_mode_is_purely_greedy(self):
        """When training=False, action must be strictly argmax(Q[state]), regardless of epsilon."""
        agent = QLearningAgent(epsilon=1.00)  # High exploration parameter
        # Seed state 100 with known Q-values
        agent.q_table[100] = [1.0, 5.0, 3.0, 2.0, 0.5]

        for _ in range(20):
            chosen = agent.choose_action(100, training=False)
            assert chosen == 1  # Action 1 has max value 5.0

    def test_deterministic_tie_breaking(self):
        """When multiple actions tie for maximum, choose lowest index deterministically."""
        agent = QLearningAgent()
        agent.q_table[50] = [4.0, 4.0, 2.0, 4.0, 1.0]  # Actions 0, 1, 3 tied

        assert agent.choose_action(50, training=False) == 0

    def test_invalid_state_id_rejection(self):
        """Invalid state IDs must be rejected with clear exceptions."""
        agent = QLearningAgent()
        with pytest.raises(ValueError):
            agent.choose_action(-1)
        with pytest.raises(ValueError):
            agent.choose_action(960)
        with pytest.raises(TypeError):
            agent.choose_action("0")  # type: ignore


class TestBellmanUpdates:
    """Verify manual calculation of non-terminal and terminal temporal difference updates."""

    def test_exact_non_terminal_update(self):
        """Verify non-terminal update equation: Q(s,a) <- Q(s,a) + alpha * (r + gamma * max(Q(s')) - Q(s,a))."""
        agent = QLearningAgent(alpha=0.10, gamma=0.95)

        state = 10
        action = 3
        reward = -2.0
        next_state = 25

        # Set known values
        agent.q_table[state, action] = 1.00
        agent.q_table[next_state] = [0.0, 4.0, 8.0, 2.0, 1.0]  # max is 8.0

        # Target = -2.0 + 0.95 * 8.0 = -2.0 + 7.6 = 5.6
        # TD Error = 5.6 - 1.0 = 4.6
        # New Q = 1.0 + 0.10 * 4.6 = 1.46
        td_error = agent.update(state, action, reward, next_state, terminated=False)

        assert np.isclose(td_error, 4.6)
        assert np.isclose(agent.q_table[state, action], 1.46)

    def test_exact_terminal_update(self):
        """Verify terminal update equation: Q(s,a) <- Q(s,a) + alpha * (r - Q(s,a)) without bootstrapping."""
        agent = QLearningAgent(alpha=0.20, gamma=0.95)

        state = 15
        action = 1
        reward = -10.0
        next_state = 99

        agent.q_table[state, action] = 0.00
        agent.q_table[next_state] = [100.0, 200.0, 300.0, 400.0, 500.0]  # Must be ignored!

        # Target = -10.0
        # TD Error = -10.0 - 0.0 = -10.0
        # New Q = 0.0 + 0.20 * (-10.0) = -2.0
        td_error = agent.update(state, action, reward, next_state, terminated=True)

        assert np.isclose(td_error, -10.0)
        assert np.isclose(agent.q_table[state, action], -2.0)

    def test_repeated_updates_converge_to_target(self):
        """Repeated updates for a fixed transition converge asymptotically to the Bellman target."""
        agent = QLearningAgent(alpha=0.10, gamma=0.90)
        target_val = -5.0

        for _ in range(200):
            agent.update(state_id=1, action=2, reward=target_val, next_state_id=2, terminated=True)

        assert np.isclose(agent.q_table[1, 2], target_val, atol=1e-4)


class TestEpsilonDecay:
    """Verify episode-level exponential decay and minimum exploration floor."""

    def test_epsilon_decay_and_floor(self):
        """Epsilon decays multiplicatively and clamps at epsilon_min."""
        agent = QLearningAgent(epsilon=1.00, epsilon_min=0.05, epsilon_decay=0.50)

        assert agent.decay_epsilon() == 0.50
        assert agent.decay_epsilon() == 0.25
        assert agent.decay_epsilon() == 0.125
        assert agent.decay_epsilon() == 0.0625
        # Fifth decay: 0.0625 * 0.5 = 0.03125 -> clamped to floor 0.05
        assert agent.decay_epsilon() == 0.05
        # Subsequent decays stay at floor
        assert agent.decay_epsilon() == 0.05


class TestPolicyExtraction:
    """Verify policy extraction from learned Q-table."""

    def test_greedy_policy_extraction(self):
        """get_policy returns an array of shape (960,) containing argmax actions."""
        agent = QLearningAgent(n_states=960, n_actions=5)
        # Assign deterministic maximums to arbitrary states
        agent.q_table[0, 2] = 10.0
        agent.q_table[1, 4] = 15.0
        agent.q_table[959, 1] = 8.0

        policy = agent.get_policy()
        assert policy.shape == (960,)
        assert policy.dtype == np.int64
        assert policy[0] == 2
        assert policy[1] == 4
        assert policy[959] == 1


class TestModelPersistence:
    """Verify save and load roundtrip fidelity."""

    def test_save_and_load_roundtrip(self):
        """Saved agent restored from .npz must have identical Q-values, params, and policy."""
        agent = QLearningAgent(
            n_states=960,
            n_actions=5,
            alpha=0.15,
            gamma=0.92,
            epsilon=0.45,
            epsilon_min=0.08,
            epsilon_decay=0.990,
            seed=1234,
        )
        # Populate some Q-table values
        agent.q_table[10, 2] = -4.5
        agent.q_table[500, 4] = +12.3
        agent.q_table[950, 0] = -0.05

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_agent.npz"
            saved_path = agent.save(filepath)
            assert saved_path.exists()

            loaded_agent = QLearningAgent.load(saved_path, seed=1234)

            assert loaded_agent.n_states == agent.n_states
            assert loaded_agent.n_actions == agent.n_actions
            assert loaded_agent.alpha == agent.alpha
            assert loaded_agent.gamma == agent.gamma
            assert loaded_agent.epsilon == agent.epsilon
            assert loaded_agent.epsilon_min == agent.epsilon_min
            assert loaded_agent.epsilon_decay == agent.epsilon_decay

            np.testing.assert_array_equal(loaded_agent.q_table, agent.q_table)
            np.testing.assert_array_equal(loaded_agent.get_policy(), agent.get_policy())
