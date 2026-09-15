import numpy as np

class QLearningAgent:
    def __init__(self, action_space, alpha=0.1, gamma=0.95, epsilon=1.0, min_epsilon=0.05, epsilon_decay=0.995):
        self.action_space = action_space
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        
        # Q-table: dictionary mapping state tuples to action-value arrays
        self.q_table = {}

    def _discretize_state(self, state):
        """
        Converts continuous state variables into discrete bins for the Q-table.
        State = (temperature, workload, ambient_temperature, cooling_level)
        """
        temp, workload, ambient, cooling = state
        
        # Temperature binning
        if temp < 22:
            t_bin = 0
        elif 22 <= temp < 24:
            t_bin = 1
        elif 24 <= temp < 26:
            t_bin = 2
        elif 26 <= temp < 28:
            t_bin = 3
        elif 28 <= temp < 30:
            t_bin = 4
        else:
            t_bin = 5
            
        # Workload binning
        if workload < 20:
            w_bin = 0
        elif 20 <= workload < 40:
            w_bin = 1
        elif 40 <= workload < 60:
            w_bin = 2
        elif 60 <= workload < 80:
            w_bin = 3
        else:
            w_bin = 4
            
        # Ambient temperature binning
        if ambient < 20:
            a_bin = 0
        elif 20 <= ambient < 25:
            a_bin = 1
        elif 25 <= ambient < 30:
            a_bin = 2
        else:
            a_bin = 3
            
        # Cooling level binning
        if cooling < 20:
            c_bin = 0
        elif 20 <= cooling < 40:
            c_bin = 1
        elif 40 <= cooling < 60:
            c_bin = 2
        elif 60 <= cooling < 80:
            c_bin = 3
        else:
            c_bin = 4
            
        return (t_bin, w_bin, a_bin, c_bin)

    def _get_q_values(self, state_discrete):
        """Returns the Q-values for a given state. Initializes if not present."""
        if state_discrete not in self.q_table:
            self.q_table[state_discrete] = np.zeros(len(self.action_space))
        return self.q_table[state_discrete]

    def choose_action(self, state, evaluate=False):
        """
        Epsilon-greedy action selection.
        If evaluate=True, it will always pick the best action (epsilon=0).
        """
        state_discrete = self.discretize_state(state)
        
        if not evaluate and np.random.rand() < self.epsilon:
            return np.random.choice(self.action_space)
            
        q_values = self._get_q_values(state_discrete)
        
        # Handle ties randomly instead of always picking the first one
        best_actions = np.argwhere(q_values == np.amax(q_values)).flatten()
        return np.random.choice(best_actions)

    def discretize_state(self, state):
        return self._discretize_state(state)

    def learn(self, state, action, reward, next_state):
        """
        Updates the Q-table using the Bellman equation.
        """
        state_discrete = self._discretize_state(state)
        next_state_discrete = self._discretize_state(next_state)
        
        # Current Q-value
        current_q = self._get_q_values(state_discrete)[action]
        
        # Max Q-value for next state
        next_max_q = np.max(self._get_q_values(next_state_discrete))
        
        # Q-learning update formula
        new_q = current_q + self.alpha * (reward + self.gamma * next_max_q - current_q)
        
        self.q_table[state_discrete][action] = new_q
        
    def decay_epsilon(self):
        """Decays exploration rate."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
