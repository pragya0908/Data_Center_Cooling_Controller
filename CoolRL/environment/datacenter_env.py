import numpy as np
from .workload_generator import WorkloadGenerator

class DataCenterEnv:
    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            
        self.workload_gen = WorkloadGenerator(seed=seed)
        
        # State variables
        self.temperature = 24.0
        self.workload = 0.0
        self.ambient_temp = 25.0
        self.cooling_level = 50.0  # 0 to 100
        
        # Environment limits
        self.max_steps = 240  # Represents one full day (24 hours * 10 steps/hr)
        self.current_step = 0
        
        # Actions
        # 0: Decrease 20%, 1: Decrease 10%, 2: Maintain, 3: Increase 10%, 4: Increase 20%
        self.action_space = [0, 1, 2, 3, 4]
        
    def reset(self):
        """Resets the environment to initial conditions."""
        self.current_step = 0
        self.workload_gen.reset()
        
        self.temperature = 24.0
        self.workload = self.workload_gen.get_workload()
        self.ambient_temp = np.random.uniform(22, 28)
        self.cooling_level = 50.0
        
        return self._get_state()
        
    def _get_state(self):
        """Returns the current state."""
        return (
            round(self.temperature, 2),
            round(self.workload, 2),
            round(self.ambient_temp, 2),
            round(self.cooling_level, 2)
        )
        
    def step(self, action):
        """
        Executes one time step in the environment.
        """
        # 1. Apply action to update cooling level
        cooling_changes = {0: -20, 1: -10, 2: 0, 3: 10, 4: 20}
        change = cooling_changes.get(action, 0)
        self.cooling_level += change
        self.cooling_level = np.clip(self.cooling_level, 0, 100)
        
        # 2. Update workload and ambient temp
        self.workload = self.workload_gen.get_workload()
        # Ambient temp fluctuates slowly
        self.ambient_temp += np.random.uniform(-0.5, 0.5)
        self.ambient_temp = np.clip(self.ambient_temp, 20, 35)
        
        # 3. Thermal model: Calculate new temperature
        # Base formula: Temp_new = Temp_old + Heat(workload) + Ambient_effect - Cooling_effect + Noise
        heat_from_workload = (self.workload / 100.0) * 3.0       # Max +3.0 C
        ambient_effect = (self.ambient_temp - 25.0) * 0.1        # Influence of outside temp
        cooling_effect = (self.cooling_level / 100.0) * 4.0      # Max -4.0 C
        noise = np.random.uniform(-0.2, 0.2)
        
        # We assume baseline heat retention if cooling is off
        base_warming = 1.0
        
        self.temperature = (self.temperature 
                            + base_warming 
                            + heat_from_workload 
                            + ambient_effect 
                            - cooling_effect 
                            + noise)
        
        # Hard limits on temperature simulation (physics constraints)
        self.temperature = np.clip(self.temperature, 15, 45)
        
        # 4. Calculate energy consumed
        # Base energy + energy proportional to cooling level
        energy_consumed = 5.0 + (self.cooling_level / 100.0) * 10.0
        
        # 5. Calculate reward
        reward = self._calculate_reward(energy_consumed)
        
        # 6. Check termination
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        # Info dictionary for metrics
        info = {
            'energy': round(energy_consumed, 2),
            'overheating': self.temperature > 30.0,
            'safe_range': 22.0 <= self.temperature <= 27.0
        }
        
        return self._get_state(), reward, done, info
        
    def _calculate_reward(self, energy_consumed):
        """Calculates the reward based on thermal safety and energy cost."""
        reward = 0
        
        # Temperature component
        if 22.0 <= self.temperature <= 27.0:
            reward += 10  # Safe range
        elif 27.0 < self.temperature <= 30.0:
            reward += 3   # Slightly warm
        elif self.temperature < 22.0:
            reward -= 3   # Too cold (wasting energy)
        else:
            reward -= 20  # Critical / Overheating
            
        # Energy penalty component (encourages less cooling when safe)
        # We subtract a fraction of the energy consumed
        energy_penalty = energy_consumed * 0.2
        reward -= energy_penalty
        
        return round(reward, 2)
