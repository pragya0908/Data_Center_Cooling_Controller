import numpy as np

class WorkloadGenerator:
    def __init__(self, seed=None):
        """
        Generates realistic workload patterns for the data center.
        """
        if seed is not None:
            np.random.seed(seed)
            
        self.time_step = 0
        
        # Define a daily profile (e.g., 24 hours mapped to simulation steps)
        # We can map each step to an hour or a fraction of an hour.
        # Let's say one episode is 240 steps (10 steps per hour).
        self.base_workloads = {
            'night': (15, 30),
            'morning': (30, 50),
            'working_hours': (60, 85),
            'peak': (85, 100)
        }
        
    def _get_time_of_day_category(self):
        # 240 steps = 24 hours, so 10 steps = 1 hour
        hour = (self.time_step // 10) % 24
        
        if 0 <= hour < 7:
            return 'night'
        elif 7 <= hour < 10:
            return 'morning'
        elif 10 <= hour < 18:
            if hour == 14 or hour == 15:
                return 'peak'
            return 'working_hours'
        elif 18 <= hour < 22:
            return 'working_hours'
        else:
            return 'night'

    def get_workload(self):
        """
        Returns the workload for the current time step (0 to 100).
        """
        category = self._get_time_of_day_category()
        min_w, max_w = self.base_workloads[category]
        
        # Base workload with some random variation
        workload = np.random.uniform(min_w, max_w)
        
        # 5% chance of a sudden spike
        if np.random.rand() < 0.05:
            workload = np.random.uniform(90, 100)
            
        # 5% chance of a sudden drop
        if np.random.rand() < 0.05:
            workload = np.random.uniform(10, 20)
            
        # Ensure it stays within bounds
        workload = np.clip(workload, 0, 100)
        
        self.time_step += 1
        return round(workload, 2)
    
    def reset(self):
        """Resets the workload generator to time step 0."""
        self.time_step = 0
        return self.get_workload()
