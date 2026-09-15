class RuleBasedController:
    def __init__(self):
        """
        A baseline controller that uses simple IF-THEN rules based on temperature.
        Action space:
        0: -20%, 1: -10%, 2: 0%, 3: +10%, 4: +20%
        """
        pass
        
    def choose_action(self, state):
        """
        State = (temperature, workload, ambient, cooling_level)
        """
        temp = state[0]
        
        # Simple rule logic based on temperature
        if temp > 29.0:
            return 4  # Increase cooling by 20%
        elif temp > 27.0:
            return 3  # Increase cooling by 10%
        elif temp < 22.0:
            return 0  # Decrease cooling by 20%
        elif temp < 24.0:
            return 1  # Decrease cooling by 10%
        else:
            return 2  # Maintain cooling
