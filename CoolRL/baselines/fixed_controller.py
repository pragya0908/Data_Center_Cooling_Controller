class FixedController:
    def __init__(self, fixed_level=70.0):
        """
        A baseline controller that always maintains a fixed cooling level.
        The action space expects index-based actions:
        0: -20%, 1: -10%, 2: 0%, 3: +10%, 4: +20%
        """
        self.fixed_level = fixed_level

    def choose_action(self, state, current_cooling):
        """
        Returns the action index required to stay as close as possible to the fixed level.
        """
        diff = self.fixed_level - current_cooling
        
        # Closest matching action
        if diff <= -20:
            return 0
        elif diff <= -10:
            return 1
        elif diff >= 20:
            return 4
        elif diff >= 10:
            return 3
        else:
            return 2  # maintain
