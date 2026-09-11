class FixedTimeAgent:
    """
    Traditional pre-timed traffic light controller.
    Switches phase periodically after a fixed cycle duration.
    """
    def __init__(self, switch_interval=3):
        # Har 3 control steps (3 x 10s = 30s) par phase switch hoga
        self.switch_interval = switch_interval
        self.current_step = 0
        self.current_phase = 0

    def predict(self, observation, deterministic=True):
        self.current_step += 1
        if self.current_step >= self.switch_interval:
            self.current_phase = 1 - self.current_phase
            self.current_step = 0
        return self.current_phase, None

    def reset(self):
        self.current_step = 0
        self.current_phase = 0