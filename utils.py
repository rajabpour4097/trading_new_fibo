


class BotState:
    def __init__(self):
        self.reset()

    def reset(self):
        self.fib_levels = None
        self.true_position = False
        self.last_touched_705_point_up = None
        self.last_touched_705_point_down = None