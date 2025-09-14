


class BotState:
    def __init__(self):
        self.reset()

    def reset(self):
        self.fib_levels = None
        self.true_position = False
        self.last_touched_705_point_up = None
        self.last_touched_705_point_down = None
        # New tracking fields for richer logging/context
        self.fib_built_time = None           # timestamp when fib was (re)built
        self.fib0_last_update_time = None    # timestamp when 0.0 was last updated (extend)
        self.fib1_time = None                # timestamp of leg1 end (anchor for 1.0)
        self.fib1_price = None               # price of 1.0 at build
        self.last_second_touch_705_point_up = None    # row of second 0.705 touch in bullish
        self.last_second_touch_705_point_down = None  # row of second 0.705 touch in bearish