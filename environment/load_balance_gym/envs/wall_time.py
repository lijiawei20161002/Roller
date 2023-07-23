class WallTime(object):
    def __init__(self):
        self.curr_time = 0.0

    def update(self, new_time):
        self.curr_time = new_time

    def reset(self):
        self.curr_time = 0.0