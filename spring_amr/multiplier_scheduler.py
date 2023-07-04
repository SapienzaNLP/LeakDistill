class MultiplierScheduler:
    def __call__(self, with_step=True):
        ret_value = self.value

        if with_step:
            self.step()
        return ret_value

    def __init__(self, init_value, final_value=None, scheduler_params=None): #, scheduler_direction, scheduler_steps=None):
        self.value = self.init_value = init_value
        self.final_value = init_value if final_value is None else final_value

        # Initial step function
        self.step = self.general_step
        self.step_index = 0
        self.total_steps = 0

        if scheduler_params is None:
            return

        if 'type' not in scheduler_params:
            raise 'Scheduler type is mandatory'

        if scheduler_params['type'] == 'constant':
            return

        if 'steps' not in scheduler_params:
            raise 'Number of steps is mandatory'

        self.total_steps = scheduler_params['steps']

        if scheduler_params['type'] == 'linear':
            self.increment = (self.final_value - self.init_value) / self.total_steps
            self.step = self.linear_scheduler
        elif scheduler_params['type'] == 'step_func':
            self.step = self.step_func_scheduler

    def general_step(self):
        self.step_index += 1

    def linear_scheduler(self):
        if self.step_index < self.total_steps:
            self.value += self.increment
            self.general_step()

    def step_func_scheduler(self):
        if self.step_index < self.total_steps:
            self.general_step()
        else:
            self.value = self.final_value


if __name__ == '__main__':
    alpha = MultiplierScheduler(0, 20, scheduler_params = {'type': 'step_func', 'steps': 100})
    for i in range(110):
        print(alpha())
