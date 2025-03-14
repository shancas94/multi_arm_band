import numpy as np
class UpperConfidenceBound:
    def __init__(self, True_mean_values, total_number_of_steps):
        self.n_arms = np.size(True_mean_values)
        self.true_mean_values = True_mean_values
        self.total_number_of_steps = total_number_of_steps
        self.no_of_times_arm_selected = np.zeros(self.n_arms)
        self.mean_reward_per_arm = np.zeros(self.n_arms)
        self.current_step = 0
        self.mean_reward = np.zeros(total_number_of_steps + 1)
        self.current_reward = 0

    def select_arm(self):
        if self.current_step < self.n_arms:
            arm_selected = self.current_step
        else:
            upper_bound_values = self.mean_reward_per_arm + np.sqrt(2 * np.log(self.current_step) / self.no_of_times_arm_selected)
            arm_selected = np.argmax(upper_bound_values)
        
        self.current_step += 1
        self.no_of_times_arm_selected[arm_selected] += 1
        self.current_reward = np.random.normal(self.true_mean_values[arm_selected], 2)
        self.mean_reward[self.current_step] = self.mean_reward[self.current_step - 1] + \
            ((1 / self.current_step) * (self.current_reward - self.mean_reward[self.current_step - 1]))
        self.mean_reward_per_arm[arm_selected] = self.mean_reward_per_arm[arm_selected] + \
            ((1 / self.no_of_times_arm_selected[arm_selected]) * (self.current_reward - self.mean_reward_per_arm[arm_selected]))

    def play_the_game(self):
        for i in range(self.total_number_of_steps):
            self.select_arm()

    def restart(self):
        self.current_step = 0
        self.no_of_times_arm_selected = np.zeros(self.n_arms)
        self.mean_reward_per_arm = np.zeros(self.n_arms)
        self.mean_reward = np.zeros(self.total_number_of_steps + 1)
        self.current_reward = 0