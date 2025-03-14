import numpy as np
class ThompsonSampling:
    def __init__(self, True_mean_values, total_number_of_steps):
        self.n_arms = np.size(True_mean_values)
        self.true_mean_values = True_mean_values
        self.total_number_of_steps = total_number_of_steps
        self.successes = np.zeros(self.n_arms)
        self.failures = np.zeros(self.n_arms)
        self.current_step = 0
        self.mean_reward = np.zeros(total_number_of_steps + 1)
        self.current_reward = 0

    def select_arm(self):
        beta_samples = [np.random.beta(self.successes[i] + 1, self.failures[i] + 1) for i in range(self.n_arms)]
        arm_selected = np.argmax(beta_samples)
        
        self.current_step += 1
        reward = np.random.normal(self.true_mean_values[arm_selected], 2)
        self.current_reward = reward
        self.mean_reward[self.current_step] = self.mean_reward[self.current_step - 1] + \
            ((1 / self.current_step) * (self.current_reward - self.mean_reward[self.current_step - 1]))
        
        if reward > 0:
            self.successes[arm_selected] += 1
        else:
            self.failures[arm_selected] += 1

    def play_the_game(self):
        for i in range(self.total_number_of_steps):
            self.select_arm()

    def restart(self):
        self.current_step = 0
        self.successes = np.zeros(self.n_arms)
        self.failures = np.zeros(self.n_arms)
        self.mean_reward = np.zeros(self.total_number_of_steps + 1)
        self.current_reward = 0