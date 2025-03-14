
import numpy as np
import matplotlib.pyplot as plt
# Multi arm bandit problem solution using epsilon greedy algorithm
class MultiArmBanditProblem:
    def __init__(self, True_mean_values , epsilon , total_number_of_steps ):
        self.n_arms = np.size(True_mean_values)
        self.true_mean_values = True_mean_values
        self.epsilon = epsilon
        self.current_step = 0
        self.total_number_of_steps = total_number_of_steps
        self.no_of_times_arm_selected=np.zeros(self.n_arms)
        self.mean_reward_per_arm = np.zeros(self.n_arms)
        self.current_reward = 0
        self.mean_reward =np.zeros(total_number_of_steps+1)
    
    def select_arm(self):
        probability_of_drawing = np.random.rand()
        # If it's the first step or if the random number is less than or equal to epsilon, explore
        if self.current_step == 0 or probability_of_drawing <= self.epsilon:
            arm_selected = np.random.choice(self.n_arms)
        else:  # Otherwise, exploit
            arm_selected = np.argmax(self.mean_reward_per_arm)
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

            