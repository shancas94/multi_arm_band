import numpy as np
import matplotlib.pyplot as plt
from Multi_Arm_Bandit import MultiArmBanditProblem
np.random.seed(42)  # Ensuring reproducibility
epsilon1 = 0
epsilon2 = 0.1
epsilon3 = 0.2
epsilon4 = 0.3
total_number_of_steps = 2000
test_cases = {
    "5 Arms": np.random.normal(0, 1, 5),
    "10 Arms": np.random.normal(0, 1, 10),
    "20 Arms": np.random.normal(0, 1, 20),
}

for case, true_mean_values in test_cases.items():
    print(f"\nTesting {case} with epsilon={epsilon1} and {total_number_of_steps} plays")

    bandit = MultiArmBanditProblem(true_mean_values, epsilon1, total_number_of_steps)
    bandit.play_the_game()
    epsilon1MeanReward = bandit.mean_reward

    bandit = MultiArmBanditProblem(true_mean_values, epsilon2, total_number_of_steps)
    bandit.play_the_game()
    epsilon2MeanReward = bandit.mean_reward

    bandit = MultiArmBanditProblem(true_mean_values, epsilon3, total_number_of_steps)
    bandit.play_the_game()
    epsilon3MeanReward = bandit.mean_reward

    bandit = MultiArmBanditProblem(true_mean_values, epsilon4, total_number_of_steps)
    bandit.play_the_game()
    epsilon4MeanReward = bandit.mean_reward
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

for idx, (case, true_mean_values) in enumerate(test_cases.items()):
    ax = axes[idx]
    for epsilon in [epsilon1, epsilon2, epsilon3, epsilon4]:
        bandit = MultiArmBanditProblem(true_mean_values, epsilon, total_number_of_steps)
        bandit.play_the_game()
        mean_reward = bandit.mean_reward
        ax.plot(np.arange(total_number_of_steps + 1), mean_reward, linewidth=2, label=f'epsilon={epsilon}')
    
    ax.set_xscale("log")
    ax.set_xlabel('Steps')
    ax.set_ylabel('Average reward')
    ax.set_title(f'Rewards for {case}')
    ax.legend()

plt.tight_layout()
plt.savefig('results_combined.png', dpi=300)
plt.show()

best_epsilons = {}
most_selected_arms = {}

for case, true_mean_values in test_cases.items():
    best_mean_reward = -np.inf
    best_epsilon = None
    best_bandit = None
    for epsilon in [epsilon1, epsilon2, epsilon3, epsilon4]:
        bandit = MultiArmBanditProblem(true_mean_values, epsilon, total_number_of_steps)
        bandit.play_the_game()
        mean_reward = bandit.mean_reward[-1]  # Get the final mean reward
        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            best_epsilon = epsilon
            best_bandit = bandit
    best_epsilons[case] = best_epsilon
    most_selected_arms[case] = np.argmax(best_bandit.no_of_times_arm_selected)

# Plotting the best epsilon values for each case
fig, ax = plt.subplots(figsize=(10, 6))
cases = list(best_epsilons.keys())
epsilons = list(best_epsilons.values())
ax.bar(cases, epsilons, color='skyblue')
ax.set_xlabel('Test Cases')
ax.set_ylabel('Best Epsilon')
ax.set_title('Best Epsilon Value for Each Test Case')
plt.tight_layout()
plt.savefig('best_epsilons.png', dpi=300)
plt.show()

# Plotting the most selected arms for each case
fig, ax = plt.subplots(figsize=(10, 6))
arms = list(most_selected_arms.values())
ax.bar(cases, arms, color='lightgreen')
ax.set_xlabel('Test Cases')
ax.set_ylabel('Most Selected Arm')
ax.set_title('Most Selected Arm for Each Test Case')
plt.tight_layout()
plt.savefig('most_selected_arms.png', dpi=300)
plt.show()

