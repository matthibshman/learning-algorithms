from functools import reduce
import math
import sys
import numpy as np


class Bandit:
    def __init__(self, update, num_bandits=10, value=0.0, initial_estimate=0.0, step_size=0.1):
        self.value = np.zeros(10)
        self.current_estimate = np.zeros(10)
        self.times_pulled = np.zeros(10)
        self.step_size = step_size
        self.update = update
        self.num_bandits = num_bandits

    def reset(self, value=0.0, initial_estimate=0.0, step_size=0.1):
        self.value = np.zeros(10)
        self.current_estimate = np.zeros(10)
        self.times_pulled = np.zeros(10)
        self.step_size = step_size

    def pull(self, arm):
        self.times_pulled[arm] += 1.0
        reward = np.random.normal(self.value[arm], 1.0)
        self.current_estimate[arm] += self.update(self, arm, reward)
        return reward

    def perform_step(self, noise_std_dev=0.01, epsilon=0.1):
        self.value += np.random.normal(0.0, noise_std_dev, 10)

        arm, optimal_arm = self.pickBandit(epsilon)
        reward = self.pull(arm)

        return (reward, 1.0 if math.isclose(self.value[arm], self.value[optimal_arm]) else 0.0)

    def pickBandit(self, epsilon):
        optimal_arm = np.argmax(self.value)
        curr_arm = (
            np.random.choice(self.num_bandits)
            if np.random.uniform() < epsilon
            else np.argmax(self.current_estimate)
        )
        return (curr_arm, optimal_arm)


def update_estimate_sample_average(bandit: Bandit, arm, reward):
    return (reward - bandit.current_estimate[arm]) / bandit.times_pulled[arm]


def update_estimate_step_size(bandit: Bandit, arm, reward):
    return bandit.step_size * (reward - bandit.current_estimate[arm])


def play(num_runs=300, num_iterations=10000, num_bandits=10):
    results = np.zeros((4, num_iterations))
    sample_average_bandit = Bandit(update_estimate_sample_average, num_bandits)
    step_size_bandit = Bandit(update_estimate_step_size, num_bandits)

    for _ in range(num_runs):
        for iteration in range(num_iterations):
            reward, optimal_action = sample_average_bandit.perform_step()
            results[0][iteration] += reward
            results[1][iteration] += optimal_action

            reward, optimal_action = step_size_bandit.perform_step()
            results[2][iteration] += reward
            results[3][iteration] += optimal_action
        sample_average_bandit.reset()
        step_size_bandit.reset()

    return results / num_runs


if __name__ == "__main__":
    file_name = sys.argv[1]

    results = play()
    np.savetxt(file_name, results)
