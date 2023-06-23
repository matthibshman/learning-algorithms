from functools import reduce
import math
import sys
import numpy as np
import random


class Bandit:
    def __init__(self, update, value=0.0, initial_estimate=0.0, step_size=0.1):
        self.value = value
        self.current_estimate = initial_estimate
        self.times_pulled = 0.0
        self.step_size = step_size
        self.update = update

    def reset(self, value=0.0, initial_estimate=0.0, step_size=0.1):
        self.value = value
        self.current_estimate = initial_estimate
        self.times_pulled = 0.0
        self.step_size = step_size

    def update_value(self, noise):
        self.value += noise

    def pull(self):
        self.times_pulled += 1.0
        reward = np.random.normal(self.value, 1.0)
        self.update_estimate(reward)
        return reward

    def update_estimate(self, reward):
        self.current_estimate += self.update(self, reward)


def pickBandit(bandits, epislon):
    # optimal_bandit = reduce(
    #     lambda bandit1, bandit2: bandit1 if bandit1.value > bandit2.value else bandit2, bandits,
    # )
    optimal_bandit = bandits[0]
    for bandit in bandits:
        optimal_bandit = optimal_bandit if optimal_bandit.value > bandit.value else bandit 

    if np.random.uniform() < epislon:
        curr_bandit = random.choice(bandits)
    else:
        # bandit = reduce(
        #     lambda bandit1, bandit2: bandit1
        #     if bandit1.current_estimate > bandit2.current_estimate
        #     else bandit2,
        #     bandits,
        # )
        curr_bandit = bandits[0]
        for bandit in bandits:
            curr_bandit = curr_bandit if curr_bandit.current_estimate > bandit.current_estimate else bandit 

    return (curr_bandit, optimal_bandit)


def update_estimate_sample_average(bandit: Bandit, reward) -> float:
    return (reward - bandit.current_estimate) / bandit.times_pulled


def update_estimate_step_size(bandit: Bandit, reward) -> float:
    return bandit.step_size * (reward - bandit.current_estimate)


def play(num_runs=300, num_iterations=10000, num_bandits=10):
    results = np.zeros((4, num_iterations))
    sample_average_bandits = [Bandit(update_estimate_sample_average) for _ in range(num_bandits)]
    step_size_bandits = [Bandit(update_estimate_step_size) for _ in range(num_bandits)]

    for _ in range(num_runs):
        for iteration in range(num_iterations):
            reward, optimal_action = perform_step(sample_average_bandits)
            results[0][iteration] += reward
            results[1][iteration] += optimal_action

            reward, optimal_action = perform_step(step_size_bandits)
            results[2][iteration] += reward
            results[3][iteration] += optimal_action
        for bandit in sample_average_bandits:
            bandit.reset()
        for bandit in step_size_bandits:
            bandit.reset()

    return results / num_runs


def perform_step(bandits, noise_std_dev=0.01, epsilon=0.1):
    for bandit in bandits:
        bandit.update_value(np.random.normal(0.0, noise_std_dev, 1))

    bandit, optimal_bandit = pickBandit(bandits, epsilon)
    reward = bandit.pull()

    return (reward, 1.0 if math.isclose(bandit.value, optimal_bandit.value) else 0.0)


if __name__ == "__main__":
    file_name = sys.argv[1]

    results = play()
    np.savetxt(file_name, results)
