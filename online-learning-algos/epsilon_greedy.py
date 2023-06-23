import numpy as np
from numpy import random


class EpsilonGreedy:
    def __init__(self, avg, num_iter, C=1):
        self.means = avg
        self.optimal = np.argmax(avg)
        self.num_arms = avg.size
        self.best_arm = np.argmax(avg)
        self.C = C
        sort = np.sort(avg)[::-1]
        self.delta = sort[0] - sort[1]

        self.num_iter = num_iter
        self.time = 0.0
        self.eps = 1
        self.emp_means = np.zeros(self.num_arms)
        self.num_pulls = np.zeros(self.num_arms)
        self.cum_reg = np.zeros(self.num_iter)

    def get_best_arm(self):
        return np.argmax(self.emp_means)

    def update_stats(self, rew, arm):
        self.emp_means[arm] = (self.emp_means[arm] * self.num_pulls[arm] + rew) / (
            self.num_pulls[arm] + 1.0
        )
        self.num_pulls[arm] += 1.0
        self.time += 1.0
        self.eps = min(1, (self.C * self.num_arms) / (self.time * self.delta**2))

        return None

    def update_regret(self, rew_vec, arm):
        self.cum_reg[int(self.time)] = self.cum_reg[int(self.time - 1)] + (
            rew_vec[self.optimal] - rew_vec[arm]
        )

        return None

    def get_arm(self):
        arm = self.time % self.num_arms

        if self.time >= self.num_arms:
            arm = (
                random.randint(self.num_arms)
                if random.uniform() <= self.eps
                else self.get_best_arm()
            )

        return arm

    def iterate(self, rew_vec):
        arm = self.get_arm()
        arm = int(arm)

        self.update_regret(rew_vec, arm)
        self.update_stats(rew_vec[arm], arm)

        return None
