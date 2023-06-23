import numpy as np


class ExploreThenCommit:
    def __init__(self, avg, m, num_iter):

        self.means = avg
        self.optimal = np.argmax(avg)
        self.m = m
        self.num_arms = avg.size
        self.best_arm = np.argmax(avg)
        self.num_iter = num_iter
        self.time = 0.0
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

        return None

    def update_regret(self, rew_vec, arm):
        self.cum_reg[int(self.time)] = self.cum_reg[int(self.time - 1)] + (
            rew_vec[self.optimal] - rew_vec[arm]
        )

        return None

    def get_arm(self):
        return (
            (self.time % self.num_arms)
            if self.time <= self.m * self.num_arms
            else self.get_best_arm()
        )

    def iterate(self, rew_vec):
        arm = self.get_arm()
        arm = int(arm)

        self.update_regret(rew_vec, arm)
        self.update_stats(rew_vec[arm], arm)

        return None
