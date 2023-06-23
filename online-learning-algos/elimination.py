import numpy as np


class Elimination:
    def __init__(self, avg, num_iter):
        self.means = avg
        self.num_iter = num_iter
        self.num_arms = avg.size
        self.best_arm = np.argmax(avg)
        self.time = 0.0
        self.A = np.arange(self.num_arms)
        self.cum_reg = [0]
        self.m = np.ceil(
            2 ** (2 * self.time)
            * np.log(max(np.exp(1), self.num_arms * self.num_iter * 2 ** (-2 * self.time)))
        )
        self.iter = 0
        self.time2 = 0.0
        self.emp_means = np.zeros(self.num_arms)
        self.num_pulls = np.zeros(self.num_arms)

    def get_best_arm(self):
        return self.A[int(self.time2 % len(self.A))]

    def update_stats(self, rew, arm):
        self.emp_means[arm] = (self.emp_means[arm] * self.num_pulls[arm] + rew) / (
            self.num_pulls[arm] + 1.0
        )
        self.num_pulls[arm] += 1.0
        self.time2 += 1.0

        return None

    def update_elimination(self):
        self.m = np.ceil(
            2 ** (2 * self.time)
            * np.log(max(np.exp(1), self.num_arms * self.num_iter * 2 ** (-2 * self.time)))
        )
        max_mean = np.amax(self.emp_means)
        elim_bound = 1 / (2**self.time)
        new_A = []
        for arm in self.A:
            if self.emp_means[arm] + elim_bound >= max_mean:
                new_A.append(arm)

        self.A = np.asarray(new_A)
        return None

    def update_regret(self, rew_vec, arm):
        self.cum_reg.append(self.cum_reg[int(self.iter)] + (rew_vec[self.best_arm] - rew_vec[arm]))

        return None

    def iterate(self, rew_vec):
        arm = int(self.get_best_arm())

        self.update_regret(rew_vec, arm)
        self.update_stats(rew_vec[arm], arm)
        self.iter += 1

        return None
