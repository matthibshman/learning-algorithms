import numpy as np
from numpy import random


class ThompsonSampling:
    def __init__(self, avg):
        self.means = avg
        self.num_arms = avg.size
        self.best_arm = np.argmax(self.means)
        self.alpha = np.ones(self.num_arms)
        self.beta = np.ones(self.num_arms)
        self.cum_reg = [0]

    def get_best_arm(
        self,
    ):
        thetas = random.beta(self.alpha, self.beta)
        return np.argmax(thetas)

    def update(self, arm, rew):
        if rew > 0:
            self.alpha[arm] += 1.0
        else:
            self.beta[arm] += 1.0

        return None

    def update_regret(self, arm, rew_vec):
        self.cum_reg.append(self.cum_reg[-1] + (rew_vec[self.best_arm] - rew_vec[arm]))

        return None

    def iterate(self, rew_vec):
        arm = int(self.get_best_arm())
        self.update_regret(arm, rew_vec)
        self.update(arm, rew_vec[arm])

        return None
