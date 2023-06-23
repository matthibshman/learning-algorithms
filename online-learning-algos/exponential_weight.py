import numpy as np
from numpy import random


class Exp3(object):
    def __init__(self, avg, eta):
        self.means = avg
        self.num_arms = avg.size
        self.eta = eta
        self.best_arm = np.argmax(self.means)
        self.num_plays = np.zeros(self.num_arms)
        self.S = np.zeros(self.num_arms)
        self.cum_reg = [0]
        self.probs_arr = np.full(self.S.shape, 1 / self.num_arms)

    def get_best_arm(self):
        return random.choice(self.num_arms, p=self.probs_arr)

    def update_exp3(self, arm, rew_vec):
        exponents = np.exp(self.eta * self.S)
        self.probs_arr = exponents / np.sum(exponents)

        self.S = self.S + np.ones(self.S.shape)
        self.S[arm] = self.S[arm] - ((1 - rew_vec[arm]) / self.probs_arr[arm])
        return None

    def update_regret(self, arm, rew_vec):
        self.cum_reg.append(self.cum_reg[-1] + (rew_vec[self.best_arm] - rew_vec[arm]))

        return None

    def iterate(self, rew_vec):
        arm = int(self.get_best_arm())
        self.update_regret(arm, rew_vec)
        self.update_exp3(arm, rew_vec)

        return None
