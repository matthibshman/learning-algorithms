import numpy as np


class UpperConfidenceBound(object):
    def __init__(self, avg):
        self.means = avg
        self.num_arms = avg.size
        self.best_arm = np.argmax(self.means)
        self.time = 0.0
        self.ucb_arr = 1e5 * np.ones(self.num_arms)
        self.emp_means = np.zeros(self.num_arms)
        self.num_pulls = np.zeros(self.num_arms)
        self.cum_reg = [0]

    def get_best_arm(self):
        return np.argmax(self.ucb_arr)

    def update_stats(self, rew, arm):
        self.emp_means[arm] = (self.emp_means[arm] * self.num_pulls[arm] + rew) / (
            self.num_pulls[arm] + 1.0
        )
        self.num_pulls[arm] += 1.0
        self.time += 1.0

        return None

    def update_bounds(self):
        log_f = np.log(1.0 + self.time * np.log(np.log(self.time)))
        if (self.num_pulls > 0).sum() == self.num_pulls.size:
            # all arms pulled at least once, update all at once
            exploration_bonus = np.sqrt((2 * log_f) / self.num_pulls)
            self.ucb_arr = self.emp_means + exploration_bonus
        else:
            # some arm still not pulled, update individually to avoid divide-by-zero
            for arm in range(self.num_arms):
                if self.num_pulls[arm] > 0:
                    exploration_bonus = np.sqrt((2 * log_f) / self.num_pulls[arm])
                    self.ucb_arr[arm] = self.emp_means[arm] + exploration_bonus

        return None

    def update_regret(self, rew_vec, arm):
        self.cum_reg.append(self.cum_reg[int(self.time)] + (rew_vec[self.best_arm] - rew_vec[arm]))

        return None

    def iterate(self, rew_vec):
        arm = int(self.get_best_arm())

        self.update_regret(rew_vec, arm)
        self.update_stats(rew_vec[arm], arm)
        self.update_bounds()

        return None
