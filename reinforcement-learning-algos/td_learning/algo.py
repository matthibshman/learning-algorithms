import numpy as np
from policy import Policy
import sys


class ValueFunctionWithApproximation(object):
    def __call__(self, s) -> float:
        """
        return the value of given state; \hat{v}(s)

        input:
            state
        output:
            value of the given state
        """
        raise NotImplementedError()

    def update(self, alpha, G, s_tau):
        """
        Implement the update rule;
        w <- w + \alpha[G- \hat{v}(s_tau;w)] \nabla\hat{v}(s_tau;w)

        input:
            alpha: learning rate
            G: TD-target
            s_tau: target state for updating (yet, update will affect the other states)
        ouptut:
            None
        """
        raise NotImplementedError()


def semi_gradient_n_step_td(
    env,  # open-ai environment
    gamma: float,
    pi: Policy,
    n: int,
    alpha: float,
    V: ValueFunctionWithApproximation,
    num_episode: int,
):
    """
    implement n-step semi gradient TD for estimating v

    input:
        env: target environment
        gamma: discounting factor
        pi: target evaluation policy
        n: n-step
        alpha: learning rate
        V: value function
        num_episode: #episodes to iterate
    output:
        None
    """
    for _ in range(num_episode):
        T = sys.maxsize
        tau = -1
        t = 0

        init_state, _ = env.reset()  # seed=
        states = np.zeros((100, init_state.size + 1))
        rewards = np.zeros(100)
        states[0] = [init_state, 0]

        while tau != T - 1:
            if t + 1 >= len(rewards):
                states = np.pad(states, ((0, 100), (0,0)), "constant")
                rewards = np.pad(rewards, (0, 100), "constant")

            if t < T:
                states[t+1], rewards[t+1], is_terminal, _ = env.step(pi.action(states[t]))
                if is_terminal:
                    T = t + 1

            tau = t - n - 1
            if tau >= 0:
                G = np.zeros(1)
                for i in range(tau+1, min(tau + n, T)+1):
                    G += gamma ** (i - tau - 1) * rewards[i]
                if tau + n < T:
                    G += gamma**n * V(states[tau + n])
                V.update(alpha, G, states[tau])
            t += 1