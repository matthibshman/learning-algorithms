from typing import Iterable
import numpy as np
import torch


class NN(torch.nn.Module):
    def __init__(self, input_dims, output_dims):
        super().__init__()
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(input_dims, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, output_dims),
        ).float()

    def forward(self, input):
        return self.classifier(torch.tensor(input).float())


class PiApproximationWithNN:
    def __init__(self, state_dims, num_actions, alpha):
        """
        state_dims: the number of dimensions of state space
        action_dims: the number of possible actions
        alpha: learning rate
        """
        self.state_dims = state_dims
        self.actions = np.arange(num_actions)

        self.nn = NN(state_dims, num_actions)
        self.optim = torch.optim.Adam(self.nn.parameters(), lr=alpha)
        self.softmax = torch.nn.Softmax(dim=-1)

    def __call__(self, s) -> int:
        self.nn.eval()
        action_probs = self.softmax(self.nn(s)).detach().numpy()
        action = np.random.choice(self.actions, p=action_probs)
        return int(action)

    def update(self, s, a, gamma_t, delta):
        """
        s: state S_t
        a: action A_t
        gamma_t: gamma^t
        delta: G-v(S_t,w)
        """
        self.nn.train()
        self.optim.zero_grad()

        forward_output = self.softmax(self.nn(s))
        logprob = torch.log(forward_output)
        loss = -gamma_t * delta * logprob[int(a)]
        loss.backward()
        self.optim.step()


class Baseline(object):
    """
    The dumbest baseline; a constant for every state
    """

    def __init__(self, b):
        self.b = b

    def __call__(self, s) -> float:
        return self.b

    def update(self, s, G):
        pass


class VApproximationWithNN(Baseline):
    def __init__(self, state_dims, alpha):
        """
        state_dims: the number of dimensions of state space
        alpha: learning rate
        """
        self.nn = NN(state_dims, 1)
        self.optim = torch.optim.Adam(self.nn.parameters(), lr=alpha)
        self.loss = torch.nn.MSELoss()

    def __call__(self, s) -> float:
        self.nn.eval()
        return self.nn(s).detach().numpy()[0]

    def update(self, s, G):
        self.nn.train()
        self.optim.zero_grad()

        forward_output = self.nn(s)
        label = torch.tensor(G).float()
        loss = self.loss(forward_output, label)
        loss.backward()
        self.optim.step()


def REINFORCE(
    env,  # open-ai environment
    gamma: float,
    num_episodes: int,
    pi: PiApproximationWithNN,
    V: Baseline,
) -> Iterable[float]:
    """
    implement REINFORCE algorithm with and without baseline.

    input:
        env: target environment; openai gym
        gamma: discount factor
        num_episode: # of episodes to iterate
        pi: policy
        V: baseline
    output:
        a list that includes the G_0 for every episodes.
    """
    g_0 = np.zeros(num_episodes)
    for episode in range(num_episodes):
        T, states, actions, rewards = generate_episode(env, pi)

        for t in range(T):
            G = 0
            for k in range(t + 1, T + 1):
                G += gamma ** (k - t - 1) * rewards[k]
            if t == 0:
                g_0[episode] = G
            delta = G - V(states[t])
            V.update(states[t], G)
            pi.update(states[t], actions[t], gamma**t, delta)

    return g_0


def generate_episode(env, pi: PiApproximationWithNN):
    states = np.zeros((500, pi.state_dims))
    actions = np.zeros(500)
    rewards = np.zeros(500)

    init_state = env.reset()
    T = 0
    states[T] = init_state
    is_terminal = False
    while not is_terminal:
        if T + 1 >= len(rewards):
            states = np.pad(states, ((0, 500), (0, 0)), "constant")
            actions = np.pad(actions, (0, 500), "constant")
            rewards = np.pad(rewards, (0, 500), "constant")

        actions[T] = pi(states[T])
        states[T + 1], rewards[T + 1], is_terminal, _ = env.step(int(actions[T]))
        T += 1
    rewards[T] = 1

    return (T, states, actions, rewards)
