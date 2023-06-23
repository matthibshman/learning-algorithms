from algo import ValueFunctionWithApproximation

import torch


class ValueFunctionWithNN(ValueFunctionWithApproximation):
    class NN(torch.nn.Module):
        def __init__(self, state_dims):
            super().__init__()
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(state_dims, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 1),
            ).float()

        def forward(self, input):
            return self.classifier(torch.tensor(input).float())

    def __init__(self, state_dims):
        """
        state_dims: the number of dimensions of state space
        """
        self.nn = self.NN(state_dims)
        self.optim = torch.optim.Adam(self.nn.parameters(), lr=0.001)
        self.loss = torch.nn.MSELoss()

    def __call__(self, s):
        self.nn.eval()

        return self.nn(s).detach().numpy()[0]

    def update(self, alpha, G, s_tau):
        self.nn.train()
        self.optim.zero_grad()

        forward_output = self.nn(s_tau)
        label = torch.tensor(G).float()
        loss = self.loss(forward_output, label)
        loss.backward()
        self.optim.step()

        return None
