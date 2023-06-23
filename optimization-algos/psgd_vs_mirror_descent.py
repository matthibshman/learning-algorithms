import numpy as np
import numpy.linalg as la


def subgrad(X, val):
    grad = np.dot(X.T, np.sign(val))
    return grad


# code adapted from algorithm in https://stanford.edu/~jduchi/projects/DuchiShSiCh08.pdf
def simplex_projection(beta, z=1):
    u = np.sort(beta)[::-1]
    rho = 1
    for j in range(u.shape[0]):
        if u[j] - (1 / (j + 1)) * (np.sum(u[0 : j + 1]) - z) > 0:
            rho = j + 1
    theta = (1 / rho) * (np.sum(u[0:rho]) - z)
    return np.maximum(beta - theta, 0)


def psgd(beta, X, y, eta):
    value = np.dot(X, beta) - y
    beta_next = beta - eta * subgrad(X, value)
    return simplex_projection(beta_next)


def mirror(beta, X, y, eta):
    value = np.dot(X, beta) - y
    z = beta * np.exp(-eta * subgrad(X, value))
    return z / la.norm(z, 1)


def objective_value(X, y, beta):
    return la.norm(np.dot(X, beta) - y, 1)


def descent(update, X, y, eta=1e-4, T=int(500)):
    beta = np.full((X.shape[1]), 1 / X.shape[0])
    objective = []
    for t in range(T):
        beta = update(beta, X, y, eta)

        # record objective value
        if (t % 1 == 0) or (t == T - 1):
            objective.append(objective_value(X, y, beta))

    return beta, objective


# algorithms = [("psgd", psgd, 1e-4), ("mirror", mirror, 1e-3)]
# for label, function, eta in algorithms:
#   beta, objective = descent(function, X, y, eta)
