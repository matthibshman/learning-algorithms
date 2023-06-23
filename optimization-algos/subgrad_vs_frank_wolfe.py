import numpy as np
import numpy.random as rn
import numpy.linalg as la


def grad(x, A, b):
    return np.dot(A.T, np.dot(A, x) - b) / A.shape[0]


def subgradient(x, A, b, lam, t=None, c=1e-4):
    mse_loss = grad(x, A, b)
    reg_loss = np.sign(x)
    reg_loss[x == 0] = rn.uniform(-1.0, 1.0, len(reg_loss[x == 0]))
    return x - c * (mse_loss + lam * reg_loss)


def frank_wolfe(x, A, b, lam, t):
    gamma = 2 / (t + 1)
    loss = grad(x, A, b)
    y = np.zeros(A.shape[1])
    i = np.argmax(np.abs(loss))
    y[i] = lam if loss[i] <= 0 else -lam
    return (1 - gamma) * x + gamma * y


def descent(update, A, b, reg, T=int(1e4)):
    x = np.zeros(A.shape[1])
    z = np.zeros(A.shape[1])
    error = []
    l1 = []
    for t in range(T):
        x = update(x, A, b, reg, t + 1)

        # record error and l1 norm
        if (t % 1 == 0) or (t == T - 1):
            error.append(la.norm(np.dot(A, x) - b))
            l1.append(np.sum(abs(x)))
            assert not np.isnan(error[-1])

    return x, error, l1


# algorithms = [("frank-wolfe ($\gamma$=5)", frank_wolfe, 5), ("subgradient", subgradient, 1e-6)][]
# for _, function, lam in algorithms:
#   _, error, l1 = descent(function, A, b, lam, T=int(1e4))
