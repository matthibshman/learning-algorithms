import numpy as np
import numpy.random as rn
import numpy.linalg as la


def subgradient(x, A, b, lam, beta=None, z=None, c=1e-4):
    mseLoss = np.dot(A.T, np.dot(A, x) - b) / A.shape[0]
    regLoss = np.sign(x)
    regLoss[x == 0] = rn.uniform(-1.0, 1.0, len(regLoss[x == 0]))
    return x - c * (mseLoss + lam * regLoss), z


def ista(x, A, b, lam, beta, z=None):
    z = x - 1 / beta * np.dot(A.T, np.dot(A, x) - b)
    alpha = lam / beta
    return np.maximum(np.abs(z) - alpha, 0) * np.sign(z), z


def fista(x, A, b, lam, beta, z, gamma=0.9):
    z_next, _ = ista(x, A, b, lam, beta)
    return (1 - gamma) * z_next + gamma * z, z_next


def descent(update, A, b, reg, beta, T=int(1e4)):
    x = np.zeros(A.shape[1])
    z = np.zeros(A.shape[1])
    error = []
    l1 = []
    for t in range(T):
        x, z = update(x, A, b, reg, beta, z)

        # record error and l1 norm
        if (t % 1 == 0) or (t == T - 1):
            error.append(la.norm(np.dot(A, x) - b))
            l1.append(np.sum(abs(x)))

    return x, error, l1

# algorithms = [("ISTA", ista), ("FISTA", fista), ("subgradient", subgradient)]
# eigenvalues, _ = la.eig(np.dot(A.T, A))
# beta = max(eigenvalues)
# for label, function in algorithms:
  # _, error_sg, _ = descent(function, A, b, 1e-6, beta, T=T)