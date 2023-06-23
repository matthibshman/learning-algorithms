import numpy as np
import numpy.linalg as la


def f(x, M, b):
    return 0.5 * np.matmul(x.T, np.matmul(M, x)) - np.matmul(b.T, x)


def gradient(x, M, b):
    return np.matmul(M, x) - b


def bfgs(x, H, M, b):
    grad_x = gradient(x, M, b)
    d = -np.matmul(H, grad_x)
    eta = exact_line_search(d, M, grad_x)
    x_next = x + eta * d
    s = x_next - x
    y = gradient(x_next, M, b) - grad_x
    p = 1 / (np.matmul(y.T, s)[0][0])
    left_update = np.identity(x.shape[0]) - (p * np.matmul(s, y.T))
    right_update = np.identity(x.shape[0]) - (p * np.matmul(y, s.T))
    H_next = left_update @ H @ right_update + (p * np.matmul(s, s.T))
    return x_next, H_next


def exact_line_search(direction, M, grad_x):
    return (np.matmul(-grad_x.T, direction) / np.matmul(direction.T, np.matmul(M, direction)))[0][0]


def descent(update, M, b, x_star, T=int(200)):
    convergence = np.zeros(T)
    x = np.expand_dims(np.random.uniform(low=-1, high=1, size=x_star.shape[0]), axis=1)
    H = np.identity(x.shape[0])
    for t in range(T):
        x, H = update(x, H, M, b)
        dist = np.log(la.norm(x - x_star) ** 2)
        if dist > -30:
            convergence[t] = dist
        else:
            return convergence[:t]

    return convergence


# convergence_m1 = descent(bfgs, M1_bfgs.to_numpy(), b1_bfgs.to_numpy(), x_bfgs.to_numpy())
# convergence_m2 = descent(bfgs, M2_bfgs.to_numpy(), b2_bfgs.to_numpy(), x_bfgs.to_numpy())
