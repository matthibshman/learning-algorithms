import numpy as np
from sklearn.feature_selection import SelectKBest, chi2, f_classif


def grad(x, Y, beta, mu):
    N = x.shape[0]
    weights = -np.dot(x, beta)
    softmax = np.expm1(weights) / np.sum(np.expm1(weights), axis=1, keepdims=True)
    grad = (np.dot(x.T, (Y - softmax))) / N
    l2_reg = 2 * mu * beta
    return grad + l2_reg


def gradient(x, Y, beta, alpha, mu=0.001, beta_prev=None, gamma=None):
    return beta - alpha * grad(x, Y, beta, mu), None


def evaluate(x, Y, beta):
    N = x.shape[0]
    # Y onehot-encoded to "select" the beta parameter for each data point
    # weights = np.sum(np.dot(x, np.dot(beta, Y.T)))
    # significant speed up when calculating the sum through the trace
    weights = np.trace(x @ beta @ Y.T)
    logs = np.sum(np.log(np.sum(np.expm1(np.dot(-x, beta)), axis=1)))
    return (weights + logs) / N


def nesterov(x, Y, beta, alpha, mu, beta_prev, gamma):
    momentum = gamma * (beta - beta_prev)
    return beta + momentum - alpha * grad(x, Y, beta, mu), beta


def onehot_encode(y):
    Y = np.zeros((y.size, 20))
    Y[np.arange(y.size), y] = 1
    return Y


def descent(update, train, test, mu=0.001, alpha=0.5, T=int(500)):
    train_loss = np.zeros(T)
    test_loss = np.zeros(T)
    # Y onehot-encoded to allow identifying class by matrix multiplication
    Y = onehot_encode(train.Y)
    Y_test = onehot_encode(test.Y)
    beta = np.zeros((train.X.shape[1], Y.shape[1]))
    beta_prev = np.zeros((train.X.shape[1], Y.shape[1]))
    for t in range(T):
        beta, beta_prev = update(train.X, Y, beta, alpha, mu, beta_prev=beta_prev, gamma=0.9)
        train_loss[t] = evaluate(train.X, Y, beta)
        test_loss[t] = evaluate(test.X, Y_test, beta)

    return train_loss, test_loss


# grad_train_loss, grad_test_loss = descent(gradient, train, test, mu)
# nest_train_loss, nest_test_loss = descent(nesterov, train, test, mu)


def accelerated_minibatch_stochastic_gradient(
    x, Y, beta, alpha, mu, beta_prev, gamma=0.9, batch_size=1000
):
    idxs = np.random.randint(0, high=x.shape[0], size=batch_size)
    grads = np.zeros(beta.shape)
    for idx in idxs:
        random_x, random_Y = x[idx].reshape(-1, x.shape[1]), Y[idx].reshape(-1, Y.shape[1])
        grads += grad(random_x, random_Y, beta, mu)
    grads = grads / batch_size
    momentum = gamma * (beta - beta_prev)
    return beta + momentum - alpha * grads, beta


def stochastic_gradient(x, Y, beta, alpha, mu, beta_prev=None):
    idx = np.random.randint(x.shape[0])
    random_x, random_Y = x[idx].reshape(-1, x.shape[1]), Y[idx].reshape(-1, Y.shape[1])
    return beta - alpha * grad(random_x, random_Y, beta, mu), beta_prev


def descent_sparse(
    update, train, test, mu=0.001, alpha=0.01, iters=int(200), eval_iters=10, adaptive_lr=False
):
    test_loss = np.zeros(int(iters / eval_iters))
    # Y onehot-encoded to allow identifying class by matrix multiplication
    Y = onehot_encode(train.Y)
    Y_test = onehot_encode(test.Y)
    beta = np.zeros((train.X.shape[1], Y.shape[1]))
    beta_prev = np.zeros((train.X.shape[1], Y.shape[1]))
    for t in range(iters):
        beta, beta_prev = update(train.X, Y, beta, alpha, mu, beta_prev)

        if t == 0 or t % eval_iters == 0:
            test_loss[int(t / eval_iters)] = evaluate(test.X, Y_test, beta)

        if adaptive_lr and t > 0 and t % 4000 == 0:
            alpha *= 0.1

    return test_loss


# f_stat = SelectKBest(f_classif, k=2000)
# X_train = f_stat.fit_transform(train.X, train.Y)
# X_test = f_stat.transform(test.X)
# train.X = X_train.toarray()
# test.X = X_test.toarray()

# label, function, mu, alpha, iterations, T = ("Optimized SGD", accelerated_minibatch_stochastic_gradient, 1e-2, 1e-2, 5000, 500)
# optimized_test_loss = descent_sparse(function, train, test, mu, alpha, iters=int(iterations), eval_iters=int(T), adaptive_lr=True)
# label, function, mu, alpha, iterations, T = ("Standard SGD", stochastic_gradient, 1e-2, 1e-2, 5000000, 500 * 1000)
# standard_test_loss = descent_sparse(function, train, test, mu, alpha, iters=int(iterations), eval_iters=int(T))
