import numpy as np

def grad(x, Y, beta, mu, axis=1):
  N = x.shape[0]
  weights = -np.dot(x, beta)
  softmax = np.exp(weights) / np.sum(np.exp(weights), axis=axis, keepdims=True)
  grad = (np.dot(x.T, (Y - softmax))) / N
  l2_reg = 2 * mu * beta
  return grad + l2_reg

def gradient(x, Y, beta, alpha, mu, T=None):
  return beta - alpha * grad(x, Y, beta, mu)

def stochastic_gradient(x, Y, beta, alpha, mu, T=None):
  idx = np.random.randint(x.shape[0])
  random_x, random_Y = x[idx].reshape(-1, x.shape[1]), Y[idx].reshape(-1, Y.shape[1])
  return beta - alpha * grad(random_x, random_Y, beta, mu)

def stochastic_variance_reduced_gradient(x, Y, beta, alpha, mu, T=10):
  y_beta = beta
  y_grad = grad(x, Y, y_beta, mu)
  beta_sum = np.zeros(beta.shape)
  for t in range(T):
    idx = np.random.randint(x.shape[0])
    random_x, random_Y = x[idx].reshape(-1, x.shape[1]), Y[idx].reshape(-1, Y.shape[1])
    beta = beta - alpha * (grad(random_x, random_Y, beta, mu) - (grad(random_x, random_Y, y_beta, mu) - y_grad))
    beta_sum += beta
  return beta_sum / T

def evaluate(x, Y, beta):
  N = x.shape[0]
  weights = np.trace(x @ beta @ Y.T)
  logs = np.sum(np.log(np.sum(np.exp(np.dot(-x, beta)), axis=1)))
  return (weights + logs) / N

def onehot_encode(y):
  Y = np.zeros((y.size, 20))
  Y[np.arange(y.size), y] = 1
  return Y

def descent(update, train, test, mu=0.001, alpha=0.01, iters=int(200), svrg_iters=10):
  train_loss = np.zeros(iters)
  test_loss = np.zeros(iters)
  # Y onehot-encoded to allow identifying class by matrix multiplication
  Y = onehot_encode(train.Y)
  Y_test = onehot_encode(test.Y)
  beta = np.zeros((train.X.shape[1], Y.shape[1]))
  for t in range(iters):
    beta = update(train.X, Y, beta, alpha, mu, svrg_iters)
    train_loss[t] = evaluate(train.X, Y, beta)
    test_loss[t] = evaluate(test.X, Y_test, beta)

  return train_loss, test_loss

# methods = [("GD", gradient, 1e-2, 1e-2, 100, 0, train.X.shape[0]),
#            ("SGD", stochastic_gradient, 1e-3, 1e-3, 50000, 0, 1),
#            ("SVRG", stochastic_variance_reduced_gradient, 1e-3, 1e-3, 25, 1000, train.X.shape[0] + 1000)]
# for method in methods:
#     _, function, mu, alpha, iterations, T, _ = method
#     grad_train_loss, _ = descent(function, train, test, mu, alpha, iters=int(iterations), svrg_iters=int(T))