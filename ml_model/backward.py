import numpy as np
from convolution import conv_bwd, pool_bwd, params

def backward(z3, h1, h2, flat, x, c1, a1, xc1, col1, p1, c2, a2, xc2, col2, y, lr=0.001):
    B = z3.shape[0]
    e = np.exp(z3 - z3.max(1, keepdims=True))
    proba = e / e.sum(1, keepdims=True)
    loss = -np.sum(y * np.log(proba + 1e-9)) / B

    dZ3 = (proba - y) / B
    dW3, db3 = dZ3.T @ h2, dZ3.sum(0)
    dZ2 = (dZ3 @ params["W3"]) * (h2 > 0)
    dW2, db2 = dZ2.T @ h1, dZ2.sum(0)
    dZ1 = (dZ2 @ params["W2"]) * (h1 > 0)
    dW1, db1 = dZ1.T @ flat, dZ1.sum(0)

    dp2 = (dZ1 @ params["W1"]).reshape(B, 64, 6, 6)
    da2c = pool_bwd(dp2, xc2)
    da2 = np.zeros_like(c2)
    da2[:, :, :da2c.shape[2], :da2c.shape[3]] = da2c
    dK2, dp1 = conv_bwd(da2 * (c2 > 0), col2, params["K2"], p1.shape, 3)

    da1 = pool_bwd(dp1, xc1)
    dK1, _ = conv_bwd(da1 * (c1 > 0), col1, params["K1"], x.shape, 3)

    for k, g in [("W3",dW3),("b3",db3),("W2",dW2),("b2",db2),
                 ("W1",dW1),("b1",db1),("K2",dK2),("K1",dK1)]:
        params[k] -= lr * g
    return loss
