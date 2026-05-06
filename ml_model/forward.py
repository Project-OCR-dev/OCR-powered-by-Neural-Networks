import numpy as np
from convolution import conv_fwd, pool_fwd, params

def forward(imgs):
    B = imgs.shape[0]
    x = imgs[:, np.newaxis]
    c1, col1 = conv_fwd(x, params["K1"])
    a1 = np.maximum(0, c1)
    p1, xc1 = pool_fwd(a1)
    c2, col2 = conv_fwd(p1, params["K2"])
    a2 = np.maximum(0, c2)
    p2, xc2 = pool_fwd(a2)
    flat = p2.reshape(B, -1)
    h1 = np.maximum(0, flat @ params["W1"].T + params["b1"])
    h2 = np.maximum(0, h1 @ params["W2"].T + params["b2"])
    z3 = h2 @ params["W3"].T + params["b3"]
    return z3, h1, h2, flat, x, c1, a1, xc1, col1, p1, c2, a2, xc2, col2
