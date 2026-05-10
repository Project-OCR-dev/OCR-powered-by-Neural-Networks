import numpy as np
from numpy.lib.stride_tricks import as_strided

params = {
    "K1": np.random.randn(32, 1, 3, 3) * np.sqrt(2 / (1 * 3 * 3)),
    "K2": np.random.randn(64, 32, 3, 3) * np.sqrt(2 / (32 * 3 * 3)),
    "W1": np.random.randn(256, 2304) * np.sqrt(2 / 2304),
    "b1": np.zeros(256),
    "W2": np.random.randn(128, 256) * np.sqrt(2 / 256),
    "b2": np.zeros(128),
    "W3": np.random.randn(62, 128) * np.sqrt(2 / 128),
    "b3": np.zeros(62),
}

def im2col(x, k):
    B, C, H, W = x.shape
    Ho, Wo = H - k + 1, W - k + 1
    s = x.strides
    return np.ascontiguousarray(
        as_strided(x, (B, Ho, Wo, C, k, k), (s[0], s[2], s[3], s[1], s[2], s[3]))
        .reshape(B, Ho * Wo, C * k * k)
    )

def col2im(dc, shape, k):
    B, C, H, W = shape
    Ho, Wo = H - k + 1, W - k + 1
    dx = np.zeros(shape)
    dc_r = dc.reshape(B, Ho, Wo, C, k, k)
    for di in range(k):
        for dj in range(k):
            dx[:, :, di:di+Ho, dj:dj+Wo] += dc_r[:, :, :, :, di, dj].transpose(0, 3, 1, 2)
    return dx

def conv_fwd(x, K):
    B, _, H, W = x.shape
    Co, _, k, _ = K.shape
    Ho, Wo = H - k + 1, W - k + 1
    cols = im2col(x, k)
    return (cols @ K.reshape(Co, -1).T).transpose(0, 2, 1).reshape(B, Co, Ho, Wo), cols

def conv_bwd(dout, cols, K, x_shape, k):
    B, Co, Ho, Wo = dout.shape
    Ci = K.shape[1]
    d = dout.transpose(0, 2, 3, 1).reshape(-1, Co)
    dK = (d.T @ cols.reshape(-1, Ci * k * k)).reshape(K.shape)
    dx = col2im((d @ K.reshape(Co, -1)).reshape(B, Ho * Wo, Ci * k * k), x_shape, k)
    return dK, dx

def pool_fwd(x):
    B, C, H, W = x.shape
    H2, W2 = H // 2, W // 2
    xc = x[:, :, :H2*2, :W2*2]
    return xc.reshape(B, C, H2, 2, W2, 2).max(axis=(3, 5)), xc

def pool_bwd(dout, xc):
    B, C, H2, W2 = dout.shape
    xr = xc.reshape(B, C, H2, 2, W2, 2)
    mask = xr == xr.max(axis=(3, 5), keepdims=True)
    mask = mask / np.maximum(1, mask.sum(axis=(3, 5), keepdims=True))
    return (mask * dout[:, :, :, np.newaxis, :, np.newaxis]).reshape(B, C, H2*2, W2*2)
