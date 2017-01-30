import numpy as np
import torch

def uniform_init(shape):
    """
    Create a shared object of a numpy array.
    """
    if len(shape) == 1:
        value = np.zeros(shape)  # bias are initialized with zeros
    else:
        drange = np.sqrt(6. / (np.sum(shape)))
        value = drange * np.random.uniform(low=-1.0, high=1.0, size=shape)
    return torch.from_numpy(value)

def orthogonal_init(shape):
    """
    Orthogonal weights.
    """
    W = np.random.randn(shape)
    u, s, v = np.linalg.svd(W)
    return torch.from_numpy(u)
