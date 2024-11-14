import numpy as np

def batch_indexing(v: np.ndarray, i: np.ndarray, dim: int = 1) -> np.ndarray:
    """
    v: a[*B, m, *]
    i: i[*B, *n]
    out: a[*B, *n, *]

    out[*b, *n, ...] = v[*b, i[*b, *n], ...]
    """
    if dim < 0:
        dim = len(v.shape) + dim
    
    B, [M, *rest] = v.shape[:dim], v.shape[dim:]
    Bi, N = i.shape[:dim], i.shape[dim:]

    assert B == Bi

    if not B:
        return v[i]

    Bsize = np.prod(B.shape)

    vv = v.view(-1, *rest)
    ii = i.view(Bsize, -1)
    ii = ii + np.arange(Bsize)[:,None] * M

    return vv[ii].view(*B, *N, *rest)

def batch_grid_indexing(v, i, dim_start, dim_end=None):
    """
    v: ndarray (*B, *m, *)
    i: int ndarray (*B, *n, |*m|)
    out: ndarray (*B, *n, *)
    
    out[j, *k, ...] = v[j, i[j, *k, 0], i[j, *k, 1], ... i[j, *k, |*m|], ...]
    """
    if dim_start < 0:
        dim_start = len(v.shape) + dim_start
    if dim_end is None:
        dim_end = len(v.shape)
    if dim_end < 0:
        dim_end = len(v.shape) + dim_end
    B, M, V = v.shape[:dim_start], v.shape[dim_start:dim_end], v.shape[dim_end:]
    Bi, N, lenM = i.shape[:dim_start], i.shape[dim_start:-1], i.shape[-1]

    assert B == Bi
    assert len(M) == lenM
    
    Bsize = 1
    for d in B:
        Bsize *= d
    
    Mcoeff = np.ones(lenM, dtype=np.int64)
    Msize = 1
    for d in range(lenM):
        Mcoeff[:d] *= M[d]
        Msize *= M[d]
    
    vv = v.reshape(-1, *V)
    ii = np.inner(i, Mcoeff).reshape(Bsize, -1)
    ii = ii + np.arange(Bsize)[:,None] * Msize
    
    return vv[ii].reshape(*B, *N, *V)
