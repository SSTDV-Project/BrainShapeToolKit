import torch

def batch_indexing(v: torch.Tensor, i: torch.IntTensor, dim: int = 1) -> torch.Tensor:
    """
    v: tensor (*B, m, *)
    i: int tensor (*B, *n)
    out: tensor (*B, *n, *)

    out[j, *k,...] = v[j, i[j, *k], ...]
    """
    if dim < 0:
        dim = len(v.shape) + dim
    B, [M, *rest] = v.shape[:dim], v.shape[dim:]
    Bi, N = i.shape[:dim], i.shape[dim:]

    assert B == Bi

    if not B:
        return v[i]
    
    Bsize = 1
    for d in B:
        Bsize *= d

    vv = v.view(-1, *rest)
    ii = i.view(Bsize, -1)
    ii = ii + torch.arange(Bsize, device=i.device)[:,None] * M

    return vv[ii].view(*B, *N, *rest)
