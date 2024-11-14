import numpy as np
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

def batch_index_add_(v, i, s, dim: int = 1):
    """
    v: tensor (*B, m, *n)
    i: int tensor (*B, *k)
    s: tensor (*B, *k, *n)

    v[*B, i[*B, k], ...] += s[*B, k, ...]
    """
    if dim < 0:
        dim = len(v.shape) + dim
    B, M, N = v.shape[:dim], v.shape[dim], v.shape[dim+1:]
    Bi, K = i.shape[:dim], i.shape[dim:]
    value_dims = len(N)
    if value_dims == 0:
        Bs, Ks, Ns = s.shape[:dim], s.shape[dim:], N
    else:
        Bs, Ks, Ns = s.shape[:dim], s.shape[dim:-value_dims], s.shape[-value_dims:]

    assert B == Bi
    assert B == Bs
    assert K == Ks 
    assert N == Ns
    
    Bsize = 1
    for d in B:
        Bsize *= d
    
    Ksize = 1
    for d in K:
        Ksize *= d

    vv = v.view(Bsize * M, *N)
    ii = i.view(Bsize, Ksize)
    ss = s.view(Bsize * Ksize, *N)
    iii = ii + torch.arange(Bsize, device=i.device)[:,None] * M

    vv.index_add_(0, iii.view(-1), ss)

def np2torch(arr):
    if isinstance(arr, np.ndarray):
        return torch.from_numpy(arr)
    # otherwise, do nothing
    return arr

def torch2np(arr):
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    # otherwise, do nothing
    return arr

def laplacian_cot(verts, faces):
    """
    Compute the cotangent laplacian

    Inspired by https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/loss/mesh_laplacian_smoothing.html

    Parameters
    ----------
    verts : torch.Tensor [*B, V, 3]
        Vertex positions.
    faces : torch.Tensor [*B, F, 3]
        array of triangle faces.
    """
    verts = np2torch(verts)
    faces = np2torch(faces).long()

    # V = sum(V_n), F = sum(F_n)
    batch_shape = verts.shape[:-2]
    V, F = verts.shape[-2], faces.shape[-2] 

    face_verts = batch_indexing(verts, faces, -2) # [*B, F, 3, 3]
    v0, v1, v2 = face_verts[...,0,:], face_verts[...,1,:], face_verts[...,2,:] # [*B, F, 3]

    # Side lengths of each triangle, of shape (sum(F_n),)
    # A is the side opposite v1, B is opposite v2, and C is opposite v3
    A = (v1 - v2).norm(dim=-1) # [*B, F]
    B = (v0 - v2).norm(dim=-1)
    C = (v0 - v1).norm(dim=-1)

    # Area of each triangle (with Heron's formula); shape is (sum(F_n),)
    s = 0.5 * (A + B + C)
    # note that the area can be negative (close to 0) causing nans after sqrt()
    # we clip it to a small positive value
    area = (s * (s - A) * (s - B) * (s - C)).clamp_(min=1e-12).sqrt() # [*B, F]

    # Compute cotangents of angles, of shape (sum(F_n), 3)
    A2, B2, C2 = A * A, B * B, C * C
    cota = (B2 + C2 - A2) / area
    cotb = (A2 + C2 - B2) / area
    cotc = (A2 + B2 - C2) / area
    cot = torch.stack([cota, cotb, cotc], dim=-1) # [*B, F, 3]
    cot /= 4.0

    # Construct a sparse matrix by basically doing:
    # L[v1, v2] = cota
    # L[v2, v0] = cotb
    # L[v0, v1] = cotc
    ii = faces[..., [1, 2, 0]]
    jj = faces[..., [2, 0, 1]]
    idx = torch.stack([ii, jj], dim=0).view(2, -1) #*batch_shape, F * 3)
    L = torch.sparse.DoubleTensor(idx, cot.view(*batch_shape, -1), (*batch_shape, V, V))

    # Make it symmetric; this means we are also setting
    # L[v2, v1] = cota
    # L[v0, v2] = cotb
    # L[v1, v0] = cotc
    L += L.mT

#     # Add the diagonal indices
#     vals = torch.sparse.sum(L, dim=0).to_dense()
#     indices = torch.arange(V)
#     idx = torch.stack([indices, indices], dim=0)
#     L = torch.sparse.FloatTensor(idx, vals, (V, V)) - L
    
    # For each vertex, compute the sum of areas for triangles containing it.
#     idx = faces.view(-1)
    idx = faces # [*B, F, 3]
    inv_areas = torch.zeros(*batch_shape, V, dtype=area.dtype, device=verts.device) # [*B, V]
#     val = torch.stack([area] * 3, dim=-1).view(-1)
    val = torch.stack([area] * 3, dim=-1) # [*B, F, 3]
    
#     inv_areas.scatter_add_(0, idx, val)
    batch_index_add_(inv_areas, idx, val, dim=-1)
    idx = inv_areas > 0
    inv_areas[idx] = 1.0 / inv_areas[idx]
    inv_areas = inv_areas.view(*batch_shape, V, 1)
    
    vals = torch.sparse.sum(L, dim=-2).to_dense()
    indices = torch.arange(V, device=verts.device)
    idx = torch.stack([indices, indices], dim=0)
    
    L = L.to_dense()
    L = torch.diag(torch.sum(L, dim=-1)) - L

    return L, inv_areas

def eigen_decomposition(L, iA):
#     Lsum = torch.sum(L, dim=-1)
#     M = (torch.diag(Lsum) - L)
    eigenvalues, eigenbases = torch.linalg.eigh(L)
    return eigenvalues, eigenbases

def get_eigenbases(verts, faces):
    L, iA = laplacian_cot(verts, faces)
    EV, EB = eigen_decomposition(L, iA)
    return EV.numpy(), EB.numpy()

def cart2spect(x, EB):
    return EB.T @ x

def spect2cart(u, EB):
    return EB @ u
