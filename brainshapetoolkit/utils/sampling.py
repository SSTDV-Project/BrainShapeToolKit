import numpy as np

from . import indexing

class UniformSurfaceSampler:
    def __init__(self, v, f, return_barycentric=False):
        """
        v: f[*B, V, D=3] - vertex positions
        f: i[*B, F, 3] - face indices
        """
        vf = indexing.batch_indexing(v, f, -2) # f[*B,F,3,D] - tri positions
        edge = vf[...,[1,2],:] - vf[...,[0,0],:] # f[*B,F,2,D]
        area_vec = np.cross(edge[...,0,:], edge[...,1,:], axis=-1) # f[*B,F,D]
        area = np.linalg.norm(area_vec, axis=-1) # f[*B,F]

        area_cum = np.cumsum(area, axis=-1) # f[*B,F]
        prob_cum = area_cum / area_cum[...,[-1]]

        self.vf = vf
        self.batch_shape = self.vf.shape[:-3]
        self.prob_cum = prob_cum
        self.return_barycentric = return_barycentric
    
    def get(self, N):
        e = np.random.rand(*self.batch_shape, N) # f[*B,N]
        f_idx = np.searchsorted(self.prob_cum, e) # i[*B,N]
        face = indexing.batch_indexing(self.vf, f_idx, dim=-3) #f[*B,N,3,D]

        w = np.random.rand(*self.batch_shape, N, 3) # f[*B,N,3]
        w1 = w[...,:2].sum(axis=-1) >= 1
        w[w1,:] = 1 - w[w1,:]
        w[...,2] = 1 - w[...,0] - w[...,1] # barycentric coord

        point = np.matmul(w[...,None,:], face) # f[*B,N,1,D]
        point = point.squeeze(-2)
        if self.return_barycentric:
            return point, f_idx, w
        return point

def sample_points(verts, faces, N, return_barycentric=False):
    sampler = UniformSurfaceSampler(verts, faces, return_barycentric)
    out = sampler.get(N)
    return out

