import numpy as np
from sklearn.preprocessing import normalize

PI = np.pi

def check_values(arr):
    """return true if tensor doesn't contain NaN or Inf"""
    return not (np.any(np.isnan(arr)).item() or np.any(np.isinf(arr)).item())

def mean_value_coordinates_3D(vertices, faces, query, verbose=False):
    """
    Tao Ju et.al. MVC for 3D triangle meshes
    params:
        vertices (B,N,3)
        faces    (B,F,3)
        query    (B,P,3)
    return:
        wj       (B,P,N)
    """
    B, F, _ = faces.shape
    _, P, _ = query.shape
    _, N, _ = vertices.shape
    # u_i = p_i - x (B,P,N,3)
    uj = vertices[:,None,:,:] - query[:,:,None,:]
    # \|u_i\| (B,P,N,1)
    dj = np.linalg.norm(uj, axis=-1, keepdims=True)
    uj = uj / dj
    # gather triangle B,P,F,3,3
    # ui = torch.gather(uj.unsqueeze(2).expand(-1,-1,F,-1,-1),
    #                    3,
    #                    faces.unsqueeze(1).unsqueeze(-1).expand(-1,P,-1,-1,3))
    ui = np.take_along_axis(np.broadcast_to(uj[:,:,None,:,:], (B,P,F,N,3)),
                            np.broadcast_to(faces[:,None,:,:,None], (B,P,F,3,3)),
                            3)
    # li = \|u_{i+1}-u_{i-1}\| (B,P,F,3)
    li = np.linalg.norm(ui[:,:,:,[1, 2, 0],:] - ui[:, :, :,[2, 0, 1],:], axis=-1)
    eps = 2e-5
    li = np.clip(li, -(2 - eps), 2 - eps)
    # li = torch.where(li>=2, li-(li.detach()-(2-eps)), li)
    # li = torch.where(li<=-2, li-(li.detach()+(2-eps)), li)
    
    # asin(x) is inf at +/-1
    # θi =  2arcsin[li/2] (B,P,F,3)
    theta_i = 2*np.arcsin(li/2)
    assert(check_values(theta_i))
    
    # B,P,F,1
    h = np.sum(theta_i, axis=-1, keepdims=True)/2
    # wi← sin[θi]d{i−1}d{i+1}
    # (B,P,F,3) ci ← (2sin[h]sin[h−θi])/(sin[θ_{i+1}]sin[θ_{i−1}])−1
    ci = 2*np.sin(h)*np.sin(h-theta_i)/(np.sin(theta_i[:,:,:,[1, 2, 0]])*np.sin(theta_i[:,:,:,[2, 0, 1]]))-1

    # NOTE: because of floating point ci can be slightly larger than 1, causing problem with sqrt(1-ci^2)
    # NOTE: sqrt(x)' is nan for x=0, hence use eps
    eps = 1e-5
    ci = np.clip(ci, -1, 1)
    # ci = torch.where(ci>=1, ci-(ci.detach()-(1-eps)), ci)
    # ci = torch.where(ci<=-1, ci-(ci.detach()+(1-eps)), ci)
    # si← sign[det[u1,u2,u3]]sqrt(1-ci^2)
    # (B,P,F)*(B,P,F,3)

    si = np.sign(np.linalg.det(ui))[...,None]*np.sqrt(1-ci**2)  # sqrt gradient nan for 0
    assert(check_values(si))
    # (B,P,F,3)
    # di = torch.gather(dj.unsqueeze(2).squeeze(-1).expand(-1,-1,F,-1), 3,
    #                   faces.unsqueeze(1).expand(-1,P,-1,-1))
    di = np.take_along_axis(np.broadcast_to(dj[:,:,None,:,0], (B,P,F,N)),
                            np.broadcast_to(faces[:,None,:,:], (B,P,F,3)),
                            3)
    assert(check_values(di))
    # if si.requires_grad:
    #     vertices.register_hook(save_grad("mvc/dv"))
    #     li.register_hook(save_grad("mvc/dli"))
    #     theta_i.register_hook(save_grad("mvc/dtheta"))
    #     ci.register_hook(save_grad("mvc/dci"))
    #     si.register_hook(save_grad("mvc/dsi"))
    #     di.register_hook(save_grad("mvc/ddi"))

    # wi← (θi −c[i+1]θ[i−1] −c[i−1]θ[i+1])/(disin[θi+1]s[i−1])
    # B,P,F,3
    # CHECK is there a 2* in the denominator
    wi = (theta_i-ci[:,:,:,[1,2,0]]*theta_i[:,:,:,[2,0,1]]-ci[:,:,:,[2,0,1]]*theta_i[:,:,:,[1,2,0]])/(di*np.sin(theta_i[:,:,:,[1,2,0]])*si[:,:,:,[2,0,1]])
    # if ∃i,|si| ≤ ε, set wi to 0. coplaner with T but outside
    # ignore coplaner outside triangle
    # alternative check
    # (B,F,3,3)
    # triangle_points = torch.gather(vertices.unsqueeze(1).expand(-1,F,-1,-1), 2, faces.unsqueeze(-1).expand(-1,-1,-1,3))
    # # (B,P,F,3), (B,1,F,3) -> (B,P,F,1)
    # determinant = dot_product(triangle_points[:,:,:,0].unsqueeze(1)-query.unsqueeze(2),
    #                           torch.cross(triangle_points[:,:,:,1]-triangle_points[:,:,:,0],
    #                                       triangle_points[:,:,:,2]-triangle_points[:,:,:,0], dim=-1).unsqueeze(1), dim=-1, keepdim=True).detach()
    # # (B,P,F,1)
    # sqrdist = determinant*determinant / (4 * sqrNorm(torch.cross(triangle_points[:,:,:,1]-triangle_points[:,:,:,0], triangle_points[:,:,:,2]-triangle_points[:,:,:,0], dim=-1), keepdim=True))

    wi = np.where(np.any(np.abs(si) <= 1e-5, axis=-1, keepdims=True), 0, wi)
    # wi = torch.where(sqrdist <= 1e-5, torch.zeros_like(wi), wi)

    # if π −h < ε, x lies on t, use 2D barycentric coordinates
    # inside triangle
    inside_triangle = (PI-h).squeeze(-1)<1e-4
    # set all F for this P to zero
    wi = np.where(np.any(inside_triangle, axis=-1, keepdims=True)[...,None], 0, wi)
    # CHECK is it di https://www.cse.wustl.edu/~taoju/research/meanvalue.pdf or li http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.516.1856&rep=rep1&type=pdf
    wi = np.where(np.broadcast_to(inside_triangle[...,None],(*inside_triangle.shape,wi.shape[-1])), np.sin(theta_i)*di[:,:,:,[2,0,1]]*di[:,:,:,[1,2,0]], wi)

    # sum over all faces face -> vertex (B,P,F*3) -> (B,P,N)
    # wj = scatter_add(wi.reshape(B,P,-1).contiguous(), faces.unsqueeze(1).expand(-1,P,-1,-1).reshape(B,P,-1), 2, out_size=(B,P,N))
    wj = np.zeros((B,P,N), dtype=wi.dtype)
    np.add.at(wj, 
              (np.arange(B)[:,None,None], np.arange(P)[None,:,None], np.broadcast_to(faces[:,None,:,:], (B,P,F,3)).reshape((B,P,-1))), 
              wi.reshape(B,P,-1))

    # close to vertex (B,P,N)
    close_to_point = dj.squeeze(-1) < 1e-8
    # set all F for this P to zero
    wj = np.where(np.any(close_to_point, axis=-1, keepdims=True), np.zeros_like(wj), wj)
    wj = np.where(close_to_point, np.ones_like(wj), wj)

    # (B,P,1)
    sumWj = np.sum(wj, axis=-1, keepdims=True)
    sumWj = np.where(sumWj==0, np.ones_like(sumWj), sumWj)

    wj_normalised = wj / sumWj
    # if wj.requires_grad:
    #     saved_variables["mvc/wi"] = wi
    #     wi.register_hook(save_grad("mvc/dwi"))
    #     wj.register_hook(save_grad("mvc/dwj"))
    if verbose:
        return wj_normalised, wi
    else:
        return wj_normalised