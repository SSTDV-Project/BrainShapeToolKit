import os

import numpy as np
import nibabel as nib

from nibabel.freesurfer.io import read_geometry

CACHE_PATH = '.'

def load_surface(path):
    verts, faces, metadata = read_geometry(path, read_metadata=True)

    ## Alignment
    verts += metadata['cras']
    return verts, faces

def load_deform(path):
    forward_deform_nib = nib.load(path)
    forward_deform, affine_matrix = forward_deform_nib.get_fdata(), forward_deform_nib.affine
    return forward_deform, affine_matrix

def get_cached_value(cache_key, value_in_lambda, is_dict=True, quiet=True):
    cache_file = os.path.join(CACHE_PATH, cache_key)
    try:
        value = np.load(cache_file, allow_pickle=True)
        if is_dict:
            value = value.item()
        if not quiet:
            print(f'Cached value for "{cache_key}" loaded!')
        return value
    except FileNotFoundError:
        if not quiet:
            print(f'Cached value NOT FOUND for "{cache_key}": processing..')
        value = value_in_lambda()
        np.save(cache_file, value)
        return value

def idx2spatial(idx, affine):
    return idx.astype(np.float64) @ affine[:3, :3].T + affine[:3, 3]

def spatial2idx(spatial, affine):
    return np.linalg.solve(affine[:3, :3], (spatial - affine[:3, 3]).T).T

def trilinear(data, xyz):
    '''
    xyz: array with coordinates inside data
    data: 3d volume
    returns: interpolated data values at coordinates
    '''
    data_dims = len(data.shape)
    if data_dims == 3:
        data = data[...,None]
    ijk = xyz.astype(np.int64)
    i, j, k = ijk[:,0], ijk[:,1], ijk[:,2]
    i, j, k = np.clip(i, 0, data.shape[0]-2), np.clip(j, 0, data.shape[1]-2), np.clip(k, 0, data.shape[2]-2)
    V000 = data[ i   , j   ,  k   ].astype(np.int64)
    V100 = data[(i+1), j   ,  k   ].astype(np.int64)
    V010 = data[ i   ,(j+1),  k   ].astype(np.int64)
    V001 = data[ i   , j   , (k+1)].astype(np.int64)
    V101 = data[(i+1), j   , (k+1)].astype(np.int64)
    V011 = data[ i   ,(j+1), (k+1)].astype(np.int64)
    V110 = data[(i+1),(j+1),  k   ].astype(np.int64)
    V111 = data[(i+1),(j+1), (k+1)].astype(np.int64)
    xyz = xyz - ijk
    x, y, z = xyz[:,[0]], xyz[:,[1]], xyz[:,[2]]
    Vxyz = (V000 * (1 - x)*(1 - y)*(1 - z)
            + V100 * x * (1 - y) * (1 - z) +
            + V010 * (1 - x) * y * (1 - z) +
            + V001 * (1 - x) * (1 - y) * z +
            + V101 * x * (1 - y) * z +
            + V011 * (1 - x) * y * z +
            + V110 * x * y * (1 - z) +
            + V111 * x * y * z)
    if data_dims == 3:
        Vxyz = Vxyz[...,0]
    return Vxyz

def apply_deform(atlas_shape, forward_deform, affine_matrix):
    atlas_verts, atlas_faces = atlas_shape

    ## Project deformation
    verts_idx = spatial2idx(atlas_verts, affine_matrix)
    new_verts = trilinear(forward_deform, verts_idx)
    return new_verts, atlas_faces

def apply_deform_point(points, forward_defrom, affine_matrix):
    points_idx = spatial2idx(points, affine_matrix)
    new_points = trilinear(forward_defrom, points_idx)
    return new_points
