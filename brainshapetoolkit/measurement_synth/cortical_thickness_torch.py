import pickle
import numpy as np
import torch

from torch_scatter import scatter

from ..utils import geometry, sampling, indexing_torch
from ..utils.closest_triangle_cuda import closest_triangle


def closest_triangle_on_surface(surface, points):
    verts, faces = [torch.from_numpy(arr).cuda() for arr in surface]
    bmin, bmax = verts.min(dim=0)[0], verts.max(dim=0)[0]
    bcenter = (bmin + bmax) * 0.5
    bextent = (bmax - bcenter).max()
    scale = 1. / bextent

    faces = faces.long()
    vf = indexing_torch.batch_indexing(verts, faces, dim=-2)
    points = torch.from_numpy(points).cuda()
    
    vf = (vf - bcenter) * scale
    points = (points - bcenter) * scale
    return closest_triangle(vf, points)

def point_distance(from_, to_):
    from scipy.spatial import KDTree
    # from_, to_ = cp.array(from_), cp.array(to_)
    tree = KDTree(to_)
    dd, _ = tree.query(from_, k=1)
    return dd

def project_to_tri(proj_surface, points, thickness):
    num_faces = proj_surface[1].shape[0]
    thickness_torch = torch.from_numpy(thickness).cuda()
    proj_tri_mean = torch.zeros((num_faces,), dtype=float).cuda()
    proj_tri_sqmean = torch.zeros((num_faces,), dtype=float).cuda()
    
    _,_, idx_torch = closest_triangle_on_surface(proj_surface, points)

    proj_tri_mean = scatter(thickness_torch, idx_torch, dim=0, out=proj_tri_mean, reduce='mean')
    proj_tri_sqmean = scatter(thickness_torch ** 2, idx_torch, dim=0, out=proj_tri_sqmean, reduce='mean')
    proj_tri_std = (proj_tri_sqmean - proj_tri_mean**2) ** 0.5
    return proj_tri_mean.cpu().numpy(), proj_tri_std.cpu().numpy()

def compute_thickness(surface, project_surface=None, N=300_000):
    pial_points = sampling.sample_points(*surface['pial'], N)
    white_points = sampling.sample_points(*surface['white'], N)
    
    pial_thickness = point_distance(pial_points, white_points)
    white_thickness = point_distance(white_points, pial_points)

    if project_surface is not None:
        return project_to_tri(
            project_surface, 
            np.concatenate([pial_points, white_points], axis=0), 
            np.concatenate([pial_thickness, white_thickness], axis=0)
        )

def surface_to_parc(proj_surface, centroid_EB, thickness, parc_voxels):
    thickness_mean, thickness_std = thickness
    parc_pos, parc_val = parc_voxels
    vals = np.unique(parc_val)
    vals.sort()
    vals = torch.from_numpy(vals).cuda()
    num_vals = vals.shape[0]
    num_spect = centroid_EB.shape[1]
    
    parc_spect = torch.zeros((num_vals, num_spect), dtype=float).cuda()
    parc_thickness_mean = torch.zeros((num_vals,), dtype=float).cuda()
    parc_thickness_std = torch.zeros((num_vals,), dtype=float).cuda()

    _,_, tri_id = closest_triangle_on_surface(proj_surface, parc_pos)

    # interpolants = [centroid_EB, thickness_mean, thickness_std]
    centroid_EB = torch.from_numpy(centroid_EB).cuda()
    thickness_mean = torch.from_numpy(thickness_mean).cuda()
    thickness_std = torch.from_numpy(thickness_std).cuda()

    parc_pos_EB = centroid_EB[tri_id]
    parc_pos_thickness_mean = thickness_mean[tri_id]
    parc_pos_thickness_std = thickness_std[tri_id]

    parc_idx = torch.searchsorted(vals, torch.from_numpy(parc_val).cuda())
    parc_spect = scatter(parc_pos_EB, parc_idx, dim=0, out=parc_spect, reduce='mean')
    parc_thickness_mean = scatter(parc_pos_thickness_mean, parc_idx, dim=0, out=parc_thickness_mean, reduce='mean')
    parc_thickness_std = scatter(parc_pos_thickness_std, parc_idx, dim=0, out=parc_thickness_std, reduce='mean')

    return parc_spect.cpu().numpy(), parc_thickness_mean.cpu().numpy(), parc_thickness_std.cpu().numpy()

class CorticalThicknessSynthesizer:
    def __init__(self, template_shape, K=32, N=300_000, **kwargs):
        self.projection_surface = template_shape
        self.projection_EB = kwargs['projection_EB'] if 'projeciton_EB' in kwargs else geometry.get_eigenbases(*self.projection_surface)[1]
        self.K = K
        self.N = N

        self.mean_model = kwargs.get('mean_model', None)
        self.std_model = kwargs.get('std_model', None)
    
    def save(self, path):
        state = {
            'projection_surface': self.projection_surface,
            'projection_EB': self.projection_EB,
            'K': self.K,
            'N': self.N,
            'mean_model': self.mean_model,
            'std_model': self.std_model,
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f, protocol=5)
    
    @classmethod
    def load(cls, path):
        state = None
        with open(path, 'rb') as f:
            state = pickle.load(f)
        out = CorticalThicknessSynthesizer(
            state['projection_surface'],
            K=state['K'],
            N=state['N'],
            projection_EB=state['projection_EB'],
            mean_model=state['mean_model'],
            std_model=state['std_model'],
        )
        return out

    def make_datapoint(self, subject_surfaces, projection_surface, parc_voxels, parc_stats=None):
        return {
            'subject_surfaces': subject_surfaces,
            'projection_surface': projection_surface,
            'parc_voxels': parc_voxels,
            'parc_stats': parc_stats,
        }

    def train(self, datapoints_iter=None, projection_result=None):
        if projection_result is None:
            assert datapoints_iter is not None
            projection_result = []
            for datapoints in datapoints_iter:
                projection_result.append(self._process_datapoint(**datapoints))

        self.last_processed_data = projection_result
        train_features, train_gt = self._get_io(projection_result, K=self.K)
        self.mean_model = self._get_model()
        self.std_model = self._get_model()
        print('Training...')
        self._train_model(self.mean_model, train_features, train_gt[:,0])
        self._train_model(self.std_model, train_features, train_gt[:,1])
        # self.mean_model.fit(train_features, train_gt[:,0])
        # self.std_model.fit(train_features, train_gt[:,1])
        print('Done!')
        
        return projection_result
        
    def sample(self, datapoints_iter=None, processed_datapoints=None):
        if processed_datapoints is None:
            processed_datapoints = [self._process_datapoint(**datapoint) for datapoint in datapoints_iter]
        num_points = len(processed_datapoints)
        features, _ = self._get_io(processed_datapoints, K=self.K, exclude_gt=True)
        
        pred_mean = self.mean_model.predict(features).reshape(num_points, -1)
        pred_std = self.std_model.predict(features).reshape(num_points, -1)
        return pred_mean, pred_std

    def _process_datapoint(self, subject_surfaces, projection_surface, parc_voxels, parc_stats=None):
        # print(f'Computing thickness...')
        thickness = compute_thickness(subject_surfaces, projection_surface, N=self.N)

        # print(f'Projecting thickness to parcellations...')
        centroid_EB = self.projection_EB[projection_surface[1]].mean(axis=1)
        parc_spect, *parc_thickness_stats = surface_to_parc(projection_surface, centroid_EB, thickness, parc_voxels)

        return {
            'parc_thickness': parc_thickness_stats,
            'parc_spect': parc_spect,
            'gt_stats': parc_stats
        }

    def _get_io(self, results, K, exclude_gt=False):
        syn_thickness = np.concatenate([np.stack(result['parc_thickness'], axis=-1) for result in results])
        syn_spect = np.concatenate([result['parc_spect'][:,2:2+K] for result in results])
        syn_features = np.concatenate([syn_thickness, syn_spect], axis=1)
        if exclude_gt:
            return syn_features, None

        gt_thickness = np.concatenate([np.stack(result['gt_stats'], axis=-1) for result in results])
        return syn_features, gt_thickness

    def _get_model(self):
        import xgboost as xgb
        return xgb.XGBRegressor(
            n_estimators=300_000,
            max_depth=2,
        )

    def _train_model(self, model, features, out):
        valid_vals = (~np.isnan(out)) &  (out > 0)
        valid_features = features[valid_vals,:]
        valid_out = out[valid_vals]
        model.fit(valid_features, valid_out)
        return model
