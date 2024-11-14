import pickle
import numpy as np

from ..utils import geometry, sampling

def point_distance(from_, to_):
    from scipy.spatial import KDTree
    tree = KDTree(to_)
    dd, _ = tree.query(from_, k=1)
    return dd

def project_to_tri(proj_surface, points, thickness):
    num_faces = proj_surface[1].shape[0]
    proj_tri_count = np.zeros((num_faces,), dtype=int)
    proj_tri_sum = np.zeros((num_faces,), dtype=float)
    proj_tri_sqsum = np.zeros((num_faces,), dtype=float)
    
    import trimesh
    proj_mesh = trimesh.Trimesh(*proj_surface)
    _, _, tri_id = trimesh.proximity.closest_point(proj_mesh, points)
    np.add.at(proj_tri_count, tri_id, 1)
    np.add.at(proj_tri_sum, tri_id, thickness)
    np.add.at(proj_tri_sqsum, tri_id, thickness ** 2)

    proj_tri_count[proj_tri_count == 0] += 1
    proj_tri_mean = proj_tri_sum / proj_tri_count
    proj_tri_std = ((proj_tri_sqsum / proj_tri_count) - proj_tri_mean ** 2) ** 0.5
    return proj_tri_mean, proj_tri_std

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
    num_vals = vals.shape[0]
    num_spect = centroid_EB.shape[1]
    
    parc_count = np.zeros((num_vals,), dtype=int)
    parc_spect = np.zeros((num_vals, num_spect), dtype=float)
    parc_thickness_mean = np.zeros((num_vals,), dtype=float)
    parc_thickness_std = np.zeros((num_vals,), dtype=float)

    import trimesh
    proj_mesh = trimesh.Trimesh(*proj_surface)
    _, _, tri_id = trimesh.proximity.closest_point(proj_mesh, parc_pos)

    # interpolants = [centroid_EB, thickness_mean, thickness_std]

    parc_pos_EB = centroid_EB[tri_id]
    parc_pos_thickness_mean = thickness_mean[tri_id]
    parc_pos_thickness_std = thickness_std[tri_id]

    parc_idx = np.searchsorted(vals, parc_val)
    np.add.at(parc_count, parc_idx, 1)
    np.add.at(parc_spect, parc_idx, parc_pos_EB)
    np.add.at(parc_thickness_mean, parc_idx, parc_pos_thickness_mean)
    np.add.at(parc_thickness_std, parc_idx, parc_pos_thickness_std)

    parc_spect /= parc_count[...,None]
    parc_thickness_mean /= parc_count
    parc_thickness_std /= parc_count
    return parc_spect, parc_thickness_mean, parc_thickness_std

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

        train_features, train_gt = self._get_io(projection_result, K=self.K)
        self.mean_model = self._get_model()
        self.std_model = self._get_model()
        print('Training...')
        self.mean_model.fit(train_features, train_gt[:,0])
        self.std_model.fit(train_features, train_gt[:,1])
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
            n_estimators=1_000_000,
            max_depth=1,
        )
