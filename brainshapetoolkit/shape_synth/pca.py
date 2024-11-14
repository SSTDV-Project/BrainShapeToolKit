import pickle
import numpy as np
from sklearn.decomposition import PCA

from ..utils import geometry, mvc

class PCAShapeSynthesizer:
    def __init__(self, template_shape=None, **kwargs):
        self.template_shape = template_shape
        self.template_EB = kwargs.get('template_EB', None)
        self.pca = kwargs.get('pca', None)
        self.trained_datapoints = kwargs.get('trained_datapoints', None)
        
        self.shape_keys = None
        if self.template_shape is not None:
            self.shape_keys = [*self.template_shape.keys()]
            if self.template_EB is None:
                self.template_EB = {
                    key: geometry.get_eigenbases(*template_shape[key])[1]
                    for key in self.shape_keys
                }
    
    def save(self, path):
        state = {
            'template_shape': self.template_shape,
            'template_EB': self.template_EB,
            'pca': self.pca,
            'trained_datapoints': self.trained_datapoints,
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f, protocol=5)
    
    @classmethod
    def load(cls, path):
        state = None
        with open(path, 'rb') as f:
            state = pickle.load(f)
        template_shape = state.pop('template_shape')
        out = PCAShapeSynthesizer(
            template_shape,
            **state,
        )
        return out
    
    def train(self, subjects_generator, K):
        """
        subject_generator: Python generator that yields surfaces {'pial': (V,F), 'white': (V,F) ...}, must be finite
        K: the number of principal components to use
        ---------------------------
        out: trained PCA model
        """
        deforms = (self._surfaces_to_deforms(surfaces) for surfaces in subjects_generator)
        spectvecs = (self._deforms_to_spectvec(d) for d in deforms)
        datapoints = np.stack([*spectvecs], axis=0)

        self.pca = PCA(n_components=K)
        self.trained_datapoints = self.pca.fit_transform(datapoints)
        return self.pca
    
    def sample(self, N=None, sample_latents=None):
        """
        N: the number of samples to generate
        ----------------------------
        out: [surface] * N, where surface = {key: (verts, faces) ...}
        """
        var = np.array(self.pca.explained_variance_)
        n_components = var.shape[0]

        if sample_latents is None:
            sample_latents = np.random.randn(N, n_components) * (var ** 0.5)
        sample_spects = self.pca.inverse_transform(sample_latents)
        sample_spects = [sample_spects[i] for i in range(sample_spects.shape[0])]
        sample_deforms = [self._spectvec_to_deforms(vec) for vec in sample_spects]
        sample_surfaces = [self._deforms_to_surfaces(d) for d in sample_deforms]
        return sample_surfaces
    
    def deform_points(self, sample_surfaces, template_points, torch_device=None):
        """
        sample_surfaces: {'pial': (V,F), ...}
        template_points: f[N, 3]
        ----------------------------
        out: f[N, 3]
        """
        deforms = self._surfaces_to_deforms(sample_surfaces)
        pial_deform = deforms['pial'] # f[V, 3]
        verts, faces = self.template_shape['pial']
        verts = verts[None,:,:]
        faces = faces[None,:,:]
        query = template_points[None,:,:]
        if torch_device is not None:
            import torch
            from ..utils import mvc_torch
            point_mvc = mvc_torch.mean_value_coordinates_3D(
                vertices = torch.from_numpy(verts).to(torch_device),
                faces = torch.from_numpy(faces).long().to(torch_device),
                query = torch.from_numpy(query).to(torch_device),
            )[0,:,:]
            point_deform = point_mvc @ torch.from_numpy(pial_deform).to(torch_device)
            point_deform = point_deform.cpu().numpy()
        else:
            point_mvc = mvc.mean_value_coordinates_3D(
                vertices=verts[None,:,:],
                faces=faces[None,:,:],
                query=template_points[None,:,:],
            )[0,:,:] # [N, V]
            point_deform = point_mvc @ pial_deform
        sample_points = template_points + point_deform
        return sample_points
    
    def _surfaces_to_deforms(self, subject_surfaces):
        deforms = {
            key: subject_surfaces[key][0] - self.template_shape[key][0]
            for key in self.shape_keys
        }
        return deforms

    def _deforms_to_surfaces(self, deforms):
        surfaces = {
            key: (verts + deforms[key], faces) 
            for key, (verts, faces) in self.template_shape.items()
        }
        return surfaces
    
    def _deforms_to_spectvec(self, deforms):
        spects = [
            geometry.cart2spect(deforms[key], self.template_EB[key])
            for key in self.shape_keys
        ]
        spectvecs = [
            spect.reshape(*spect.shape[:-2], -1)
            for spect in spects
        ]
        spectvec = np.concatenate(spectvecs, axis=-1)
        return spectvec
    
    def _spectvec_to_deforms(self, spectvec):
        sizes = [self.template_EB[key].shape[-1] * 3 for key in self.shape_keys]
        C = [0] + np.cumsum(sizes).tolist()
        spectvecs = [
            spectvec[..., C[i]:C[i+1]] 
            for i in range(len(self.shape_keys))
        ]
        spects = [
            vec.reshape(*vec.shape[:-1], -1, 3)
            for vec in spectvecs
        ]
        deforms = {
            key: geometry.spect2cart(spects[i], self.template_EB[key])
            for i, key in enumerate(self.shape_keys)
        }
        return deforms

