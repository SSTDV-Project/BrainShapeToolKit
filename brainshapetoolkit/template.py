import numpy as np
import functools

class Template:
    def __init__(self, freesurfer_path, shape_data_or_path):
        self.freesurfer_path = freesurfer_path
        self.shape_data = shape_data_or_path
        if isinstance(shape_data_or_path, str):
            self.shape_data = np.load(shape_data_or_path, allow_pickle=True).item()

        self.keys = [
            ('original_shapes', None),
            *[(key, length) for key in ['simplified_shapes', 'convex_hulls', 'semi_hulls'] for length in self.shape_data[key]]
        ]

        self._shape_keys = self.shape_data['shape_keys']
    
    def get_surface(self, primary_key, secondary_key=None):
        return {
            'pial': self.get_pial(primary_key, secondary_key),
            'white': self.get_white(primary_key, secondary_key),
        }

    def _get_surface(self, primary_key, secondary_key, lh_shape_key, rh_shape_key):
        shape = self.shape_data[primary_key]
        if secondary_key:
            shape = shape[secondary_key]
        
        atlas_lh = shape[lh_shape_key]
        atlas_rh = shape[rh_shape_key]
    
        atlas_shapes = [atlas_lh, atlas_rh]
        stacked_verts = [atlas_shape[0] for atlas_shape in atlas_shapes]
        stacked_faces = []
        v_offset = 0
        for v, f in atlas_shapes:
            stacked_faces.append(f + v_offset)
            v_offset += v.shape[0]
        
        atlas_shape = np.concatenate(stacked_verts, axis=0), np.concatenate(stacked_faces, axis=0)
        return atlas_shape

    @functools.cache
    def get_pial(self, primary_key, secondary_key=None):
        """
        Pial mesh from given keys.
        If `primary_key` is given as integer, `self.keys[primary_key]` is used as keys.
        Otherwise, arguments are used as keys
        """
        if isinstance(primary_key, int):
            primary_key, secondary_key = self.keys[primary_key]

        return self._get_surface(
            primary_key, secondary_key, 
            self._shape_keys[0], self._shape_keys[1]
        )
        
    @functools.cache
    def get_white(self, primary_key, secondary_key=None):
        """
        White matter mesh from given keys.
        If `primary_key` is given as integer, `self.keys[primary_key]` is used as keys.
        Otherwise, arguments are used as keys
        """
        if isinstance(primary_key, int):
            primary_key, secondary_key = self.keys[primary_key]
        
        return self._get_surface(
            primary_key, secondary_key, 
            self._shape_keys[2], self._shape_keys[3]
        )
