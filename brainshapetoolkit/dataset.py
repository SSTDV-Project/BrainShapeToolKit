import numpy as np
import os
import functools
import nibabel as nib
import pandas as pd

from .utils.misc import apply_deform, idx2spatial
from .utils.indexing import batch_grid_indexing
from .template import Template

FWD_FIELD_NAME = 'fwd_field.mgz'

PARC_NAMES = {
    1: 'bankssts',
    2: 'caudalanteriorcingulate',
    3: 'caudalmiddlefrontal',
    5: 'cuneus',
    6: 'entorhinal',
    7: 'fusiform',
    8: 'inferiorparietal',
    9: 'inferiortemporal',
    10: 'isthmuscingulate',
    11: 'lateraloccipital',
    12: 'lateralorbitofrontal',
    13: 'lingual',
    14: 'medialorbitofrontal',
    15: 'middletemporal',
    16: 'parahippocampal',
    17: 'paracentral',
    18: 'parsopercularis',
    19: 'parsorbitalis',
    20: 'parstriangularis',
    21: 'pericalcarine',
    22: 'postcentral',
    23: 'posteriorcingulate',
    24: 'precentral',
    25: 'precuneus',
    26: 'rostralanteriorcingulate',
    27: 'rostralmiddlefrontal',
    28: 'superiorfrontal',
    29: 'superiorparietal',
    30: 'superiortemporal',
    31: 'supramarginal',
    32: 'frontalpole',
    33: 'temporalpole',
    34: 'transversetemporal',
    35: 'insula',
}

TOTAL_PARCS = len(PARC_NAMES) * 2

APARC_VAL_TO_NAMES = {
    **{1000 + idx: name for idx, name in PARC_NAMES.items()},
    **{2000 + idx: name for idx, name in PARC_NAMES.items()},
}

APARC_VAL_TO_LUT_NAMES = {
    **{1000 + idx: f'ctx-lh-{name}' for idx, name in PARC_NAMES.items()},
    **{2000 + idx: f'ctx-rh-{name}' for idx, name in PARC_NAMES.items()},
}

class RegistrationDataset:
    def __init__(self, template: Template, atlas_key, reg_root, dirs=None, deform_filename=FWD_FIELD_NAME):
        self.template = template
        self.atlas_key = atlas_key
        self.root = reg_root
        self.deform_filename = deform_filename

        if dirs is None:
            dirs = os.listdir(reg_root)
        self.dirs = dirs
    
    def _get_deform_filepath(self, dir):
        return os.path.join(self.root, dir, self.deform_filename)

    def check_files(self):
        all_checked = True
        for dir in self.dirs:
            path = self._get_deform_filepath(dir)
            if not os.path.isfile(path):
                print(f'File not found! ({path})')
                all_checked = False
        return all_checked

    @functools.cache
    def get_deform(self, dir):
        path = self._get_deform_filepath(dir)
        deform_nib = nib.load(path)
        deform, affine_matrix = deform_nib.get_fdata(), deform_nib.affine
        return deform, affine_matrix
    
    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, idx):
        dir = self.dirs[idx]
        return self.get_surfaces(dir)

    def iter_surfaces(self):
        for dir in self.dirs:
            yield dir, self.get_surfaces(dir, self.atlas_key)
    
    @functools.cache
    def get_surfaces(self, dir, atlas_key=None):
        if atlas_key is None:
            atlas_key = self.atlas_key
        template_shape = self.template.get_surface(*atlas_key)

        if dir is None:
            return template_shape
        
        deforms = self.get_deform(dir)

        deformed_shape = {
            key: apply_deform(surface, *deforms) for key, surface in template_shape.items()
        }

        return deformed_shape

class ParcellationDataset:
    def __init__(self, data_root, dirs=None, dir_fspath=None, dir2fspath_fn=None, parc_names=None):
        """
        data_root: path_like
        dirs: list[str], relevant subdirectories within data_root, all if set None
        dir_path: dict[str, path_like], path to freesurfer output for each dirname
        dir2path_fn: str -> path_like
        """
        self.root = data_root

        if dirs is None:
            dirs = os.listdir(data_root)
        self.dirs = dirs

        if dir2fspath_fn is not None:
            dir_fspath = {dirname: dir2fspath_fn(dirname) for dirname in self.dirs}
        if dir_fspath is None:
            dir_fspath = {dirname: dirname for dirname in self.dirs}

        self.dir_fspath = dir_fspath

    def _get_paths(self, dirname):
        fs_path = os.path.join(self.root, self.dir_fspath[dirname])
        lh_path = os.path.join(fs_path, 'stats', 'lh.aparc.stats')
        rh_path = os.path.join(fs_path, 'stats', 'rh.aparc.stats')
        parc_path = os.path.join(fs_path, 'mri', 'aparc+aseg.mgz')
        return lh_path, rh_path, parc_path

    def check_files(self):
        all_checked = True
        for dir in self.dirs:
            paths = self._get_paths(dir)
            for path in paths:
                if not os.path.isfile(path):
                    print(f'File not found! ({path})')
                    all_checked = False
        return all_checked

    @functools.cache
    def get_parc_stats(self, dirname):
        lh_path, rh_path, _ = self._get_paths(dirname)

        mean, std = [np.nan] * TOTAL_PARCS, [np.nan] * TOTAL_PARCS
        for path, offset in zip([lh_path, rh_path], [0, len(PARC_NAMES)]):
            tab = pd.read_csv(path, sep='\s+', comment='#', header=None)
            tab_names = tab[0].tolist()
            tab_means = tab[4].tolist()
            tab_stds = tab[5].tolist()
            dict_means = {name: val for name, val in zip(tab_names, tab_means)}
            dict_stds = {name: val for name, val in zip(tab_names, tab_stds)}
            for idx, name in enumerate(PARC_NAMES.values()):
                mean[idx + offset] = dict_means.get(name, np.nan)
                std[idx + offset] = dict_stds.get(name, np.nan)
        mean, std = np.array(mean), np.array(std)
        # print(f"{dirname = }, {mean.shape = }, {std.shape = }")
        return mean, std
    
    @functools.cache
    def get_parc_voxels(self, dirname):
        _, _, parc_path = self._get_paths(dirname)
        parc_result = nib.load(parc_path)
        dims = parc_result.shape
        idx = np.stack(np.mgrid[:dims[0], :dims[1], :dims[2]], axis=-1).reshape(-1,3)
        pos = idx2spatial(idx, parc_result.affine)
        val = batch_grid_indexing(parc_result.get_fdata(), idx, dim_start=0, dim_end=3)

        parc_filter = ((val > 1000) & (val < 1099)) | ((val > 2000) & (val < 2099))
        pos = pos[parc_filter,:]
        val = val[parc_filter]
        return pos, val
    

