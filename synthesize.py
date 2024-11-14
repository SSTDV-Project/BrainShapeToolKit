import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm, trange

import torch

import brainshapetoolkit as bstk
from brainshapetoolkit.shape_synth.pca import PCAShapeSynthesizer
from brainshapetoolkit.measurement_synth.cortical_thickness_torch import CorticalThicknessSynthesizer

from scipy import stats

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str)

args = parser.parse_args()

import configparser
config = configparser.ConfigParser()
config.read(args.config)

TEMPLATE_ATLAS_PATH = config['FilePaths']['TEMPLATE_ATLAS']
TEMPLATE_SHAPES_PATH = config['FilePaths']['TEMPLATE_SHAPES']
DATASET_PATH = config['FilePaths']['DATASET']
DATASET_CSV_PATH = config['FilePaths']['DATASET_CSV']


OUTPUT_PATH = config['FilePaths']['OUTPUT']
OUTPUT_MODELS = os.path.join(OUTPUT_PATH, 'models')
OUTPUT_SYNTHDATA = os.path.join(OUTPUT_PATH, 'synth_data')
OUTPUT_REALDATA = os.path.join(OUTPUT_PATH, 'real_data')

Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_SYNTHDATA).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_REALDATA).mkdir(parents=True, exist_ok=True)

TARGET_GROUP = config['DatasetConfig']['TARGET_GROUP']

GPU_INDEX = int(config['HardwareConfig']['GPU_INDEX'])

NUM_SAMPLES = int(config['SynthesisConfig']['NUM_SAMPLES'])

SHAPE_PCA_COMPONENTS = int(config['ShapeSynthesisConfig']['PCA_COMPONENTS'])
SHAPE_LOAD_MODEL = int(config['ShapeSynthesisConfig'].get('LOAD_MODEL', 0)) > 0
SHAPE_SAVE_MODEL = int(config['ShapeSynthesisConfig'].get('SAVE_MODEL', 0)) > 0
SHAPE_NOISE_RATIO = float(config['ShapeSynthesisConfig'].get('NOISE_RATIO', 1.0))

SYNTH_SPECTRAL_DIMS = int(config['MeasurementsSynthesisConfig']['SPECTRAL_DIMS'])
XGB_NUM_ESTIMATORS = int(config['MeasurementsSynthesisConfig'].get('XGB_NUM_ESTIMATORS', -1))
SYNTH_LOAD_MODEL = int(config['MeasurementsSynthesisConfig'].get('LOAD_MODEL', 0)) > 0
SYNTH_SAVE_MODEL = int(config['MeasurementsSynthesisConfig'].get('SAVE_MODEL', 0)) > 0

SHAPE_DEFAULT_MODEL_PATH = os.path.join(OUTPUT_MODELS, f'SHAPE-{TARGET_GROUP}.model')
SYNTH_DEFAULT_MODEL_PATH = os.path.join(OUTPUT_MODELS, f'SYNTH-{TARGET_GROUP}.model')

SHAPE_PRETRAINED_MODEL_PATH = config['ShapeSynthesisConfig'].get('PRETRAINED_MODEL_PATH', SHAPE_DEFAULT_MODEL_PATH)
SYNTH_PRETRAINED_MODEL_PATH = config['MeasurementsSynthesisConfig'].get('PRETRAINED_MODEL_PATH', SYNTH_DEFAULT_MODEL_PATH)

print(f"Shape model path will be : {SHAPE_PRETRAINED_MODEL_PATH}")
print(f"Cortical thickness model will be : {SYNTH_PRETRAINED_MODEL_PATH}")

OUTPUT_PATH_MEAN = os.path.join(OUTPUT_SYNTHDATA, f'synth_{TARGET_GROUP}_mean.csv')
OUTPUT_PATH_STD = os.path.join(OUTPUT_SYNTHDATA, f'synth_{TARGET_GROUP}_std.csv')
print(f"Mean cortical thickness output will be : {OUTPUT_PATH_MEAN}")
print(f"Std. cortical thickness output will be : {OUTPUT_PATH_STD}")

REALDATA_PATH_MEAN = os.path.join(OUTPUT_REALDATA, f'real_{TARGET_GROUP}_mean.csv')
REALDATA_PATH_STD = os.path.join(OUTPUT_REALDATA, f'real_{TARGET_GROUP}_std.csv')

device = torch.device(f'cuda:{GPU_INDEX}')

data_pd = pd.read_csv(DATASET_CSV_PATH)
datagroup = {
    'ALL': data_pd,
    'F60': data_pd[(data_pd['Sex'] == 'F') & (data_pd['Age'] < 75)],
    'F80': data_pd[(data_pd['Sex'] == 'F') & (data_pd['Age'] > 75)],
    'M60': data_pd[(data_pd['Sex'] == 'M') & (data_pd['Age'] < 75)],
    'M80': data_pd[(data_pd['Sex'] == 'M') & (data_pd['Age'] > 75)],
}

def get_group_dirs(group):
    return group['Image Data ID'].tolist()


template = bstk.template.Template(TEMPLATE_ATLAS_PATH, TEMPLATE_SHAPES_PATH)
TARGET_SHAPE_KEY = template.keys[12]

template_parc_dataset = bstk.dataset.ParcellationDataset(
    os.path.join(TEMPLATE_ATLAS_PATH, '..'),
    dirs = ['icbm152-freesurfer'],
)
template_parc_voxels = template_parc_dataset.get_parc_voxels(template_parc_dataset.dirs[0])

group_pd = datagroup[TARGET_GROUP]
dataset = bstk.dataset.RegistrationDataset(
    template, 
    TARGET_SHAPE_KEY, 
    DATASET_PATH, 
    dirs=get_group_dirs(group_pd),
)

parc_dataset = bstk.dataset.ParcellationDataset(
    DATASET_PATH,
    dirs=get_group_dirs(group_pd),
    dir2fspath_fn=lambda d: os.path.join(d, f'{d}_FS'),
)

print("Checking dataset...")
check_dataset = dataset.check_files()
check_parc_dataset = parc_dataset.check_files()

if not (check_dataset and check_parc_dataset):
    print("Some files are missing!")
    exit(1)


shape_gen = None
if SHAPE_LOAD_MODEL and os.path.exists(SHAPE_PRETRAINED_MODEL_PATH):
    print("Loading shape model...")
    shape_gen = PCAShapeSynthesizer.load(SHAPE_PRETRAINED_MODEL_PATH)
else:
    print("Building shape model...")
    template_surfaces = dataset.get_surfaces(None, TARGET_SHAPE_KEY)
    train_surfaces = [dataset.get_surfaces(dirname, TARGET_SHAPE_KEY) for dirname in tqdm(dataset.dirs)]

    shape_gen = PCAShapeSynthesizer(template_surfaces)
    shape_gen.train(train_surfaces, K=SHAPE_PCA_COMPONENTS)
    if SHAPE_SAVE_MODEL:
        print("Saving shape model...")
        shape_gen.save(SHAPE_DEFAULT_MODEL_PATH)

explained_variance_ratio = sum(shape_gen.pca.explained_variance_ratio_)
print(f"Shape model: explained_variance = {explained_variance_ratio:.7f}")




def deform_points_div(sample_surfaces, DIV=64):
    template_pos = template_parc_voxels[0]
    sample_pos = np.zeros_like(template_pos)
    for i in range(DIV):
        template_pos_div = template_pos[i::DIV,:]
        sample_pos[i::DIV,:] = shape_gen.deform_points(sample_surfaces, template_pos_div, torch_device=device)
    return sample_pos


def ks_test_ignore_nan(x, y):
    valid_x = x[~np.isnan(x)]
    valid_y = y[~np.isnan(y)]
    return stats.kstest(valid_x, valid_y)

template_project_surfaces = dataset.get_surfaces(None, TARGET_SHAPE_KEY)['pial']
ct_gen = None

print("Loading GT cortical thickness data...")
gt_stats = [parc_dataset.get_parc_stats(dirname) for dirname in parc_dataset.dirs]
gt_stats = np.stack([stat[0] for stat in gt_stats], axis=0), np.stack([stat[1] for stat in gt_stats], axis=0)


if SYNTH_LOAD_MODEL and os.path.exists(SYNTH_PRETRAINED_MODEL_PATH):
    print("Loading cortical thickness model...")
    ct_gen = CorticalThicknessSynthesizer.load(SYNTH_PRETRAINED_MODEL_PATH)
else:
    print("Building cortical thickness model...")
    ct_gen = CorticalThicknessSynthesizer(template_project_surfaces, K=SYNTH_SPECTRAL_DIMS)
    if XGB_NUM_ESTIMATORS > 0:
        def custom_model():
            import xgboost as xgb
            return xgb.XGBRegressor(
                device='cuda',
                n_estimators=XGB_NUM_ESTIMATORS,
                max_depth=1,
            )
        ct_gen._get_model = custom_model

    print("Loading datapoints...")
    recon_shapes = shape_gen.sample(sample_latents=shape_gen.trained_datapoints)
    recon_parc_voxels = [
        (deform_points_div(sample_surfaces), template_parc_voxels[1])
        for sample_surfaces in tqdm(recon_shapes)
    ]
    print("Processing datapoints...")
    
    train_datapoints = [ct_gen.make_datapoint(
        subject_surfaces=shape,
        projection_surface=shape['pial'],
        parc_voxels=parc_voxels,
        parc_stats=parc_dataset.get_parc_stats(dirname)
    ) for shape, parc_voxels, dirname in zip(recon_shapes, recon_parc_voxels, dataset.dirs)]

    print("Training cortical thickness model...")
    processed_datapoints = ct_gen.train(datapoints_iter=tqdm(train_datapoints))
    if SYNTH_SAVE_MODEL:
        print("Saving cortical thickness model...")
        ct_gen.save(SYNTH_DEFAULT_MODEL_PATH)


    print ("Testing for trained stats...")
    pred_stats = ct_gen.sample(processed_datapoints=processed_datapoints)
    pred_stats = tuple(stat.reshape(50,-1) for stat in pred_stats)

    ks = ks_test_ignore_nan(
        pred_stats[0].reshape(-1),
        gt_stats[0].reshape(-1),
    )
    print(ks)

print("Generating synthetic shapes...")

if SHAPE_NOISE_RATIO == 1.0:
    synth_shapes = shape_gen.sample(N=NUM_SAMPLES)
else:
    var = np.array(shape_gen.pca.explained_variance_)
    recon_latent = shape_gen.trained_datapoints
    synth_noise = np.random.randn(*recon_latent.shape) * (var ** 0.5)

    synth_latent = (1.0 - SHAPE_NOISE_RATIO) * recon_latent + SHAPE_NOISE_RATIO * synth_noise

    synth_shapes = shape_gen.sample(sample_latents=synth_latent)

print("Generating pseudo-parcellation results...")
synth_parc_voxels = [
    (deform_points_div(sample_surfaces), template_parc_voxels[1])
    for sample_surfaces in tqdm(synth_shapes)
]

print("Processing datapoints from synthetic shapes...")
syn_datapoints = [ct_gen.make_datapoint(
    subject_surfaces=shape,
    projection_surface=shape['pial'],
    parc_voxels=parc_voxels,
) for shape, parc_voxels in zip(synth_shapes, synth_parc_voxels)]
syn_proc_data = [ct_gen._process_datapoint(**datapoint) for datapoint in tqdm(syn_datapoints)]

print("Synthesizing measurements from cortical thickness model...")
syn_stats = ct_gen.sample(processed_datapoints=syn_proc_data)

print("Saving synthetic cortical thickness stats...")

names = [*bstk.dataset.APARC_VAL_TO_LUT_NAMES.items()]
cols = [name for idx, name in names]

syn_stats_mean_pd = pd.DataFrame(syn_stats[0], columns=cols)
syn_stats_std_pd = pd.DataFrame(syn_stats[1], columns=cols)

syn_stats_mean_pd.to_csv(OUTPUT_PATH_MEAN)
syn_stats_std_pd.to_csv(OUTPUT_PATH_STD)

gt_stats_mean_pd = pd.DataFrame(gt_stats[0], columns=cols)
gt_stats_std_pd = pd.DataFrame(gt_stats[1], columns=cols)

gt_stats_mean_pd.to_csv(REALDATA_PATH_MEAN)
gt_stats_std_pd.to_csv(REALDATA_PATH_STD)


print("Done!")
