import os
import configparser
import numpy as np
import pandas as pd
from pathlib import Path
import brainshapetoolkit as bstk
from tqdm import tqdm

# Import the synthesizers (adjust import path as needed)
from brainshapetoolkit.shape_synth.pca import PCAShapeSynthesizer
from brainshapetoolkit.longitudinal_synth.mlm import MLMLongitudinalSynthesizer


def get_group_dirs(df):
    """Generate directory names from subject ID and months."""
    return df.apply(lambda row: f"{row['Subject']}/{row['Months']:02d}", axis=1).tolist()


def load_config(config_path):
    """Load configuration from INI file."""
    config = configparser.ConfigParser()
    config.read(config_path)
    return config


def prepare_metadata(df):
    """Prepare metadata DataFrame for MLM training."""
    metadata = df.copy()
    
    # Ensure proper column names (match notebook)
    metadata = metadata.rename(columns={
        'Subject': 'subject',
        'Months': 'months', 
        'Group': 'group',
        'Sex': 'sex',
        'Age': 'baseage'
    })
    
    # Calculate baseline age (age at month 0 for each subject)
    # The notebook does this by taking every other row
    metadata['baseage'] = metadata['baseage'].astype(float)
    
    # For each subject, propagate baseline age to all timepoints
    baseline_ages = {}
    for subject in metadata['subject'].unique():
        subject_data = metadata[metadata['subject'] == subject].sort_values('months')
        # Get age at first visit (month 0 or earliest)
        baseline_age = subject_data.iloc[0]['baseage']
        baseline_ages[subject] = baseline_age
    
    metadata['baseage'] = metadata['subject'].map(baseline_ages)
    
    return metadata


def main(config_path='config.ini'):
    """Main execution function."""
    
    # Load configuration
    config = load_config(config_path)
    
    # Extract paths
    template_atlas_path = config['FilePaths']['TEMPLATE_ATLAS']
    template_shapes_path = config['FilePaths']['TEMPLATE_SHAPES']
    dataset_path = config['FilePaths']['DATASET']
    dataset_csv_path = config['FilePaths']['DATASET_CSV']
    output_path = config['FilePaths']['OUTPUT']
    
    # Extract configs
    target_group = config['DatasetConfig']['TARGET_GROUP']
    num_samples = int(config['SynthesisConfig']['NUM_SAMPLES'])
    pca_components = int(config['ShapeSynthesisConfig']['PCA_COMPONENTS'])
    
    # Shape synthesis configs
    load_shape_model = int(config['ShapeSynthesisConfig']['LOAD_MODEL'])
    save_shape_model = int(config['ShapeSynthesisConfig']['SAVE_MODEL'])
    shape_model_path = config['ShapeSynthesisConfig'].get(
        'PRETRAINED_MODEL_PATH', 
        os.path.join(output_path, 'models', f'SHAPE-{target_group}.model')
    ).format(target_group)
    
    # Longitudinal synthesis configs
    load_long_model = int(config['LongitudinalSynthesisConfig'].get('LOAD_MODEL', '0'))
    save_long_model = int(config['LongitudinalSynthesisConfig'].get('SAVE_MODEL', '1'))
    long_model_path = config['LongitudinalSynthesisConfig'].get(
        'PRETRAINED_MODEL_PATH',
        os.path.join(output_path, 'models', f'SYNTH-{target_group}.model')
    ).format(target_group)
    
    # MLM formula and settings
    mlm_formula = config['LongitudinalSynthesisConfig'].get(
        'FORMULA', 
        'months * group + baseage + sex'
    )
    groups_col = config['LongitudinalSynthesisConfig'].get('GROUPS_COL', 'subject')
    
    # Create output directories
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, 'models'), exist_ok=True)
    
    print("="*60)
    print("Longitudinal Brain Shape Synthesis")
    print("="*60)
    print(f"Template Atlas: {template_atlas_path}")
    print(f"Template Shapes: {template_shapes_path}")
    print(f"Dataset: {dataset_path}")
    print(f"Dataset CSV: {dataset_csv_path}")
    print(f"Target Group: {target_group}")
    print(f"PCA Components: {pca_components}")
    print("="*60)
    
    # Load template
    print("\n[1/6] Loading template...")
    template = bstk.template.Template(template_atlas_path, template_shapes_path)
    target_shape_key = template.keys[3]  # ('simplified_shapes', 4) from notebook
    print(f"Target shape key: {target_shape_key}")
    
    # Load dataset metadata
    print("\n[2/6] Loading dataset metadata...")
    data_pd = pd.read_csv(dataset_csv_path)
    print(f"Loaded {len(data_pd)} observations from {data_pd['Subject'].nunique()} subjects")
    
    # Filter by target group if needed
    if target_group != 'ALL':
        data_pd = data_pd[data_pd['Group'] == target_group]
        print(f"Filtered to {len(data_pd)} observations for group {target_group}")
    
    # Create dataset object
    print("\n[3/6] Creating registration dataset...")
    dataset = bstk.dataset.RegistrationDataset(
        template,
        target_shape_key,
        dataset_path,
        dirs=get_group_dirs(data_pd),
        deform_filename='easyreg/fwd_field.mgz'
    )
    
    # Verify files exist
    if not dataset.check_files():
        raise FileNotFoundError("Some required dataset files are missing!")
    print("All dataset files verified")
    
    # Get template surfaces
    template_surfaces = dataset.get_surfaces(None, target_shape_key)
    
    # Step 1: Train or load PCA Shape Synthesizer
    print("\n[4/6] Setting up PCA Shape Synthesizer...")
    
    if load_shape_model and os.path.exists(shape_model_path):
        print(f"Loading pre-trained PCA model from: {shape_model_path}")
        pca_synthesizer = PCAShapeSynthesizer.load(shape_model_path)
    else:
        print("Training new PCA model...")
        pca_synthesizer = PCAShapeSynthesizer(template_shape=template_surfaces)
        
        # Create generator for training
        def surface_generator():
            for dirname in tqdm(dataset.dirs, desc="Loading surfaces"):
                yield dataset.get_surfaces(dirname, target_shape_key)
        
        pca_synthesizer.train(surface_generator(), K=pca_components)
        print(f"PCA trained with {pca_components} components")
        print(f"Explained variance: {pca_synthesizer.pca.explained_variance_ratio_.sum():.3f}")
        
        if save_shape_model:
            print(f"Saving PCA model to: {shape_model_path}")
            pca_synthesizer.save(shape_model_path)
    
    # Step 2: Train or load MLM Longitudinal Synthesizer
    print("\n[5/6] Setting up MLM Longitudinal Synthesizer...")
    
    if load_long_model and os.path.exists(long_model_path):
        print(f"Loading pre-trained MLM model from: {long_model_path}")
        mlm_synthesizer = MLMLongitudinalSynthesizer.load(long_model_path)
    else:
        print("Training new MLM model...")
        mlm_synthesizer = MLMLongitudinalSynthesizer(pca_synthesizer)
        
        # Prepare metadata
        metadata = prepare_metadata(data_pd)
        
        # Create generator for training (reuse surfaces)
        def surface_generator():
            for dirname in tqdm(dataset.dirs, desc="Loading surfaces for MLM"):
                yield dataset.get_surfaces(dirname, target_shape_key)
        
        # Train MLM
        mlm_synthesizer.train(
            surface_generator(),
            metadata=metadata,
            formula=mlm_formula,
            groups_col=groups_col,
            verbose=True
        )
        
        if save_long_model:
            print(f"Saving MLM model to: {long_model_path}")
            mlm_synthesizer.save(long_model_path)
    
    # Step 3: Generate synthetic trajectories
    print("\n[6/6] Generating synthetic trajectories...")
    
    # Example 1: Generate trajectory for a new synthetic subject
    print("\nExample 1: Synthetic CN Male, age 75, 24-month trajectory")
    months_trajectory = np.arange(0, 25, 3)  # 0, 3, 6, ..., 24 months
    
    synthetic_surfaces = mlm_synthesizer.synthesize_trajectory(
        subject_id='SYNTHETIC_CN_MALE_75',
        baseline_age=75.0,
        group='CN',
        sex='M',
        months=months_trajectory,
        subject_specific=False  # Population-level prediction
    )
    
    print(f"Generated {len(synthetic_surfaces)} timepoints")
    
    # Save synthetic trajectory (optional)
    trajectory_output = os.path.join(output_path, 'synthetic_trajectories')
    os.makedirs(trajectory_output, exist_ok=True)
    
    # Example: Save as numpy arrays
    for i, (month, surfaces) in enumerate(zip(months_trajectory, synthetic_surfaces)):
        month_dir = os.path.join(trajectory_output, 'SYNTHETIC_CN_MALE_75', f'{month:02d}')
        os.makedirs(month_dir, exist_ok=True)
        
        for key, (verts, faces) in surfaces.items():
            np.save(os.path.join(month_dir, f'{key}_verts.npy'), verts)
            np.save(os.path.join(month_dir, f'{key}_faces.npy'), faces)
    
    print(f"Saved synthetic trajectory to: {trajectory_output}")
    
    # Example 2: Generate multiple trajectories
    print("\nExample 2: Generating multiple synthetic subjects...")
    
    synthetic_subjects = []
    for group in ['CN', 'MCI', 'AD']:
        for sex in ['M', 'F']:
            for age in [65, 75, 85]:
                subject_id = f'SYNTHETIC_{group}_{sex}_{age}'
                surfaces = mlm_synthesizer.synthesize_trajectory(
                    subject_id=subject_id,
                    baseline_age=float(age),
                    group=group,
                    sex=sex,
                    months=np.array([0, 12, 24]),
                    subject_specific=False
                )
                synthetic_subjects.append({
                    'subject_id': subject_id,
                    'group': group,
                    'sex': sex,
                    'age': age,
                    'surfaces': surfaces
                })
    
    print(f"Generated {len(synthetic_subjects)} synthetic subject trajectories")
    
    # Save summary statistics
    print("\n[Summary] Model Statistics:")
    print(mlm_synthesizer.get_model_summary(component_idx=0))  # Show first component as example
    
    # Save metadata about synthetic subjects
    synth_metadata = pd.DataFrame([
        {
            'subject_id': s['subject_id'],
            'group': s['group'],
            'sex': s['sex'],
            'baseline_age': s['age'],
            'n_timepoints': len(s['surfaces'])
        }
        for s in synthetic_subjects
    ])
    synth_metadata.to_csv(
        os.path.join(output_path, f'synthetic_metadata_{target_group}.csv'),
        index=False
    )
    
    print("\n" + "="*60)
    print("Synthesis Complete!")
    print(f"Output saved to: {output_path}")
    print("="*60)
    
    return mlm_synthesizer, synthetic_subjects


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Longitudinal Brain Shape Synthesis')
    parser.add_argument('--config', type=str, default='config.ini',
                       help='Path to configuration file')
    args = parser.parse_args()
    
    mlm_synthesizer, synthetic_subjects = main(args.config)
