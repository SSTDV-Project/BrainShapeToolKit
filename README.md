BrainShapeToolKit verification 2024
===================================

## Install dependencies
```bash
$ conda env create -f docker/environment.yml
$ conda activate bstk
```

## Data synthesis
```bash
python ./synthesize.py --config ./configs/config_example.ini
```

## Synthetic data evaluation
```bash
python ./evaluate.py --gt_path ./data/real_data/real_ALL_{}.csv --syn_path ./data/synth_data/synth_ALL_{}.csv --output_path ./data/eval/
```

## Additional files
* A template shape file containing simplified meshes from the brain atlas is required. Additional repository constructing the template shape will be available soon.
