BrainShapeToolKit
=================

[[Technical Report](https://github.com/SSTDV-Project/BrainShapeToolKit/blob/master/docs/static/pdfs/tech-report.pdf)]
[[Synthetic dataset](https://github.com/SSTDV-Project/BrainShapeToolKit/tree/master/opendata/cortical_thickness)]

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

## Longitudinal shape synthesis
```bash
python ./synthesize_longitudinal.py --config ./configs/config_example_longitudinal.ini
```

## Additional files
* A template shape file containing simplified meshes from the brain atlas is required. Additional repository constructing the template shape will be available soon.


## Acknowledgement

> 이 논문은 2024년도 정부(과학기술정보통신부)의 재원으로 정보통신기획평가원의 지원을 받아 수행된 연구임 (No.00223446, 목적 맞춤형 합성데이터 생성 및 평가기술 개발)

> This work was supported by Institute for Information & communications Technology Promotion(IITP) grant funded by the Korea government(MSIT) (No.00223446, Development of object-oriented synthetic data generation and evaluation methods)
