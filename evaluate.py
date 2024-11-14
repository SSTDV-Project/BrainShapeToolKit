import os
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from pathlib import Path

from sdmetrics.reports.single_table import QualityReport

import argparse

def load_csv(path, **kwargs):
    tab = pd.read_csv(path, **kwargs)
    tab = tab.drop(tab.columns[0], axis=1)
    return tab

def sample_from_meanstd(mean, std, n_sample=None, K=10):
    cols = mean.columns.tolist()
    mean_np = mean.to_numpy()
    std_np = std.to_numpy()
    
    if n_sample is None:
        n_sample = np.random.randn(K, *mean_np.shape)
    sample = mean_np + n_sample * std_np
    sample = sample.reshape(-1, mean_np.shape[-1])

    sample = pd.DataFrame(sample, columns=cols)
    return sample

def KL_mc(p, q, n=100_000):
    points = p.resample(n)
    p_pdf = p.pdf(points)
    q_pdf = q.pdf(points)
    return np.log(p_pdf / q_pdf).mean()

def filter_nan(x):
    return x[~np.isnan(x)]

def evaluate_KLdiv(real, syn, num_samples=100_000):
    from scipy import stats
    real_np = real.to_numpy()
    syn_np = syn.to_numpy()

    cols = real_np.shape[-1]
    results = []
    for col in trange(cols):
        colname = real.columns[col]
        real_col = real_np[:,col]
        syn_col = syn_np[:,col]
        
        real_gk = stats.gaussian_kde(filter_nan(real_col))
        syn_gk = stats.gaussian_kde(syn_col)
        kldiv = KL_mc(syn_gk, real_gk, n=num_samples)
        # print(f'colname = {colname} : kl_div = {kldiv:.7f}')
        results.append({
            'Column': colname,
            'KL Divergence': kldiv,
        })
    return pd.DataFrame(results)

def _load_files(path_template: str):
    mean_path = path_template.format('mean')
    stdev_path = path_template.format('std')

    print(f'- mean data from `{mean_path}`')
    mean_pd = load_csv(mean_path)

    print(f'- stdev data from `{stdev_path}`')
    stdev_pd = load_csv(stdev_path)
    return mean_pd, stdev_pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_path', type=str)
    parser.add_argument('--syn_path', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--sample_k', type=int, default=10)
    parser.add_argument('--kldiv_numsamples', type=int, default=100_000)

    parser.add_argument('--metric', type=str, default='all', choices=['all', 'kscomp', 'kldiv'])

    args = parser.parse_args()

    GT_MEAN_PATH = args.gt_path.format('mean')
    GT_STDEV_PATH = args.gt_path.format('std')

    SYN_MEAN_PATH = args.syn_path.format('mean')
    SYN_STDEV_PATH = args.syn_path.format('std')

    RUN_KL_DIV = args.metric in ['all', 'kldiv']
    RUN_KS_COMP = args.metric in ['all', 'kscomp']

    print('Loading GT data')
    gt_mean, gt_stdev = _load_files(args.gt_path)

    print('Loading Synthetic data')
    syn_mean, syn_stdev = _load_files(args.syn_path)

    print('Sampling from mean/stdev')
    gt_sample = sample_from_meanstd(gt_mean, gt_stdev, K=args.sample_k)
    syn_sample = sample_from_meanstd(syn_mean, syn_stdev, K=args.sample_k)

    Path(args.output_path).mkdir(parents=True, exist_ok=True)
    gt_sample.to_csv(os.path.join(args.output_path, 'gt_sample.csv'))
    syn_sample.to_csv(os.path.join(args.output_path, 'syn_sample.csv'))

    if RUN_KL_DIV:
        print('Measuring KL divergence')
        kldiv_result = evaluate_KLdiv(gt_sample, syn_sample, num_samples=args.kldiv_numsamples)
        kldiv_result.to_csv(os.path.join(args.output_path, 'kldiv_result.csv'))

    if RUN_KS_COMP:
        print('Running KS test')
        cols = gt_sample.columns.tolist()
        pd_metadata = {
            "columns": {colname: {"sdtype": "numerical"} for colname in cols}
        }
        report = QualityReport()
        report.generate(gt_sample, syn_sample, pd_metadata)
        
        kscomp_result = report.get_details(property_name='Column Shapes')
        kscomp_result.to_csv(os.path.join(args.output_path, 'kscomp_result.csv'))


    print('')
    print('RESULT#####################################')
    if RUN_KS_COMP:
        mean_kscomp = kscomp_result['Score'].mean()
        kscomp_accepted = mean_kscomp >= 0.9
        print(f'* Column Average KS Complement = {mean_kscomp:.7f} ')
        print(f' - {"Accepted" if kscomp_accepted else "Rejected"} by the quality threshold (>= 0.9)')
    if RUN_KL_DIV:
        mean_kldiv = kldiv_result['KL Divergence'].mean()
        kldiv_accepted = mean_kldiv <= 0.03
        print(f'* Column Average KL Divergence = {mean_kldiv:.7f}')
        print(f' - {"Accepted" if kldiv_accepted else "Rejected"} by the quality threshold (<= 0.03)')
