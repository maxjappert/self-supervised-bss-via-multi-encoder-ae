import sys

import numpy as np
from sympy.physics.control.control_plots import plt

results_basis = np.load('../results/results_basis.npy')
results_bss = np.load('../results/results_bss.npy')
results_bss_linear = np.load('../results/results_bss_linear.npy')
results_noise = np.load('../results/results_noise.npy')
results_prior_samples = np.load('../results/results_prior_samples.npy')
results_nmf = np.load('../results/results_nmf.npy')

results = np.stack((results_basis, results_bss, results_bss_linear, results_noise, results_prior_samples, results_nmf))

k = results_basis.shape[0]

metrics = {'sdr': 0,
           'isr': 1,
           'sir': 2,
           'sar': 3}

assert results.shape == (6, k, len(metrics.keys()), 900)

for metric in metrics.keys():
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel('Method')
    ax1.set_ylabel(metric.upper())
    ax1.set_title(f'{metric.upper()} of the discussed Source Separation Approaches', fontsize=16)

    res1 = ax1.boxplot(
        results[:, 0, metrics[metric], :].T, positions=np.arange(6)-0.2, widths=0.2,
        patch_artist=True, label='Source 1'
    )
    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(res1[element], color='k')

    for patch in res1['boxes']:
        patch.set_facecolor('tab:blue')


    # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    # ax2.set_ylabel('SDR', color='tab:orange')
    res2 = ax1.boxplot(
        results[:, 1, metrics[metric], :].T, positions=np.arange(6)+0.2, widths=0.2,
        patch_artist=True, label='Source 2'
    )
    ##from https://stackoverflow.com/a/41997865/2454357
    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(res2[element], color='k')

    for patch in res2['boxes']:
        patch.set_facecolor('tab:orange')

    # ax1.set_xlim([-0.55, 11.55])
    ax1.set_xticks(range(6))#
    ax1.set_xticklabels(['BASIS', 'AE-BSS', 'Linear AE-BSS', 'Noise', 'Samples from Prior', 'NMF'])
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(f'weight_experiment_{metric}.png', dpi=300)

