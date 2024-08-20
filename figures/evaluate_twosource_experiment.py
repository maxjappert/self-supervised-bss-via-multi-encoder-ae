import sys

import numpy as np
from sympy.physics.control.control_plots import plt

vae_name = 'toy'

results_basis = np.load(f'../results/results_basis_{vae_name}.npy')
results_basis_finetuned = np.load(f'../results/results_basis_finetuned_{vae_name}.npy')
results_basis_opti = np.load(f'../results/results_basis_opti_{vae_name}.npy')
results_bss = np.load(f'../results/results_bss_{vae_name}.npy')
results_bss_linear = np.load(f'../results/results_bss_linear_{vae_name}.npy')
results_noise = np.load(f'../results/results_noise_{vae_name}.npy')
results_prior_samples = np.load(f'../results/results_prior_samples_{vae_name}.npy')
results_nmf = np.load(f'../results/results_nmf_{vae_name}.npy')

results = np.stack((results_basis, results_basis_finetuned, results_basis_opti, results_bss, results_bss_linear, results_noise, results_prior_samples, results_nmf))

k = results_basis.shape[0]

metrics = {'sdr': 0,
           'isr': 1,
           'sir': 2,
           'sar': 3}

assert results.shape == (8, k, len(metrics.keys()), results.shape[3])

data_name = 'Toy' if vae_name.__contains__('toy') else 'MUSDB18'

for metric in metrics.keys():
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel('Method')
    ax1.set_ylabel(metric.upper())
    ax1.set_title(f'{metric.upper()} of the discussed Source Separation methods on {data_name} data', fontsize=16)

    res1 = ax1.boxplot(
        results[:, 0, metrics[metric], :].T, positions=np.arange(8)-0.2, widths=0.2,
        patch_artist=True, label='Source 2'
    )
    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(res1[element], color='k')

    for patch in res1['boxes']:
        patch.set_facecolor('tab:blue')


    # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    # ax2.set_ylabel('SDR', color='tab:orange')
    res2 = ax1.boxplot(
        results[:, 1, metrics[metric], :].T, positions=np.arange(8)+0.2, widths=0.2,
        patch_artist=True, label='Source 2'
    )
    ##from https://stackoverflow.com/a/41997865/2454357
    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(res2[element], color='k')

    for patch in res2['boxes']:
        patch.set_facecolor('tab:orange')

    # ax1.set_xlim([-0.55, 11.55])
    ax1.set_xticks(range(8))#
    ax1.set_xticklabels(['BASIS', 'BASIS Finetuned', 'BASIS Optimised', 'AE-BSS', 'Linear AE-BSS', 'Noise', 'Samples from Prior', 'NMF'])
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(f'small_twosource_{metric}_{vae_name}.png', dpi=300)

