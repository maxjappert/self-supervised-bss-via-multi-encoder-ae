import sys

import numpy as np
from mir_eval.chord import evaluate
from sympy.physics.control.control_plots import plt

evaluated_sigmas = np.load('../results/sigma_Ls_evaluated.npy')
basis_sigma_l = np.load('../results/basis_sigma_toy_experiment_results.npy')

num_evaluated = basis_sigma_l.shape[1]

data_for_plot = []
positions = []
labels = []

jump_distance = 1

metrics = {'sdr': 0,
           'isr': 1,
           'sir': 2,
           'sar': 3}


for metric in metrics.keys():
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel('$\sigma_L$')
    ax1.set_ylabel(metric.upper())
    ax1.set_title(f'Influence of BASIS $\sigma_L$ on {metric.upper()}', fontsize=16)

    res1 = ax1.boxplot(
        basis_sigma_l[0:20, 0, metrics[metric], :].T, positions=np.arange(len(evaluated_sigmas)) - 0.2, widths=0.2,
        patch_artist=True, label='Source 1'
    )
    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(res1[element], color='k')

    for patch in res1['boxes']:
        patch.set_facecolor('tab:blue')


    # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    # ax2.set_ylabel('SDR', color='tab:orange')
    res2 = ax1.boxplot(
        basis_sigma_l[0:20, 1, metrics[metric], :].T, positions=np.arange(len(evaluated_sigmas)) + 0.2, widths=0.2,
        patch_artist=True, label='Source 2'
    )
    ##from https://stackoverflow.com/a/41997865/2454357
    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(res2[element], color='k')

    for patch in res2['boxes']:
        patch.set_facecolor('tab:orange')

    # ax1.set_xlim([-0.55, 11.55])
    ax1.set_xticks(range(len(evaluated_sigmas)))#
    ax1.set_xticklabels(evaluated_sigmas)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(f'sigma_toy_experiment_{metric}.png', dpi=300)

