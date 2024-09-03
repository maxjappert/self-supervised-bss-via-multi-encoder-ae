import sys

import numpy as np
from sympy.physics.control.control_plots import plt

vae_name = 'toy'

results_basis = np.load(f'../results/results_basis_rochester_novideo_128.npy')
results_basis_video = np.load(f'../results/results_basis_rochester_video_128.npy')

results = np.stack((results_basis, results_basis_video))
results = np.mean(results, axis=1)

k = 2

metrics = {'sdr': 0,
           'isr': 1,
           'sir': 2,
           'sar': 3}

assert results.shape == (2, len(metrics.keys()), results.shape[2])

data_name = 'Toy' if vae_name.__contains__('toy') else 'MUSDB18'

for metric in metrics.keys():
    fig = plt.figure(figsize=(10, 6))

    plt.ylabel(metric.upper())
    plt.title(f'{metric.upper()} Comparison of Using Video with $\\beta = 128$ vs. No Video', fontsize=16)

    res1 = plt.boxplot(
        results[:, metrics[metric], :].T
    )

    # ax1.set_xlim([-0.55, 11.55])
    plt.xticks([1, 2], ['No Video', 'Video'])
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(f'video_experiment_{metric}.png', dpi=300)

    print(metric)
    print(f'No Video: {np.mean(results[0, metrics[metric], :])} +- {np.std(results[0, metrics[metric], :])}')
    print(f'Video: {np.mean(results[1, metrics[metric], :])} +- {np.std(results[1, metrics[metric], :])}')
    print()

