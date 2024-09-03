import sys

import numpy as np
import matplotlib.pyplot as plt

name = 'musdb'

title_name = ''

if name == 'toy':
    title_name = 'Toy'
elif name == 'musdb':
    title_name = 'MUSDB18'
elif name == 'video':
    title_name = 'URMP'
else:
    print('ERror')
    sys.exit(-1)

if name == 'video':
    evaluated_weights = np.concatenate([np.load('../results/gradient_video_weights_evaluated_0.npy'), np.load('../results/gradient_video_weights_evaluated.npy')], axis=0)# [20:]
    basis_weights = np.concatenate([np.load('../results/gradient_video_weights_0.npy'), np.load('../results/gradient_video_weights.npy')], axis=0)# [20:]
else:
    evaluated_weights = np.load(f'../results/gradient_{name}_weights_evaluated.npy')
    basis_weights = np.load(f'../results/gradient_{name}_weights.npy')

metrics = {'sdr': 0, 'isr': 1, 'sir': 2, 'sar': 3}
jump_distance = 1

for metric in metrics.keys():
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel('$\lambda$')
    ax1.set_ylabel(metric.upper())
    ax1.set_title(f'Influence of $\lambda$ on {metric.upper()} using {title_name} data', fontsize=16)

    mean_values = np.mean(np.mean(basis_weights[:, :, metrics[metric], :], axis=1), axis=1)
    std_values = np.std(np.mean(basis_weights[:, :, metrics[metric], :], axis=1), axis=1)

    # Printing mean ± std for each weight
    print(metric)
    for i, (mean, std) in enumerate(zip(mean_values, std_values)):
        print(f'Weight {evaluated_weights[i*jump_distance]:.2f}: {mean:.2f} ± {std:.2f}')


    ax1.errorbar(
        np.arange(len(evaluated_weights)//jump_distance),
        mean_values,
        yerr=std_values,
        fmt='-o',
        color='k',
        ecolor='red',
        capsize=5
    )

    ax1.set_xticks(range(len(evaluated_weights)//jump_distance))
    ax1.set_xticklabels([round(weight) for weight in evaluated_weights[::jump_distance]])
    ax1.grid(True, linestyle='--', alpha=0.7)

    fig.tight_layout()
    plt.savefig(f'gradient_weight_experiment_{name}_{metric}.png', dpi=300)
