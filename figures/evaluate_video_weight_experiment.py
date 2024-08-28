import numpy as np
import matplotlib.pyplot as plt

evaluated_weights = np.load('../results/video_weights_evaluated.npy')
basis_weights = np.load('../results/video_weights.npy')

metrics = {'sdr': 0, 'isr': 1, 'sir': 2, 'sar': 3}
jump_distance = 1

for metric in metrics.keys():
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel('$\lambda$')
    ax1.set_ylabel(metric.upper())
    ax1.set_title(f'Influence of video weight on {metric.upper()}', fontsize=16)

    mean_values = np.mean(np.mean(basis_weights[:, :, metrics[metric], :], axis=1), axis=1)
    std_values = np.std(np.mean(basis_weights[:, :, metrics[metric], :], axis=1), axis=1)

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
    plt.savefig(f'video_weight_experiment_{metric}.png', dpi=300)
