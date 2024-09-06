import sys

import numpy as np
from sympy.physics.control.control_plots import plt
from statsmodels.graphics.boxplots import violinplot


vae_name = 'toy'

results_basis = np.load(f'../results/results_basis_rochester_novideo_128.npy')
results_basis_video = np.load(f'../results/results_basis_rochester_video_128.npy')

results = np.stack((results_basis, results_basis_video))
# results = np.mean(results, axis=1)

k = 2

offset = 1

metrics = {'sdr': 0,
           'isr': 1,
           'sir': 2,
           'sar': 3}

assert results.shape == (len(results), 2, len(metrics.keys()), results.shape[3])

data_name = 'Toy' if vae_name.__contains__('toy') else 'MUSDB18'


for metric in metrics.keys():
    # Initialize the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Iterate over each result
    for i in range(results.shape[0]):
        # Split the data for the left and right channels
        data_left = results[i, 0, metrics[metric], :]  # .flatten()  # Flatten across all metrics
        data_right = results[i, 1, metrics[metric], :]  # .flatten()

        # Create a violin plot for each
        violinplot([data_left], positions=[i + offset], show_boxplot=False, side='left', ax=ax,
                   plot_opts={'violin_fc': 'C0'})
        violinplot([data_right], positions=[i + offset], show_boxplot=False, side='right', ax=ax,
                   plot_opts={'violin_fc': 'C1'})

    plt.ylabel(metric.upper())

    custom_labels = ['', 'No Video', 'Video', '']
    tick_positions = [i for i in range(len(custom_labels))]
    plt.xticks(ticks=tick_positions, labels=custom_labels)
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.subplots_adjust(left=0.5)
    # ax1.set_xlim([-0.55, 11.55])

    plt.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(f'video_experiment_{metric}_violin.png', dpi=300)

    print(metric)
    print(f'No Video: {np.mean(results[0, :, metrics[metric], :])} +- {np.std(results[0, :, metrics[metric], :], ddof=1) / np.sqrt(len(results[0, :, metrics[metric], :].flatten()))}')
    print(f'Video: {np.mean(results[1, :, metrics[metric], :])} +- {np.std(results[1, :, metrics[metric], :], ddof=1) / np.sqrt(len(results[1, :, metrics[metric], :].flatten()))}')
    print()

