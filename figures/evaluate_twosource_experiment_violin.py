import sys
import numpy as np
import pandas as pd
import sns
from sympy.physics.control.control_plots import plt
import matplotlib.patches as mpatches
from statsmodels.graphics.boxplots import violinplot

vae_name = 'musdb'

results_basis = np.load(f'../results/results_basis_{vae_name}.npy')
results_basis_finetuned = np.load(f'../results/results_basis_finetuned_{vae_name}.npy')
results_bss = np.load(f'../results/results_bss_{vae_name}.npy')
results_bss_linear = np.load(f'../results/results_bss_linear_{vae_name}.npy')
results_noise = np.load(f'../results/results_noise_{vae_name}.npy')
results_prior_samples = np.load(f'../results/results_prior_samples_{vae_name}.npy')
results_nmf = np.load(f'../results/results_nmf_{vae_name}.npy')

results = np.stack((results_basis, results_basis_finetuned, results_bss, results_bss_linear, results_noise, results_prior_samples, results_nmf))

k = results_basis.shape[0]

offset = 0.5

metrics = {'sdr': 0, 'isr': 1, 'sir': 2, 'sar': 3}

assert results.shape == (len(results), k, len(metrics.keys()), results.shape[3])

data_name = 'Toy' if vae_name.__contains__('toy') else 'MUSDB18'

for metric in metrics.keys():
    # Initialize the plot
    fig, ax = plt.subplots(figsize=(10, 6))


    # Iterate over each result
    for i in range(results.shape[0]):
        # Split the data for the left and right channels
        data_left = results[i, 0, metrics[metric], :]#.flatten()  # Flatten across all metrics
        data_right = results[i, 1, metrics[metric], :]#.flatten()

        # Create a violin plot for each
        violinplot([data_left], positions=[i+offset], show_boxplot=False, side='left', ax=ax, plot_opts={'violin_fc':'C0'})
        violinplot([data_right], positions=[i+offset], show_boxplot=False, side='right', ax=ax, plot_opts={'violin_fc':'C1'})

        # Offset the right channel to create an asymmetric effect
        # plt.gca().collections[-1].set_offsets([i + 0.25, 0])

    # Create custom legend handles
    left_patch = mpatches.Patch(color='blue', label='Left Channel')
    right_patch = mpatches.Patch(color='red', label='Right Channel')

    # Add the legend t the plot
    # plt.legend(handles=[left_patch, right_patch], loc='upper right')

    # Set the x-axis
    plt.xticks(range(results.shape[0]), [f'Result {i + 1}' for i in range(results.shape[0])])

    # Set labels and show plot
    # plt.xlabel('Results')
    plt.ylabel(metric.upper())

    custom_labels = ['BASIS', 'BASIS Finetuned', 'AE-BSS', 'Linear AE-BSS', 'Noise', 'Samples from Prior', 'NMF']
    tick_positions = [i + offset for i in range(len(custom_labels))]
    plt.xticks(ticks=tick_positions, labels=custom_labels)
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    # ax.set_xticks(range(len(custom_labels)))
    # ax.set_xticklabels(custom_labels)
    # plt.grid(True)
    # plt.legend(True)
    plt.title(f'{metric.upper()} of the Discussed Source Separation Methods on {data_name} Data')
    plt.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(f'small_twosource_{metric}_{vae_name}_violin.png', dpi=300)
