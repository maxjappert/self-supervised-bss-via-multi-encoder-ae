import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy import signal
from scipy.ndimage import rotate

from functions_prior import PriorDataset, MultiModalDataset

dataset_name = 'musdb_18_prior'

train_datasets = [
    PriorDataset('train', image_h=128, image_w=128, name=dataset_name, num_stems=4, debug=False, stem_type=i + 1)
    for i in range(4)]

gt_data = [train_datasets[stem_index][18]['spectrogram'] for stem_index in [0, 1, 2, 3]]
gt_m = torch.sum(torch.cat(gt_data), dim=0).numpy()

if dataset_name == 'musdb_18_prior':
    fig, axs = plt.subplots(1, 5, figsize=(10, 2))

    axs[0].imshow(rotate(gt_m.squeeze(), angle=180), cmap='grey')
    axs[0].set_title('Mixture')
    axs[0].set_xticks([])
    axs[0].set_yticks([])

    titles = ['Drums', 'Bass', 'Other', 'Vocals']

    for i in range(1, 5):
        axs[i].imshow(rotate(gt_data[i-1].squeeze(), angle=180), cmap='grey')
        axs[i].set_title(titles[i-1])
        axs[i].set_xticks([])
        axs[i].set_yticks([])

    plt.savefig('figures/waves_musdb.png')
elif dataset_name == 'toy_dataset':
    fig, axs = plt.subplots(2, 5, figsize=(10, 4))

    axs[0, 0].imshow(rotate(gt_m.squeeze(), angle=180), cmap='grey')
    axs[0, 0].set_title('Mixture')
    axs[0, 0].set_xticks([])
    axs[0, 0].set_yticks([])

    titles = ['Sine', 'Sawtooth', 'Square', 'Triangle']

    for i in range(1, 5):
        axs[0, i].imshow(rotate(gt_data[i-1].squeeze(), angle=180), cmap='grey')
        axs[0, i].set_title(titles[i-1])
        axs[0, i].set_xticks([])
        axs[0, i].set_yticks([])

    freq = 5
    t = np.linspace(0, 1, 1000)

    sine_wave = np.sin(2 * np.pi * freq * t)
    sawtooth_wave = signal.sawtooth(2 * np.pi * freq * t)
    square_wave = signal.square(2 * np.pi * freq * t)
    triangle_wave = signal.sawtooth(2 * np.pi * freq * t, 0.5)
    mix = sine_wave + sawtooth_wave + square_wave + triangle_wave

    waves = [mix, sine_wave, sawtooth_wave, square_wave, triangle_wave]


    for i in range(5):
        axs[1, i].plot(t, waves[i], color='black')
        # axs[1, i].set_title(titles[i])
        axs[1, i].set_xticks([])
        axs[1, i].set_yticks([])
        axs[1, i].grid(True)

    axs[0, 0].set_ylabel('Time-frequency Space')
    axs[1, 0].set_ylabel('Real Space')
    plt.savefig('figures/waves.png')
else:
    while True:
        sample = dataset[random.randint(0, len(dataset) - 1)]
        if sample['label'] == 1:
            break

    dataset = MultiModalDataset('train', normalise=hps_video['normalise'], fps=hps_video['fps'])
    gt_xs = sample['sources'].numpy()
    gt_m = np.sum(gt_xs, axis=0)

    fig, axs = plt.subplots(1, 5, figsize=(10, 2))

    axs[0].imshow(rotate(gt_m.squeeze(), angle=180), cmap='grey')
    axs[0].set_title('Mixture')
    axs[0].set_xticks([])
    axs[0].set_yticks([])

    titles = ['Drums', 'Bass', 'Other', 'Vocals']

    for i in range(1, 5):
        axs[i].imshow(rotate(gt_data[i - 1].squeeze(), angle=180), cmap='grey')
        axs[i].set_title(titles[i - 1])
        axs[i].set_xticks([])
        axs[i].set_yticks([])

    plt.savefig('figures/waves_musdb.png')
