import os
import sys

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import NMF
import soundfile as sf

from evaluation_metric_functions import compute_spectral_snr
from functions import evaluate_separation_ability

def create_combined_image(S_mix_gt, S1_approx, S2_approx, S1_gt, S2_gt, output_path):
    fig, axes = plt.subplots(3, 2, figsize=(10, 15))

    # Display the spectrograms
    librosa.display.specshow(S_mix_gt, x_axis='time', y_axis='log', ax=axes[0, 0])
    axes[0, 0].set_title('Original Mix')

    librosa.display.specshow((S1_approx + S2_approx) / 2, x_axis='time', y_axis='log', ax=axes[0, 1])
    axes[0, 1].set_title('Approximate Mix')

    librosa.display.specshow(S1_approx, x_axis='time', y_axis='log', ax=axes[1, 0])
    axes[1, 0].set_title('Approximate Source 1')

    librosa.display.specshow(S2_approx, x_axis='time', y_axis='log', ax=axes[1, 1])
    axes[1, 1].set_title('Approximate Source 2')

    librosa.display.specshow(S1_gt, x_axis='time', y_axis='log', ax=axes[2, 0])
    axes[2, 0].set_title('Ground Truth Source 1')

    librosa.display.specshow(S2_gt, x_axis='time', y_axis='log', ax=axes[2, 1])
    axes[2, 1].set_title('Ground Truth Source 2')

    # Save the combined image
    plt.tight_layout()

    if not os.path.exists('images'):
        os.mkdir('images')

    plt.savefig(f'images/{output_path}')
    plt.close()

data_path = 'data/musdb18_two_sources/validation'

total_snr = 0
num_data = len(os.listdir(data_path))

for i, data in enumerate(os.listdir(data_path)):

    S_mix_gt = np.array(Image.open(os.path.join(data_path, data, 'mix.png')))
    S_mix_gt = np.mean(S_mix_gt, axis=2)

    S1_gt = np.array(Image.open(os.path.join(data_path, data, 'stems', os.listdir(os.path.join(data_path, data, 'stems'))[0])))
    S2_gt = np.array(Image.open(os.path.join(data_path, data, 'stems', os.listdir(os.path.join(data_path, data, 'stems'))[1])))

    S1_gt = np.mean(S1_gt, axis=2)
    S2_gt = np.mean(S2_gt, axis=2)

    nmf = NMF(n_components=2, init='nndsvdar', max_iter=5000, random_state=0)
    W = nmf.fit_transform(S_mix_gt)
    H = nmf.components_

    # Reconstruct sources
    S1_approx = np.dot(W[:, 0:1], H[0:1, :])
    S2_approx = np.dot(W[:, 1:2], H[1:2, :])

    total_snr += evaluate_separation_ability([S1_approx, S2_approx], [S1_gt, S2_gt], compute_spectral_snr)

    if i % 10000 == 0:
        create_combined_image(S_mix_gt, S1_approx, S2_approx, S1_gt, S2_gt, f'nmf_{i}.png')
        print(f'After {i+1} validation images: ~{total_snr / (i+1)} ')


print(total_snr / num_data)


