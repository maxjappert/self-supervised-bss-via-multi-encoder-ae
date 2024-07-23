import os
import random
import sys

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import NMF
import soundfile as sf

from evaluation_metric_functions import compute_spectral_metrics
from functions import evaluate_separation_ability, create_combined_image, metric_index_mapping

data_path = 'data/musdb18_two_sources/validation'

total_nmf_sdr = 0
total_random_sdr = 0
num_data = len(os.listdir(data_path))
counter = 0

def nmf_approx_two_sources(S_mix_gt):
    if len(S_mix_gt.shape) == 3:
        S_mix_gt = np.mean(S_mix_gt, axis=2)

    nmf = NMF(n_components=2, init='nndsvdar', max_iter=1000, random_state=0)
    W = nmf.fit_transform(S_mix_gt)
    H = nmf.components_

    # Reconstruct sources
    S1_approx = np.dot(W[:, 0:1], H[0:1, :])
    S2_approx = np.dot(W[:, 1:2], H[1:2, :])

    return S1_approx, S2_approx

# Random benchmark

if __name__ == '__main__':

    for i, data in enumerate(os.listdir(data_path)):

        if random.random() < 0.95:
            continue

        counter += 1

        S_mix_gt = np.array(Image.open(os.path.join(data_path, data, 'mix.png')))
        S1_approx, S2_approx = nmf_approx_two_sources(S_mix_gt)

        S1_gt = np.array(
            Image.open(os.path.join(data_path, data, 'stems', 'S0.png')))
        S2_gt = np.array(
            Image.open(os.path.join(data_path, data, 'stems', 'S1.png')))

        S1_gt = np.mean(S1_gt, axis=2)
        S2_gt = np.mean(S2_gt, axis=2)

        #total_sdr += evaluate_separation_ability([S1_approx, S2_approx], [S1_gt, S2_gt], compute_spectral_snr)

        phases = [np.load(os.path.join(data_path, data, 'stems', 'S0_phase.npy')), np.load(os.path.join(data_path, data, 'stems', 'S1_phase.npy'))]
        total_nmf_sdr += np.mean(compute_spectral_metrics([S1_gt, S2_gt], [S1_approx, S2_approx], phases)[metric_index_mapping['sdr']])

        random_image1 = np.random.randint(0, 256, size=(1025, 431))
        random_image2 = np.random.randint(0, 256, size=(1025, 431))

        total_random_sdr += np.mean(compute_spectral_metrics([S1_gt, S2_gt], [random_image1, random_image2], phases)[metric_index_mapping['sdr']])

        if counter % 10 == 0:
            create_combined_image(S_mix_gt, S1_approx, S2_approx, S1_gt, S2_gt, f'nmf_{i}.png')
            print(f'After {counter} validation images: ~{total_nmf_sdr / counter} ')
            print(f'Benchmark Random SDR: ~{total_random_sdr / counter} ')

