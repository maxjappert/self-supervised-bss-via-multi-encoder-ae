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

from evaluation_metric_functions import compute_spectral_snr
from functions import evaluate_separation_ability, create_combined_image

data_path = 'data/musdb18_two_sources/validation'

total_snr = 0
num_data = len(os.listdir(data_path))
counter = 0

for i, data in enumerate(os.listdir(data_path)):

    if random.random() < 0.95:
        continue

    counter += 1

    S_mix_gt = np.array(Image.open(os.path.join(data_path, data, 'mix.png')))
    S_mix_gt = np.mean(S_mix_gt, axis=2)

    S1_gt = np.array(Image.open(os.path.join(data_path, data, 'stems', os.listdir(os.path.join(data_path, data, 'stems'))[0])))
    S2_gt = np.array(Image.open(os.path.join(data_path, data, 'stems', os.listdir(os.path.join(data_path, data, 'stems'))[1])))

    S1_gt = np.mean(S1_gt, axis=2)
    S2_gt = np.mean(S2_gt, axis=2)

    nmf = NMF(n_components=2, init='nndsvdar', max_iter=1000, random_state=0)
    W = nmf.fit_transform(S_mix_gt)
    H = nmf.components_

    # Reconstruct sources
    S1_approx = np.dot(W[:, 0:1], H[0:1, :])
    S2_approx = np.dot(W[:, 1:2], H[1:2, :])

    total_snr += evaluate_separation_ability([S1_approx, S2_approx], [S1_gt, S2_gt], compute_spectral_snr)

    if counter % 10 == 0:
        create_combined_image(S_mix_gt, S1_approx, S2_approx, S1_gt, S2_gt, f'nmf_{i}.png')
        print(f'After {counter} validation images: ~{total_snr / counter} ')


