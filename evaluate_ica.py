import os

import numpy as np
from PIL import Image
from sklearn.decomposition import FastICA

from audio_spectrogram_conversion_functions import spectrogram_to_audio
from functions import create_combined_image

data_path = 'data/musdb18_two_sources/validation'

data_paths = [data_path + '/' + name for name in os.listdir(data_path)]

data_path1 = data_paths[5]
data_path2 = data_paths[8]

mix1 = np.array(Image.open(data_path1 + '/mix.png')).mean(axis=-1)
mix2 = np.array(Image.open(data_path2 + '/mix.png')).mean(axis=-1)

stems1 = f'{data_path1}/stems/{os.listdir(f"{data_path1}/stems")[0]}'
stems2 = f'{data_path1}/stems/{os.listdir(f"{data_path1}/stems")[1]}'

source_1_gt = np.array(Image.open(stems1)).mean(axis=-1)
source_2_gt = np.array(Image.open(stems2)).mean(axis=-1)

# Center the data
mix1 = mix1 - np.mean(mix1, axis=0)
source_1_gt = source_1_gt - np.mean(source_1_gt, axis=0)
source_2_gt = source_2_gt - np.mean(source_2_gt, axis=0)

mixed_signals = np.stack((mix1, mix2), axis=-1)  # Shape will be (2, 1025, 216)

# Reshape to (216, 2050) where 216 is the number of time steps and 2050 is 2 * 1025 (frequency bins * number of signals)
mixed_signals = mixed_signals.reshape(-1, 2)

# Center the data
#mean = np.mean(mixed_signals, axis=0)
#centered_signals = mixed_signals - mean

# Initialize ICA
ica = FastICA(n_components=2)

# Fit and transform the data
separated_sources = ica.fit_transform(mixed_signals)  # Shape will be (216*1025, 2)


# Reshape back to the original spectrogram dimensions
source_1 = separated_sources[:, 0].reshape(1025, 216)
source_2 = separated_sources[:, 1].reshape(1025, 216)

print("Range of the source_1:", source_1.min(), "to", source_1.max())
print("Range of the source_1_gt:", source_1_gt.min(), "to", source_1_gt.max())

spectrogram_to_audio(source_1, 'ica_s1.wav')
spectrogram_to_audio(source_2, 'ica_s2.wav')
spectrogram_to_audio(source_1_gt, 'ica_s1_gt.wav')
spectrogram_to_audio(source_2_gt, 'ica_s2_gt.wav')
spectrogram_to_audio(mix1, 'ica_mix1_gt.wav')
spectrogram_to_audio(mix2, 'ica_mix2_gt.wav')


create_combined_image(mix1, source_1, source_2, source_1_gt, source_2_gt, 'ica.png')
