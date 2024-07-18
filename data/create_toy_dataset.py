import os
import random
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import spectrogram
from scipy.io.wavfile import write

sys.path.append('../')

from audio_spectrogram_conversion_functions import audio_to_spectrogram

# Function to create waves
def create_wave(wave_type, freq, duration, sr):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    if wave_type == 'sine':
        wave = np.sin(2 * np.pi * freq * t)
    elif wave_type == 'sawtooth':
        wave = signal.sawtooth(2 * np.pi * freq * t)
    elif wave_type == 'square':
        wave = signal.square(2 * np.pi * freq * t)
    elif wave_type == 'triangle':
        wave = signal.sawtooth(2 * np.pi * freq * t, 0.5)
    return wave


# Directory structure and parameters
dataset_dir = Path('toy_dataset')
splits = ['train', 'test', 'val']
num_datas = {'train': 10000, 'test': 1000, 'val': 1000}
wave_types = ['sine', 'sawtooth', 'square', 'triangle']
sr = 16000
duration = 5

for split in splits:
    split_dir = dataset_dir / split
    os.makedirs(split_dir, exist_ok=True)

    for i in range(1, num_datas[split]):  # Create 5 combinations for each split
        comb_dir = split_dir / f'combination{i}'
        mix_dir = comb_dir / 'mix.png'
        stems_dir = comb_dir / 'stems'
        os.makedirs(stems_dir, exist_ok=True)

        mix_wave = np.zeros(int(sr * duration))
        for j, wave_type in enumerate(wave_types):
            wave = create_wave(wave_type, random.randint(200, 1500), duration, sr)
            mix_wave += wave
            audio_to_spectrogram(wave, f'stem{j + 1}', save_to_file=True, dest_folder=stems_dir, sr=sr)

        audio_to_spectrogram(mix_wave, 'mix', save_to_file=True, dest_folder=comb_dir, sr=sr)
