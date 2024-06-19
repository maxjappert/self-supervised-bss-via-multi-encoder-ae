# Dataset: split into 10 second windows
import os
import pickle
import sys
import random

import librosa
import numpy as np
import psutil
import torch
from PIL import Image

#data = {'mix': [], 'sep1': [], 'sep2': [], 'sep3': [], 'sep4': []}
slice_start_indexes = []

chunk_length = 5  # seconds
sample_rate = 22050  # Sample rate for librosa
n_fft = 2048  # FFT window size
hop_length = 512  # Hop length


def slice_audio_file_to_chunks(file_path):
    # Load audio file
    audio, sr = librosa.load(file_path, sr=sample_rate)
    total_length = chunk_length * sr
    chunks = []

    for start in range(0, len(audio), total_length):
        end = start + total_length
        chunk = audio[start:end]

        # Pad if the chunk is shorter than the desired length
        if len(chunk) < total_length:
            padding = total_length - len(chunk)
            chunk = np.pad(chunk, (0, padding), 'constant')

        # Convert chunk to spectrogram
        S = librosa.stft(chunk, n_fft=n_fft, hop_length=hop_length)
        #spectrogram = np.abs(spectrogram)
        S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
        S_db_normalized = (S_db + 80) / 80 * 255
        S_db_normalized = S_db_normalized.astype(np.uint8)

        chunks.append(S_db_normalized)

    return chunks

def log_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / (1024 * 1024)} MB")

    return process.memory_info().rss / (1024 * 1024)

def extract(split, num_sources):

    #data = [[]]

    #for _ in range(num_sources):
    #    data.append([])

    folder_counter = 0

    new_folder_counter = 0

    parent_folder = 'slakh_two_sources' if num_sources == 2 else 'slakh2100_flac_redux'

    new_parent = os.path.join('slakh_two_sources_preprocessed', split)


    if not os.path.exists(new_parent):
        os.mkdir('slakh_two_sources_preprocessed')
        os.mkdir(new_parent)
    else:
        print('Folder already exists. Aborting.')
        return

    for folder in os.listdir(f'{parent_folder}/{split}'):
        if len(os.listdir(f'{parent_folder}/{split}/{folder}/stems')) == num_sources\
                and random.random() < 0.2:
            print(f'Processing {folder}')

            folder_counter += 1
            chunks_master = slice_audio_file_to_chunks(f'{parent_folder}/{split}/{folder}/mix.flac')
            #data[0].extend(chunks_master)

            idxs = []

            for i, chunk in enumerate(chunks_master):
                img = Image.fromarray(chunk)
                new_folder = os.path.join(new_parent, f'Track{new_folder_counter}')
                os.mkdir(new_folder)
                os.mkdir(os.path.join(new_folder, 'stems'))
                img.save(os.path.join(new_folder, 'mix.png'))
                idxs.append(new_folder_counter)
                new_folder_counter += 1

            for i, stem in enumerate(os.listdir(f'{parent_folder}/{split}/{folder}/stems')):
                chunks_stem = slice_audio_file_to_chunks(f'{parent_folder}/{split}/{folder}/stems/{stem}')

                for j, chunk in enumerate(chunks_stem):
                    img = Image.fromarray(chunk)
                    img.save(os.path.join(new_parent, f'Track{idxs[j]}', 'stems', f'S0{i}.png'))

                #data[i+1].extend(chunks_stem)

    print(f'Found {folder_counter} songs with {num_sources} sources. Total of {new_folder_counter} {chunk_length} second long data points.')


extract(sys.argv[2], int(sys.argv[1]))
