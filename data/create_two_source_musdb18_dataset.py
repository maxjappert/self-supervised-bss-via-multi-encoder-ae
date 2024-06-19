import os
import shutil
import subprocess
from itertools import combinations
from pathlib import Path

import librosa
import numpy as np
import matplotlib.pyplot as plt

sample_rate = 22050
chunk_length = 5  # in seconds
n_fft = 2048
hop_length = 512

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

        chunks.append(chunk)

    return chunks

def audio_to_spectrogram(audio, name, save_to_file=False, dest_folder=None):
    # Convert to spectrogram
    S = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)

    # Normalize the spectrogram to 0-255
    S_db_normalized = (S_db + 80) / 80 * 255
    S_db_normalized = S_db_normalized.astype(np.uint8)

    # Save the spectrogram to a PNG file
    if save_to_file and dest_folder:
        plt.imsave(dest_folder / f'{name}.png', S_db_normalized, cmap='gray')

    return S_db_normalized

def create_two_sources_dataset(input_folder, output_folder, split_ratio=0.8):
    # Define paths
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)

    # Create necessary output directories
    output_train = output_folder / "train"
    output_test = output_folder / "test"
    output_validation = output_folder / "validation"
    output_train.mkdir(parents=True, exist_ok=True)
    output_test.mkdir(parents=True, exist_ok=True)
    output_validation.mkdir(parents=True, exist_ok=True)

    # Helper function to extract stems, slice into chunks, and create mix
    def process_file(file_path, dest_folder, comb_counter):
        stems_folder = dest_folder / "temp_stems"
        stems_folder.mkdir(parents=True, exist_ok=True)

        # Extract all stems
        stem_paths = []
        for i in range(4):
            stem_path = stems_folder / f"S0{i + 1}.wav"
            cmd = f"ffmpeg -i '{file_path}' -map 0:{i + 1} -y '{stem_path}'"
            try:
                subprocess.run(cmd, shell=True, check=True)
                stem_paths.append(stem_path)
            except subprocess.CalledProcessError as e:
                print(f"Error processing {file_path}: {e}")
                return comb_counter

        # Slice stems into 5-second chunks
        stem_chunks = [slice_audio_file_to_chunks(stem) for stem in stem_paths]

        # Create all combinations of 2 stems
        combs = list(combinations(range(4), 2))
        for (idx1, idx2) in combs:
            num_chunks = min(len(stem_chunks[idx1]), len(stem_chunks[idx2]))
            for chunk_idx in range(num_chunks):
                comb_counter += 1
                mix_folder = dest_folder / f"combination{comb_counter}"
                mix_folder.mkdir(parents=True, exist_ok=True)
                mix_stems_folder = mix_folder / "stems"
                mix_stems_folder.mkdir(parents=True, exist_ok=True)

                chunk1 = stem_chunks[idx1][chunk_idx]
                chunk2 = stem_chunks[idx2][chunk_idx]

                # Convert chunks to spectrograms and save
                s1_spectrogram = audio_to_spectrogram(chunk1, f"S{idx1 + 1}_{chunk_idx + 1}", save_to_file=True,
                                                      dest_folder=mix_stems_folder)
                s2_spectrogram = audio_to_spectrogram(chunk2, f"S{idx2 + 1}_{chunk_idx + 1}", save_to_file=True,
                                                      dest_folder=mix_stems_folder)

                # Merge the two chunks into a single audio
                mix_audio = chunk1 + chunk2
                mix_spectrogram = audio_to_spectrogram(mix_audio, f"mix", save_to_file=True,
                                                       dest_folder=mix_folder)

        # Remove the original extracted stems
        shutil.rmtree(stems_folder)
        return comb_counter

    # Process train files
    comb_counter = 0
    train_files = list((input_folder / "train").glob("*.stem.mp4"))
    for file_path in train_files:
        comb_counter = process_file(file_path, output_train, comb_counter)

    # Process test files
    test_files = list((input_folder / "test").glob("*.stem.mp4"))
    for file_path in test_files:
        comb_counter = process_file(file_path, output_test, comb_counter)

    # Split train files into train and validation based on split ratio
    validation_count = int((1 - split_ratio) * len(train_files))
    validation_files = train_files[:validation_count]
    for file_path in validation_files:
        comb_counter = process_file(file_path, output_validation, comb_counter)


if __name__ == "__main__":
    input_folder = "musdb18"
    output_folder = "musdb18_two_sources"
    create_two_sources_dataset(input_folder, output_folder)
