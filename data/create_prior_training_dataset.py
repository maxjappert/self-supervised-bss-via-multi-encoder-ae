import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import librosa
import numpy as np

sys.path.append('../')
from audio_spectrogram_conversion_functions import audio_to_spectrogram, chunk_length


def slice_audio_file_to_chunks(file_path, sample_rate=44100):
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


def process_file(file_path, dest_folder, counter):
    stems_folder = dest_folder / "temp_stems"
    stems_folder.mkdir(parents=True, exist_ok=True)

    # Extract all stems
    stem_paths = []
    for i in range(1, 5):
        stem_path = stems_folder / f"S0{i}.wav"
        cmd = f"ffmpeg -i '{file_path}' -map 0:{i} -y '{stem_path}'"
        try:
            subprocess.run(cmd, shell=True, check=True)
            stem_paths.append(stem_path)
        except subprocess.CalledProcessError as e:
            print(f"Error processing {file_path}: {e}")
            return counter

    # Slice stems into 5-second chunks
    stem_chunks = [slice_audio_file_to_chunks(stem_path) for stem_path in stem_paths]

    # Process chunks and save spectrograms
    num_chunks = min(len(chunks) for chunks in stem_chunks)
    for chunk_idx in range(num_chunks):
        chunks = [stem_chunks[stem_idx][chunk_idx] for stem_idx in range(4)]

        # Skip if any stem is silent
        if any(np.all(np.isclose(chunk, 0)) for chunk in chunks):
            continue

        counter += 1
        chunk_folder = dest_folder / f"datapoint{counter}"
        chunk_folder.mkdir(parents=True, exist_ok=True)

        # Convert chunks to spectrograms and save
        for stem_idx, chunk in enumerate(chunks):
            audio_to_spectrogram(chunk, f"stem{stem_idx + 1}", save_to_file=True, dest_folder=chunk_folder)

    # Remove the original extracted stems
    shutil.rmtree(stems_folder)
    return counter


def create_spectrogram_dataset(input_folder, output_folder, split_ratios):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)

    # Create necessary output directories
    splits = ['train', 'val', 'test']
    for split in splits:
        (output_folder / split).mkdir(parents=True, exist_ok=True)

    # Get all audio files and shuffle
    audio_files = list(input_folder.glob("**/*.stem.mp4"))
    random.shuffle(audio_files)

    # Split files based on split_ratios
    total_files = len(audio_files)
    train_split = int(split_ratios['train'] * total_files)
    val_split = int(split_ratios['val'] * total_files) + train_split

    splits_files = {
        'train': audio_files[:train_split],
        'val': audio_files[train_split:val_split],
        'test': audio_files[val_split:]
    }

    # Process files and create spectrograms
    counter = 0
    for split, files in splits_files.items():
        for file_path in files:
            counter = process_file(file_path, output_folder / split, counter)


if __name__ == "__main__":
    input_folder = "musdb18"
    output_folder = "musdb_18_prior"
    split_ratios = {'train': 0, 'val': 0.9, 'test': 0.1}

    create_spectrogram_dataset(input_folder, output_folder, split_ratios)
