import itertools
import os
import random
import sys
from pathlib import Path

import cv2
import librosa
import numpy as np
from moviepy.editor import VideoFileClip

sys.path.append('../')
from audio_spectrogram_conversion_functions import audio_to_spectrogram

sr = 22050

def process_video(video_path, audio_sources, name, indices, chunk_duration=5):
    # Create output directories if they don't exist
    # os.makedirs(output_dir, exist_ok=True)

    # Load video
    print(name)
    video = VideoFileClip(video_path)
    video_duration = int(video.duration)

    # Process each 5-second chunk
    for counter, start_time in enumerate(range(0, video_duration, chunk_duration)):
        end_time = min(start_time + chunk_duration, video_duration)
        chunk_clip = video.subclip(start_time, end_time)

        if random.random() < split:
            output_dir = os.path.join('rochester_preprocessed', 'train', f'{name}_{indices[0]}_{indices[1]}_{counter}')
        else:
            output_dir = os.path.join('rochester_preprocessed', 'val', f'{name}_{indices[0]}_{indices[1]}_{counter}')

        os.makedirs(output_dir)

        # Save video chunk
        chunk_video_path = os.path.join(output_dir, f'video.mp4')
        chunk_clip.write_videofile(chunk_video_path, codec="libx264", audio_codec="aac")

        # Extract frames (assuming equal number of frames per chunk)
        frames = []
        cap = cv2.VideoCapture(chunk_video_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        # Process each audio source
        for i, source in enumerate(audio_sources):
            # print(source)

            # Load and process audio chunk
            # audio_chunk = source.subclip(start_time, end_time)

            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)

            audio_chunk = source[start_sample:end_sample]

            # Save spectrogram
            # spectrogram_path = os.path.join(output_dir, f'spectrogram_{source.name}_{start_time}-{end_time}.png')
            audio_to_spectrogram(audio_chunk, f's{i+1}', save_to_file=True, dest_folder=Path(output_dir), sr=22050)


# def main(input_dir, output_dir, audio_sources):
#     os.makedirs(output_dir, exist_ok=True)
#
#     train_dir = os.path.join(output_dir, 'train')
#     val_dir = os.path.join(output_dir, 'val')
#
#     os.makedirs(train_dir, exist_ok=True)
#     os.makedirs(val_dir, exist_ok=True)
#
#     for video_file in os.listdir(input_dir):
#         video_path = os.path.join(input_dir, video_file)
#
#         if "train" in video_file:
#             process_video(video_path, audio_sources, train_dir)
#         elif "val" in video_file:
#             process_video(video_path, audio_sources, val_dir)


if __name__ == "__main__":
    master_folder = 'rochester_preprocessed'

    os.makedirs(f'{master_folder}', exist_ok=True)
    os.makedirs(f'{master_folder}/train', exist_ok=True)
    os.makedirs(f'{master_folder}/val', exist_ok=True)

    split = 0.8

    for datapoint in os.listdir('rochester'):
        if datapoint[0] == '.':
            continue

        folder = os.path.join('rochester', datapoint)

        num_sources = datapoint.count('_') - 1

        video_path = os.path.join(folder, f'Vid_{datapoint}.mp4')

        name_prefix = f'{datapoint.split("_")[0]}_{datapoint.split("_")[1]}'

        source_shorts = datapoint.split('_')[2:]

        source_paths = []

        for source_idx in range(num_sources):
            source_paths.append(os.path.join(folder, f'AuSep_{source_idx+1}_{source_shorts[source_idx]}_{name_prefix}.wav'))

        combinations = list(itertools.combinations(range(num_sources), 2))

        for combination in combinations:
            # audio_source_indexes = [combination[0], combination[1]]

            audio_sources = [librosa.load(source_paths[combination[0]], sr=sr)[0], librosa.load(source_paths[combination[1]], sr=sr)[0]]
            # librosa.load(audio_path, sr=None)

            process_video(video_path, audio_sources, datapoint, combination)

