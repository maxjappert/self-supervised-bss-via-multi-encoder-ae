import os
import random
import shutil
import sys

from pydub import AudioSegment

from data.convert_slakh_dataset_to_pickle import slice_audio_file_to_chunks


def combine_stems(stem1_path, stem2_path, output_path=None):
    stem1 = AudioSegment.from_file(stem1_path)
    stem2 = AudioSegment.from_file(stem2_path)
    combined = stem1.overlay(stem2)
    
    if output_path:
        combined.export(output_path, format="flac")
    
    return combined


def create_two_source_mixes(dataset_path, output_path):
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    else:
        print('Output path already exists. Aborting.')
        return
    counter = 0
    for track_folder in os.listdir(dataset_path):
        track_path = os.path.join(dataset_path, track_folder)
        if os.path.isdir(track_path):
            stems_folder = os.path.join(track_path, "stems")
            stems = [os.path.join(stems_folder, stem) for stem in os.listdir(stems_folder) if stem.endswith('.flac')]
            print(f'Processing {track_path}')

            os.mkdir('tmp')
            for i in range(len(stems)):
                for j in range(i + 1, len(stems)):
                    if random.random() < 0.0001:
                        stem1 = stems[i]
                        stem2 = stems[j]
                        track_name = f'Track{counter}'
                        os.mkdir(os.path.join(output_path, track_name))
                        output_file = os.path.join(output_path, track_name, f"mix.flac")
                        combine_stems(stem1, stem2, output_file)
                        os.mkdir(os.path.join(output_path, track_name, 'stems'))
                        shutil.copy(stem1, os.path.join(output_path, track_name, 'stems'))
                        shutil.copy(stem2, os.path.join(output_path, track_name, 'stems'))
                        counter += 1


# Create the two-source mixes
create_two_source_mixes(f"slakh2100_flac_redux/{sys.argv[1]}", f"slakh_two_sources/{sys.argv[1]}")
