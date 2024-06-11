from audio_spectrogram_conversion_functions import spectrogram_to_audio
from functions import *

for i in range(6):
    spectrogram_to_audio(f'first_spectro_{i}.png', sr=22050, output_filename=f'first_spectro_{i}', from_file=True)
    spectrogram_to_audio(f'second_spectro_{i}.png', sr=22050, output_filename=f'second_spectro_{i}', from_file=True)
    spectrogram_to_audio(f'third_spectro_{i}.png', sr=22050, output_filename=f'third_spectro_{i}', from_file=True)
