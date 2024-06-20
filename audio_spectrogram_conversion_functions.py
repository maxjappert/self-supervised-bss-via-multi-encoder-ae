import librosa
import numpy as np
import matplotlib.pyplot as plt
import scipy
from PIL import Image


def audio_to_spectrogram(audio_file, save_to_file=False):
    # Load the audio file
    y, sr = librosa.load(audio_file)
    name = audio_file.stem

    # Convert to spectrogram
    S = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)

    # Normalize the spectrogram to 0-255
    S_db_normalized = (S_db + 80) / 80 * 255
    S_db_normalized = S_db_normalized.astype(np.uint8)

    # Save the spectrogram to a PNG file
    if save_to_file:
        plt.imsave(f' images / spectrogram_{name}.png', S_db_normalized, cmap='gray')
        np.save(f'images / sr_{name}.npy', sr)

    return S_db_normalized, sr


def spectrogram_to_audio(spectrogram, sr, output_filename, from_file=False):
    # Load the spectrogram image
    if from_file:
        img = Image.open(f'images/{spectrogram}').convert('L')
        S_db_imported = np.array(img)
        #sr = np.load(sr).item()
    else:
        S_db_imported = spectrogram

    # Convert back to the original dB scale
    S_db_imported = S_db_imported.astype(np.float32) / 255 * 80 - 80

    # Convert back to amplitude
    S_imported = librosa.db_to_amplitude(S_db_imported)

    # Use Griffin-Lim algorithm to approximate the phase and reconstruct the audio
    audio_signal = librosa.core.griffinlim(S_imported)

    # Write the output to a WAV file
    scipy.io.wavfile.write(f'wavs/{output_filename}.wav', sr, np.array(audio_signal * 32767, dtype=np.int16))


#spectrogram, sr = audio_to_spectrogram('01_Jupiter_vn_vc/AuSep_1_vn_01_Jupiter.wav', save_to_file=True)
#spectrogram_to_audio('spectrogram_AuSep_1_vn_01_Jupiter.png', 'sr_AuSep_1_vn_01_Jupiter.npy', 'aaa', from_file=True)
