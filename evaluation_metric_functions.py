import math
import numbers
import sys

import fast_bss_eval
import librosa
import mir_eval
import numpy as np
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim

from audio_spectrogram_conversion_functions import spectrogram_to_audio

def compute_spectral_snr(reference_spectrogram, noisy_spectrogram):
    """
    Compute the Spectral Signal-to-Noise Ratio (SNR) between the reference and noisy spectrograms.

    Parameters:
    reference_spectrogram (np.ndarray): The original source spectrogram.
    noisy_spectrogram (np.ndarray): The noisy spectrogram.

    Returns:
    float: The average spectral SNR value in dB.
    """
    # Ensure the spectrograms are the same shape
    assert reference_spectrogram.shape == noisy_spectrogram.shape, "Spectrograms must have the same shape"

    epsilon = 1e-7

    # Compute power spectra
    ref_power = np.abs(reference_spectrogram) ** 2
    noise_power = np.abs(reference_spectrogram - noisy_spectrogram) ** 2

    # Compute SNR for each frequency bin and time frame
    snr_spectrum = 10 * np.log10(np.maximum(ref_power / np.maximum(noise_power, epsilon), epsilon))

    # Average SNR across all frequency bins and time frames
    avg_snr = np.mean(snr_spectrum)

    if avg_snr == math.nan:
        print('snr is nan')

    return avg_snr


def save_spectrogram_to_file(spectrogram, filename):
    """
    Saves a spectrogram image.
    :param spectrogram: 2D array constituting a spectrogram.
    :param filename: Name of file within the images folder.
    :return: None.
    """
    plt.imsave(f'images/{filename}', spectrogram, cmap='gray')



def compute_spectral_sdr(reference, estimated):
    """
    Calculate the Signal-to-Distortion Ratio (SDR) for 2D numpy arrays of spectrograms.

    :param reference: 2D numpy array (spectrogram) of the reference signal
    :param estimated: 2D numpy array (spectrogram) of the estimated signal
    :return: SDR value in dB
    """

    assert reference.shape == estimated.shape, "Spectrograms must have the same shape"

    # Ensure inputs are numpy arrays
    reference = np.array(reference)
    estimated = np.array(estimated)

    # Compute the numerator and denominator for SDR
    numerator = np.sum(reference ** 2)
    denominator = np.sum((reference - estimated) ** 2)

    if np.isnan(reference).any():
        print('Reference in SDR contains NaN. Returning SDR of 0.')
        return 0
    if np.isnan(estimated).any():
        print('Prediction in SDR contains NaN. Returning SDR of 0.')
        return 0

    eps = 1e-7

    try:
        # Avoid division by zero
        if denominator == 0 and not np.isclose(reference, np.zeros_like(reference)).all():
            print(np.isclose(reference, estimated).all())
            print('Sus accuracy')
            print(reference.shape)
            save_spectrogram_to_file(reference, 'sus_reference.png')
            save_spectrogram_to_file(estimated, 'sus_estimated.png')
            return float('inf')
        elif denominator == 0:
            return 0
        else:
            sdr = 10 * np.log10(np.maximum(numerator / np.maximum(denominator, eps), eps))

            return sdr
    except Exception as e:
        print(e)
        print(denominator)


def compute_ssim(gts, approxs):
    return ssim(gts, approxs, data_range=approxs.max() - approxs.min())

