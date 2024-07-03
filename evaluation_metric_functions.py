import math
import numbers
import sys
import warnings

import librosa
import mir_eval
import numpy as np
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim

from audio_spectrogram_conversion_functions import spectrogram_to_audio


def compute_spectral_metrics(gt_spectros, approx_spectros):
    """
    Compute spectral metrics for source separation evaluation.

    This function calculates the Signal to Distortion Ratio (SDR), Signal to Interference Ratio (SIR),
    and Signal to Artifacts Ratio (SAR) between the ground truth and approximated spectrograms using
    `mir_eval.separation.bss_eval_sources`.

    Args:
        gt_spectros (list of np.ndarray): List of ground truth spectrograms. Each element in the list
            should be a 2D numpy array representing a spectrogram.
        approx_spectros (list of np.ndarray): List of approximated spectrograms. Each element in the
            list should be a 2D numpy array representing a spectrogram.

    Returns:
        tuple: A tuple containing the following elements:
            - SDR (np.ndarray): Signal to Distortion Ratio for each source.
            - SIR (np.ndarray): Signal to Interference Ratio for each source.
            - SAR (np.ndarray): Signal to Artifacts Ratio for each source.
            - perm (np.ndarray): Optimal permutation of estimated sources to match the ground truth sources.

    Raises:
        AssertionError: If the lengths of `gt_spectros` and `approx_spectros` are not the same.
    """

    assert len(gt_spectros) == len(approx_spectros)

    gt_wavs = []
    approx_wavs = []

    for i in range(len(gt_spectros)):
        gt_wavs.append(spectrogram_to_audio(gt_spectros[i], None))
        approx_wavs.append(spectrogram_to_audio(approx_spectros[i], None))

    return mir_eval.separation.bss_eval_sources(np.vstack(gt_wavs), np.vstack(approx_wavs))


def compute_si_sdr(gt, approx):
    # Ensure inputs are numpy arrays
    true_source = np.asarray(gt)
    estimated_source = np.asarray(approx)

    # Zero-mean normalization
    true_source = true_source - np.mean(true_source)
    estimated_source = estimated_source - np.mean(estimated_source)

    # Optimal scaling factor
    scaling_factor = np.dot(estimated_source, true_source) / np.dot(estimated_source, estimated_source)

    # Scaled estimated source
    scaled_estimated_source = scaling_factor * estimated_source

    # Compute the error signal
    error_signal = true_source - scaled_estimated_source

    # Compute SI-SDR
    sdr_value = 10 * np.log10(np.sum(true_source ** 2) / np.sum(error_signal ** 2))

    return sdr_value


def save_spectrogram_to_file(spectrogram, filename):
    """
    Saves a spectrogram image.
    :param spectrogram: 2D array constituting a spectrogram.
    :param filename: Name of file within the images folder.
    :return: None.
    """
    plt.imsave(f'images/{filename}', spectrogram, cmap='gray')


def compute_spectral_sdr(reference, estimated):
    warnings.warn(
        f"compute_spectral_sdr is deprecated and will be removed in a future version",
        category=DeprecationWarning,
        stacklevel=2
    )
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

