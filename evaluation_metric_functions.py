import math
import sys

import librosa
import mir_eval
import numpy as np
from skimage.metrics import structural_similarity as ssim

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

    epsilon = 1e-10

    # Compute power spectra
    ref_power = np.abs(reference_spectrogram) ** 2
    noise_power = np.abs(reference_spectrogram - noisy_spectrogram) ** 2

    # Compute SNR for each frequency bin and time frame
    snr_spectrum = 10 * np.log10(np.maximum(ref_power / np.maximum(noise_power, epsilon), epsilon))

    # Average SNR across all frequency bins and time frames
    avg_snr = np.mean(snr_spectrum)

    return avg_snr


def compute_bss_metrics(reference_spectrogram, estimated_spectrogram, sr=22050, n_fft=2048,
                                          hop_length=512):
    """
    Compute BSS metrics (SDR, SIR, SAR, ISR) between the reference and estimated spectrograms.

    Parameters:
    reference_spectrograms (np.ndarray): The original source spectrograms (shape: num_sources x freq_bins x time_frames).
    estimated_spectrograms (np.ndarray): The separated (estimated) spectrograms (shape: num_sources x freq_bins x time_frames).
    sr (int): Sampling rate of the signals.
    n_fft (int): FFT window size.
    hop_length (int): Number of samples between successive frames.

    Returns:
    tuple: A tuple containing numpy arrays for SDR, SIR, SAR, and ISR for each source.
    """
    # Ensure the spectrograms are the same shape
    assert reference_spectrogram.shape == estimated_spectrogram.shape, "Spectrograms must have the same shape"

    #num_sources, _, _ = reference_spectrograms.shape

    # Convert spectrograms back to time-domain signals
    reference_signal = librosa.istft(reference_spectrogram, hop_length=hop_length)
    estimated_signal = librosa.istft(estimated_spectrogram, hop_length=hop_length)

    # Because all-zero signals aren't allowed
    try:
        # Compute BSS metrics
        sdr, sir, sar, isr = mir_eval.separation.bss_eval_sources(reference_signal, estimated_signal)
        return sdr[0]
    except ValueError:
        return -math.inf


def calculate_ssim(a, b):
    return ssim(a, b, data_range=b.max() - b.min())

    #return {
    #    'total_mse': total_mse,
    #    'mean_mse': mean_mse,
    #    'matching_indices': list(zip(row_ind, col_ind))
    #}

