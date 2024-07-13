import copy
import math
import numbers
import os
import sys
import warnings

import librosa
import mir_eval
import numpy as np
import torch
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from skimage.metrics import structural_similarity as ssim

from audio_spectrogram_conversion_functions import spectrogram_to_audio


def compute_spectral_metrics(gt_spectros, approx_spectros, phases):
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

    assert len(gt_spectros) == len(approx_spectros) == len(phases)

    if type(gt_spectros) is torch.Tensor:
        gt_spectros = gt_spectros.detach().cpu().numpy()
        approx_spectros = approx_spectros.detach().cpu().numpy()
        gt_spectros_new = []
        approx_spectros_new = []

        threshold = 5
        for i in range(len(gt_spectros)):
            if i > threshold:
                break
            gt_spectros_new.append(np.array(gt_spectros[i].squeeze()))
            approx_spectros_new.append(np.array(approx_spectros[i].squeeze()))

        gt_spectros = gt_spectros_new
        approx_spectros = approx_spectros_new

    gt_wavs = []
    approx_wavs = []

    for i in range(len(gt_spectros)):
        gt_wavs.append(spectrogram_to_audio(gt_spectros[i], None, phase=phases[i]))
        approx_wavs.append(spectrogram_to_audio(approx_spectros[i], None, phase=phases[i]))

    separation = mir_eval.separation.bss_eval_sources(np.vstack(gt_wavs), np.vstack(approx_wavs))

    return separation


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


def visualise_predictions(x_gt, x_i_gts, x_pred, x_i_preds: list, name='test'):
    """
    Save an image file containing a visualisation of the separation.
    :param x_gt: Mixed ground truth spectrogram.
    :param x_i_gts: Array of ground truth stem spectrograms.
    :param x_pred: Approximated mixed spectrogram.
    :param x_i_preds: Approximated stem spectrograms as array.
    :param name: Filename.
    :return: None.
    """

    assert len(x_i_preds) == len(x_i_gts)

    num_sources = len(x_i_preds)

    fig = plt.figure(figsize=(2*num_sources, 6))
    grid = ImageGrid(fig, 111,
                    nrows_ncols=(2, 1 + num_sources),
                    axes_pad=0.15,
                    )

    #labels = ['Mixed', 'Circle', 'Triangle']
    #images = [sample, circle, triangle, None, x_pred] + x_i_preds
    images = [x_gt] + x_i_gts + [x_pred] + x_i_preds
    y_labels = ['True', 'Pred.']
    for i, (ax, im) in enumerate(zip(grid, images)):
        #if i < num_sources + 1:
        #    ax.set_title(labels[i])
        #if i % 4 == 0:
        #    ax.set_ylabel(y_labels[(i)//4])
        #if i+1 == len(images):
        #    ax.set_title('(Dead Enc.)', color='gray', fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(im, cmap='gray')

    if not os.path.exists('images'):
        os.mkdir('images')

    plt.savefig(f'images/{name}.png')
    #plt.show()
