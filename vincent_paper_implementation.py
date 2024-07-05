import sys

import matplotlib.pyplot as plt
import numpy as np
import fast_bss_eval
import scipy
from PIL import Image
from scipy.io import wavfile
import mir_eval

from audio_spectrogram_conversion_functions import spectrogram_to_audio, audio_to_spectrogram
from evaluation_metric_functions import compute_spectral_sdr
from functions import load_model, TwoSourcesDataset, test, get_linear_separation, get_reconstruction, \
    save_spectrogram_to_file

def get_reconstruction_set(index, model_name='musdb18_linear_evaluated_optimal_final'):
    model = load_model(model_name)

    dataset = TwoSourcesDataset(split='validation', name='musdb18_two_sources')

    datapoint = dataset[index]

    recon_spectro_mix = get_reconstruction(model, datapoint)
    recon_spectro_s1, recon_spectro_s2 = get_linear_separation(model, datapoint)

    gt_spectro_mix = datapoint[0].numpy().squeeze()
    gt_spectro_s1 = datapoint[1].numpy().squeeze()
    gt_spectro_s2 = datapoint[2].numpy().squeeze()

    save_spectrogram_to_file(gt_spectro_mix, 'spectro_gt_mix.png')
    save_spectrogram_to_file(gt_spectro_s1, 'spectro_gt_s1.png')
    save_spectrogram_to_file(gt_spectro_s2, 'spectro_gt_s2.png')

    phases = dataset.get_phase(index)

    wav_gt_mix = spectrogram_to_audio('spectro_gt_mix.png', 'wav_gt_mix', phase=phases[0], from_file=True)
    wav_gt_s1 = spectrogram_to_audio('spectro_gt_s1.png', 'wav_gt_s1', phase=phases[1], from_file=True)
    wav_gt_s2 = spectrogram_to_audio('spectro_gt_s2.png', 'wav_gt_s2', phase=phases[2], from_file=True)

    save_spectrogram_to_file(recon_spectro_mix, 'spectro_recon_mix.png')
    save_spectrogram_to_file(recon_spectro_s1, 'spectro_recon_s1.png')
    save_spectrogram_to_file(recon_spectro_s2, 'spectro_recon_s2.png')

    wav_recon_mix = spectrogram_to_audio('spectro_recon_mix.png',  'wav_recon_mix', phase=phases[0], from_file=True)
    wav_recon_s1 = spectrogram_to_audio('spectro_recon_s1.png', 'wav_recon_s1', phase=phases[1], from_file=True)
    wav_recon_s2 = spectrogram_to_audio('spectro_recon_s2.png', 'wav_recon_s2', phase=phases[2], from_file=True)

    return wav_gt_mix, wav_gt_s1, wav_gt_s2, wav_recon_mix, wav_recon_s1, wav_recon_s2, gt_spectro_mix, gt_spectro_s1, gt_spectro_s2, recon_spectro_mix, recon_spectro_s1, recon_spectro_s2


wav_gt_mix, wav_gt_s1, wav_gt_s2, wav_recon_mix, wav_recon_s1, wav_recon_s2, gt_spectro_mix, gt_spectro_s1, gt_spectro_s2, recon_spectro_mix, recon_spectro_s1, recon_spectro_s2 = get_reconstruction_set(2)
wav_gt_mix1, wav_gt_s11, wav_gt_s21, wav_recon_mix1, wav_recon_s11, wav_recon_s21, _, _, _, _, _, _ = get_reconstruction_set(2)
wav_gt_mix2, wav_gt_s12, wav_gt_s22, wav_recon_mix2, wav_recon_s12, wav_recon_s22, _, _, _, _, _, _ = get_reconstruction_set(2)

audio_to_spectrogram(wav_recon_s1, 'spectro_gt_s1_reconverted.png')

def compute_decomposition(wav_gt_s1, wav_gt_s2, wav_recon_s1, wav_recon_s2, source_idx):
    """
    Compute the decomposition of a source signal reconstruction as described by  Vincent et al. (2006).
    """

    wav_gt_sources = [wav_gt_s1, wav_gt_s2]
    wav_recon_sources = [wav_recon_s1, wav_recon_s2]

    s_hat_j = wav_recon_sources[source_idx]
    s_j = wav_gt_sources[source_idx]

    R = np.zeros((2, 2))

    for i in range(2):
        for j in range(2):
            R[i, j] = np.dot(wav_gt_sources[i], wav_gt_sources[j])

    # compute c

    c = np.conjugate(np.linalg.inv(R)@[np.dot(s_hat_j, wav_gt_s1), np.dot(s_hat_j, wav_gt_s2)]).T

    s = np.vstack(wav_gt_sources)

    P_s_s_hat_j = np.conjugate(c).T@s

    n1 = wav_recon_s1 - wav_gt_s1
    n2 = wav_recon_s2 - wav_gt_s2

    P_sn_s_hat_j = P_s_s_hat_j + np.dot(s_hat_j, n1) * n1 / np.sum(n1)**2 + np.dot(s_hat_j, n2) * n2 / np.sum(n2)**2

    s_target = np.dot(s_hat_j, s_j) * s_j / np.sum(s_j) ** 2
    e_interf = P_s_s_hat_j - s_target
    e_noise = P_sn_s_hat_j - P_s_s_hat_j
    e_artif = s_hat_j - P_sn_s_hat_j

    assert np.allclose(s_hat_j, s_target + e_interf + e_noise + e_artif)

    return s_target, e_interf, e_noise, e_artif


def compute_sdr(s_target, e_interf, e_noise, e_artif):
    return 10 * np.log10(np.sum(s_target) ** 2 / np.sum(e_interf + e_noise + e_artif) ** 2)


for _ in range(5):
    wav_gt_mix, wav_gt_s1, wav_gt_s2, wav_recon_mix, wav_recon_s1, wav_recon_s2, gt_spectro_mix, gt_spectro_s1, gt_spectro_s2, recon_spectro_mix, recon_spectro_s1, recon_spectro_s2 = get_reconstruction_set(2)

    s1_target, e1_interf, e1_noise, e1_artif = compute_decomposition(wav_gt_s1, wav_gt_s2, wav_recon_s1, wav_recon_s2, 0)
    s2_target, e2_interf, e2_noise, e2_artif = compute_decomposition(wav_gt_s1, wav_gt_s2, wav_recon_s1, wav_recon_s2, 1)

    print(f'Manual SDR: [{compute_sdr(s1_target, e1_interf, e1_noise, e1_artif)}, {compute_sdr(s2_target, e2_interf, e2_noise, e2_artif)}]')

    print(f'Old spectral SDR: [{compute_spectral_sdr(gt_spectro_s1, recon_spectro_s1)}, {compute_spectral_sdr(gt_spectro_s2, recon_spectro_s2)}]')

    #print(f'Manual SI-SDR: [{compute_si_sdr(wav_gt_s1, wav_recon_s2)}, {compute_si_sdr(wav_gt_s2, wav_recon_s2)}]')

    print('\nMir_eval_SDR:')
    print(mir_eval.separation.bss_eval_sources(np.vstack([wav_gt_s1, wav_gt_s2]), np.vstack([wav_recon_s1, wav_recon_s2])))
    print('\n')

#print(mir_eval.separation.bss_eval_sources(np.vstack([wav_gt_s1, wav_gt_s2]), np.vstack([wav_recon_s1, wav_recon_s2])))
#print(mir_eval.separation.bss_eval_sources(np.vstack([wav_gt_s2, wav_gt_s1]), np.vstack([wav_recon_s1, wav_recon_s2])))
#print(mir_eval.separation.bss_eval_sources(np.vstack([wav_gt_s11, wav_gt_s21]), np.vstack([wav_recon_s11, wav_recon_s21])))
#print(mir_eval.separation.bss_eval_sources(np.vstack([wav_gt_s12, wav_gt_s22]), np.vstack([wav_recon_s12, wav_recon_s22])))
