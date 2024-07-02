import matplotlib.pyplot as plt
import numpy as np
import fast_bss_eval
import scipy
from PIL import Image
from scipy.io import wavfile

from audio_spectrogram_conversion_functions import spectrogram_to_audio, audio_to_spectrogram, \
    numpy_audio_to_spectrogram
from evaluation_metric_functions import compute_spectral_sdr
from functions import load_model, TwoSourcesDataset, test, get_linear_separation, get_reconstruction, \
    save_spectrogram_to_file

def get_reconstruction_set(index, model_name='musdb18_linear_evaluated_optimal_final'):
    model = load_model(model_name)

    val_dataset = TwoSourcesDataset(split='validation', name='musdb18_two_sources')

    datapoint = val_dataset[index]

    recon_spectro_mix = get_reconstruction(model, datapoint)
    recon_spectro_s1, recon_spectro_s2 = get_linear_separation(model, datapoint)

    gt_spectro_mix = datapoint[0].numpy().squeeze()
    gt_spectro_s1 = datapoint[1].numpy().squeeze()
    gt_spectro_s2 = datapoint[2].numpy().squeeze()

    save_spectrogram_to_file(gt_spectro_mix, 'spectro_gt_mix.png')
    save_spectrogram_to_file(gt_spectro_s1, 'spectro_gt_s1.png')
    save_spectrogram_to_file(gt_spectro_s2, 'spectro_gt_s2.png')

    wav_gt_mix = spectrogram_to_audio('spectro_gt_mix.png', 22050, 'wav_gt_mix', from_file=True)
    wav_gt_s1 = spectrogram_to_audio('spectro_gt_s1.png', 22050, 'wav_gt_s1', from_file=True)
    wav_gt_s2 = spectrogram_to_audio('spectro_gt_s2.png', 22050, 'wav_gt_s2', from_file=True)

    save_spectrogram_to_file(recon_spectro_mix, 'spectro_recon_mix.png')
    save_spectrogram_to_file(recon_spectro_s1, 'spectro_recon_s1.png')
    save_spectrogram_to_file(recon_spectro_s2, 'spectro_recon_s2.png')

    wav_recon_mix = spectrogram_to_audio('spectro_recon_mix.png', 22050, 'wav_recon_mix', from_file=True)
    wav_recon_s1 = spectrogram_to_audio('spectro_recon_s1.png', 22050, 'wav_recon_s1', from_file=True)
    wav_recon_s2 = spectrogram_to_audio('spectro_recon_s2.png', 22050, 'wav_recon_s2', from_file=True)

    return wav_gt_mix, wav_gt_s1, wav_gt_s2, wav_recon_mix, wav_recon_s1, wav_recon_s2

wav_gt_mix, wav_gt_s1, wav_gt_s2, wav_recon_mix, wav_recon_s1, wav_recon_s2 = get_reconstruction_set(19)
wav_gt_mix1, wav_gt_s11, wav_gt_s21, wav_recon_mix1, wav_recon_s11, wav_recon_s21 = get_reconstruction_set(19)
wav_gt_mix2, wav_gt_s12, wav_gt_s22, wav_recon_mix2, wav_recon_s12, wav_recon_s22 = get_reconstruction_set(19)

print(np.isclose(wav_gt_mix1, wav_gt_mix2).sum())
print(np.isclose(numpy_audio_to_spectrogram(wav_gt_mix1), numpy_audio_to_spectrogram(wav_gt_mix2)).sum())

plt.imsave(f'images/v1.png', numpy_audio_to_spectrogram(wav_gt_mix1), cmap='gray')
plt.imsave(f'images/v2.png', numpy_audio_to_spectrogram(wav_gt_mix2), cmap='gray')

#print(wav_gt_s11)
#print(wav_gt_s2)
#print(wav_recon_s1)
#print(wav_recon_s2)

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


s1_target, e1_interf, e1_noise, e1_artif = compute_decomposition(wav_gt_s1, wav_gt_s2, wav_recon_s1, wav_recon_s2, 0)
s2_target, e2_interf, e2_noise, e2_artif = compute_decomposition(wav_gt_s1, wav_gt_s2, wav_recon_s1, wav_recon_s2, 1)

# print(s1_target)
# print(e1_interf)
# print(e1_noise)
# print(e1_artif)
#
# scipy.io.wavfile.write(f'wavs/s1.wav', 22050, np.array(wav_gt_s1, dtype=np.int16))
# scipy.io.wavfile.write(f'wavs/s1_target.wav', 22050, np.array(s1_target, dtype=np.int16))
# scipy.io.wavfile.write(f'wavs/e1_interf.wav', 22050, np.array(e1_interf, dtype=np.int16))
# scipy.io.wavfile.write(f'wavs/e1_noise.wav', 22050, np.array(e1_noise, dtype=np.int16))
# scipy.io.wavfile.write(f'wavs/e1_artif.wav', 22050, np.array(e1_artif, dtype=np.int16))
#
numpy_audio_to_spectrogram(wav_gt_s1, name='s1.png')
numpy_audio_to_spectrogram(s1_target, name='s1_target.png')
numpy_audio_to_spectrogram(e1_interf, name='e1_interf.png')
numpy_audio_to_spectrogram(e1_noise, name='e1_noise.png')
numpy_audio_to_spectrogram(e1_artif, name='e1_artif.png')
numpy_audio_to_spectrogram(e1_interf + e1_noise + e1_artif, name='sdr_denominator')


def compute_sdr(s_target, e_interf, e_noise, e_artif):
    return 10 * np.log10(np.sum(s_target) ** 2 / np.sum(e_interf + e_noise + e_artif) ** 2)


print(compute_sdr(s1_target, e1_interf, e1_noise, e1_artif))
print(compute_sdr(s2_target, e2_interf, e2_noise, e2_artif))

