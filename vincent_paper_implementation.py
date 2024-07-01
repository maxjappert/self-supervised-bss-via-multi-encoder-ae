import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.io import wavfile

from audio_spectrogram_conversion_functions import spectrogram_to_audio
from evaluation_metric_functions import compute_spectral_sdr
from functions import load_model, TwoSourcesDataset, test, get_linear_separation, get_reconstruction, \
    save_spectrogram_to_file

#gt_spectro = np.array(Image.open('data/musdb18_two_sources/train/combination25436/mix.png'))

model = load_model('musdb18_linear_evaluated_optimal_final')

val_dataset = TwoSourcesDataset(split='validation', name='musdb18_two_sources')

datapoint = val_dataset[19]

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

wav_gt_sources = [wav_gt_s1, wav_gt_s2]

# s_hat = s_target + e_interf + e_noise + e_artif

# for s1:

R = np.zeros((2, 2))

for i in range(2):
    for j in range(2):
        R[i, j] = np.dot(wav_gt_sources[i], wav_gt_sources[j])

# compute c

c = np.conjugate(np.linalg.inv(R)@[np.dot(wav_recon_s1, wav_gt_s1), np.dot(wav_recon_s1, wav_gt_s2)]).T

s = np.vstack(wav_gt_sources)

P_s_s_hat_j = np.conjugate(c).T@s

n1 = wav_recon_s1 - wav_gt_s1
n2 = wav_recon_s2 - wav_gt_s2

P_sn_s_hat_j = P_s_s_hat_j + np.dot(wav_recon_s1, n1) * n1 / np.sum(n1)**2 + np.dot(wav_recon_s1, n2) * n2 / np.sum(n2)**2

s1_target = np.dot(wav_recon_s1, wav_gt_s1) * wav_gt_s1 / np.sum(wav_gt_s1) ** 2
e_interf = P_s_s_hat_j - s1_target
e_noise = P_sn_s_hat_j - P_s_s_hat_j
e_artif = wav_recon_s1 - P_sn_s_hat_j

print(s1_target + e_interf + e_noise + e_artif)
print(wav_recon_s1)


#sdr = 10 * np.log10(sum(s1_target) ** 2 / sum(e_interf) ** 2)

