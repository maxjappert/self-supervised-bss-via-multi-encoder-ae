import pickle
import sys

from scipy.io import wavfile

from audio_spectrogram_conversion_functions import *
from functions import *
import torch.nn.functional as F
import fast_bss_eval
#with open('data/slakh_train2.pkl', 'rb') as file:
#    loaded_list_of_lists = pickle.load(file)
#    #print(loaded_list_of_lists)
#
#print(loaded_list_of_lists.shape)
#
#sys.exit(0)

#example = F.avg_pool2d(loaded_list_of_lists[3][1].unsqueeze(0), kernel_size=2)

#print(example.shape)

#print(loaded_list_of_lists.shape)

#
#save_spectrogram_to_file(example.squeeze(), 'example.png')
#
#spectrogram_to_audio('example.png', 22050, 'hello', from_file=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_sources = 2

name = 'musdb18_linear_evaluated_optimal_final'
dataset_name = 'musdb18_two_sources'

print(f'name: {name}')
print(f'dataset: {dataset_name}')

# image_height=1025, image_width=216

channel_options = [[8, 16], [8, 16, 32], [8, 16, 32, 64], [8, 16, 32, 64, 128], [24, 48, 96, 144], [24, 48, 96, 196],
                   [24, 48, 96, 144, 196], [16, 32, 64, 128, 256]]

hps = {'sep_lr': 1.0, 'zero_lr': 0.42, 'hidden': 512, 'channel_index': 2, 'norm_type': 'none', 'weight_decay': 0.0001, 'sep_norm': 'L1', 'batch_size': 32, 'lr': 0.001, 'normalisation': 'minmax', 'linear': True}

#model, _, _ = train(dataset_train=TwoSourcesDataset(split='train', name='musdb18_two_sources'), batch_size=28, lr=1e-4, hidden=512, dataset_val=TwoSourcesDataset(split='validation', name='musdb18_two_sources'), channels=[24, 48, 96, 144, 196], num_encoders=num_sources, image_height=1025, image_width=216, visualise=True, test_save_step=1, name=name, linear=True)
#model, _, _ = train(dataset_train=TwoSourcesDataset(split='train', name='musdb18_two_sources', normalisation=hps['normalisation']), batch_size=hps['batch_size'], lr=hps['lr'], hidden=hps['hidden'], dataset_val=TwoSourcesDataset(split='validation', name='musdb18_two_sources', normalisation=hps['normalisation']), channels=channel_options[hps['channel_index']], num_encoders=num_sources, image_height=1025, image_width=216, visualise=True, test_save_step=1, name=name, linear=hps['linear'])

with open(f'hyperparameters/{name}.json', 'r') as file:
    hps = json.load(file)

model = model_factory(linear=hps['linear'], channels=hps['channels'], hidden=hps['hidden'], num_encoders=num_sources, image_height=1025, image_width=216, norm_type=hps['norm_type'], use_weight_norm=hps['use_weight_norm']).to(device)

model.load_state_dict(torch.load(f'checkpoints/{name}_best_sdr.pth', map_location=device))

sdr1 = test(model, TwoSourcesDataset(split='validation', name=dataset_name), visualise=True, name=f'first_spectro_{name}', num_samples=1, single_file=False, linear=hps['linear'], metric_function=compute_spectral_sdr, random_visualisation=True)
sdr2 = test(model, TwoSourcesDataset(split='validation', name=dataset_name), visualise=True, name=f'second_spectro_{name}', num_samples=1, single_file=False, linear=hps['linear'], metric_function=compute_spectral_sdr, random_visualisation=True)
sdr3 = test(model, TwoSourcesDataset(split='validation', name=dataset_name), visualise=True, name=f'third_spectro_{name}', num_samples=1, single_file=False, linear=hps['linear'], metric_function=compute_spectral_sdr, random_visualisation=True)

print(f'SDRs:')
print(sdr1)
print(sdr2)
print(sdr3)

if not os.path.exists('wavs'):
    os.mkdir('wavs')

spectrogram_to_audio(f'first_spectro_{name}_mix.png', sr=22050, output_filename=f'first_spectro_{name}_mix', from_file=True)
spectrogram_to_audio(f'second_spectro_{name}_mix.png', sr=22050, output_filename=f'second_spectro_{name}_mix', from_file=True)
spectrogram_to_audio(f'third_spectro_{name}_mix.png', sr=22050, output_filename=f'third_spectro_{name}_mix', from_file=True)
spectrogram_to_audio(f'first_spectro_{name}_mix_gt.png', sr=22050, output_filename=f'first_spectro_{name}_mix_gt', from_file=True)
spectrogram_to_audio(f'second_spectro_{name}_mix_gt.png', sr=22050, output_filename=f'second_spectro_{name}_mix_gt', from_file=True)
spectrogram_to_audio(f'third_spectro_{name}_mix_gt.png', sr=22050, output_filename=f'third_spectro_{name}_mix_gt', from_file=True)

for i in range(num_sources):
    spectrogram_to_audio(f'first_spectro_{name}_{i}.png', sr=22050, output_filename=f'first_spectro_{name}_{i}', from_file=True)
    spectrogram_to_audio(f'second_spectro_{name}_{i}.png', sr=22050, output_filename=f'second_spectro_{name}_{i}', from_file=True)
    spectrogram_to_audio(f'third_spectro_{name}_{i}.png', sr=22050, output_filename=f'third_spectro_{name}_{i}', from_file=True)
    spectrogram_to_audio(f'first_spectro_{name}_{i}_gt.png', sr=22050, output_filename=f'first_spectro_{name}_{i}_gt', from_file=True)
    spectrogram_to_audio(f'second_spectro_{name}_{i}_gt.png', sr=22050, output_filename=f'second_spectro_{name}_{i}_gt', from_file=True)
    spectrogram_to_audio(f'third_spectro_{name}_{i}_gt.png', sr=22050, output_filename=f'third_spectro_{name}_{i}_gt', from_file=True)

