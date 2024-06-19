import pickle
import sys

from audio_spectrogram_conversion_functions import *
from functions import *
import torch.nn.functional as F
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

num_sources = 2

name = 'musdb18_linear'
dataset_name = 'musdb18_two_sources'

# image_height=1025, image_width=216

#model, _, _ = train(dataset_train=TwoSourcesDataset(split='train', name='musdb18_two_sources'), batch_size=32, dataset_val=TwoSourcesDataset(split='validation'), channels=[24, 48, 96, 144], num_encoders=num_sources, image_height=1025, image_width=216, visualise=True, test_save_step=10, name='musdb18_linear', linear=True)

with open(f'{name}.json', 'r') as file:
    hps = json.load(file)

model = get_model(linear=hps['linear'], channels=hps['channels'], hidden=hps['hidden'], num_encoders=num_sources, image_height=1025, image_width=216, norm_type=hps['norm_type'], use_weight_norm=hps['use_weight_norm']).to('cuda')

model.load_state_dict(torch.load(f'{name}_best.pth'))

test(model, TwoSourcesDataset(split='validation', name=dataset_name), visualise=True, name=f'first_spectro_{name}', num_samples=1, single_file=False, linear=hps['linear'])
test(model, TwoSourcesDataset(split='validation', name=dataset_name), visualise=True, name=f'second_spectro_{name}', num_samples=1, single_file=False, linear=hps['linear'])
test(model, TwoSourcesDataset(split='validation', name=dataset_name), visualise=True, name=f'third_spectro_{name}', num_samples=1, single_file=False, linear=hps['linear'])

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
