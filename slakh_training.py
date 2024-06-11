import pickle
import sys

from audio_spectrogram_conversion_functions import spectrogram_to_audio
from functions import *
import torch.nn.functional as F
with open('data/slakh_validation6.pkl', 'rb') as file:
    loaded_list_of_lists = pickle.load(file)
    #print(loaded_list_of_lists)

print(loaded_list_of_lists[3][1].shape)
#example = F.avg_pool2d(loaded_list_of_lists[3][1].unsqueeze(0), kernel_size=2)

#print(example.shape)

#print(loaded_list_of_lists.shape)

#
#save_spectrogram_to_file(example.squeeze(), 'example.png')
#
#spectrogram_to_audio('example.png', 22050, 'hello', from_file=True)

num_sources = 6

model, _, _ = train(dataset_train=SlakhDataset(split='train', num_sources=num_sources), batch_size=4, dataset_val=SlakhDataset(split='validation', num_sources=num_sources), channels=[24, 48, 90, 138], num_encoders=num_sources, image_height=1025, image_width=216, visualise=True, test_save_step=10, name='first_spectrogram')

with open('first_spectrogram.json', 'r') as file:
    hps = json.load(file)

model = get_model(channels=hps['channels'], num_encoders=num_sources, image_height=512, image_width=108, norm_type=hps['norm_type'], use_weight_norm=hps['use_weight_norm']).to('cuda')

model.load_state_dict(torch.load('first_spectrogram_best.pth'))
test(model, SlakhDataset(split='validation', num_sources=6), visualise=True, name='first_spectro', num_samples=1, single_file=False)
test(model, SlakhDataset(split='validation', num_sources=6), visualise=True, name='second_spectro', num_samples=1, single_file=False)
test(model, SlakhDataset(split='validation', num_sources=6), visualise=True, name='third_spectro', num_samples=1, single_file=False)

spectrogram_to_audio(f'first_spectro_mix.png', sr=22050, output_filename=f'first_spectro_mix', from_file=True)

for i in range(6):
    spectrogram_to_audio(f'first_spectro_{i}.png', sr=22050, output_filename=f'first_spectro_{i}', from_file=True)
    spectrogram_to_audio(f'second_spectro_{i}.png', sr=22050, output_filename=f'second_spectro_{i}', from_file=True)
    spectrogram_to_audio(f'third_spectro_{i}.png', sr=22050, output_filename=f'third_spectro_{i}', from_file=True)
