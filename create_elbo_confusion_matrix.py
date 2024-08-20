import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import pandas as pd

from functions_prior import PriorDataset
from separate_new import get_vaes

import seaborn as sn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_name = 'toy_opti'
dataset_name = 'toy_dataset' if model_name.__contains__('toy') else 'musdb_18_prior'
image_h = 64
image_w = 64

test_datasets = [PriorDataset('test', image_h=image_h, image_w=image_w, name=dataset_name, num_stems=4, debug=False,
                               stem_type=i + 1) for i in range(4)]

num_samples = 128

print(f'Number of samples: {num_samples}')

vaes = get_vaes(model_name, [0, 1, 2, 3])

confusion_matrix = np.zeros((4, 4))

for vae_idx, vae in enumerate(vaes):
    for dataset_idx, dataset in enumerate(test_datasets):
        dataloader = DataLoader(dataset, batch_size=num_samples, shuffle=True, num_workers=12)
        batch = next(iter(dataloader))
        batch = batch['spectrogram'].float().to(device)
        assert batch.shape[0] == num_samples
        # spectrograms = dataset[0:num_samples]['spectrogram'].float().to(device)
        elbo = vae.log_prob(batch)

        confusion_matrix[vae_idx, dataset_idx] += elbo

        confusion_matrix[vae_idx, dataset_idx] /= num_samples

print(confusion_matrix)

confusion_matrix = np.exp(confusion_matrix) / np.sum(np.exp(confusion_matrix), axis=1, keepdims=True)

print(confusion_matrix)

stems = ['Sine', 'Sawtooth', 'Square', 'Triangle'] if dataset_name == 'toy_dataset' else ['Drums', 'Bass', 'Other', 'Vocals']

df_cm = pd.DataFrame(confusion_matrix, index=stems, columns=stems)
# plt.figure(figsize=(10,7))
sn.set(font_scale=1.4) # for label size
sn.heatmap(df_cm, annot=False, annot_kws={"size": 16}) # font size

plt.savefig(f'figures/confusion_matrix_elbo_{model_name}.png')
