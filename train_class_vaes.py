import sys

import torch
from torch.nn import MSELoss
from torch.utils.data import DataLoader

from functions_prior import PriorDataset, train_vae

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

image_h = 64
image_w = 64

batch_size = 256

if len(sys.argv) > 1:
    stem_type = int(sys.argv[1])
else:
    stem_type = None

dataset_train = PriorDataset('train', debug=False, name='toy_dataset', image_h=image_h, image_w=image_w, stem_type=stem_type)
dataset_val = PriorDataset('val', debug=False, name='toy_dataset', image_h=image_h, image_w=image_w, stem_type=stem_type)

dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)
dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, num_workers=4)

appendix = f'_stem{stem_type}' if stem_type is not None else ''

train_vae(dataloader_train,
          dataloader_val,
          strides=[1, 1, 1, 1, 1],
          lr=0.001,
          channels=[4, 8, 16, 32, 64],
          name=f'toy_1d{appendix}',
          criterion=MSELoss(),
          epochs=500,
          latent_dim=1,
          visualise=True,
          image_h=image_h,
          image_w=image_w,
          recon_weight=1)

