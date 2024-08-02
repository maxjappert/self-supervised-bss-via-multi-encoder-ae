import sys

import torch
from torch.nn import MSELoss
from torch.utils.data import DataLoader

from functions_prior import PriorDataset, train_vae

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

image_h = 1024
image_w = 128

batch_size = 128

debug = False

if debug:
    print('Debug mode activated')

finetune = sys.argv[1] == 'finetune'

print(f'Finetuning is {finetune}')

for stem_type in [1, 2, 3, 4]:
    dataset_train = PriorDataset('train', debug=debug, name='toy_dataset', image_h=image_h, image_w=image_w, stem_type=stem_type)
    dataset_val = PriorDataset('val', debug=debug, name='toy_dataset', image_h=image_h, image_w=image_w, stem_type=stem_type)

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=12)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, num_workers=12)

    appendix = f'_stem{stem_type}' if stem_type is not None else ''

    tr_vae(dataloader_train,
              dataloader_val,
              strides=[1, 1, 1],
              lr=0.0001,
              channels=[4, 8, 16],
              name=f'toy_fullsize{appendix}',
              criterion=MSELoss(),
              epochs=200,
              latent_dim=8,
              visualise=True,
              image_h=image_h,
              image_w=image_w,
              recon_weight=10,
              finetune=True)

