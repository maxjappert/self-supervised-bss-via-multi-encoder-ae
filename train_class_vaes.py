import argparse
import sys

import torch
from torch.nn import MSELoss
from torch.utils.data import DataLoader

from functions_prior import PriorDataset, train_vae

parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')

parser.add_argument('--name')      # option that takes a
parser.add_argument('--stem', default=None)
parser.add_argument('--latent_dim', default=8)

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

image_h = 64
image_w = 64

batch_size = 128

dataset_name = 'toy_dataset' if args.name.__contains__('toy') else 'musdb_18_prior'

dataset_train = PriorDataset('train', debug=False, name=dataset_name, image_h=image_h, image_w=image_w, stem_type=int(args.stem))
dataset_val = PriorDataset('val', debug=False, name=dataset_name, image_h=image_h, image_w=image_w, stem_type=int(args.stem))

dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)
dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, num_workers=4)

appendix = f'_stem{args.stem}' if args.stem is not None else ''

train_vae(dataloader_train,
          dataloader_val,
          strides=[1, 1, 1, 1],
          lr=0.001,
          channels=[4, 8, 16, 32],
          name=args.name + appendix,
          criterion=MSELoss(),
          epochs=500,
          latent_dim=int(args.latent_dim),
          visualise=True,
          image_h=image_h,
          image_w=image_w,
          recon_weight=10)

