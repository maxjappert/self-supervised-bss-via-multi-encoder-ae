import argparse
import sys

import torch
from torch.nn import MSELoss
from torch.utils.data import DataLoader

from functions_prior import PriorDataset, train_vae

def train_class_vaes(name: str, stem: int):

    hps_opti_toy = {'latent_dim': 12, 'depth': 3, 'num_channels': 12, 'stride_ends': 1, 'stride_middle': 2, 'kernel_size': 7, 'batch_norm': False}
    hps_opti_musdb = {'latent_dim': 40, 'depth': 2, 'num_channels': 12, 'stride_ends': 2, 'stride_middle': 1, 'kernel_size': 5, 'batch_norm': False}
    # parser = argparse.ArgumentParser(
    #                     prog='ProgramName',
    #                     description='What the program does',
    #                     epilog='Text at the bottom of help')
    #
    # parser.add_argument('--name')      # option that takes a
    # parser.add_argument('--stem', default=None)
    # parser.add_argument('--latent_dim', default=8)

    # args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    image_h = 64
    image_w = 64

    batch_size = 256

    dataset_name = 'toy_dataset' if name.__contains__('toy') else 'musdb_18_prior'

    dataset_train = PriorDataset('train', debug=False, name=dataset_name, image_h=image_h, image_w=image_w, stem_type=int(stem))
    dataset_val = PriorDataset('val', debug=False, name=dataset_name, image_h=image_h, image_w=image_w, stem_type=int(stem))

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=12)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, num_workers=12)

    appendix = f'_stem{stem}' if stem is not None else ''

    num_channels = hps_opti_toy['num_channels']
    depth = hps_opti_toy['depth']

    channels = [num_channels**i for i in range(1, depth+1)]

    strides = []

    for i in range(depth):
        if i == 0 or i == depth - 1:
            strides.append(hps_opti_toy['stride_ends'])
        else:
            strides.append(hps_opti_toy['stride_middle'])

    train_vae(dataloader_train,
              dataloader_val,
              strides=strides,
              lr=0.001,
              kernel_sizes=[hps_opti_toy['kernel_size'] for _ in range(depth)],
              channels=channels,
              name=name + appendix,
              criterion=MSELoss(),
              epochs=500,
              latent_dim=hps_opti_toy['latent_dim'],
              visualise=True,
              image_h=image_h,
              image_w=image_w,
              batch_norm=hps_opti_toy['batch_norm'],
              recon_weight=10)

# train_vae(dataloader_train,
#           dataloader_val,
#           strides=[1, 1, 1, 1],
#           lr=0.001,
#           channels=[4, 8, 16, 32],
#           name=args.name + appendix,
#           criterion=MSELoss(),
#           epochs=500,
#           latent_dim=int(args.latent_dim),
#           visualise=True,
#           image_h=image_h,
#           image_w=image_w,
#           recon_weight=10)

