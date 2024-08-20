import argparse
import sys

import torch
from torch.nn import MSELoss
from torch.utils.data import DataLoader

from functions_prior import PriorDataset, train_vae, PriorDatasetVideo

def collate_fn(batch):
  return {
      'spectrogram': torch.stack([x['spectrogram'] for x in batch]),
  }

def train_video_vae(name: str, stem: str):

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

    batch_size = 128

    dataset_train = PriorDatasetVideo('train', debug=False, image_h=image_h, image_w=image_w, stem_type=stem)
    dataset_val = PriorDatasetVideo('val', debug=False, image_h=image_h, image_w=image_w, stem_type=stem)

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn)

    appendix = f'_{stem}' if stem is not None else ''

    train_vae(dataloader_train,
              dataloader_val,
              strides=[1, 1, 1],
              lr=0.001,
              kernel_sizes=[3, 3, 3],
              channels=[8, 16, 32],
              name=name + appendix,
              criterion=MSELoss(),
              epochs=500,
              latent_dim=32,
              visualise=True,
              image_h=image_h,
              image_w=image_w,
              batch_norm=False,
              recon_weight=10)

train_video_vae(sys.argv[1], sys.argv[1])

# train_video_vae('vn, 'vn')
# train_video_vae('vc', 'vc')
# train_video_vae('fl', 'fl')
# train_video_vae('cl', 'cl')
# train_video_vae('tp', 'tp')
# train_video_vae('tb', 'tb')
# train_video_vae('sax','sax')
# train_video_vae('ob', 'ob')


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

