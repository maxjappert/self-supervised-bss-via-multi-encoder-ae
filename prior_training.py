import sys

import torch
from torch import nn
from torch.nn import BCELoss, MSELoss
from torch.utils.data import DataLoader

from functions_prior import train_vae, PriorDataset, train_classifier, VAE, SDRLoss

debug = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

image_h = 64
image_w = 64

batch_size = 256

dataset_train = PriorDataset('train', debug=debug, name='musdb_18_prior', image_h=image_h, image_w=image_w)
dataset_val = PriorDataset('val', debug=debug, name='musdb_18_prior', image_h=image_h, image_w=image_w)

dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=12)
dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, num_workers=12)

train_vae(dataloader_train,
          dataloader_val,
          strides=[1, 1, 1],
          lr=0.001,
          channels=[4, 8, 16],
          name=f'musdb_tiny_optimal_2_full_all_stems',
          criterion=MSELoss(),
          epochs=200,
          latent_dim=84,
          visualise=True,
          image_h=image_h,
          image_w=image_w)

print()

for i in range(4):
    dataset_train = PriorDataset('train', debug=debug, name='musdb_18_prior', image_h=image_h, image_w=image_w, stem_type=i+1)
    dataset_val = PriorDataset('val', debug=debug, name='musdb_18_prior', image_h=image_h, image_w=image_w, stem_type=i+1)

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=12)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, num_workers=12)

    train_vae(dataloader_train,
              dataloader_val,
              strides=[1, 1, 1],
              lr=0.001,
              channels=[4, 8, 16],
              name=f'musdb_tiny_optimal_2_full_stem{i+1}',
              criterion=MSELoss(),
              epochs=200,
              latent_dim=84,
              visualise=True,
              image_h=image_h,
              image_w=image_w)

    print()

dataset_train = PriorDataset('train', debug=debug, name='toy_dataset', image_h=image_h, image_w=image_w)
dataset_val = PriorDataset('val', debug=debug, name='toy_dataset', image_h=image_h, image_w=image_w)

dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=12)
dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, num_workers=12)

# train_vae(dataloader_train,
#           dataloader_val,
#           kernel_size=hps['kernel_size'],
#           cyclic_lr=True,
#           lr=1e-05,
#           channels=channels[hps['channel_index']],
#           name='cyclic_toy',
#           criterion=SDRLoss(),
#           epochs=50,
#           contrastive_loss=True,
#           use_blocks=True,
#           latent_dim=512,
#           kld_weight=hps['beta'],
#           visualise=True,
#           image_h=1024,
#           image_w=128)

# train_vae(dataloader_train,
#           dataloader_val,
#           kernel_sizes=3,
#           cyclic_lr=False,
#           lr=1e-03,
#           channels=[32, 64, 64, 64],
#           name='cyclic_toy_small',
#           criterion=MSELoss(),
#           epochs=50,
#           contrastive_loss=False,
#           use_blocks=False,
#           latent_dim=8,
#           kld_weight=1,
#           visualise=True,
#           image_h=28,
#           image_w=28)

# train_vae(dataloader_train,
#           dataloader_val,
#           # kernel_sizes=[3, 3, 3, 3, 3],
#           strides=[1, 1, 1, 1, 1],
#           # cyclic_lr=True,
#           lr=0.001,
#           channels=[4, 8, 16, 32, 64],
#           name='toy_tiny_optimised',
#           epochs=50,
#           latent_dim=4,
#           visualise=True,
#           image_h=image_h,
#           image_w=image_w)

#
#vae_block_l1_contrastive = train_vae(dataloader_train, dataloader_val, lr=1e-6, name='test_entangled_block_l1_contrastive', criterion=nn.L1Loss(), epochs=30, contrastive_loss=True, use_blocks=True)
##train_classifier(dataloader_train, dataloader_val, vae_block, lr=1e-3, name='classifier_blocks')
#
#vae_normal_l1_contrastive = train_vae(dataloader_train, dataloader_val, lr=1e-6, name='test_entangled_nonblock_l1_contrastive', criterion=nn.L1Loss(), epochs=30, contrastive_loss=True, use_blocks=False)
##train_classifier(dataloader_train, dataloader_val, vae_normal, lr=1e-3, name='classifier_normal')
#
#vae_block_sdr_contrastive = train_vae(dataloader_train, dataloader_val, lr=1e-6, name='test_entangled_block_sdr_contrastive', criterion=SDRLoss(), epochs=30, contrastive_loss=True, use_blocks=True)
#vae_normal_sdr_contrastive = train_vae(dataloader_train, dataloader_val, lr=1e-6, name='test_entangled_nonblock_sdr_contrastive', criterion=SDRLoss(), epochs=30, contrastive_loss=True, use_blocks=False)
#
#vae_block_l1 = train_vae(dataloader_train, dataloader_val, lr=1e-6, name='test_entangled_block_l1', criterion=nn.L1Loss(), epochs=30, contrastive_loss=False, use_blocks=True)
##train_classifier(dataloader_train, dataloader_val, vae_block, lr=1e-3, name='classifier_blocks')
#
#vae_normal_l1 = train_vae(dataloader_train, dataloader_val, lr=1e-6, name='test_entangled_nonblock_l1', criterion=nn.L1Loss(), epochs=30, contrastive_loss=False, use_blocks=False)
##train_classifier(dataloader_train, dataloader_val, vae_normal, lr=1e-3, name='classifier_normal')
#
#vae_block_sdr = train_vae(dataloader_train, dataloader_val, lr=1e-6, name='test_entangled_block_sdr', criterion=SDRLoss(), epochs=30, contrastive_loss=False, use_blocks=True)
#vae_normal_sdr = train_vae(dataloader_train, dataloader_val, lr=1e-6, name='test_entangled_nonblock_sdr', criterion=SDRLoss(), epochs=30, contrastive_loss=False, use_blocks=False)
#
#
#train_classifier(dataloader_train, dataloader_val, None, lr=1e-3, epochs=20, name='classifier_naive', naive=True, pretrained=False)
