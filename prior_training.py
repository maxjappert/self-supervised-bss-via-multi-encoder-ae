import sys

import torch
from torch import nn
from torch.nn import BCELoss, MSELoss
from torch.utils.data import DataLoader

from functions_prior import train_vae, PriorDataset, train_classifier, VAE, SDRLoss

debug = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

channels = [[32, 64, 128, 256, 512], [16, 32, 64, 128, 256], [16, 32, 64, 128],
                   [8, 16, 32, 64, 128, 256, 512], [8, 16, 32, 64, 128, 256], [16, 32, 64, 128], [16, 32, 64]]

hps = {'latent_dim': 448, 'channel_index': 3, 'batch_size': 16, 'lr': 1e-04, 'kernel_size': 5, 'beta': 1e-06}

dataset_train = PriorDataset('train', debug=debug, name='musdb_18_prior')
dataset_val = PriorDataset('val', debug=debug, name='musdb_18_prior')

dataloader_train = DataLoader(dataset_train, batch_size=hps['batch_size'], shuffle=True, num_workers=12)
dataloader_val = DataLoader(dataset_val, batch_size=hps['batch_size'], shuffle=True, num_workers=12)


# train_vae(dataloader_train,
#           dataloader_val,
#           kernel_size=hps['kernel_size'],
#           lr=1e-05,
#           cyclic_lr=True,
#           channels=channels[hps['channel_index']],
#           name='cyclic_musdb',
#           criterion=SDRLoss(),
#           epochs=50,
#           contrastive_loss=True,
#           use_blocks=True,
#           contrastive_weight=hps['beta'],
#           latent_dim=512,
#           kld_weight=hps['beta'],
#           visualise=True,
#           image_h=1024,
#           image_w=384)

# train_vae(dataloader_train,
#           dataloader_val,
#           kernel_size=hps['kernel_size'],
#           lr=1e-05,
#           cyclic_lr=True,
#           channels=channels[hps['channel_index']],
#           name='cyclic_musdb_kl',
#           criterion=SDRLoss(),
#           epochs=50,
#           contrastive_loss=False,
#           use_blocks=True,
#           contrastive_weight=hps['beta'],
#           latent_dim=128,
#           kld_weight=1,
#           visualise=True,
#           image_h=1024,
#           image_w=384)

dataset_train = PriorDataset('train', debug=debug, name='toy_dataset', image_h=1024, image_w=128)
dataset_val = PriorDataset('val', debug=debug, name='toy_dataset', image_h=1024, image_w=128)

dataloader_train = DataLoader(dataset_train, batch_size=128, shuffle=True, num_workers=12)
dataloader_val = DataLoader(dataset_val, batch_size=128, shuffle=True, num_workers=12)

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

train_vae(dataloader_train,
          dataloader_val,
          kernel_size=hps['kernel_size'],
          cyclic_lr=True,
          lr=1e-05,
          channels=channels[hps['channel_index']],
          name='cyclic_toy_kl',
          criterion=SDRLoss(),
          epochs=50,
          contrastive_loss=False,
          use_blocks=True,
          latent_dim=128,
          kld_weight=1,
          visualise=True,
          image_h=1024,
          image_w=128)

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
