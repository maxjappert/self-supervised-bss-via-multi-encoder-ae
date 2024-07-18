import sys

import torch
from torch import nn
from torch.utils.data import DataLoader

from functions_prior import train_vae, PriorDataset, train_classifier, VAE, SDRLoss

debug = False

device = torch.device('cuda')

hps = {}
channels = []

dataset_train = PriorDataset('train', debug=debug, name='musdb_18_prior')
dataset_val = PriorDataset('val', debug=debug, name='musdb_18_prior')

dataloader_train = DataLoader(dataset_train, batch_size=hps['batch_size'], shuffle=True, num_workers=12)
dataloader_val = DataLoader(dataset_val, batch_size=hps['batch_size'], shuffle=True, num_workers=12)

train_vae(dataloader_train,
          dataloader_val,
          kernel_size=hps['kernel_size'],
          lr=hps['lr'],
          channels=channels[hps['channel_index']],
          name='optimal1_musdb',
          criterion=SDRLoss,
          epochs=30,
          contrastive_loss=hps['contrastive_loss'],
          use_blocks=True,
          contrastive_weight=hps['beta'],
          latent_dim=hps['latent_dim'],
          kld_weight=hps['kld_weight'],
          visualise=True,
          image_h=1024,
          image_w=384)

dataset_train = PriorDataset('train', debug=debug, name='toy_dataset', image_h=1024, image_w=128)
dataset_val = PriorDataset('val', debug=debug, name='toy_dataset', image_h=1024, image_w=128)

dataloader_train = DataLoader(dataset_train, batch_size=hps['batch_size'], shuffle=True, num_workers=12)
dataloader_val = DataLoader(dataset_val, batch_size=hps['batch_size'], shuffle=True, num_workers=12)

train_vae(dataloader_train,
          dataloader_val,
          kernel_size=hps['kernel_size'],
          lr=hps['lr'],
          channels=channels[hps['channel_index']],
          name='optimal1_toy',
          criterion=SDRLoss,
          epochs=30,
          contrastive_loss=hps['contrastive_loss'],
          use_blocks=True,
          contrastive_weight=hps['beta'],
          latent_dim=hps['latent_dim'],
          kld_weight=hps['kld_weight'],
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
