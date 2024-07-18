import sys

import torch
from torch import nn
from torch.utils.data import DataLoader

from functions_prior import train_vae, PriorDataset, train_classifier, VAE, SDRLoss

debug = False

device = torch.device('cuda')

dataset_train = PriorDataset('train', debug=debug)
dataset_val = PriorDataset('val', debug=debug)

#train_classifier(dataloader_train, dataloader_val, None, lr=1e-3, name='classifier_naive_pretrained', naive=True, pretrained=True)

dataloader_train = DataLoader(dataset_train, batch_size=16, shuffle=True, num_workers=12)
dataloader_val = DataLoader(dataset_val, batch_size=16, shuffle=True, num_workers=12)

train_vae(dataloader_train, dataloader_val, kernel_size=3, lr=1e-6, channels=[8, 16, 32], name='test_entangled_block_l1_contrastive', criterion=nn.L1Loss(), epochs=30, contrastive_loss=True, use_blocks=True)

sys.exit(0)

vae_block_l1_contrastive = train_vae(dataloader_train, dataloader_val, lr=1e-6, name='test_entangled_block_l1_contrastive', criterion=nn.L1Loss(), epochs=30, contrastive_loss=True, use_blocks=True)
#train_classifier(dataloader_train, dataloader_val, vae_block, lr=1e-3, name='classifier_blocks')

vae_normal_l1_contrastive = train_vae(dataloader_train, dataloader_val, lr=1e-6, name='test_entangled_nonblock_l1_contrastive', criterion=nn.L1Loss(), epochs=30, contrastive_loss=True, use_blocks=False)
#train_classifier(dataloader_train, dataloader_val, vae_normal, lr=1e-3, name='classifier_normal')

vae_block_sdr_contrastive = train_vae(dataloader_train, dataloader_val, lr=1e-6, name='test_entangled_block_sdr_contrastive', criterion=SDRLoss(), epochs=30, contrastive_loss=True, use_blocks=True)
vae_normal_sdr_contrastive = train_vae(dataloader_train, dataloader_val, lr=1e-6, name='test_entangled_nonblock_sdr_contrastive', criterion=SDRLoss(), epochs=30, contrastive_loss=True, use_blocks=False)

vae_block_l1 = train_vae(dataloader_train, dataloader_val, lr=1e-6, name='test_entangled_block_l1', criterion=nn.L1Loss(), epochs=30, contrastive_loss=False, use_blocks=True)
#train_classifier(dataloader_train, dataloader_val, vae_block, lr=1e-3, name='classifier_blocks')

vae_normal_l1 = train_vae(dataloader_train, dataloader_val, lr=1e-6, name='test_entangled_nonblock_l1', criterion=nn.L1Loss(), epochs=30, contrastive_loss=False, use_blocks=False)
#train_classifier(dataloader_train, dataloader_val, vae_normal, lr=1e-3, name='classifier_normal')

vae_block_sdr = train_vae(dataloader_train, dataloader_val, lr=1e-6, name='test_entangled_block_sdr', criterion=SDRLoss(), epochs=30, contrastive_loss=False, use_blocks=True)
vae_normal_sdr = train_vae(dataloader_train, dataloader_val, lr=1e-6, name='test_entangled_nonblock_sdr', criterion=SDRLoss(), epochs=30, contrastive_loss=False, use_blocks=False)


train_classifier(dataloader_train, dataloader_val, None, lr=1e-3, epochs=20, name='classifier_naive', naive=True, pretrained=False)
