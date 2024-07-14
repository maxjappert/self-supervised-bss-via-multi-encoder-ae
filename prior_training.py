import torch
from torch.utils.data import DataLoader

from functions_prior import train_vae, PriorDataset, train_classifier, VAE

debug = False

device = torch.device('cuda')

dataset_train = PriorDataset('train', debug=debug)
dataset_val = PriorDataset('val', debug=debug)
dataloader_train = DataLoader(dataset_train, batch_size=16, shuffle=True, num_workers=12)
dataloader_val = DataLoader(dataset_val, batch_size=16, shuffle=True, num_workers=12)

vae_block = train_vae(dataloader_train, dataloader_val, lr=1e-4, name='test_entangled_block', epochs=20, contrastive_loss=True, use_blocks=True)
train_classifier(dataloader_train, dataloader_val, vae_block, lr=1e-4, name='classifier_blocks')

print()
vae_normal = train_vae(dataloader_train, dataloader_val, lr=1e-4, name='test_entangled_nonblock', epochs=20, contrastive_loss=True, use_blocks=False)
train_classifier(dataloader_train, dataloader_val, vae_normal, lr=1e-4, name='classifier_normal')
