from torch.utils.data import DataLoader

from functions_prior import train_vae, PriorDataset

debug = True

dataset_train = PriorDataset('train', debug=debug)
dataset_val = PriorDataset('val', debug=debug)
dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True, num_workers=12)
dataloader_val = DataLoader(dataset_val, batch_size=32, shuffle=True, num_workers=12)

train_vae(dataloader_train, dataloader_val, lr=1e-3, name='test')
