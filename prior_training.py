from torch.utils.data import DataLoader

from functions_prior import train_vae, PriorDataset

dataset_train = PriorDataset('train')
dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)

train_vae(dataloader_train, dataloader_train)
