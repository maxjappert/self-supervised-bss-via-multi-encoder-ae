import torch
from torch.utils.data import DataLoader

from functions_prior import MultiModalDataset, train_video

def collate_fn(batch):
  return {
      'video': torch.stack([x['video'] for x in batch]),
      'sources': torch.stack([x['sources'] for x in batch]),
      'label': torch.tensor([x['label'] for x in batch])
  }

dataset_train = MultiModalDataset('train')
dataset_val = MultiModalDataset('val')

batch_size = 2

dataloader_train = DataLoader(dataset_train, batch_size=batch_size, num_workers=4, collate_fn=collate_fn)
dataloader_val = DataLoader(dataset_val, batch_size=batch_size, num_workers=4, collate_fn=collate_fn)

train_video(dataloader_train, dataloader_val)