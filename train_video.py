import torch
from torch.utils.data import DataLoader

from functions_prior import MultiModalDataset, train_video

def collate_fn(batch):
  return {
      'video': torch.stack([x['video'] for x in batch]),
      'sources': torch.stack([x['sources'] for x in batch]),
      'label': torch.tensor([x['label'] for x in batch])
  }

debug = False

dataset_train = MultiModalDataset('train', debug=debug)
dataset_val = MultiModalDataset('val', debug=debug)

batch_size = 1

dataloader_train = DataLoader(dataset_train, batch_size=batch_size, num_workers=8, collate_fn=collate_fn)
dataloader_val = DataLoader(dataset_val, batch_size=batch_size, num_workers=8, collate_fn=collate_fn)

train_video(dataloader_train, dataloader_val, lr=1e-04, name='video_model')
