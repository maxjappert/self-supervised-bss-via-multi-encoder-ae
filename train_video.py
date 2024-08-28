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

if debug:
    print('DEBUG MODE ACTIVATED')

dataset_train = MultiModalDataset('train', debug=debug, fps=15, normalise=True)
dataset_val = MultiModalDataset('val', debug=debug, fps=15, normalise=True)

# print(len(dataset_train))

# batch_size = 3

# dataloader_train = DataLoader(dataset_train, batch_size=batch_size, num_workers=20, collate_fn=collate_fn)
# dataloader_val = DataLoader(dataset_val, batch_size=batch_size, num_workers=20, collate_fn=collate_fn)

# train_video(dataloader_train, dataloader_val, lr=1e-04, name='video_model_raft_resnet', use_optical_flow=True, use_resnet=True, epochs=50)

# train_video(dataloader_train, dataloader_val, lr=1e-04, name='video_model_raft', use_optical_flow=True, use_resnet=False, epochs=50)

batch_size = 12

dataloader_train = DataLoader(dataset_train, batch_size=batch_size, num_workers=24, collate_fn=collate_fn)
dataloader_val = DataLoader(dataset_val, batch_size=batch_size, num_workers=24, collate_fn=collate_fn)

# train_video(dataloader_train, dataloader_val, lr=1e-04, name='video_model_simple', use_optical_flow=False, use_resnet=False, epochs=50)

train_video(dataloader_train, dataloader_val, lr=1e-04, name='video_model_resnet', use_optical_flow=False, use_resnet=True, epochs=50)
