import json

import torch

from functions import save_spectrogram_to_file
from functions_prior import PriorDataset, VAE

debug = False
device = torch.device('cpu')# torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_val = PriorDataset('val', debug=debug)

names = ['musdb_small_newelbo']

for name in names:
    print(name)

    if name.__contains__('toy'):
        dataset_name = 'toy_dataset'
    else:
        dataset_name = 'musdb_18_prior'

    hps = json.load(open(f'hyperparameters/{name}.json'))

    image_h = hps['image_h']
    image_w = hps['image_w']

    vae = VAE(latent_dim=hps['hidden'], use_blocks=False, image_h=image_h, image_w=image_w, kernel_sizes=hps['kernel_sizes'], strides=hps['strides'], channels=hps['channels']).to(device)

    vae.load_state_dict(torch.load(f'checkpoints/{name}.pth', map_location=device))

    dataset_val = PriorDataset('val', debug=debug, name=dataset_name, image_h=image_h, image_w=image_w)

    idx = 93
    datapoint = dataset_val[idx]

    output, _, _ = vae(datapoint['spectrogram'].unsqueeze(dim=0).to(device).float())

    print(datapoint['spectrogram'].squeeze().detach().cpu().numpy())
    print(output.squeeze().detach().cpu().numpy())

    save_spectrogram_to_file(datapoint['spectrogram'].squeeze().detach().cpu().numpy(), f'aaa_gt_{name}.png')
    save_spectrogram_to_file(output.squeeze().detach().cpu().numpy(), f'aaa_recon_{name}.png')
