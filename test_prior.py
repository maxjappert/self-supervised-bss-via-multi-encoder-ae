import torch

from functions import save_spectrogram_to_file
from functions_prior import PriorDataset, VAE

debug = False
device = torch.device('cpu')# torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_val = PriorDataset('val', debug=debug)

names = ['optimal1_toy', 'optimal1_toy_contrastive', 'optimal1_musdb', 'optimal1_musdb_contrastive']

for name in names:
    print(name)

    if name.__contains__('toy'):
        image_w = 128
        dataset_name = 'toy_dataset'
    else:
        image_w = 384
        dataset_name = 'musdb_18_prior'

    vae = VAE(latent_dim=448, image_h=1024, image_w=image_w, kernel_size=5, channels=[8, 16, 32, 64, 128, 256, 512]).to(device)

    vae.load_state_dict(torch.load(f'checkpoints/{name}.pth', map_location=device))

    dataset_val = PriorDataset('val', debug=debug, name=dataset_name, image_h=1024, image_w=image_w)

    idx = 19
    datapoint = dataset_val[idx]

    output, _, _ = vae(datapoint['spectrogram'].unsqueeze(dim=0).to(device).float())

    print(datapoint['spectrogram'].squeeze().detach().cpu().numpy())
    print(output.squeeze().detach().cpu().numpy())

    save_spectrogram_to_file(datapoint['spectrogram'].squeeze().detach().cpu().numpy(), f'aaa_gt_{name}.png')
    save_spectrogram_to_file(output.squeeze().detach().cpu().numpy(), f'aaa_recon_{name}.png')
