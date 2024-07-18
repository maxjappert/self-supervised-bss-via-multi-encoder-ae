import torch

from functions import save_spectrogram_to_file
from functions_prior import PriorDataset, VAE

debug = False
device = torch.device('cpu')# torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_val = PriorDataset('val', debug=debug)

vae = VAE(latent_dim=256).to(device)

name = 'beta_eval_0'

print(name)

vae.load_state_dict(torch.load(f'checkpoints/{name}.pth', map_location=device))

idx = 12
datapoint = dataset_val[idx]

output, _, _ = vae(datapoint['spectrogram'].unsqueeze(dim=0).to(device).float())

print(datapoint['spectrogram'].squeeze().detach().cpu().numpy())
print(output.squeeze().detach().cpu().numpy())

save_spectrogram_to_file(datapoint['spectrogram'].squeeze().detach().cpu().numpy(), f'aaa_gt_{name}.png')
save_spectrogram_to_file(output.squeeze().detach().cpu().numpy(), f'aaa_recon_{name}.png')
