from torch.utils.data import DataLoader

from functions_prior import train_vae, SDRLoss, PriorDataset

latent_dim = 256

kld_weights = [1e-4] # [0, 1e-5, 1e-3, 1e-1, 1, 2, 4, 6, 8, 10]

debug = False

dataset_train = PriorDataset('train', debug=debug)
dataset_val = PriorDataset('val', debug=debug)

dataloader_train = DataLoader(dataset_train, batch_size=16, shuffle=True, num_workers=12)
dataloader_val = DataLoader(dataset_val, batch_size=16, shuffle=True, num_workers=12)

for beta in kld_weights:
    print(f'beta = {beta}')

    train_vae(dataloader_train, dataloader_val, lr=1e-5, name=f'beta_eval_{str(beta).replace(".", "_")}', criterion=SDRLoss(), kld_weight=beta, epochs=10, contrastive_loss=False, use_blocks=True, latent_dim=latent_dim)
