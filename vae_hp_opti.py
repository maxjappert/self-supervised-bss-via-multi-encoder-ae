import math
import traceback

import optuna
import torch
from torch.nn import BCELoss
from torch.utils.data import DataLoader

from evaluation_metric_functions import compute_spectral_sdr
from functions import model_factory, CircleTriangleDataset, train, test, TwoSourcesDataset
from functions_prior import PriorDataset, train_vae, SDRLoss


def objective(trial):
    #channel_options = [[2, 4, 8], [4, 8, 16], [8, 16, 32], [16, 32, 64], [32, 64, 128], [64, 128, 256], [24, 48, 96, 144], [16, 32, 64, 128], [16, 32, 64, 128, 256], [24, 48, 96, 144, 196]]

    channel_options = [[32, 64, 128, 256, 512], [16, 32, 64, 128, 256], [16, 32, 64, 128],
                       [8, 16, 32, 64, 128, 256, 512], [8, 16, 32, 64, 128, 256], [16, 32, 64, 128], [16, 32, 64]]

    # Suggest hyperparameters
    latent_dim = trial.suggest_int('latent_dim', 32, 1024, step=32)
    channel_index = trial.suggest_categorical('channel_index', list(range(len(channel_options))))
    batch_size = trial.suggest_categorical('batch_size', [2 ** i for i in range(3, 7)])
    lr = trial.suggest_categorical('lr', [10 ** (-i) for i in range(4, 7)])
    kernel_size = trial.suggest_categorical('kernel_size', [3, 5, 7, 9])
    beta = trial.suggest_categorical('beta', [10 ** (-i) for i in range(1, 7)])
    contrastive_loss = trial.suggest_categorical('contrastive_loss', [True, False])
    contrastive_weight = trial.suggest_categorical('contrastive_weight', [10 ** (-i) for i in range(0, 7)])
    #criterion = trial.suggest_categorical('criterion', [SDRLoss, BCELoss])

    # Load dataset
    dataset_train = PriorDataset('train', debug=True, name='toy_dataset')
    dataset_val = PriorDataset('val', debug=True, name='toy_dataset')

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=12)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, num_workers=12)

    try:
        # Train model
        model, best_val_sdr = train_vae(dataloader_train,
                                        dataloader_val,
                                        lr=lr,
                                        latent_dim=latent_dim,
                                        kernel_size=kernel_size,
                                        contrastive_loss=contrastive_loss,
                                        contrastive_weight=contrastive_weight,
                                        visualise=False,
                                        channels=channel_options[channel_index],
                                        kld_weight=beta,
                                        verbose=True,
                                        epochs=5,
                                        criterion=SDRLoss,
                                        image_h=1024,
                                        image_w=128)

    except torch.cuda.OutOfMemoryError:
        print('CUDA out of Memory. Skipping')
        return -math.inf
    except Exception as e:
        print(f"Caught an exception: {e}. Skipping.")
        traceback.print_exc()
        return -math.inf

    # Return the best validation loss
    return best_val_sdr


# Set up the Optuna study
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=200)

# Print the best hyperparameters
print("Best hyperparameters: ", study.best_params)
