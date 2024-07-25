import math
import random
import traceback

import numpy as np
import optuna
import torch
from torch.nn import BCELoss
from torch.utils.data import DataLoader

from evaluation_metric_functions import compute_spectral_sdr
from functions import model_factory, CircleTriangleDataset, train, test, TwoSourcesDataset
from functions_prior import PriorDataset, train_vae, SDRLoss


def objective(trial):
    #channel_options = [[2, 4, 8], [4, 8, 16], [8, 16, 32], [16, 32, 64], [32, 64, 128], [64, 128, 256], [24, 48, 96, 144], [16, 32, 64, 128], [16, 32, 64, 128, 256], [24, 48, 96, 144, 196]]

    channel_options = [[4, 8], [4, 8, 16], [4, 8, 16, 32], [4, 8, 16, 32, 64], [4, 8, 16, 32, 64, 128]]

    # Suggest hyperparameters
    latent_dim = trial.suggest_int('latent_dim', 4, 256, step=4)
    channel_index = trial.suggest_categorical('channel_index', list(range(len(channel_options))))
    batch_size = trial.suggest_categorical('batch_size', [2 ** i for i in range(6, 9)])
    lr = trial.suggest_categorical('lr', [10 ** (-i) for i in range(2, 7)])
    stride = trial.suggest_categorical('stride', [1, 2, 3])
    # kernel_options = [random.choice([3, 5, 7, 9]) for _ in range(depth)]
    # kernel_sizes = trial.suggest_categorical('kernel_sizes', str(kernel_options)[1::3])
    # stride_options = [random.choice([1, 2, 3]) for _ in range(depth)]
    # strides = trial.suggest_categorical('strides', stride_options[1::3])

    # beta = trial.suggest_categorical('beta', [10 ** (-i) for i in range(1, 7)])
    # contrastive_loss = trial.suggest_categorical('contrastive_loss', [True, False])
    # contrastive_weight = trial.suggest_categorical('contrastive_weight', [10 ** (-i) for i in range(0, 7)])
    # criterion = trial.suggest_categorical('criterion', [SDRLoss, BCELoss])

    image_h = 64
    image_w = 64

    # Load dataset
    dataset_train = PriorDataset('train', debug=True, name='musdb_18_prior', image_h=image_h, image_w=image_w)
    dataset_val = PriorDataset('val', debug=True, name='musdb_18_prior', image_h=image_h, image_w=image_w)

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=12)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, num_workers=12)

    try:
        # Train model
        model, best_recon_loss = train_vae(dataloader_train,
                                        dataloader_val,
                                        lr=lr,
                                        latent_dim=latent_dim,
                                        visualise=False,
                                        channels=channel_options[channel_index],
                                        verbose=False,
                                        epochs=20,
                                        image_h=image_h,
                                        image_w=image_w,
                                        strides=[stride for _ in range(len(channel_options[channel_index]))])

    except torch.cuda.OutOfMemoryError:
        print('CUDA out of Memory. Skipping')
        return -math.inf
    except Exception as e:
        print(f"Caught an exception: {e}. Skipping.")
        traceback.print_exc()
        return -math.inf

    # Return the best validation loss
    return best_recon_loss


# Set up the Optuna study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=200)

# Print the best hyperparameters
print("Best hyperparameters: ", study.best_params)
