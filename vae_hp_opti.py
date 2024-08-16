import math
import random
import traceback

import joblib
import numpy as np
import optuna
import torch
from torch.nn import BCELoss
from torch.utils.data import DataLoader

from evaluation_metric_functions import compute_spectral_sdr
from functions import model_factory, CircleTriangleDataset, train, test, TwoSourcesDataset
from functions_prior import PriorDataset, train_vae, SDRLoss, test_vae

def hp_opti_vae(n_trials=100, dataset_name='toy_dataset'):

    def objective(trial):
        #channel_options = [[2, 4, 8], [4, 8, 16], [8, 16, 32], [16, 32, 64], [32, 64, 128], [64, 128, 256], [24, 48, 96, 144], [16, 32, 64, 128], [16, 32, 64, 128, 256], [24, 48, 96, 144, 196]]

        # Suggest hyperparameters
        latent_dim = trial.suggest_int('latent_dim', 1, 64, step=1)
        depth = trial.suggest_int('depth', 1, 8, step=1)
        num_channels = trial.suggest_int('num_channels', 2, 32, step=2)
        # batch_size = trial.suggest_categorical('batch_size', [2 ** i for i in range(5, 9)])
        # lr = trial.suggest_categorical('lr', [10 ** (-i) for i in range(2, 7)])
        stride_ends = trial.suggest_categorical('stride_ends', [1, 2, 3])
        stride_middle = trial.suggest_categorical('stride_middle', [1, 2, 3])
        kernel_size = trial.suggest_categorical('kernel_size', [3, 5, 7])
        batch_norm = trial.suggest_categorical('batch_norm', [True, False])

        # kernel_options = [random.choice([3, 5, 7, 9]) for _ in range(depth)]
        # kernel_sizes = trial.suggest_categorical('kernel_sizes', str(kernel_options)[1::3])
        # stride_options = [random.choice([1, 2, 3]) for _ in range(depth)]
        # strides = trial.suggest_categorical('strides', stride_options[1::3])

        channels = []

        max_channels = 1024
        for i in range(1, depth+1):
            layer_channels = num_channels * 2**i
            if layer_channels <= max_channels:
                channels.append(num_channels * 2**i)
            else:
                channels.append(max_channels)

        if channels[-1] > 8192:
            return -float('inf')

        # beta = trial.suggest_categorical('beta', [10 ** (-i) for i in range(1, 7)])
        # contrastive_loss = trial.suggest_categorical('contrastive_loss', [True, False])
        # contrastive_weight = trial.suggest_categorical('contrastive_weight', [10 ** (-i) for i in range(0, 7)])
        # criterion = trial.suggest_categorical('criterion', [SDRLoss, BCELoss])

        image_h = 64
        image_w = 64

        # Load dataset
        dataset_train = PriorDataset('train', debug=False, name=dataset_name, image_h=image_h, image_w=image_w)
        dataset_val = PriorDataset('val', debug=False, name=dataset_name, image_h=image_h, image_w=image_w)

        dataloader_train = DataLoader(dataset_train, batch_size=64, shuffle=True, num_workers=6)
        dataloader_val = DataLoader(dataset_val, batch_size=64, shuffle=True, num_workers=6)

        strides = []

        for i in range(depth):
            if i == 0 or i == depth-1:
                strides.append(stride_ends)
            else:
                strides.append(stride_middle)

        try:
            # Train model
            model, best_recon_loss = train_vae(dataloader_train,
                                            dataloader_val,
                                            lr=0.001,
                                            latent_dim=latent_dim,
                                            visualise=False,
                                            channels=channels,
                                            verbose=False,
                                            epochs=5,
                                            image_h=image_h,
                                            image_w=image_w,
                                            strides=strides,
                                            kernel_sizes=[kernel_size for _ in range(depth)],
                                            batch_norm=batch_norm)

        except torch.cuda.OutOfMemoryError:
            print('CUDA out of Memory. Skipping')
            return -math.inf
        except Exception as e:
            print(f"Caught an exception: {e}. Skipping.")
            traceback.print_exc()
            return -math.inf

        mean_sdr = test_vae(model, dataset_val, num_samples=64)

        # Return the best validation loss
        return mean_sdr


    # Set up the Optuna study
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    # Print the best hyperparameters
    print("Best hyperparameters: ", study.best_params)

    joblib.dump(study, f"studies/study_vae_{dataset_name}.pkl")
