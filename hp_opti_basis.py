import math
import random
import traceback

import joblib
import numpy as np
import optuna
import torch
from mir_eval.chord import evaluate
from torch.nn import BCELoss
from torch.utils.data import DataLoader

from evaluation_metric_functions import compute_spectral_sdr
from functions import model_factory, CircleTriangleDataset, train, test, TwoSourcesDataset
from functions_prior import PriorDataset, train_vae, SDRLoss
from separate_new import evaluate_basis_ability

dataset_name = 'toy_dataset'

def hp_opti_basis(name='toy', n_trials=200):
    def objective(trial):
        T = trial.suggest_int('T', 10, 1000, log=True)
        alpha = trial.suggest_float('alpha', 1e-10, 1, log=True)
        sigma_start = trial.suggest_float('sigma_start', 0.01, 0.1, log=True)
        sigma_end = trial.suggest_float('sigma_end', 0.2, 1.0, log=True)
        delta = trial.suggest_float('delta', 1e-10, 1e-04, log=True)
        recon_weight = trial.suggest_float('recon_weight', -50, 50)
        L = trial.suggest_int('L', 5, 100)

        image_h = 64
        image_w = 64

        try:
            mean_sdr = evaluate_basis_ability(T, L, alpha,
                                              sigma_start,
                                              sigma_end,
                                              delta,
                                              recon_weight,
                                              image_h=image_h,
                                              image_w=image_w,
                                              num_samples=16,
                                              name_vae=name)

        except Exception as e:
            print(f"Caught an exception: {e}. Skipping.")
            traceback.print_exc()
            return -math.inf

        # Return the best validation loss
        return mean_sdr


    # Set up the Optuna study
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    # Print the best hyperparameters
    print("Best hyperparameters: ", study.best_params)

    joblib.dump(study, f"studies/study_basis_{dataset_name}.pkl")

