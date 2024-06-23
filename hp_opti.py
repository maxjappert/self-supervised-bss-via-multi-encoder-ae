import optuna
from functions import get_model, CircleTriangleDataset, train, test, TwoSourcesDataset


def objective(trial):

    #channel_options = [[2, 4, 8], [4, 8, 16], [8, 16, 32], [16, 32, 64], [32, 64, 128], [64, 128, 256], [24, 48, 96, 144], [16, 32, 64, 128], [16, 32, 64, 128, 256], [24, 48, 96, 144, 196]]

    channel_options = [[8, 16], [8, 16, 32], [8, 16, 32, 64], [8, 16, 32, 64, 128], [24, 48, 96, 144], [24, 48, 96, 196], [24, 48, 96, 144, 196], [16, 32, 64, 128, 256]]

    # Suggest hyperparameters
    sep_lr = trial.suggest_float('sep_lr', 0.0, 1.0, step=0.1)
    zero_lr = trial.suggest_float('zero_lr', 0.0, 0.5, step=0.01)
    hidden = trial.suggest_int('hidden', 16, 1024, step=16)
    channel_index = trial.suggest_int('channel_index', 0, len(channel_options)-1, step=1)
    norm_type = trial.suggest_categorical('norm_type', ['none', 'batch_norm', 'group_norm'])
    weight_decay = trial.suggest_categorical('weight_decay', [10**(-i) for i in range(6)])
    sep_norm = trial.suggest_categorical('sep_norm', ['L1', 'L2'])
    batch_size = trial.suggest_categorical('batch_size', [2**i for i in range(9)])
    lr = trial.suggest_categorical('lr', [10**(-i) for i in range(6)])

    channels = channel_options[channel_index]

    # Load dataset
    dataset_train = TwoSourcesDataset(split='train', name='musdb18_two_sources')
    dataset_val = TwoSourcesDataset(split='validation', name='musdb18_two_sources')

    # Train model
    model, train_losses, val_losses = train(
        channels=channels, hidden=hidden, norm_type=norm_type, num_encoders=2,
        dataset_train=dataset_train,
        dataset_val=dataset_val,
        dataset_split_ratio=0.8,
        batch_size=batch_size,
        sep_norm=sep_norm,
        sep_lr=sep_lr,
        zero_lr=zero_lr,
        lr=lr,
        lr_step_size=50,
        lr_gamma=1.0,
        weight_decay=weight_decay,
        max_epochs=5,
        verbose=False,
        num_workers=12
    )

    test_score = test(model, dataset_val, visualise=False, metric='snr')
    # Return the best validation loss
    return test_score


# Set up the Optuna study
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=200)

# Print the best hyperparameters
print("Best hyperparameters: ", study.best_params)
