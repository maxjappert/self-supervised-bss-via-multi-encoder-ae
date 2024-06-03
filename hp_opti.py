import optuna
from functions import get_model, CircleTriangleDataset, train, test


def objective(trial):

    channel_options = [[2, 4, 8], [4, 8, 16], [8, 16, 32], [16, 32, 64], [32, 64, 128], [64, 128, 256], [24, 48, 96, 144], [16, 32, 64, 128]]

    # Suggest hyperparameters
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-10, 1.0)
    z_decay = trial.suggest_loguniform('z_decay', 1e-10, 1.0)
    sep_lr = trial.suggest_loguniform('sep_lr', 1e-10, 1.0)
    zero_lr = trial.suggest_loguniform('zero_lr', 1e-10, 1.0)
    hidden = trial.suggest_int('hidden', 16, 4096, step=16)
    channel_index = trial.suggest_int('channel_index', 0, len(channel_options), step=1)
    sep_norm = trial.suggest_categorical('sep_norm', ['L1', 'L2'])
    norm_type = trial.suggest_categorical('norm_type', ['none', 'batch_norm', 'group_norm'])

    channels = channel_options[channel_index]

    # Initialize model
    model = get_model(input_channels=1, image_hw=64, channels=channels, hidden=hidden, norm_type=norm_type, num_encoders=2)

    # Load dataset
    dataset_trainval = CircleTriangleDataset()

    # Train model
    model, train_losses, val_losses = train(
        model=model,
        dataset_trainval=dataset_trainval,
        dataset_split_ratio=0.8,
        batch_size=512,
        sep_norm=sep_norm,
        sep_lr=sep_lr,
        zero_lr=zero_lr,
        lr=lr,
        lr_step_size=50,
        lr_gamma=0.1,
        weight_decay=weight_decay,
        z_decay=z_decay,
        max_epochs=60,

        verbose=False
    )

    # TODO: Split dataset into trainval and test
    test_loss = test(model, dataset_trainval, visualise=False)
    # Return the best validation loss
    return test_loss


# Set up the Optuna study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)

# Print the best hyperparameters
print("Best hyperparameters: ", study.best_params)
