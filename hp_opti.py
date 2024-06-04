import optuna
from functions import get_model, CircleTriangleDataset, train, test


def objective(trial):

    channel_options = [[2, 4, 8], [4, 8, 16], [8, 16, 32], [16, 32, 64], [32, 64, 128], [64, 128, 256], [24, 48, 96, 144], [16, 32, 64, 128]]

    # Suggest hyperparameters
    sep_lr = trial.suggest_float('sep_lr', 0.0, 1.0, step=0.1)
    zero_lr = trial.suggest_float('zero_lr', 0.0, 0.5, step=0.01)
    hidden = trial.suggest_int('hidden', 64, 1024, step=64)
    channel_index = trial.suggest_int('channel_index', 0, len(channel_options)-1, step=1)
    norm_type = trial.suggest_categorical('norm_type', ['none', 'batch_norm', 'group_norm'])
    weight_decay = trial.suggest_float('weight_decay', 0.0, 0.0001, step=0.000001)

    channels = channel_options[channel_index]

    # Load dataset
    dataset_trainval = CircleTriangleDataset()

    # Train model
    model, train_losses, val_losses = train(
        channels=channels, hidden=hidden, norm_type=norm_type, num_encoders=2,
        dataset_trainval=dataset_trainval,
        dataset_split_ratio=0.8,
        batch_size=512,
        sep_norm='L1',
        sep_lr=sep_lr,
        zero_lr=zero_lr,
        lr=0.001,
        lr_step_size=50,
        lr_gamma=1.0,
        weight_decay=weight_decay,
        max_epochs=60,
        verbose=False
    )

    # TODO: Split dataset into trainval and test
    test_loss = test(model, dataset_trainval, visualise=False)
    # Return the best validation loss
    return test_loss


# Set up the Optuna study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=200)

# Print the best hyperparameters
print("Best hyperparameters: ", study.best_params)


