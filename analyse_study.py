import pickle

import joblib
import optuna
from plotly.io import show

study = joblib.load(f"studies/study_basis_toy.pkl")

# Print the best trial details
best_trial = study.best_trial
print(f"Best trial number: {best_trial.number}")
print(f"Best value: {best_trial.value}")
print("Best hyperparameters: ", best_trial.params)

# Example: Contour plot for two hyperparameters
# optuna.visualization.plot_contour(study, params=['param1', 'param2'])

# Example: Optimization history
fig = optuna.visualization.plot_contour(study)
show(fig)

fig = optuna.visualization.plot_contour(study)

