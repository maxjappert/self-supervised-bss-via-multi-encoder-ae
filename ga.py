import random
from datetime import datetime
import numpy as np

from functions import get_model, train, CircleTriangleDataset, test

print('lesgoo')

def create_individual():
    return {key: random.choice(values) for key, values in hyperparameter_ranges.items()}

def create_population(size):
    return [create_individual() for _ in range(size)]

hyperparameter_ranges = {
    'channels': [[2, 4, 8], [4, 8, 16], [8, 16, 32], [16, 32, 64], [32, 64, 128], [64, 128, 256], [24, 48, 96, 144]],
    'hidden': [4, 16, 32, 64, 128, 256, 512, 1024, 2048],
    'norm_type': ['none', 'batch_norm', 'group_norm'],
    'sep_norm': ['L1', 'L2'],
    'sep_lr': [0, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'zero_lr': [0, 0.0001, 0.001, 0.01, 0.1, 1.0],
    'lr': [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0],
    'weight_decay': [0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0],
    'z_decay': [0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0],
}

dataset = CircleTriangleDataset()

def fitness(hps):
    model, _, val_losses = train(get_model(channels=hps['channels'], hidden=hps['hidden'], norm_type=hps['norm_type']), dataset, batch_size=512, sep_norm=hps['sep_norm'], sep_lr=hps['sep_lr'],
          zero_lr=hps['zero_lr'], lr=hps['lr'], weight_decay=hps['weight_decay'], z_decay=hps['z_decay'], max_epochs=5, verbose=False)

    # The fitness is the inverse final validation loss
    if val_losses[-1] == 0:
        return float('inf')

    prediction_error = test(model, dataset, visualise=False)

    return 1.0 / prediction_error

def select_pair(population, fitnesses):
    selected = random.choices(population, weights=fitnesses, k=2)
    return selected[0], selected[1]

def crossover(parent1, parent2):
    child = {}
    for key in parent1:
        child[key] = parent1[key] if random.random() > 0.5 else parent2[key]
    return child

def mutate(individual, mutation_rate):
    for key in individual:
        if random.random() < mutation_rate:
            individual[key] = random.choice(hyperparameter_ranges[key])
    return individual

def genetic_algorithm(pop_size=10, generations=50, mutation_rate=0.1):
    population = create_population(pop_size)

    for generation in range(generations):
        fitnesses = [fitness(ind) for ind in population]
        new_population = []

        for _ in range(pop_size):
            parent1, parent2 = select_pair(population, fitnesses)
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            new_population.append(child)

        print(f'\n[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]  Generation {generation}, high score {max(fitnesses)}')
        print(population[np.argmax(fitnesses)])

        population = new_population

genetic_algorithm(pop_size=30)
