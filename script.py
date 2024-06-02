import random

from functions import *

print('lesgoo')

#model = get_model()

#model = train(model, CircleTriangleDataset(), batch_size=1024, max_epochs=10, name='model')
#print('here')
#test(model, CircleTriangleDataset())

def create_individual():
    return {key: random.choice(values) for key, values in hyperparameter_ranges.items()}

def create_population(size):
    return [create_individual() for _ in range(size)]

dataset = CircleTriangleDataset()

hyperparameter_ranges = {
    'batch_size': [32, 64, 128, 256, 512, 1024],
    'channels': [[16, 32, 64], [32, 64, 128], [64, 128, 256]],
    'hidden': [128, 256, 512, 1024],
    'norm_type': ['none', 'batch_norm', 'layer_norm', 'group_norm'],
    'sep_norm': ['L1', 'L2'],
    'sep_lr': [0.1, 0.5, 1.0],
    'zero_lr': [0.001, 0.01, 0.1],
    'lr': [1e-4, 1e-3, 1e-2],
    'weight_decay': [1e-6, 1e-5, 1e-4],
    'z_decay': [1e-3, 1e-2, 1e-1],
}


def fitness(hps):
    _, _, val_losses = train(get_model(), dataset, batch_size=hps['batch_size'], sep_norm=hps['sep_norm'], sep_lr=hps['sep_lr'],
          zero_lr=hps['zero_lr'], lr=hps['lr'], weight_decay=hps['weight_decay'], z_decay=hps['z_decay'], max_epochs=5, verbose=False)

    # The fitness is the inverse final validation loss
    return 1.0 / val_losses[-1]


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

def genetic_algorithm(pop_size=1000, generations=50, mutation_rate=0.1):
    population = create_population(pop_size)

    for generation in range(generations):
        fitnesses = [fitness(ind) for ind in population]
        new_population = []

        for _ in range(pop_size):
            parent1, parent2 = select_pair(population, fitnesses)
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            new_population.append(child)

        print(f'Generation {generation}, high score {max(fitnesses)}')

        population = new_population


genetic_algorithm()
