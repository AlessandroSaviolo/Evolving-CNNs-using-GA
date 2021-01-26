import tensorflow as tf
import os
from network import Network
from inout import compute_parent
from random import randint, sample
from utilities import load_dataset, order_indexes, plot_training, plot_statistics, load_network
from copy import deepcopy

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)      # suppress messages from Tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def initialize_population(population_size, dataset):
    print("----->Initializing Population")
    daddy = compute_parent(dataset)                                 # load parent from input
    population = [daddy]
    for it in range(1, population_size):
        population.append(daddy.asexual_reproduction(it, dataset))

    # sort population on ascending order based on fitness
    return sorted(population, key=lambda cnn: cnn.fitness)


def selection(k, population, num_population):
    if k == 0:                                              # elitism selection
        print("----->Elitism selection")
        return population[0], population[1]
    elif k == 1:                                            # tournament selection
        print("----->Tournament selection")
        i = randint(0, num_population - 1)
        j = i
        while j < num_population - 1:
            j += 1
            if randint(1, 100) <= 50:
                return population[i], population[j]
        return population[i], population[0]
    else:                                                   # proportionate selection
        print("----->Proportionate selection")
        cum_sum = 0
        for i in range(num_population):
            cum_sum += population[i].fitness
        perc_range = []
        for i in range(num_population):
            count = 100 - int(100 * population[i].fitness / cum_sum)
            for j in range(count):
                perc_range.append(i)
        i, j = sample(range(1, len(perc_range)), 2)
        while i == j:
            i, j = sample(range(1, len(perc_range)), 2)
        return population[perc_range[i]], population[perc_range[j]]


def crossover(parent1, parent2, it):
    print("----->Crossover")
    child = Network(it)

    first, second = None, None
    if randint(0, 1):
        first = parent1
        second = parent2
    else:
        first = parent2
        second = parent1

    child.block_list = deepcopy(first.block_list[:randint(1, len(first.block_list) - 1)]) \
                       + deepcopy(second.block_list[randint(1, len(second.block_list) - 1):])

    order_indexes(child)                            # order the indexes of the blocks

    return child


def genetic_algorithm(num_population, num_generation, num_offspring, dataset):
    print("Genetic Algorithm")

    population = initialize_population(num_population, dataset)

    print("\n-------------------------------------")
    print("Initial Population:")
    for cnn in population:
        print(cnn.name, ': ', cnn.fitness)
    print("--------------------------------------\n")

    # for printing statistics about fitness and number of parameters of the best individual
    stats = [(population[0].fitness, population[0].model.count_params())]

    for gen in range(1, num_generation + 1):

        '''
            k is the selection parameter:
                k = 0 -> elitism selection
                k = 1 -> tournament selection
                k = 2 -> proportionate selection
        '''
        k = randint(0, 2)

        print("\n------------------------------------")
        print("Generation", gen)
        print("-------------------------------------")

        for c in range(num_offspring):

            print("\nCreating Child", c)

            parent1, parent2 = selection(k, population, num_population)                 # selection
            print("Selected", parent1.name, "and", parent2.name, "for reproduction")

            child = crossover(parent1, parent2, c + num_population)                     # crossover
            print("Child has been created")

            print("----->Soft Mutation")
            child.layer_mutation(dataset)                                               # mutation
            child.parameters_mutation()
            print("Child has been mutated")

            model = child.build_model()                                                 # evaluation

            while model == -1:
                child = crossover(parent1, parent2, c + num_population)
                child.block_mutation(dataset)
                child.layer_mutation(dataset)
                child.parameters_mutation()
                model = child.build_model()

            child.train_and_evaluate(model, dataset)

            if child.fitness < population[-1].fitness:                                  # evolve population
                print("----->Evolution: Child", child.name, "with fitness", child.fitness, "replaces parent ", end="")
                print(population[-1].name, "with fitness", population[-1].fitness)
                name = population[-1].name
                population[-1] = deepcopy(child)
                population[-1].name = name
                population = sorted(population, key=lambda net: net.fitness)
            else:
                print("----->Evolution: Child", child.name, "with fitness", child.fitness, "is discarded")

        stats.append((population[0].fitness, population[0].model.count_params()))

    print("\n\n-------------------------------------")
    print("Final Population")
    print("-------------------------------------\n")
    for cnn in population:
        print(cnn.name, ': ', cnn.fitness)

    print("\n-------------------------------------")
    print("Stats")
    for i in range(len(stats)):
        print("Best individual at generation", i + 1, "has fitness", stats[i][0], "and parameters", stats[i][1])
    print("-------------------------------------\n")

    # plot the fitness and the number of parameters of the best individual at each iteration
    plot_statistics(stats)

    return population[0]


def main():
    batch_size = 32                         # the number of training examples in one forward/backward pass
    num_classes = 10                        # number of cifar-10 dataset classes
    epochs = 3                              # number of forward and backward passes of all the training examples

    '''
        dataset contains the hyper parameters for loading data and the dataset:
            dataset = {
                'batch_size': batch_size,
                'num_classes': num_classes,
                'epochs': epochs,
                'x_train': x_train,
                'x_test': x_test,
                'y_train': y_train,
                'y_test': y_test
            }
    '''
    dataset = load_dataset(batch_size, num_classes, epochs)

    num_population = 4
    num_generation = 4
    num_offspring = 2

    # plot the best model obtained
    optCNN = genetic_algorithm(num_population, num_generation, num_offspring, dataset)

    # plot the training and validation loss and accuracy
    num_epoch = 3
    model = optCNN.build_model()
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(dataset['x_train'],
                        dataset['y_train'],
                        batch_size=dataset['batch_size'],
                        epochs=num_epoch,
                        validation_data=(dataset['x_test'], dataset['y_test']),
                        shuffle=True)
    optCNN.model = model                                        # model
    optCNN.fitness = history.history['val_loss'][-1]            # fitness

    print("\n\n-------------------------------------")
    print("The initial CNN has been evolved successfully in the individual", optCNN.name)
    print("-------------------------------------\n")
    daddy = load_network('parent_0')
    model = tf.keras.models.load_model('parent_0.h5')
    print("\n\n-------------------------------------")
    print("Summary of initial CNN")
    print(model.summary())
    print("Fitness of initial CNN:", daddy.fitness)

    print("\n\n-------------------------------------")
    print("Summary of evolved individual")
    print(optCNN.model.summary())
    print("Fitness of the evolved individual:", optCNN.fitness)
    print("-------------------------------------\n")

    plot_training(history)


if __name__ == '__main__':
    main()
