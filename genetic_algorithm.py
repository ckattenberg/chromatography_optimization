import random
import interface
import numpy as np
import numpy.random as npr
import helper_functions as hf
# Representation of an individual? A list of floating point numbers
# If n is the number of segments, there are n + 1 phi values between 0 and 1,
# 1 t_init value and n delta_t values.

# Selection strategies

def roulette_wheel_selection(population, population_fitness):
    # We need N parent pairs, where N is the population size? yes
    N = len(population)
    # Calculate the fitness of each chromosome in the population
    total_fitness = sum(population_fitness)
    selection_probabilities = [f/total_fitness for f in population_fitness]

    # Create the parent pairs (list of lists)
    parents = []
    for _ in range(N):
        # Choose 2 parents
        parent1 = population[npr.choice(len(population), p=selection_probabilities)]
        parent2 = population[npr.choice(len(population), p=selection_probabilities)]
        parents.append([parent1, parent2])
    return(parents)

def truncation_selection():
    pass

def tournament_selection():
    pass


def single_point_crossover(parents):
    l = len(parents[0][0])
    N = len(parents)
    children = []

    for i in range(N):
        child = [0] * l
        parent1 = parents[i][0]
        parent2 = parents[i][1]

        # Generate the child and append to list of children
        crossover_point = random.randint(0, l)
        for j in range(l):
            if(j < crossover_point):
                child[j] = parent1[j]
            else:
                child[j] = parent2[j]

        children.append(child)
    return(children)


# Mutation rate is a number between 0 and 1
# Mutation means here replacing with new random value
def mutate(children, mutation_rate, bounds):
    l = len(children[0])

    new_population = []

    for child in children:
        for i in range(l):
            # To mutate or not to mutate, that is the question.
            if(random.random() < mutation_rate):
                gene_bounds = bounds[i]
                # Mutate the gene
                child[i] = random.uniform(gene_bounds[0], gene_bounds[1])

        new_population.append(child)
    return(new_population)


def ga(iterations, segments):

    mutation_rate = 0.4

    bounds = hf.get_bounds(segments)

    # 1. Initialize population
    population = hf.initialize_population_random_uniform(30, bounds)
    # Main algorithm loop
    for i in range(iterations):

        # 2. calculate scores on crf (objective function)
        population_fitness = hf.evaluate_population(population)
        print(population_fitness, "\n")
        # 2. Selection
        parents = roulette_wheel_selection(population, population_fitness)
        # 3. Crossover
        children = single_point_crossover(parents)
        # 4. Mutation
        #old_population = population
        population = mutate(children, mutation_rate, bounds)
        # 5. Back to step 2

    # Return best solution in last generation?
    print(population, "\n")
    print(population_fitness, "\n")
    population_fitness = evaluate_population(population)
    first_max_fitness = max(population_fitness)
    max_index = population_fitness.index(first_max_fitness)
    # Which values need to be returned? in a list i think, see meta_experiment
    # Change this to keep track of best solution after each generation.
    return_list = [population[max_index], first_max_fitness]

    # return_list = [iters, runtime, -res.fun, list(res.x), func_vals, runtimes_cumulative]

    print(return_list)
    return(return_list)
    # Runtime, solution score, solution itself

#ga(10000, 3)
