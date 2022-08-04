def get_bounds(segments):

    # Create bounds list
    bounds = []
    # Add phi bounds
    for i in range(segments + 1):
        bounds.append([0.0, 1.0])
    # Add t_init bounds
    bounds.append([0.0, 5.0])
    # add delta_t bounds
    for i in range(segments):
        bounds.append([0.1, 20.0])

    return(bounds)


# Bounds format [[low, high], [low, high], etc]
def initialize_population_random_uniform(N, bounds):
    #l = len(bounds)
    #number_of_segments = int((l - 2)/2)
    population = []

    for _ in range(N):

        chromosome = []

        for gene_bounds in bounds:
            gene = random.uniform(gene_bounds[0], gene_bounds[1])
            chromosome.append(gene)

        population.append(chromosome)

    return(population)


# Evaluate the fitness of each individual in the population,
def evaluate_population(population):
    population_fitness = []
    for chromosome in population:
        fitness = interface.interface(np.array(chromosome))
        population_fitness.append(fitness)

    return(population_fitness)
