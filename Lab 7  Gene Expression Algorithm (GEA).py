import numpy as np

# Define the Problem (objective function to optimize)
def fitness_function(x):
    return x**2  # Maximizing f(x) = x^2

# Initialize Parameters
population_size = 20
num_genes = 5
mutation_rate = 0.05
crossover_rate = 0.7
num_generations = 100

# Initialize Population
def initialize_population(pop_size, num_genes, min_val, max_val):
    return np.random.uniform(min_val, max_val, (pop_size, num_genes))

# Evaluate Fitness
def evaluate_fitness(population):
    return np.array([fitness_function(np.sum(individual)) for individual in population])

# Crossover (Single-point crossover)
def crossover(parent1, parent2):
    if np.random.random() < crossover_rate:
        crossover_point = np.random.randint(1, len(parent1))
        offspring1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        offspring2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
        return offspring1, offspring2
    return parent1, parent2

# Mutation (Gaussian mutation)
def mutate(offspring):
    if np.random.random() < mutation_rate:
        mutation_value = np.random.normal(0, 1)
        offspring += mutation_value
    return offspring

# GEA Iteration
def run_gea():
    population = initialize_population(population_size, num_genes, -10, 10)
    fitness_values = evaluate_fitness(population)
    
    best_solution = population[np.argmax(fitness_values)]
    best_fitness = np.max(fitness_values)

    for generation in range(num_generations):
        new_population = []
        
        for _ in range(population_size // 2):
            parent1, parent2 = population[np.random.choice(population_size, 2, p=fitness_values/fitness_values.sum())]
            offspring1, offspring2 = crossover(parent1, parent2)
            new_population.append(mutate(offspring1))
            new_population.append(mutate(offspring2))
        
        population = np.array(new_population)
        fitness_values = evaluate_fitness(population)
        
        max_fitness_index = np.argmax(fitness_values)
        if fitness_values[max_fitness_index] > best_fitness:
            best_fitness = fitness_values[max_fitness_index]
            best_solution = population[max_fitness_index]
        
        print(f"Generation {generation+1}/{num_generations} - Best Fitness: {best_fitness}, Best Solution: {best_solution}")
    
    return best_solution, best_fitness

best_solution, best_fitness = run_gea()
print("\nBest solution found:", best_solution)
print("Best fitness found:", best_fitness)
