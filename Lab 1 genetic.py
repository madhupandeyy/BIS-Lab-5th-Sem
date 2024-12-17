import numpy as np
import random

# 1. Define the Problem (objective function to optimize)
def fitness_function(x):
    return x**2  # Maximizing f(x) = x^2

# 2. Initialize Parameters
population_size = 100      # Number of individuals in population
mutation_rate = 0.01       # Probability of mutation
crossover_rate = 0.7       # Probability of crossover
num_generations = 200      # Number of generations
min_value = -10            # Minimum value of x
max_value = 10             # Maximum value of x

# 3. Create Initial Population
def initialize_population(pop_size, min_val, max_val):
    return np.random.uniform(min_val, max_val, pop_size)

# 4. Evaluate Fitness
def evaluate_fitness(population):
    return np.array([fitness_function(x) for x in population])

# 5. Selection
def select_parents(population, fitness_values):
    # Roulette wheel selection method
    total_fitness = np.sum(fitness_values)
    selection_prob = fitness_values / total_fitness
    selected_parents = np.random.choice(population, size=2, p=selection_prob)
    return selected_parents

# 6. Crossover (one-point crossover)
def crossover(parent1, parent2):
    if random.random() < crossover_rate:
        crossover_point = random.randint(1, len(parent1)-1)
        offspring1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        offspring2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
        return offspring1, offspring2
    return parent1, parent2

# 7. Mutation (Gaussian mutation)
def mutate(offspring):
    if random.random() < mutation_rate:
        mutation_value = np.random.normal(0, 1)  # Gaussian mutation
        offspring += mutation_value
    return offspring

# 8. Iteration (repeat generations)
def run_genetic_algorithm():
    # Initial population
    population = initialize_population(population_size, min_value, max_value)
    best_solution = None
    best_fitness = -np.inf

    for generation in range(num_generations):
        # Evaluate fitness
        fitness_values = evaluate_fitness(population)
        
        # Track best solution
        max_fitness_index = np.argmax(fitness_values)
        if fitness_values[max_fitness_index] > best_fitness:
            best_fitness = fitness_values[max_fitness_index]
            best_solution = population[max_fitness_index]
        
        # Create next generation
        next_generation = []
        for _ in range(population_size // 2):
            parent1, parent2 = select_parents(population, fitness_values)
            offspring1, offspring2 = crossover(parent1, parent2)
            next_generation.append(mutate(offspring1))
            next_generation.append(mutate(offspring2))
        
        population = np.array(next_generation)
        
        # Print progress
        print(f"Generation {generation+1}/{num_generations} - Best Fitness: {best_fitness}, Best Solution: {best_solution}")
    
    return best_solution, best_fitness

# Running the Genetic Algorithm
best_solution, best_fitness = run_genetic_algorithm()

print("\nBest solution found:", best_solution)
print("Best fitness found:", best_fitness)
