import numpy as np

# Define the Problem (objective function to optimize)
def fitness_function(x):
    return x**2  # Maximizing f(x) = x^2

# Initialize Parameters
num_nests = 20
num_iterations = 100
p = 0.25   # Discovery probability
alpha = 0.01  # Step size

# Initialize Population
def initialize_nests(num_nests, min_val, max_val):
    return np.random.uniform(min_val, max_val, num_nests)

# Generate New Solutions (Levy Flight)
def levy_flight():
    return np.random.normal(0, 1) * np.random.normal(0, 1)

# Cuckoo Search Iteration
def run_cuckoo_search():
    nests = initialize_nests(num_nests, -10, 10)
    fitness_values = np.array([fitness_function(x) for x in nests])
    best_nest = nests[np.argmax(fitness_values)]
    best_fitness = np.max(fitness_values)
    
    for iteration in range(num_iterations):
        new_nests = nests + alpha * levy_flight()
        new_fitness_values = np.array([fitness_function(x) for x in new_nests])
        
        for i in range(num_nests):
            if new_fitness_values[i] > fitness_values[i]:
                nests[i] = new_nests[i]
                fitness_values[i] = new_fitness_values[i]
        
        worst_nests = np.argsort(fitness_values)[:int(p * num_nests)]
        for i in worst_nests:
            nests[i] = np.random.uniform(-10, 10)
        
        best_nest = nests[np.argmax(fitness_values)]
        best_fitness = np.max(fitness_values)
        
        print(f"Iteration {iteration+1}/{num_iterations} - Best Fitness: {best_fitness}")
    
    return best_nest, best_fitness

best_nest, best_fitness = run_cuckoo_search()
print("\nBest nest found:", best_nest)
print("Best fitness found:", best_fitness)
