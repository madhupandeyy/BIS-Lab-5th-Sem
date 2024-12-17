import numpy as np

# Define the Problem (objective function to optimize)
def fitness_function(x):
    return x**2  # Maximizing f(x) = x^2

# Initialize Parameters
num_wolves = 30
num_iterations = 100
alpha = 0.8
beta = 0.8
delta = 0.8

# Initialize Population (Wolves)
def initialize_wolves(num_wolves, min_val, max_val):
    return np.random.uniform(min_val, max_val, (num_wolves, 1))

# Update Positions (Using Grey Wolf Leadership)
def update_position(wolf, alpha_position, beta_position, delta_position):
    r1 = np.random.random()
    r2 = np.random.random()
    a = 2 * alpha * r1 - alpha
    c = 2 * r2
    new_position = wolf + a * (alpha_position - wolf) + c * (beta_position - wolf)
    return new_position

# GWO Iteration
def run_gwo():
    wolves = initialize_wolves(num_wolves, -10, 10)
    fitness_values = np.array([fitness_function(wolf) for wolf in wolves])
    
    alpha_position = wolves[np.argmin(fitness_values)]
    beta_position = wolves[np.argsort(fitness_values)[1]]
    delta_position = wolves[np.argsort(fitness_values)[2]]

    for iteration in range(num_iterations):
        for i in range(num_wolves):
            wolves[i] = update_position(wolves[i], alpha_position, beta_position, delta_position)
            fitness_values[i] = fitness_function(wolves[i])

        sorted_indices = np.argsort(fitness_values)
        alpha_position = wolves[sorted_indices[0]]
        beta_position = wolves[sorted_indices[1]]
        delta_position = wolves[sorted_indices[2]]

        print(f"Iteration {iteration+1}/{num_iterations} - Best Fitness: {fitness_values[sorted_indices[0]]}")
    
    return alpha_position, fitness_values[sorted_indices[0]]

best_position, best_fitness = run_gwo()
print("\nBest position found:", best_position)
print("Best fitness found:", best_fitness)
