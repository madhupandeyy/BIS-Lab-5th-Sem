import numpy as np

# Define the Problem (objective function to optimize)
def fitness_function(x):
    return -x**2 + 10  # Maximizing f(x) = -x^2 + 10

# Initialize Parameters
num_particles = 30        # Number of particles
num_iterations = 100      # Number of iterations
inertia_weight = 0.7      # Inertia weight
cognitive_weight = 1.5    # Cognitive coefficient
social_weight = 1.5       # Social coefficient
min_value = -10           # Minimum value of x
max_value = 10            # Maximum value of x

# Initialize Particles
def initialize_particles(num_particles, min_val, max_val):
    positions = np.random.uniform(min_val, max_val, num_particles)
    velocities = np.random.uniform(-1, 1, num_particles)
    return positions, velocities

# Evaluate Fitness
def evaluate_fitness(positions):
    return np.array([fitness_function(x) for x in positions])

# Update Velocities and Positions
def update_velocity(position, velocity, best_position, global_best_position, inertia_weight, cognitive_weight, social_weight):
    cognitive_velocity = cognitive_weight * np.random.random() * (best_position - position)
    social_velocity = social_weight * np.random.random() * (global_best_position - position)
    return inertia_weight * velocity + cognitive_velocity + social_velocity

def update_position(position, velocity):
    return position + velocity

# PSO Iteration
def run_pso():
    positions, velocities = initialize_particles(num_particles, min_value, max_value)
    fitness_values = evaluate_fitness(positions)
    personal_best_positions = positions.copy()
    personal_best_fitness = fitness_values.copy()
    
    global_best_position = positions[np.argmax(fitness_values)]
    
    for iteration in range(num_iterations):
        for i in range(num_particles):
            fitness_values[i] = fitness_function(positions[i])
            if fitness_values[i] > personal_best_fitness[i]:
                personal_best_fitness[i] = fitness_values[i]
                personal_best_positions[i] = positions[i]
        
        global_best_position = personal_best_positions[np.argmax(personal_best_fitness)]
        
        for i in range(num_particles):
            velocities[i] = update_velocity(positions[i], velocities[i], personal_best_positions[i], global_best_position, inertia_weight, cognitive_weight, social_weight)
            positions[i] = update_position(positions[i], velocities[i])
        
        print(f"Iteration {iteration+1}/{num_iterations} - Global Best Position: {global_best_position}, Fitness: {fitness_function(global_best_position)}")
    
    return global_best_position, fitness_function(global_best_position)

best_position, best_fitness = run_pso()
print("\nBest position found:", best_position)
print("Best fitness found:", best_fitness)


