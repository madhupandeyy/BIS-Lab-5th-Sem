import numpy as np

# Define the Problem (objective function to optimize)
def fitness_function(x, y):
    return -(x**2 + y**2)  # Maximizing f(x, y) = -(x^2 + y^2)

# Initialize Parameters
grid_size = 5       # Grid size (5x5)
num_iterations = 100
neighborhood_size = 1  # Neighbors for cellular interaction (e.g., 1 means adjacent cells)

# Initialize Population (Cells)
def initialize_population(grid_size, min_val, max_val):
    return np.random.uniform(min_val, max_val, (grid_size, grid_size, 2))  # 2D grid with (x, y) positions

# Update States based on neighbors (diffusion-like update)
def update_state(cells, x, y, grid_size):
    # Gather neighbors (we assume a von Neumann neighborhood)
    neighbors = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if 0 <= x + dx < grid_size and 0 <= y + dy < grid_size and (dx != 0 or dy != 0):
                neighbors.append(cells[x + dx, y + dy])
    
    # Update cell's state (move towards the best neighbor)
    best_neighbor = max(neighbors, key=lambda n: fitness_function(n[0], n[1]))
    cells[x, y] = best_neighbor

# Parallel Cellular Iteration
def run_parallel_cellular():
    cells = initialize_population(grid_size, -10, 10)
    for iteration in range(num_iterations):
        for x in range(grid_size):
            for y in range(grid_size):
                update_state(cells, x, y, grid_size)
        
        # Evaluate fitness of the entire grid
        fitness_values = np.array([[fitness_function(cells[x, y][0], cells[x, y][1]) for y in range(grid_size)] for x in range(grid_size)])
        best_cell = np.unravel_index(np.argmax(fitness_values), fitness_values.shape)
        best_fitness = fitness_values[best_cell]
        
        print(f"Iteration {iteration+1}/{num_iterations} - Best Fitness: {best_fitness}")
    
    return cells[best_cell], best_fitness

best_cell, best_fitness = run_parallel_cellular()
print("\nBest cell found:", best_cell)
print("Best fitness found:", best_fitness)
