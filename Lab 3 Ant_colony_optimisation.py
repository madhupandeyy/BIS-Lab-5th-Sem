import numpy as np

# Define the Problem (TSP - coordinates of cities)
cities = np.array([[0, 0], [1, 2], [4, 3], [6, 1], [7, 5]])  # Example cities with (x, y) coordinates

# Initialize Parameters
num_ants = 10
num_iterations = 100
alpha = 1        # Pheromone importance
beta = 2         # Distance heuristic importance
rho = 0.1        # Pheromone evaporation rate
Q = 100          # Pheromone quantity

# Calculate distance between two cities
def calculate_distance(city1, city2):
    return np.linalg.norm(city1 - city2)

# Initialize pheromones
pheromones = np.ones((len(cities), len(cities)))

# Construct Solutions
def construct_solution():
    path = []
    visited = np.zeros(len(cities), dtype=bool)
    current_city = np.random.randint(len(cities))
    visited[current_city] = True
    path.append(current_city)
    
    while len(path) < len(cities):
        probabilities = []
        for i in range(len(cities)):
            if not visited[i]:
                pheromone = pheromones[current_city][i] ** alpha
                distance = calculate_distance(cities[current_city], cities[i]) ** (-beta)
                probabilities.append(pheromone * distance)
            else:
                probabilities.append(0)
        
        total_prob = sum(probabilities)
        probabilities = [prob / total_prob for prob in probabilities]
        next_city = np.random.choice(range(len(cities)), p=probabilities)
        visited[next_city] = True
        path.append(next_city)
        current_city = next_city
    
    return path

# Update Pheromones
def update_pheromones(paths, path_lengths):
    global pheromones
    pheromones *= (1 - rho)
    for i in range(len(paths)):
        for j in range(len(paths[i]) - 1):
            pheromones[paths[i][j], paths[i][j+1]] += Q / path_lengths[i]
            pheromones[paths[i][j+1], paths[i][j]] += Q / path_lengths[i]

# ACO Iteration
def run_aco():
    best_path = None
    best_length = float('inf')
    
    for iteration in range(num_iterations):
        paths = []
        path_lengths = []
        
        for _ in range(num_ants):
            path = construct_solution()
            path_length = sum(calculate_distance(cities[path[i]], cities[path[i+1]]) for i in range(len(path) - 1))
            paths.append(path)
            path_lengths.append(path_length)
        
        min_length_index = np.argmin(path_lengths)
        if path_lengths[min_length_index] < best_length:
            best_length = path_lengths[min_length_index]
            best_path = paths[min_length_index]
        
        update_pheromones(paths, path_lengths)
        
        print(f"Iteration {iteration+1}/{num_iterations} - Best Path Length: {best_length}")
    
    return best_path, best_length

best_path, best_length = run_aco()
print("\nBest path found:", best_path)
print("Best path length:", best_length)
