import numpy as np
from queue import PriorityQueue

def heuristic(a, b):
    """Calculate the Euclidean distance between two points."""
    return np.sqrt(sum((a - b) ** 2 for a, b in zip(a, b)))

def get_neighbors():
    """Generate all possible movements including diagonal."""
    neighbors = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if dx == 0 and dy == 0 and dz == 0:
                    continue  # Skip the current node itself
                neighbors.append((dx, dy, dz))
    return neighbors

def movement_cost(current, neighbor):
    """Calculate the cost of moving from current to neighbor."""
    dx, dy, dz = np.abs(np.array(neighbor) - np.array(current))
    if dx + dy + dz == 3:
        return np.sqrt(3)  # Diagonal in 3D
    elif dx + dy + dz == 2:
        return np.sqrt(2)  # Diagonal in 2D (plane)
    else:
        return 1  # Straight movement

def a_star(start, goal, obstacles, grid_size):
    neighbors = get_neighbors()
    
    open_set = PriorityQueue()
    open_set.put((0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    
    while not open_set.empty():
        current = open_set.get()[1]

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]  # Return reversed path
        
        for dx, dy, dz in neighbors:
            neighbor = (current[0] + dx, current[1] + dy, current[2] + dz)
            
            # Skip out of bounds or through obstacles
            if not (0 <= neighbor[0] < grid_size[0] and 0 <= neighbor[1] < grid_size[1] and 0 <= neighbor[2] < grid_size[2]):
                continue
            if neighbor in obstacles:
                continue
            
            tentative_g_score = g_score[current] + movement_cost(current, neighbor)
            
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                open_set.put((f_score[neighbor], neighbor))
    
    return None  # No path found

# Example usage
start = (0, 0, 0)
goal = (9, 9, 9)
grid_size = (10, 10, 10)
obstacles = {(x, y, z) for x in range(3, 7) for y in range(3, 7) for z in range(3, 7)}  # Define a cube obstacle

path = a_star(start, goal, obstacles, grid_size)
print("Path:", path)