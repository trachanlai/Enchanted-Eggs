from queue import PriorityQueue
from functools import lru_cache

@lru_cache(maxsize=None)  # Add memoization to the heuristic function
def heuristic(a, b):
    """Calculate the Euclidean distance between two points."""
    return ((a[0] - b[0])**2 + (a[1] - b[1])**2 + (a[2] - b[2])**2) ** 0.5

def get_neighbors():
    """Generate possible movements in 3D space, favoring larger steps."""
    return [(dx, dy, dz) for dx in [-10, 0, 10] for dy in [-10, 0, 10] for dz in [-10, 0, 10] if not (dx == dy == dz == 0)]

neighbors = get_neighbors()

def is_obstacle(x, y, z, obstacles):
    """Check if the point is inside an obstacle."""
    return (x, y, z) in obstacles

def a_star(start, goal, obstacles):
    open_set = PriorityQueue()
    open_set.put((0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while not open_set.empty():
        _, current = open_set.get()

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        for dx, dy, dz in neighbors:
            neighbor = (current[0] + dx, current[1] + dy, current[2] + dz)

            if is_obstacle(neighbor[0], neighbor[1], neighbor[2], obstacles):
                continue

            tentative_g_score = g_score[current] + heuristic(current, neighbor)

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                open_set.put((f_score[neighbor], neighbor))

    return None

def line_of_sight(start, end, obstacles):
    """Check if there's a clear line of sight between start and end points."""
    # Bresenham's Line Algorithm in 3D to check for obstacles
    points = []
    x0, y0, z0 = start
    x1, y1, z1 = end
    dx, dy, dz = abs(x1 - x0), abs(y1 - y0), abs(z1 - z0)
    xs, ys, zs = 1 if x1 > x0 else -1, 1 if y1 > y0 else -1, 1 if z1 > z0 else -1
    # Driving axis is the axis with the maximum delta
    if dx >= dy and dx >= dz:
        # x is driving axis
        p1, p2 = 2 * dy - dx, 2 * dz - dx
        while x0 != x1:
            if (x0, y0, z0) in obstacles:
                return False
            if p1 >= 0:
                y0 += ys
                p1 -= 2 * dx
            if p2 >= 0:
                z0 += zs
                p2 -= 2 * dx
            x0 += xs
            p1 += 2 * dy
            p2 += 2 * dz
    elif dy >= dx and dy >= dz:
        # y is driving axis
        p1, p2 = 2 * dx - dy, 2 * dz - dy
        while y0 != y1:
            if (x0, y0, z0) in obstacles:
                return False
            if p1 >= 0:
                x0 += xs
                p1 -= 2 * dy
            if p2 >= 0:
                z0 += zs
                p2 -= 2 * dy
            y0 += ys
            p1 += 2 * dx
            p2 += 2 * dz
    else:
        # z is driving axis
        p1, p2 = 2 * dy - dz, 2 * dx - dz
        while z0 != z1:
            if (x0, y0, z0) in obstacles:
                return False
            if p1 >= 0:
                y0 += ys
                p1 -= 2 * dz
            if p2 >= 0:
                x0 += xs
                p2 -= 2 * dz
            z0 += zs
            p1 += 2 * dy
            p2 += 2 * dx
    return True  # No obstacles found

def smooth_path(path, obstacles):
    """Remove unnecessary waypoints from the path if direct line of sight exists."""
    if not path:
        return path

    smooth_path = [path[0]]  # Always include the start point
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1:
            if line_of_sight(path[i], path[j], obstacles):
                smooth_path.append(path[j])
                i = j  # Jump to the next waypoint
                break
            j -= 1
        if j == i + 1:  # No direct path found; proceed to the next waypoint
            smooth_path.append(path[i + 1])
            i += 1
    return smooth_path

start = (-400, -400, 0)
goal = (400, 400, 0)
extra_range = 7
obstacles = {(x, y, z) for x in range(-100 - extra_range, 101 + extra_range) for y in range(-100 - extra_range, 101 + extra_range) for z in range(-100 - extra_range, 100 + extra_range)}

if line_of_sight(start, goal, obstacles):
    print("Direct path found:", [start, goal])
else:
    path = a_star(start, goal, obstacles)
    if path:
        smoothed_path = smooth_path(path, obstacles)
        print("Path found:", len(smoothed_path), smoothed_path)
    else:
        print("No path found.")