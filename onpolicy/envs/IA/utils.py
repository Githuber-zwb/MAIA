import numpy as np
import networkx as nx
import heapq
from matplotlib import pyplot as plt

def compute_dist(a, b):
    if isinstance(a, tuple):
        a = np.array(a)
    if isinstance(b, tuple):
        b = np.array(b)
    assert a.shape == b.shape
    return np.sqrt(np.sum((a - b) ** 2))

# 判断点是否在多边形内，不包括在边上
def pnpoly(vertices, testp):
    n = len(vertices)
    j = n - 1
    res = False
    for i in range(n):
        if (vertices[i][1] > testp[1]) != (vertices[j][1] > testp[1]) and \
                testp[0] < (vertices[j][0] - vertices[i][0]) * (testp[1] - vertices[i][1]) / (
                vertices[j][1] - vertices[i][1]) + vertices[i][0]:
            res = not res
        j = i
    return res

# A* algorithm implementation
def a_star(graph, start, target, heuristic):
    # Priority queue for open set (using heapq for efficient min-heap)
    open_set = []
    heapq.heappush(open_set, (0 + heuristic(start, target), start))  # (f, node)

    # Dictionary to store the shortest path from start to each node
    came_from = {}

    # g_score stores the cost of the path from start to the current node
    g_score = {node: float('inf') for node in graph}
    g_score[start] = 0

    # f_score stores the estimated total cost from start to target through current node
    f_score = {node: float('inf') for node in graph}
    f_score[start] = heuristic(start, target)

    while open_set:
        # Get the node with the lowest f_score
        _, current = heapq.heappop(open_set)

        # If we reached the target, reconstruct the path
        if current == target:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path, f_score[target]

        # Check each neighbor of the current node
        for neighbor in graph[current]:
            tentative_g_score = g_score[current] + compute_dist(current, neighbor)

            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, target)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None, float('inf')  # If no path found

# Example heuristic: Euclidean distance for grid-based paths
def heuristic(node, target):
    return compute_dist(node, target)

def find_target_index(arr, target):
    # Iterate through each index in the m*n plane
    for i in range(arr.shape[0]):  # m (first dimension)
        for j in range(arr.shape[1]):  # n (second dimension)
            # Extract the 2D slice from the 3D array (shape: (2,))
            slice_2d = arr[i, j]
            
            # Check if the slice matches the target
            if np.array_equal(slice_2d, target):
                return (i, j)  # Return the (i, j) index if match is found
    
    return None  # Return None if no match is found

def find_target(arr, target):
    dim = len(arr.shape)
    if dim == 3:
        # Iterate through each index in the m*n plane
        for i in range(arr.shape[0]):  # m (first dimension)
            for j in range(arr.shape[1]):  # n (second dimension)
                # Extract the 2D slice from the 3D array (shape: (2,))
                slice_2d = arr[i, j]
                
                # Check if the slice matches the target
                if np.array_equal(slice_2d, target):
                    return True  # Return the (i, j) index if match is found
    elif dim == 2:
        for i in range(arr.shape[0]):
            slice_2d = arr[i]
            if np.array_equal(slice_2d, target):
                return True  # Return the (i, j) index if match is found
    
    return False  # Return None if no match is found

def test_graph(graph, path = None):
    fig, ax = plt.subplots()

    # Set the axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # Plot nodes as scatter points
    for node in graph:
        x, y = node  # Unpack the coordinates of the node
        ax.scatter(x, y, c='blue', s=1, zorder=5)  # Plot the node (blue point)

    # Plot edges as lines between connected nodes
    for node, neighbors in graph.items():
        for neighbor in neighbors:
            x1, y1 = node
            x2, y2 = neighbor
            ax.plot([x1, x2], [y1, y2], c='gray', linestyle='-', linewidth=1, zorder=1)  # Plot edge (line)

    # Set the aspect of the plot to be equal to ensure the nodes are not distorted
    ax.set_aspect('equal', adjustable='box')

    if path != None:
        for i in range(len(path) - 1):
            ax.scatter(path[i][0], path[i][1], c='yellow', s=1, zorder=5)  # Plot the node
            ax.plot([path[i][0], path[i + 1][0]] , [path[i][1], path[i + 1][1]], c='red', linestyle='-', linewidth=1, zorder=1)  # Plot edge (line)
        ax.scatter(path[0][0], path[0][1], c='black', s=1, zorder=5)  # Plot the node
        ax.scatter(path[-1][0], path[-1][1], c='purple', s=1, zorder=5)  # Plot the node

    # Display the plot
    plt.show()
    # plt.pause(5) # 显示1s
    # plt.close()