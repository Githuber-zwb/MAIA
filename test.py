import random
import numpy as np
# import ipaddress

# class Iptables:
#     def __init__(self):
#         # Initialize an empty dictionary to store chains and their rules
#         self.chains = {}

#     def insert_rule(self, chain, ip_cidr, action):
#         # Insert rule at the beginning of the chain
#         if chain not in self.chains:
#             self.chains[chain] = []
#         self.chains[chain].insert(0, (ip_cidr, action))

#     def append_rule(self, chain, ip_cidr, action):
#         # Append rule at the end of the chain
#         if chain not in self.chains:
#             self.chains[chain] = []
#         self.chains[chain].append((ip_cidr, action))

#     def delete_rule(self, chain, ip_cidr):
#         # Remove the rule from the chain
#         if chain in self.chains:
#             self.chains[chain] = [rule for rule in self.chains[chain] if rule[0] != ip_cidr]

#     def goto_chain(self, chain_from, ip_cidr, chain_to):
#         # Insert a Goto rule at the end of the chain (3 elements: ip_cidr, 'G', chain_to)
#         if chain_from not in self.chains:
#             self.chains[chain_from] = []
#         self.chains[chain_from].append((ip_cidr, "G", chain_to))

#     def match_ip(self, chain, ip):
#         while chain in self.chains:
#             for rule in self.chains[chain]:
#                 rule_ip_cidr = rule[0]
#                 action = rule[1]

#                 # Convert to an ip_network object for CIDR comparison
#                 network = ipaddress.ip_network(rule_ip_cidr, strict=False)
                
#                 if ip in network:
#                     if action == "A" or action == "R":
#                         # Return Accept or Reject if matched
#                         return action
#                     elif action == "G" and len(rule) == 3:
#                         # Jump to the next chain using Goto
#                         chain = rule[2]
#                         break  # Immediately break and match in the new chain
#             else:
#                 # If no match is found in the current chain, return 'U' (Unknown)
#                 return 'U'
#         return 'U'  # If chain does not exist or no match found

# def process_iptables_commands(n, commands):
#     iptables = Iptables()
#     results = []

#     for command in commands:
#         parts = command.split()

#         if parts[0] == 'I':
#             # Insert rule
#             chain_name, ip_cidr, action = parts[1], parts[2], parts[3]
#             iptables.insert_rule(chain_name, ip_cidr, action)
        
#         elif parts[0] == 'A':
#             # Append rule
#             chain_name, ip_cidr, action = parts[1], parts[2], parts[3]
#             iptables.append_rule(chain_name, ip_cidr, action)

#         elif parts[0] == 'D':
#             # Delete rule
#             chain_name, ip_cidr = parts[1], parts[2]
#             iptables.delete_rule(chain_name, ip_cidr)

#         elif parts[0] == 'G':
#             # Goto chain
#             chain_name_from, ip_cidr, chain_name_to = parts[1], parts[2], parts[4]
#             iptables.goto_chain(chain_name_from, ip_cidr, chain_name_to)

#         elif parts[0] == 'M':
#             # Match IP
#             ip = ipaddress.ip_address(parts[1])
#             # Always start matching from the 'c0' chain
#             result = iptables.match_ip('c0', ip)
#             results.append(result)
    
#     return results
# # Example test case:
# # n = 7
# # commands = [
# #     "A c0 10.1.0.0/24 A",        # Rule 1: Accept 10.1.0.0/24 in c0
# #     "A c0 192.168.0.0/16 R",     # Rule 2: Reject 192.168.0.0/16 in c0
# #     "A c1 172.16.0.0/12 A",      # Rule 3: Accept 172.16.0.0/12 in c1
# #     "A c0 172.16.0.0/12 G c1",   # Rule 4: Goto c1 for 172.16.0.0/12 in c0
# #     "M 10.1.0.5",                # Query 1: Match IP 10.1.0.5 (Accept from rule 1)
# #     "M 192.168.1.1",             # Query 2: Match IP 192.168.1.1 (Reject from rule 2)
# #     "M 172.16.1.1"               # Query 3: Match IP 172.16.1.1 (Goto c1, then Accept from rule 3)
# # ]

# n = 2
# commands = [
#     "A c0 192.168.1.0/24 R",
#     "M 192.168.1.20"
# ]
# # Process the commands
# output = process_iptables_commands(n, commands)
# for res in output:
#     print(res)


'''
某在线游戏运营商对旗下的游戏后台数据做了快照，从中可以得到用户的登陆时间信息，希望据此快速得到历史最大的用户同时在线人数。
算法输入：每行英文逗号分隔的：一用户上线时间（简化为整数，不小于0）,下线时间（简化为整数，小于10000）；空行结束，如：
0,8
12,15
4,9
6,9
14,20
8,10
10,20

算法输出：最大同时在线人数，如：4
'''

def solve():
    ls = []
    N = int(input())
    for _ in range(N):
        ls.append(list(map(int, input().split(','))))
    ls.sort(key=lambda x: x[0])

    result = 1
    count = 1
    for i in range(1, N):
        if ls[i][0] <= ls[i - 1][1]:
            count += 1
            ls[i][1] = min(ls[i][1], ls[i - 1][1])
            result = max(result, count)
        else:
            count = 1
    return result



"""
【累加序列】
给定一个字符串数字，编写一个算法来判断组成它的数字可以形成累加序列。
一个有效的累加序列必须**至少**包含3个数。除了最开始的两个数以外，字符串中的其他数都等于它之前两个数相加的和。
说明：字符串只包含`'0'-'9'`的字符，累加序列里的数不会以0开头，所以不会出现`1, 2, 03`或者`1, 02, 3`的情况。
**Example1:**
```
Input: "112358"
Output: true
Explanation: The digits can form an additive sequence: 1, 1, 2, 3, 5, 8. 
             1 + 1 = 2, 1 + 2 = 3, 2 + 3 = 5, 3 + 5 = 8
```
**Example 2:**
```
Input: "199100199"
Output: true
Explanation: The additive sequence is: 1, 99, 100, 199. 
             1 + 99 = 100, 99 + 100 = 199
```
**Constraints:**
`1 <= num.length <= 35`
"""
def solve():
    s = input()
    s_ls = list(s)
    n = len(s)

    # startSec = 1
    # endSec = 1
    for startSec in range(1, n - 1):
        for endSec in range(startSec, n - 1):
            if s[startSec] == '0' and startSec != endSec:
                break
            # num1 = int(s[:startSec])
            # num2 = int(s[startSec:endSec + 1])
            # total = num1 + num2
            # total_s = str(total)
            # if endSec + len(total_s) >= n:
            #     break
            if isValid(s, startSec, endSec):
                return True
                
    return False

def isValid(s, startSec, endSec):
    n = len(s)
    startFirst, endFirst = 0, startSec - 1
    while startSec <= n - 1:
        num1 = int(s[startFirst:endFirst + 1])
        num2 = int(s[startSec:endSec + 1])
        num3 = num1 + num2
        num3_s = str(num3)
        startThird = endSec + 1
        endThird = endSec + len(num3_s)
        print(num1, num2, num3)
        print(startThird, endThird)
        if endThird >= n or int(s[startThird:endThird + 1]) != num3:
            break
        if endThird == n - 1:
            return True
        startFirst, endFirst = startSec, endSec
        startSec, endSec = startThird, endThird
    return False

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


import matplotlib.pyplot as plt
import collections

def plot_graph(graph):
    """
    Visualizes the graph using matplotlib.

    :param graph: The graph represented as a defaultdict(list),
                where keys are node coordinates (tuples) and values are lists of neighboring coordinates (tuples).
    """
    # Create a figure and axis for the plot
    fig, ax = plt.subplots()

    # Set the axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # Plot nodes as scatter points
    for node in graph:
        x, y = node  # Unpack the coordinates of the node
        ax.scatter(x, y, c='blue', s=100, zorder=5)  # Plot the node (blue point)

    # Plot edges as lines between connected nodes
    for node, neighbors in graph.items():
        for neighbor in neighbors:
            x1, y1 = node
            x2, y2 = neighbor
            ax.plot([x1, x2], [y1, y2], c='gray', linestyle='-', linewidth=1, zorder=1)  # Plot edge (line)

    # Set the aspect of the plot to be equal to ensure the nodes are not distorted
    ax.set_aspect('equal', adjustable='box')

    # Display the plot
    plt.show()


import heapq
import collections
import math

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
            tentative_g_score = g_score[current] + euclidean_distance(current, neighbor)

            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, target)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None, float('inf')  # If no path found

# Euclidean distance between two points
def euclidean_distance(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

# Example heuristic: Euclidean distance for grid-based paths
def heuristic(node, target):
    return euclidean_distance(node, target)

def group_crossover(parent1: np.array, parent2: np.array, n, m):
    parent1_groups = []
    parent2_groups = []
    break_points1 = np.where(parent1 == -1)[0]
    working_lines1 = np.split(parent1, break_points1)
    for i, wl in enumerate(working_lines1):
        parent1_groups.append(wl[:] if i == 0 else wl[1:])
    break_points2 = np.where(parent2 == -1)[0]
    working_lines2 = np.split(parent2, break_points2)
    for i, wl in enumerate(working_lines2):
        parent2_groups.append(wl[:] if i == 0 else wl[1:])

    # 生成随机排列
    crossover_order = random.sample(range(m), m)

    child_groups = []
    used_tasks = set()

    # 交叉
    for i, k in enumerate(crossover_order):
        if random.random() < 0.5:
            selected_group = parent1_groups[k].copy()
        else:
            selected_group = parent2_groups[k].copy()
        # 去重
        tmp = []
        for num in selected_group:
            if num not in used_tasks:
                tmp.append(num)
            used_tasks.add(num)
        if i != m - 1:
            tmp.append(-1)
        if len(tmp) != 0:
            child_groups.append(tmp)
    child_groups = np.concatenate(child_groups, dtype=int)
    
    # 将未选择的节点插入子代
    for num in range(n):
        if num not in used_tasks:
            child_groups = np.insert(child_groups, np.random.randint(child_groups.shape[0] + 1), num)

    return child_groups

def in_group_exchange(chrom, m):
    child_groups = []
    break_points = np.where(chrom == -1)[0]
    working_lines = np.split(chrom, break_points)
    for i, wl in enumerate(working_lines):
        child_groups.append(wl[:] if i == 0 else wl[1:])
    a, b = random.sample(range(len(child_groups)), 2)
    if child_groups[a].shape[0] == 0 or child_groups[b].shape[0] == 0:
        return chrom
    id1 = np.random.randint(child_groups[a].shape[0])
    id2 = np.random.randint(child_groups[b].shape[0])
    print(a,b,id1,id2)
    child_groups[a][id1], child_groups[b][id2] = child_groups[b][id2], child_groups[a][id1] 
    for i, c in enumerate(child_groups[:-1]):
        child_groups[i] = np.append(c,-1)
    chrom = np.concatenate(child_groups, dtype=int)
    return chrom

def two_opt(chrom):
    child_groups = []
    break_points = np.where(chrom == -1)[0]
    working_lines = np.split(chrom, break_points)
    for i, wl in enumerate(working_lines):
        child_groups.append(wl[:] if i == 0 else wl[1:])
    a = np.random.randint(len(child_groups))
    if child_groups[a].shape[0] < 1:
        return chrom
    id1, id2 = random.sample(range(child_groups[a].shape[0] + 1), 2)
    if id1 > id2:
        id1, id2 = id2, id1
    print(a,id1,id2)
    child_groups[a][id1:id2] = child_groups[a][id1:id2][::-1]
    for i, c in enumerate(child_groups[:-1]):
        child_groups[i] = np.append(c,-1)
    chrom = np.concatenate(child_groups, dtype=int)
    return chrom

if __name__ == "__main__":
    # print(solve())
    import numpy as np
    import math

    # a = np.array([-1,-1,1,2,3,1,4,5,6])
    # b = np.where(a==-1)[0]
    # # print(np.split(a, b))

    # a = np.array([3,2,3,4,3])
    # print(len(np.where(a == 2)[0]))


    # A = np.array([1,2,3,-1,5,4,-1,6,9,7,8,10,-1])
    # B = np.array([9,7,8,-1,5,4,6,-1,1,2,3,-1,2])

    # ids = np.where(B==-1)[0]
    # index = np.where(ids==7)[0]
    # assert len(index) == 1
    # index = index[0]
    # xs = np.where(A==-1)[0]
    # x = xs[index]
    # # print(x)
    # p1, p2 = np.random.randint(0, 10, 2)
    # print(p1, p2)

    # a = np.arange(24).reshape(2,3,4,1)
    # print(np.mean(np.mean(a, axis=0), axis=0).squeeze(-1))


    # Example graph (undirected)
    # graph = collections.defaultdict(list)

    # # Add edges (undirected graph)
    # graph[(0, 0)].append((1, 1))
    # graph[(0, 0)].append((1, 0))
    # graph[(1, 0)].append((0, 0))
    # graph[(1, 0)].append((2, 1))
    # graph[(1, 1)].append((0, 0))
    # graph[(1, 1)].append((2, 1))
    # graph[(2, 1)].append((1, 0))
    # graph[(2, 1)].append((1, 1))

    # # Test A* algorithm
    # start = (0, 0)
    # target = (2, 1)
    # path, cost = a_star(graph, start, target, heuristic)

    # print(f"Path: {path}")
    # print(f"Total cost: {cost}")

    # # Example graph where nodes are represented by coordinates (tuples)
    # graph = collections.defaultdict(list)
    # graph[(0, 0)] = [(1, 1), (1, -1)]
    # graph[(1, 1)] = [(0, 0), (2, 2)]
    # graph[(1, -1)] = [(0, 0), (2, -2)]
    # graph[(2, 2)] = [(1, 1)]
    # graph[(2, -2)] = [(1, -1)]

    # # Visualize the graph
    # plot_graph(graph)
    
    # Example usage
    # n = 10  # Number of cities
    # m = 3   # Number of salesmen
    # parent1 = np.array([0, 1, 2, -1, 3, 4, 5, 6, -1, 7, 8, 9, 10, 11, 12])
    # parent2 = np.array([3, 4, 5, -1, 0, 1, 2, -1, 6, 7, 8, 9])

    # # child = group_crossover(parent1, parent2, n, m)
    # # print("Child Chromosome:", child)

    # chrom = two_opt(parent1)
    # print(chrom)

    # arr = np.random.uniform(190, 250, 1000)
    # for i, num in enumerate(arr):
    #     arr[i] = math.ceil(num/3)
    # print(np.mean(arr))