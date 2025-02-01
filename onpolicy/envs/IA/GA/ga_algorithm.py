import numpy as np
import matplotlib.pyplot as plt
from numpy import random
import copy
import math
from onpolicy.config import get_config
from onpolicy.envs.IA.utils import compute_dist
from onpolicy.envs.IA.ia_core import World

def find_minus_in_A(A, B, p):
    """
    在A中找到B对应位置p的-1是在A中的哪一个位置出现的
    """
    assert B[p] == -1
    ids = np.where(B==-1)[0]
    index = np.where(ids==p)[0]
    assert len(index) == 1
    index = index[0]
    xs = np.where(A==-1)[0]
    x = xs[index]
    return x

def ga(world: World, NP = 100, G = 500, target = "time", show_result=False):
    N = world.field.num_working_lines + world.num_harvester - 1     # 总的节点数，作业行数+收割机数目-1，不同收割机的作业行用0分隔
    # NP: 种群数目
    # G: 最大迭代步数
    f = np.zeros((NP, N), dtype=int)  # 种群矩阵，每一行是一个染色体，每一个染色体长度是N
    F = []  # 临时存储更新过程的变量

    # 种群初始化
    for i in range(NP):
        arr = [t for t in range(world.field.num_working_lines)]
        arr += [-1 for _ in range(world.num_harvester - 1)]
        arr = np.array(arr)
        random.shuffle(arr)
        f[i, :] = arr[:]

    R = f[0, :].copy()  # 存储最优个体（对应最短路径/最短时间）
    len_path = np.zeros(NP)  # 存储个体的路径长度/转运时间
    fitness = np.zeros(NP)  # 存储正则化的适应度函数
    gen = 0  #迭代步数

    Rlength = []  # 存储每一次迭代时的最短路径
    while gen < G:
        gen += 1
        # 计算每一个个体的路程/时间
        for i in range(NP):
            chrom = f[i, :].copy()
            break_points = np.where(chrom == -1)[0]
            working_lines = np.split(chrom, break_points)
            # print(working_lines)
            harvesters_len = []
            harvesters_time = []
            for h, harv in enumerate(world.harvesters):
                harv.dispatch_tasks(working_lines[h][:] if h == 0 else working_lines[h][1:])
                length = 0
                for n in range(1, harv.nav_points.shape[0]):
                    # length += np.sqrt(np.sum((harv.nav_points[n] - harv.nav_points[n - 1]) ** 2))
                    length += compute_dist(harv.nav_points[n], harv.nav_points[n - 1])
                harvesters_len.append(length)
                harvesters_time.append(length / harv.speed)
            # print(harvesters_len)
            if target == "length":
                len_path[i] = np.sum(harvesters_len)
            elif target == "time":
                len_path[i] = np.max(harvesters_time)
            else:
                raise NotImplementedError
            
        maxlen = np.max(len_path)  # 最长路径/时间
        minlen = np.min(len_path)  # 最短路径/时间
        Rlength.append(minlen)

        # 更新最短路程
        rr = np.where(len_path == minlen)[0]
        R = f[rr[0], :].copy()

        # 计算归一化的适应度函数
        for i in range(len_path.shape[0]):
            fitness[i] = 1 - ((len_path[i] - minlen) / (maxlen - minlen + 0.001))

        # 轮盘赌选择个体
        F = []
        for i in range(NP):
            if fitness[i] >= np.random.rand():
                F.append(f[i, :].copy())

        F = np.array(F)

        # 保证种群数目不变
        while F.shape[0] < NP:
            # 从亲代中随机选择两个个体
            nnper = np.random.permutation(F.shape[0])
            A = F[nnper[0], :].copy()
            B = F[nnper[1], :].copy()

            # 交叉操作
            W = np.ceil(N / 10).astype(int)  # 交叉点数目
            p = np.random.randint(0, N - W + 1)  # Randomly select crossover range
            for i in range(W):
                if B[p + i] != -1:
                    x = np.where(A == B[p + i])[0][0]
                else:
                    # B选中的是分割点，判断这是B中的第几个分割点
                    x = find_minus_in_A(A, B, p+i)
                if A[p + i] != -1:
                    y = np.where(B == A[p + i])[0][0]
                else:
                    # A选中的是分割点，判断这是A中的第几个分割点
                    y = find_minus_in_A(B, A, p+i)
                # 交换
                A[p + i], B[p + i] = B[p + i], A[p + i]
                A[x], B[y] = B[y], A[x]
            
            # Mutation operation
            p1, p2 = np.random.randint(0, N, 2)
            while p1 == p2:
                p1, p2 = np.random.randint(0, N, 2)
            A[p1], A[p2] = A[p2], A[p1]
            B[p1], B[p2] = B[p2], B[p1]

            # Add new individuals to the population
            F = np.vstack([F, A, B])

        # Ensure the population size is NP
        if F.shape[0] > NP:
            F = F[:NP, :]

        f = F.copy()  # Update population
        f[0, :] = R.copy()  # Retain the best individual

    # Plot the best path found
    # plt.figure()
    # for i in range(N - 1):
    #     plt.plot([C[R[i], 0], C[R[i + 1], 0]], [C[R[i], 1], C[R[i + 1], 1]], 'bo-')
    # plt.plot([C[R[N - 1], 0], C[R[0], 0]], [C[R[N - 1], 1], C[R[0], 1]], 'ro-')
    # plt.title(f'Optimized shortest distance: {minlen}')
    # plt.show()

    # Plot the evolution of the fitness over generations
    # print("Best individual:", R)
    best_break_points = np.where(R == -1)[0]
    best_working_lines = np.split(R, best_break_points)
    best_individual = [best_working_lines[h][:] if h == 0 else best_working_lines[h][1:] for h in range(len(best_working_lines))]

    if show_result:
        print("Best individual:", best_individual)
        plt.figure()
        plt.plot(Rlength)
        plt.xlabel('Generation')
        plt.ylabel('Fitness value (Shortest path length)')
        plt.title('Fitness Evolution Curve')
        plt.show()

    return best_individual, Rlength

def ga_mulit(world: World, NP = 100, G = 500, P_c = 0.6, target = "time", show_result=False):
    N = world.field.num_working_lines + world.num_harvester - 1     # 总的节点数，作业行数+收割机数目-1，不同收割机的作业行用0分隔
    # NP: 种群数目
    # G: 最大迭代步数
    # P_c: 交叉概率
    f = np.zeros((NP, N), dtype=int)  # 种群矩阵，每一行是一个染色体，每一个染色体长度是N
    F = np.zeros((NP, N), dtype=int)  # 临时存储更新过程的变量

    # 种群初始化
    for i in range(NP):
        arr = [t for t in range(world.field.num_working_lines)]
        arr += [-1 for _ in range(world.num_harvester - 1)]
        arr = np.array(arr)
        random.shuffle(arr)
        f[i, :] = arr[:]

    R = f[0, :].copy()  # 存储最优个体（对应最短路径/最短时间）
    len_path = np.zeros(NP)  # 存储个体的路径长度/转运时间
    fitness = np.zeros(NP)  # 存储正则化的适应度函数
    gen = 0  #迭代步数

    Rlength = []  # 存储每一次迭代时的最短路径
    while gen < G:
        gen += 1
        # 计算每一个个体的路程/时间
        for i in range(NP):
            chrom = f[i, :].copy()
            break_points = np.where(chrom == -1)[0]
            working_lines = np.split(chrom, break_points)
            # print(working_lines)
            harvesters_len = []
            harvesters_time = []
            for h, harv in enumerate(world.harvesters):
                harv.dispatch_tasks(working_lines[h][:] if h == 0 else working_lines[h][1:])
                length = 0
                for n in range(1, harv.nav_points.shape[0]):
                    # length += np.sqrt(np.sum((harv.nav_points[n] - harv.nav_points[n - 1]) ** 2))
                    length += compute_dist(harv.nav_points[n], harv.nav_points[n - 1])
                harvesters_len.append(length)
                harvesters_time.append(length / harv.speed)
            # print(harvesters_len)
            if target == "length":
                len_path[i] = np.sum(harvesters_len)
            elif target == "time":
                len_path[i] = np.max(harvesters_time)
            else:
                raise NotImplementedError
            
        maxlen = np.max(len_path)  # 最长路径/时间
        minlen = np.min(len_path)  # 最短路径/时间
        Rlength.append(minlen)

        # 更新最短路程
        rr = np.where(len_path == minlen)[0]
        R = f[rr[0], :].copy()

        # 计算归一化的适应度函数
        for i in range(len_path.shape[0]):
            fitness[i] = 1 / len_path[i]
        total_fitness = sum(fitness)
        for i in range(len_path.shape[0]):
            fitness[i] /= total_fitness
        # print(fitness, sum(fitness))
        # print(fitness)

        # 保证种群数目不变
        # print(fitness)
        for t in range(NP):
            # 通过轮盘赌，从亲代选择两个个体
            rand_num1 = random.rand()
            cumulative_sum = 0
            for a, fitness_score in enumerate(fitness):
                cumulative_sum += fitness_score
                if cumulative_sum > rand_num1:
                    A = f[a, :].copy()
                    break
            rand_num2 = random.rand()
            cumulative_sum = 0
            for b, fitness_score in enumerate(fitness):
                cumulative_sum += fitness_score
                if cumulative_sum > rand_num2:
                    B = f[b, :].copy()
                    break
            # print(A,B,a,b)
            if random.rand() < P_c:
                # 交叉操作
                W = np.ceil(N / 10).astype(int)  # 交叉点数目
                p = np.random.randint(0, N - W + 1)  # Randomly select crossover range
                for i in range(W):
                    if B[p + i] != -1:
                        x = np.where(A == B[p + i])[0][0]
                    else:
                        # B选中的是分割点，判断这是B中的第几个分割点
                        x = find_minus_in_A(A, B, p+i)
                    if A[p + i] != -1:
                        y = np.where(B == A[p + i])[0][0]
                    else:
                        # A选中的是分割点，判断这是A中的第几个分割点
                        y = find_minus_in_A(B, A, p+i)
                    # 交换
                    A[p + i], B[p + i] = B[p + i], A[p + i]
                    A[x], B[y] = B[y], A[x]
            else:
                A = A if fitness[a] > fitness[b] else B

            # Mutation operation
            p1, p2 = np.random.randint(0, N, 2)
            while p1 == p2:
                p1, p2 = np.random.randint(0, N, 2)
            A[p1], A[p2] = A[p2], A[p1]
            # B[p1], B[p2] = B[p2], B[p1]

            # Add new individuals to the population
            F[t, :] = A.copy()

        # Ensure the population size is NP
        # if F.shape[0] > NP:
        #     F = F[:NP, :]

        f = copy.deepcopy(F)  # Update population
        f[0, :] = R.copy()  # Retain the best individual

    # Plot the best path found
    # plt.figure()
    # for i in range(N - 1):
    #     plt.plot([C[R[i], 0], C[R[i + 1], 0]], [C[R[i], 1], C[R[i + 1], 1]], 'bo-')
    # plt.plot([C[R[N - 1], 0], C[R[0], 0]], [C[R[N - 1], 1], C[R[0], 1]], 'ro-')
    # plt.title(f'Optimized shortest distance: {minlen}')
    # plt.show()

    # Plot the evolution of the fitness over generations
    # print("Best individual:", R)
    best_break_points = np.where(R == -1)[0]
    best_working_lines = np.split(R, best_break_points)
    best_individual = [best_working_lines[h][:] if h == 0 else best_working_lines[h][1:] for h in range(len(best_working_lines))]

    if show_result:
        print("Best individual:", best_individual)
        plt.figure()
        plt.plot(Rlength)
        plt.xlabel('Generation')
        plt.ylabel('Fitness value (Shortest path length)')
        plt.title('Fitness Evolution Curve')
        plt.show()

    return best_individual, Rlength

if __name__ == "__main__":
    parser = get_config()
    parser.add_argument('--scenario_name', type=str,
                        default='ia_simple', help="Which scenario to run on")
    parser.add_argument("--num_harvester", type=int, default=3, help="number of harvesters")
    parser.add_argument('--num_transporter', type=int,
                        default=2, help="number of transporters")
    all_args = parser.parse_known_args()[0]
    np.random.seed(0)
    world = World(all_args)
    _, r1 = ga(world)
    _, r2 = ga_mulit(world)

    plt.figure()
    plt.plot(r1, color='r', label='GA')
    plt.plot(r2, color='b', label='GA Multi')
    plt.xlabel('Generation')
    plt.ylabel('Fitness value (Shortest path length)')
    plt.title('Fitness Evolution Curve')
    plt.legend()
    plt.show()