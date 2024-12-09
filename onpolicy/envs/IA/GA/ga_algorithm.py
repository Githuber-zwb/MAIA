import numpy as np
import matplotlib.pyplot as plt
from numpy import random
import copy
import math
from onpolicy.config import get_config
from onpolicy.envs.IA.ia_core import World

def ga(world: World, target = "time", show_result=False):
    N = world.field.num_working_lines + world.num_harvester - 1     # 总的节点数，作业行数+收割机数目-1，不同收割机的作业行用0分隔

    NP = 100  # 种群数目
    G = 500  # 最大迭代步数
    f = np.zeros((NP, N), dtype=int)  # 种群矩阵，每一行是一个染色体，每一个染色体长度是N
    F = []  # 临时存储更新过程的变量

    # 种群初始化
    for i in range(NP):
        arr = [t for t in range(world.field.num_working_lines)]
        arr += [-1 for _ in range(world.num_harvester - 1)]
        arr = np.array(arr)
        random.shuffle(arr)
        f[i, :] = arr[:]

    R = f[0, :]  # 存储最优个体（对应最短路径/最短时间）
    len_path = np.zeros(NP)  # 存储个体的路径长度/转运时间
    fitness = np.zeros(NP)  # 存储正则化的适应度函数
    gen = 0  #迭代步数

    Rlength = []  # 存储每一次迭代时的最短路径
    while gen < G:
        gen += 1
        # 计算每一个个体的路程/时间
        for i in range(NP):
            chrom = f[i, :]
            break_points = np.where(chrom == -1)[0]
            working_lines = np.split(chrom, break_points)
            # print(working_lines)
            harvesters_len = []
            harvesters_time = []
            for h, harv in enumerate(world.harvesters):
                harv.dispatch_tasks(working_lines[h][:] if h == 0 else working_lines[h][1:])
                length = 0
                for n in range(1, harv.nav_points.shape[0]):
                    length += np.sqrt(np.sum((harv.nav_points[n] - harv.nav_points[n - 1]) ** 2))
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
        R = f[rr[0], :]

        # 计算归一化的适应度函数
        for i in range(len_path.shape[0]):
            fitness[i] = 1 - ((len_path[i] - minlen) / (maxlen - minlen + 0.001))

        # 轮盘赌选择个体
        F = []
        for i in range(NP):
            if fitness[i] >= np.random.rand():
                F.append(f[i, :])

        F = np.array(F)

        # 保证种群数目不变
        while F.shape[0] < NP:
            # 从秦代中随机选择两个个体
            nnper = np.random.permutation(F.shape[0])
            A = F[nnper[0], :]
            B = F[nnper[1], :]

            # 交叉操作
            W = np.ceil(N / 10).astype(int)  # 交叉点数目
            p = np.random.randint(0, N - W + 1)  # Randomly select crossover range
            for i in range(W):
                if B[p + i] != -1:
                    x = np.where(A == B[p + i])[0][0]
                else:
                    # B选中的是分割点，判断这是B中的第几个分割点
                    ids = np.where(B==-1)[0]
                    index = np.where(ids==p+i)[0]
                    assert len(index) == 1
                    index = index[0]
                    xs = np.where(A==-1)[0]
                    x = xs[index]
                if A[p + i] != -1:
                    y = np.where(B == A[p + i])[0][0]
                else:
                    # A选中的是分割点，判断这是A中的第几个分割点
                    ids = np.where(A==-1)[0]
                    index = np.where(ids==p+i)[0]
                    assert len(index) == 1
                    index = index[0]
                    ys = np.where(B==-1)[0]
                    y = ys[index]
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

        f = F  # Update population
        f[0, :] = R  # Retain the best individual

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

    return best_individual

if __name__ == "__main__":
    parser = get_config()
    parser.add_argument('--scenario_name', type=str,
                        default='ia_simple', help="Which scenario to run on")
    parser.add_argument("--num_harvester", type=int, default=3, help="number of harvesters")
    parser.add_argument('--num_transporter', type=int,
                        default=2, help="number of transporters")
    all_args = parser.parse_known_args()[0]
    # np.random.seed(1)
    world = World(all_args)
    ga(world, show_result=True)