import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import copy
import math
from onpolicy.config import get_config
from onpolicy.envs.IA.utils import compute_dist
from onpolicy.envs.IA.ia_core import World
import random

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

def decode_popu(population: np.array):
    """
    输入一个种群，输出解码后的种群
    """
    n = population.shape[0]
    child_groups = [[] for _ in range(n)]
    for i in range(n):
        chrom = population[i, :].copy()
        break_points = np.where(chrom == -1)[0]
        working_lines = np.split(chrom, break_points)
        for h, working_line in enumerate(working_lines):
            child_groups[i].append(working_line[:] if h == 0 else working_line[1:])
    return child_groups

def roulette_wheel_selection(population, fitness_scores):
    """
    轮盘赌选择亲代
    """
    # Calculate the total fitness
    assert len(fitness_scores) == population.shape[0]
    total_fitness = sum(fitness_scores)
    
    # Generate a random number between 0 and total fitness
    rand = np.random.uniform(0, total_fitness)
    
    # Traverse through the population and select the individual
    cumulative_sum = 0
    for i, fitness in enumerate(fitness_scores):
        cumulative_sum += fitness
        if cumulative_sum > rand:
            return i, population[i]

def group_crossover(parent1: np.array, parent2: np.array, n, m):
    """
    交叉算子
    """
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

def cross_group_trans(group):
    """
    组间单点转移算子，将某一个任务转移至另一个农机
    """
    a, b = random.sample(range(len(group)), 2)
    if group[b].shape[0] == 0:
        return group
    id1 = np.random.randint(group[a].shape[0] + 1)
    id2 = np.random.randint(group[b].shape[0])
    group[a] = np.insert(group[a], id1, group[b][id2])
    group[b] = np.delete(group[b], id2)
    return group

def cross_trans_multi(group, P_st = 0.8):
    """
    组间多点转移算子，将某一个或多个任务转移至另一个农机
    """
    if random.random() < P_st:
        return cross_group_trans(group)
    a, b = random.sample(range(len(group)), 2)
    if group[b].shape[0] == 0:
        return group
    n = np.random.randint(1, group[b].shape[0] + 1)
    start1 = np.random.randint(group[a].shape[0] + 1)
    start2 = np.random.randint(group[b].shape[0] - n + 1)
    group[a] = np.insert(group[a], start1, group[b][start2:(start2+n)].copy())
    group[b] = np.delete(group[b], slice(start2,start2+n))
    return group

def cross_group_exchange(group):
    """
    组间交换算子，交换两个作业任务
    """
    a, b = random.sample(range(len(group)), 2)
    if group[a].shape[0] == 0 or group[b].shape[0] == 0:
        return group
    id1 = np.random.randint(group[a].shape[0])
    id2 = np.random.randint(group[b].shape[0])
    group[a][id1], group[b][id2] = group[b][id2], group[a][id1] 
    return group

def cross_exchange_multi(group, P_se = 0.6):
    """
    组间多点转移算子，将某一个或多个任务转移至另一个农机
    """
    if random.random() < P_se:
        return cross_group_exchange(group)
    a, b = random.sample(range(len(group)), 2)
    if group[a].shape[0] == 0 or group[b].shape[0] == 0:
        return group
    n = np.random.randint(1, min(group[a].shape[0], group[b].shape[0]) + 1)
    start1 = np.random.randint(group[a].shape[0] - n + 1)
    start2 = np.random.randint(group[b].shape[0] - n + 1)
    tmp = group[a][start1:(start1+n)].copy()
    group[a][start1:(start1+n)] = group[b][start2:(start2+n)].copy()
    group[b][start2:(start2+n)] = tmp
    return group

def two_opt(group):
    a = np.random.randint(len(group))
    if group[a].shape[0] < 1:
        return group
    id1, id2 = random.sample(range(group[a].shape[0] + 1), 2)
    if id1 > id2:
        id1, id2 = id2, id1
    # print(a,id1,id2)
    group[a][id1:id2] = group[a][id1:id2][::-1]
    return group

def in_group_exchange(group):
    """
    组内交换算子，交换两个作业任务
    """
    a = np.random.randint(len(group))
    if group[a].shape[0] < 2:
        return group
    id1, id2 = random.sample(range(group[a].shape[0]), 2)
    group[a][id1], group[a][id2] = group[a][id2], group[a][id1] 
    return group

def in_group_order(group):
    """
    组内拍戏算子
    """
    a = np.random.randint(len(group))
    if group[a].shape[0] < 2:
        return group
    id1, id2 = random.sample(range(group[a].shape[0] + 1), 2)
    if id1 > id2:
        id1, id2 = id2, id1
    tmp = np.sort(group[a][id1:id2].copy())
    if np.random.rand() < 0.5:
        tmp = tmp[::-1]
    group[a][id1:id2] = tmp
    return group

class GA_RAW(object):
    def __init__(self, world:World, NP = 100, G = 500, target = "time"):
        self.world = world  # 求解的农田场景，包括作业地块、收割机参数信息
        self.N = world.field.num_working_lines + world.num_harvester - 1     # 总的节点数，不同收割机的作业行用-1分隔
        self.NP = NP    # 种群数目
        self.G = G  # 最大迭代步数
        self.target = target
    
    def solve(self, show_result=False):
        popu = np.zeros((self.NP, self.N), dtype=int) # 种群矩阵，每一行是一个染色体，每一个染色体长度是N
        ind_best = np.zeros(self.N, dtype=int)    # 存储迭代过程中的最优个体
        len_path = np.zeros(self.NP)  # 存储每一轮迭代时个体的路径长度/转运时间
        fitness = np.zeros(self.NP)  # 存储每一轮迭代时正则化的适应度函数
        gen = 0    #迭代步数

        # 种群初始化
        for i in range(self.NP):
            arr = [t for t in range(self.world.field.num_working_lines)]
            arr += [-1 for _ in range(self.world.num_harvester - 1)]
            arr = np.array(arr)
            np.random.shuffle(arr)
            popu[i, :] = arr[:].copy()

        ind_best = popu[0, :].copy()  # 存储最优个体（对应最短路径/最短时间）

        Rlength = []  # 存储每一次迭代时的最短路径
        while gen < self.G:
            gen += 1
            # 计算每一个个体的路程/时间
            for i in range(self.NP):
                chrom = popu[i, :].copy()
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
                if self.target == "length":
                    len_path[i] = np.sum(harvesters_len)
                elif self.target == "time":
                    len_path[i] = np.max(harvesters_time)
                else:
                    raise NotImplementedError
                
            maxlen = np.max(len_path)  # 最长路径/时间
            minlen = np.min(len_path)  # 最短路径/时间
            Rlength.append(minlen)

            # 更新最短路程
            rr = np.where(len_path == minlen)[0]
            ind_best = popu[rr[0], :].copy()

            # 计算归一化的适应度函数
            for i in range(len_path.shape[0]):
                fitness[i] = 1 - ((len_path[i] - minlen) / (maxlen - minlen + 0.001))

            # 轮盘赌选择个体
            popu_tmp = []
            for i in range(self.NP):
                if fitness[i] >= np.random.rand():
                    popu_tmp.append(popu[i, :].copy())

            popu_tmp = np.array(popu_tmp)

            # 保证种群数目不变
            while popu_tmp.shape[0] < self.NP:
                # 从亲代中随机选择两个个体
                nnper = np.random.permutation(popu_tmp.shape[0])
                A = popu_tmp[nnper[0], :].copy()
                B = popu_tmp[nnper[1], :].copy()

                # 交叉操作
                W = np.ceil(self.N / 10).astype(int)  # 交叉点数目
                p = np.random.randint(0, self.N - W + 1)  # Randomly select crossover range
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
                p1, p2 = np.random.randint(0, self.N, 2)
                while p1 == p2:
                    p1, p2 = np.random.randint(0, self.N, 2)
                A[p1], A[p2] = A[p2], A[p1]
                B[p1], B[p2] = B[p2], B[p1]

                # Add new individuals to the population
                popu_tmp = np.vstack([popu_tmp, A, B])

            # Ensure the population size is NP
            if popu_tmp.shape[0] > self.NP:
                popu_tmp = popu_tmp[:self.NP, :]

            popu = popu_tmp.copy()  # Update population
            popu[0, :] = ind_best.copy()  # Retain the best individual

        # Plot the best path found
        # plt.figure()
        # for i in range(N - 1):
        #     plt.plot([C[R[i], 0], C[R[i + 1], 0]], [C[R[i], 1], C[R[i + 1], 1]], 'bo-')
        # plt.plot([C[R[N - 1], 0], C[R[0], 0]], [C[R[N - 1], 1], C[R[0], 1]], 'ro-')
        # plt.title(f'Optimized shortest distance: {minlen}')
        # plt.show()

        # Plot the evolution of the fitness over generations
        # print("Best individual:", R)
        best_break_points = np.where(ind_best == -1)[0]
        best_working_lines = np.split(ind_best, best_break_points)
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

class GA(object):
    def __init__(self, world:World, NP = 200, G = 300, P_c = 0.6, P_m1 = 0.5, P_m2 = 0.5, P_m3 = 0.5, P_m4 = 0.8, P_m5 = 0.6, target = "time"):
        self.world = world  # 求解的农田场景，包括作业地块、收割机参数信息
        self.N = world.field.num_working_lines + world.num_harvester - 1     # 总的节点数，不同收割机的作业行用-1分隔
        self.NP = NP    # 种群数目
        self.G = G  # 最大迭代步数
        self.P_c = P_c  # 交叉概率
        self.P_m1 = P_m1    # 组间转移概率
        self.P_m2 = P_m2    # 组间交换概率
        self.P_m3 = P_m3    # 2-opt变异概率
        self.P_m4 = P_m4    # 组内交换概率
        self.P_m5 = P_m5    # 组内排序概率
        self.target = target
    
    def solve(self, show_result=False, savedir = None):
        popu = np.zeros((self.NP, self.N), dtype=int) # 种群矩阵，每一行是一个染色体，每一个染色体长度是N
        popu_tmp = np.zeros((self.NP, self.N), dtype=int) # 临时存储更新过程的变量
        ind_best = np.zeros(self.N, dtype=int)    # 存储迭代过程中的最优个体
        len_path = np.zeros(self.NP)  # 存储每一轮迭代时个体的路径长度/转运时间
        fitness = np.zeros(self.NP)  # 存储每一轮迭代时正则化的适应度函数
        gen = 0    #迭代步数

        # 种群初始化
        for i in range(self.NP):
            arr = [t for t in range(self.world.field.num_working_lines)]
            arr += [-1 for _ in range(self.world.num_harvester - 1)]
            arr = np.array(arr)
            np.random.shuffle(arr)
            popu[i, :] = arr.copy()

        ind_best = popu[0, :].copy()  # 存储最优个体（对应最短路径/最短时间）

        Rlength = []  # 存储每一次迭代时的最短路径
        while gen < self.G:
            gen += 1
            # 解码
            child_groups = decode_popu(popu)
            # 计算每一个个体的路程/时间
            for i in range(self.NP):
                harvesters_len = []
                harvesters_time = []
                for h, harv in enumerate(self.world.harvesters):
                    harv.dispatch_tasks(child_groups[i][h])
                    length = 0
                    for n in range(1, harv.nav_points.shape[0]):
                        # length += np.sqrt(np.sum((harv.nav_points[n] - harv.nav_points[n - 1]) ** 2))
                        length += compute_dist(harv.nav_points[n], harv.nav_points[n - 1])
                    harvesters_len.append(length)
                    harvesters_time.append(length / harv.speed)
                # print(harvesters_len)
                if self.target == "length":
                    len_path[i] = np.sum(harvesters_len)
                elif self.target == "time":
                    len_path[i] = np.max(harvesters_time)
                else:
                    raise NotImplementedError
                
            maxlen = np.max(len_path)  # 最长路径/时间
            minlen = np.min(len_path)  # 最短路径/时间
            Rlength.append(minlen)

            # 更新最短路程
            rr = np.where(len_path == minlen)[0]
            ind_best = popu[rr[0], :].copy()

            # 计算归一化的适应度函数
            for i in range(len_path.shape[0]):
                fitness[i] = 1 / len_path[i]
            total_fitness = sum(fitness)
            for i in range(len_path.shape[0]):
                fitness[i] /= total_fitness
            # print(fitness, sum(fitness))

            ## 交叉操作
            for t in range(self.NP):
                # 通过轮盘赌，从亲代选择两个个体
                a, A = roulette_wheel_selection(popu, fitness)
                b, B = roulette_wheel_selection(popu, fitness)
                # 如果随机数大于P_c，则选择适应度高的父代加入子代
                if np.random.rand() > self.P_c:
                    popu_tmp[t, :] = A.copy() if fitness[a] > fitness[b] else B.copy()
                else:
                    # 分组交叉操作
                    child = group_crossover(A, B, self.world.field.num_working_lines, self.world.num_harvester)
                    popu_tmp[t, :] = child.copy()

            ## 变异操作
            child_groups_tmp = decode_popu(popu_tmp)
            for t in range(self.NP):
                child_group_tmp = child_groups_tmp[t]
                if np.random.rand() < self.P_m1:    # 组间转移
                    child_group_tmp = cross_group_trans(child_group_tmp)
                    # child_group_tmp = cross_trans_multi(child_group_tmp)
                if np.random.rand() < self.P_m2:    # 组间交换
                    child_group_tmp = cross_exchange_multi(child_group_tmp)
                if np.random.rand() < self.P_m3:    # 2-opt算子
                    child_group_tmp = two_opt(child_group_tmp)
                if np.random.rand() < self.P_m4:
                    child_group_tmp = in_group_exchange(child_group_tmp)
                if np.random.rand() < self.P_m5:
                    child_group_tmp = in_group_order(child_group_tmp)

                # 编码
                for i, c in enumerate(child_group_tmp[:-1]):
                    child_group_tmp[i] = np.append(c,-1)
                chrom = np.concatenate(child_group_tmp, dtype=int)
                popu_tmp[t,:] = chrom.copy()

            popu = popu_tmp.copy()  # Update population
            popu[0, :] = ind_best.copy()  # Retain the best individual

        # print("Best individual:", R)
        best_break_points = np.where(ind_best == -1)[0]
        best_working_lines = np.split(ind_best, best_break_points)
        best_individual = [best_working_lines[h][:] if h == 0 else best_working_lines[h][1:] for h in range(len(best_working_lines))]
        # np.save(savedir + self.target + '_' + str(self.world.num_harvester)  + '.npy', np.array(Rlength))

        if show_result:
            print("Best individual:", best_individual)
            assert savedir != None
            plt.rcParams.update({'font.size': 13})
            plt.figure()
            plt.plot(Rlength)
            plt.xlabel('迭代次数')
            if self.target == "time":
                plt.ylabel('最优方案完成作业时间(s)')
            else:
                plt.ylabel('最优方案完成作业路程(m)')
            # plt.ylim((2800,3300))
            plt.title('优化目标值随迭代过程变化曲线')
            plt.savefig(savedir + self.target + '_' + str(self.world.num_harvester)  + '.png')
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
    random.seed(0)
    world = World(all_args)
    for harv in world.harvesters:
        print(harv.speed)
    # solver1 = GA_RAW(world)
    # _, r1 = solver1.solve()
    solver2 = GA(world)
    # solver2 = GA(world, target="time")
    # _, r2 = solver2.solve()
    _, r2 = solver2.solve(True, 'figs/results/ga_ST_')

    # plt.figure()
    # plt.plot(r1, color='r', label='GA')
    # plt.plot(r2, color='b', label='GA Multi')
    # plt.xlabel('Generation')
    # plt.ylabel('Fitness value (Shortest path length)')
    # plt.title('Fitness Evolution Curve')
    # plt.legend()
    # plt.show()

    # popu = np.array([[0, 1, 2, -1, 3, 4, 5, 6, -1, 7, 8, 9, 10, 11, 12], \
    #                  [12, 10, 11, 8, -1, 9, 5, 6, 7, 0, -1, 1, 2, 3, 4], \
    #                  [5, 6, 7, 8, -1, 0, 1, 2, 3, 4, -1, 9, 10, 11, 12]])
    # a = decode_popu(popu)
    # for group in a:
    #     print(in_group_order(group))