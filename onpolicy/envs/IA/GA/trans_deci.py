from onpolicy.envs.IA.IA_env import IAEnv
from onpolicy.config import get_config
from onpolicy.envs.IA.utils import test_graph, compute_path_len
from onpolicy.envs.IA.GA.ga_algorithm import decode_popu, roulette_wheel_selection, group_crossover, cross_group_trans, cross_group_exchange
import numpy as np
import random
import copy
from matplotlib import pyplot as plt
import imageio

class DemandPoint(object):
    def __init__(self, id, h, t, p, g, old_nav_point, curr_nav_point):
        self.id = id
        self.h = h  # 收割机编号
        self.t = t  # 需求点时间
        self.p = p  # 需求点位置
        self.g = g  # 连通图
        self.old_nav_point = old_nav_point
        self.curr_nav_point = curr_nav_point

def decode_chrom(env, chrom, demand_points, show_graph=False, print_sche = False, return_decode_results=False):
    """
    输入：环境env、染色体chrom和需求点列表demand_points
    chrom是一个列表，长度为运粮车数目，每一个列表中是一台运粮车服务的需求点编号
    输出：该方案对应的运粮车行驶路程、转运次数和收割机等待时间
    """
    # 首先判断染色体长度和需求点数目相同
    tmp = 0
    for c in chrom:
        tmp += c.shape[0]
    assert tmp == len(demand_points)
    assert len(chrom) == env.world.num_transporter

    num_transporter = env.world.num_transporter  # 运粮车数目

    travel_lengths = np.zeros(num_transporter,dtype=float)   # 记录每一类运粮车的行驶路程
    trans_times = np.zeros(num_transporter, dtype=int)   # 记录每一类运粮车返回机库的次数
    wait_times = np.zeros(all_args.num_harvester)   # 记录每一辆收割机的等待时间
    decode_results = [] # 记录解码结果，每一个结果是一个元组，包括需求点时间、收割机编号（-1表示机库）、响应的运粮车

    # 解码，计算每一辆运粮车的调度结果
    for t_id in range(num_transporter):
        # print("Transporter: ", t_id)
        finish_time = 0.0
        c = chrom[t_id]
        trans_times[t_id] = c.shape[0]
        # 遍历需求点
        for tmped, p_id in enumerate(c):
            demand_point = demand_points[p_id]
            assert p_id == demand_point.id
            # 导航到目标需求点
            path = env.world.transporters[t_id].search_path(demand_point.p, demand_point.g)
            # print(path)
            if show_graph:
                # test_graph(demand_point.g, path)
                test_graph(demand_point.g, path, finish_time, True, '/home/wenbo/Documents/MAIA/figs/nav_test/trans_sche_', t_id)
            # 计算路程
            p_length = compute_path_len(path)
            travel_lengths[t_id] += p_length
            # 计算转移时间
            p_time = p_length / env.world.transporters[t_id].speed
            # 计算是否产生额外等待时间
            if finish_time + p_time <= demand_point.t:
                set_off_time = demand_point.t - p_time
                finish_time = demand_point.t + env.world.harvesters[demand_point.h].capacity / env.world.transporters[t_id].transporting_speed
            else:
                set_off_time = finish_time
                wait_times[demand_point.h] += finish_time + p_time - demand_point.t
                finish_time += p_time + env.world.harvesters[demand_point.h].capacity / env.world.transporters[t_id].transporting_speed

            # 完成转运任务，更新运粮车的位置和导航点
            env.world.transporters[t_id].pos = demand_point.p.copy()
            env.world.transporters[t_id].nav_points = [demand_point.old_nav_point.copy(), demand_point.curr_nav_point.copy()]
            
            # 更新运粮车capacity
            env.world.transporters[t_id].load += env.world.harvesters[demand_point.h].capacity

            # print(env.world.transporters[t_id].load)
            # 打印决策方案
            if print_sche:
                print(f"需求点ID：{p_id}；收割机：{demand_point.h}；时间：{demand_point.t}；位置：{demand_point.p}")
                print(f"运粮车：{t_id}；出发时间：{set_off_time}；导航路程：{p_length}；导航时间：{p_time}；抵达时间：{set_off_time+p_time}；完成时间：{finish_time}；转运后容量：{env.world.transporters[t_id].load}；等待时间：{wait_times[demand_point.h]}")
                print(f"{p_id} & {demand_point.h} & {demand_point.t} & {demand_point.p} & {set_off_time} & {p_length} & {p_time} & {set_off_time+p_time} & {finish_time} & {env.world.transporters[t_id].load}")
                print("*"*10)
                decode_results.append((demand_point.h, set_off_time, t_id))

            # 如果运粮车的剩余储粮量小于最大的收割机容量，返回机库
            if (tmped < c.shape[0] - 1 and (env.world.transporters[t_id].capacity - env.world.transporters[t_id].load) < env.world.harvesters[demand_points[tmped+1].h].capacity) or tmped == c.shape[0] - 1:
                path = env.world.transporters[t_id].search_path(env.world.field.depot, demand_point.g)
                p_length = compute_path_len(path)
                set_off_time = finish_time
                if show_graph:
                    # test_graph(demand_point.g, path)
                    test_graph(demand_point.g, path, finish_time, True, '/home/wenbo/Documents/MAIA/figs/nav_test/trans_sche_', t_id)
                travel_lengths[t_id] += p_length
                trans_times[t_id] += 1
                p_time = p_length / env.world.transporters[t_id].speed
                finish_time += p_time + env.world.transporters[t_id].load / env.world.transporters[t_id].transporting_speed
                # 打印决策方案
                if print_sche:
                    print(f"机库位置：{env.world.field.depot}")
                    print(f"运粮车：{t_id}；出发时间：{set_off_time}；导航路程：{p_length}；导航时间：{p_time}；抵达时间：{set_off_time+p_time}；完成时间：{finish_time}；转运后容量：{env.world.transporters[t_id].load}")
                    print(f"/ & / & / & {env.world.field.depot} & {set_off_time} & {p_length} & {p_time} & {set_off_time+p_time} & {finish_time} & {env.world.transporters[t_id].load}")
                    print("*"*10)
                    decode_results.append((-1, set_off_time, t_id))
                # 完成转运任务，更新运粮车的位置和导航点
                env.world.transporters[t_id].pos = env.world.field.depot.copy()
                env.world.transporters[t_id].nav_points = [env.world.field.depot, env.world.field.depot_nav_point]
                # 清空粮仓
                env.world.transporters[t_id].load = 0.0

    # 恢复环境
    env.world.recover()

    if not return_decode_results:
        return travel_lengths, trans_times, wait_times
    else:
        return travel_lengths, trans_times, wait_times, decode_results

def make_order(chrom):
    child_groups = []
    break_points = np.where(chrom == -1)[0]
    working_lines = np.split(chrom, break_points)
    for h, working_line in enumerate(working_lines):
        tmp = working_line[:] if h == 0 else working_line[1:]
        tmp = np.sort(tmp)
        if h != len(working_lines) - 1:
            tmp = np.append(tmp, -1)
        child_groups.append(tmp)
    return np.concatenate(child_groups, dtype=int)

class GA_trans(object):
    def __init__(self, env:IAEnv, demand_points, NP = 30, G = 50, P_c = 0.6, P_m1 = 0.7, P_m2 = 0.7, P_m3 = 0.8):
        self.env = env # 求解的农田场景，包括作业地块、收割机参数信息
        self.NP = NP    # 种群数目
        self.N = len(demand_points) + env.world.num_transporter - 1 # 染色体长度
        self.G = G  # 最大迭代步数
        self.P_c = P_c  # 交叉概率
        self.P_m1 = P_m1    # 组间转移概率
        self.P_m2 = P_m2    # 组间交换概率
        self.P_m3 = P_m3    # 有序编码概率
        self.demand_points = demand_points
        self.wait_time_factor = env.wait_time_factor
        self.distance_factor = env.distance_factor
        self.trans_times_factor = env.trans_times_factor
    
    def solve(self, show_result=False, savedir = None, test_graph = False):
        popu = np.zeros((self.NP, self.N), dtype=int) # 种群矩阵，每一行是一个染色体，每一个染色体长度是N
        popu_tmp = np.zeros((self.NP, self.N), dtype=int) # 临时存储更新过程的变量
        ind_best = np.zeros(self.N, dtype=int)    # 存储迭代过程中的最优个体
        cost = np.zeros(self.NP)  # 存储每一轮迭代时各个个体的总cost
        fitness = np.zeros(self.NP)  # 存储每一轮迭代时正则化的适应度函数
        gen = 0    #迭代步数

        # 种群初始化
        for i in range(self.NP):
            arr = [t for t in range(len(self.demand_points))]
            arr += [-1 for _ in range(self.env.world.num_transporter - 1)]
            arr = np.array(arr)
            np.random.shuffle(arr)
            if np.random.rand() < self.P_m3:
                arr = make_order(arr)
            popu[i, :] = arr.copy()
        # 将第一个个体手工初始化
        tmp = []
        for n in range(self.env.world.num_transporter):
            c = np.arange(n, len(self.demand_points), self.env.world.num_transporter)
            if n != self.env.world.num_transporter - 1:
                c = np.append(c, -1)
            tmp.append(c.copy())
        tmp = np.concatenate(tmp, dtype=int)
        popu[0,:] = tmp.copy()

        ind_best = popu[0, :].copy()  # 存储最优个体（对应最短路径/最短时间）

        Rlength = []  # 存储每一次迭代时的最短路径
        while gen < self.G:
            # print("迭代步数：", gen)
            gen += 1
            # print(popu)
            # 解码
            child_groups = decode_popu(popu)
            # 计算每一个个体的路程/时间
            for i in range(self.NP):
                chrom = child_groups[i].copy()
                travel_lengths, trans_times, wait_times = decode_chrom(self.env, chrom, self.demand_points, test_graph)
                cost[i] = self.wait_time_factor * np.sum(wait_times) + self.distance_factor * np.sum(travel_lengths) + self.trans_times_factor * np.sum(trans_times)
                
            maxcost = np.max(cost)  # 最长路径/时间
            mincost = np.min(cost)  # 最短路径/时间
            Rlength.append(mincost)
            # print(mincost)

            # 更新最短路程
            rr = np.where(cost == mincost)[0]
            ind_best = popu[rr[0], :].copy()
            # print(ind_best)

            # 计算归一化的适应度函数
            for i in range(cost.shape[0]):
                fitness[i] = 1 / cost[i]
            total_fitness = sum(fitness)
            for i in range(cost.shape[0]):
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
                    child = group_crossover(A, B, len(self.demand_points), self.env.world.num_transporter)
                    popu_tmp[t, :] = child.copy()

            ## 变异操作
            child_groups_tmp = decode_popu(popu_tmp)
            for t in range(self.NP):
                child_group_tmp = child_groups_tmp[t].copy()
                if np.random.rand() < self.P_m1:    # 组间转移
                    child_group_tmp = cross_group_trans(child_group_tmp)
                    # child_group_tmp = cross_trans_multi(child_group_tmp)
                if np.random.rand() < self.P_m2:    # 组间交换
                    child_group_tmp = cross_group_exchange(child_group_tmp)

                # 编码
                for i, c in enumerate(child_group_tmp[:-1]):
                    child_group_tmp[i] = np.append(c,-1)
                chrom = np.concatenate(child_group_tmp, dtype=int)
                if np.random.rand() < self.P_m3:
                    chrom = make_order(chrom)
                popu_tmp[t,:] = chrom.copy()

            popu = popu_tmp.copy()  # Update population
            popu[0, :] = ind_best.copy()  # Retain the best individual

        # print("Best individual:", R)
        best_break_points = np.where(ind_best == -1)[0]
        best_working_lines = np.split(ind_best, best_break_points)
        best_individual = [best_working_lines[h][:] if h == 0 else best_working_lines[h][1:] for h in range(len(best_working_lines))]

        if show_result:
            print("Best individual:", best_individual)
            # assert savedir != None
            plt.rcParams.update({'font.size': 13})
            plt.figure()
            plt.plot(Rlength)
            plt.xlabel('迭代步数')
            plt.ylabel('最优方案运粮车转运代价')
            # plt.ylim((2800,3300))
            plt.title('运粮车转运代价随迭代过程变化曲线')
            if savedir != None:
                plt.savefig(savedir + str(self.env.world.num_harvester) + "_" + str(self.env.world.num_transporter)  + '.png')
                np.save(savedir + str(self.env.world.num_harvester) + "_" + str(self.env.world.num_transporter) + '.npy', np.array(Rlength))
            plt.show()

        return best_individual, Rlength

if __name__  == "__main__":

    np.random.seed(3)
    random.seed(1)

    parser = get_config()
    parser.add_argument('--scenario_name', type=str,
                        default='ia_simple', help="Which scenario to run on")
    parser.add_argument("--num_harvester", type=int, default=3, help="number of harvesters")
    parser.add_argument('--num_transporter', type=int,
                        default=2, help="number of transporters")
    all_args = parser.parse_known_args()[0]

    num_exp = 1    # 生成的总环境数目

    # 生成随机环境
    env_ls = []
    results = np.zeros(num_exp)
    for _ in range(num_exp):
        env = IAEnv(all_args)
        env_ls.append(env)

    # 存放每个环境中的需求点列表，每个需求点是前面定义的类
    demand_point_ls = [[] for _ in range(num_exp)]
    
    # 计算需求点，保存在demand_point_ls中
    for i in range(num_exp):
        print("Environment ", i)
        env = env_ls[i]
        tmp = []
        actions = np.zeros([env.world.num_transporter, 1])
        t = 0.0
        id = 0
        while True:
            t += all_args.decision_dt
            # img = env.render("rgb_array")[0]
            obs, rews,dones,infos = env.step(actions, no_trans_mode=True, decPt = 1.0, demand_harvs = tmp)
            if len(tmp) != 0: # 一般一个时刻只有一个需求点，但是不排除特殊情况
                # h是收割机编号
                for h in tmp:
                    # 复制连通图，并连接当前收割机位置和旧的导航点
                    g = copy.deepcopy(env.world.field.graph)
                    g[tuple(env.world.harvesters[h].pos.copy())].append(tuple(env.world.harvesters[h].old_nav_point.copy()))
                    g[tuple(env.world.harvesters[h].old_nav_point.copy())].append(tuple(env.world.harvesters[h].pos.copy()))
                    if env.world.harvesters[h].in_head_lines():
                        g[tuple(env.world.harvesters[h].pos.copy())].append(tuple(env.world.harvesters[h].curr_nav_point.copy()))
                        g[tuple(env.world.harvesters[h].curr_nav_point.copy())].append(tuple(env.world.harvesters[h].pos.copy()))
                    demand_point = DemandPoint(id, h, t, env.world.harvesters[h].pos.copy(), g, \
                                               env.world.harvesters[h].old_nav_point.copy(), \
                                               env.world.harvesters[h].curr_nav_point.copy())
                    demand_point_ls[i].append(demand_point)
                    id += 1
                tmp = []

            if np.all(dones):
                env.world.recover()
                break
        
        # for p in demand_point_ls[i]:
        #     print(p.id, p.h, p.t, p.p, p.old_nav_point, p.curr_nav_point)
    
    ## 到这一步为止，所有的环境存在env_ls中，第i个环境的所有需求点存在demand_point_ls[i]中
    for i in range(num_exp):
        print("Environment ", i)
        env = env_ls[i] # 环境
        demand_points = demand_point_ls[i]  #需求点列表
        # for h in env.world.harvesters:
        #     print(h.capacity, h.speed)
        # for t in env.world.transporters:
        #     print(t.capacity, t.speed)

        # 初始化染色体编码，每个染色体是一个列表，其中由num_trnasporter个元素，对应每个运粮车响应的需求点
        # chrom = []
        # for n in range(env.world.num_transporter):
        #     c = np.arange(n, len(demand_points), env.world.num_transporter)
        #     chrom.append(c)
        # print(chrom)

        solver = GA_trans(env, demand_points, G=30)
        # solver.solve(True, "/home/wenbo/Documents/MAIA/figs/results/trans_sche_full_rand_")

        # best_individual, _ = solver.solve()
        # print(best_individual)
        best_individual = [np.array([1,2,5,6]), np.array([0,3,4,7,8,9,10,11,12])]
        _,_,_,decpde_results = decode_chrom(env, best_individual, demand_points, print_sche=True, return_decode_results=True)
        print(decpde_results)

    #     _, r = solver.solve()
    #     results[i] = r[-1]
    #     print(r[-1])
    # print(np.mean(results))
        
    # 存下调度结果动态过程
    for i in range(num_exp):
        print("Environment ", i)
        env = env_ls[i]
        env.current_step = 0
        actions = np.zeros([env.world.num_transporter, 1])
        t = 0.0
        env.world.recover()
        img_ls = []
        while True:
            flag = [False] * env.world.num_transporter
            t += all_args.decision_dt
            img = env.render("rgb_array")[0]
            img_ls.append(img)
            # obs, rews,dones,infos = env.step(actions, static_sche_mode=True, decode_results = decpde_results)
            obs, rews,dones,infos = env.step(actions, auto_trans_mode=True, decPt=0.8, transDP=1.0)

            if np.all(dones):

                print(flag)
                for id, trans in enumerate(env.world.transporters):
                    # print(np.all(trans.pos == env.world.field.depot))
                    # print(trans.load == 0.0)
                    # if not np.all(trans.pos == env.world.field.depot):
                    #     trans.set_action(1)
                    if np.all(trans.pos == env.world.field.depot) and trans.load == 0.0:
                        # print("HERE")
                        flag[id] = True
                if np.all(flag):
                    imageio.mimsave(f"figs/auto_trans_mode.mp4", img_ls)
                    break