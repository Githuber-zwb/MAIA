import numpy as np
import math
import matplotlib.pyplot as plt
from onpolicy.config import get_config
from onpolicy.envs.IA.ia_core import World
import random
from onpolicy.envs.IA.GA.ga_algorithm import GA, decode_popu, compute_dist

def rand_algo(world: World, NP = 10000, target='time'):
    N = world.field.num_working_lines + world.num_harvester - 1
    popu = np.zeros((NP, N), dtype=int) 
    len_path = np.zeros(NP)  

    # 种群初始化
    for i in range(NP):
        arr = [t for t in range(world.field.num_working_lines)]
        arr += [-1 for _ in range(world.num_harvester - 1)]
        arr = np.array(arr)
        np.random.shuffle(arr)
        popu[i, :] = arr.copy()

    child_groups = decode_popu(popu)
    # 计算每一个个体的路程/时间
    for i in range(NP):
        harvesters_len = []
        harvesters_time = []
        for h, harv in enumerate(world.harvesters):
            harv.dispatch_tasks(child_groups[i][h])
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
    return min(len_path)

def manual_algo(world: World, target='time'):
    H = world.num_harvester
    N = world.field.num_working_lines
    harvesters_len = []
    harvesters_time = []
    total_speed = 0
    for harv in world.harvesters:
        total_speed += harv.speed
    start, end = 0, 0
    for harv in world.harvesters[:-1]:
        num_working_line = int(harv.speed / total_speed * N)
        end = start + num_working_line
        harv.dispatch_tasks(np.arange(start, end))

        length = 0
        for n in range(1, harv.nav_points.shape[0]):
            # length += np.sqrt(np.sum((harv.nav_points[n] - harv.nav_points[n - 1]) ** 2))
            length += compute_dist(harv.nav_points[n], harv.nav_points[n - 1])
        harvesters_len.append(length)
        harvesters_time.append(length / harv.speed)

        start = end
    world.harvesters[-1].dispatch_tasks(np.arange(start, world.field.num_working_lines))
    length = 0
    for n in range(1, harv.nav_points.shape[0]):
        # length += np.sqrt(np.sum((harv.nav_points[n] - harv.nav_points[n - 1]) ** 2))
        length += compute_dist(harv.nav_points[n], harv.nav_points[n - 1])
    harvesters_len.append(length)
    harvesters_time.append(length / harv.speed)
    if target == "length":
        return np.sum(harvesters_len)
    elif target == "time":
        return np.max(harvesters_time)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    parser = get_config()
    parser.add_argument('--scenario_name', type=str,
                        default='ia_simple', help="Which scenario to run on")
    parser.add_argument("--num_harvester", type=int, default=3, help="number of harvesters")
    parser.add_argument('--num_transporter', type=int,
                        default=2, help="number of transporters")
    all_args = parser.parse_known_args()[0]
    # np.random.seed(0)
    # random.seed(0)
    ## single test
    # world = World(all_args)
    # # for harv in world.harvesters:
    # #     print(harv.speed)
    # solver = GA(world)
    # # _, r = solver2.solve(True, 'figs/results/ga_full_')
    # _, r = solver.solve()

    # GA mulit test
    num_exp = 20
    results = np.zeros(num_exp)
    for i in range(num_exp):
        world = World(all_args)
        solver = GA(world)
        # solver = GA(world, target="length")
        _, r = solver.solve()
        results[i] = r[-1]
        print(i, r[-1])
    print(np.mean(results))

    # # random test
    # num_exp = 20
    # results = np.zeros(num_exp)
    # for i in range(num_exp):
    #     world = World(all_args)
    #     # results[i] = rand_algo(world, 5000)
    #     results[i] = rand_algo(world, 5000, target="length")
    #     print(i, results[i])
    # print(np.mean(results))

    # # manual test
    # num_exp = 20
    # results = np.zeros(num_exp)
    # for i in range(num_exp):
    #     world = World(all_args)
    #     # results[i] = manual_algo(world)
    #     results[i] = manual_algo(world, target="length")
    #     print(i, results[i])
    # print(np.mean(results))