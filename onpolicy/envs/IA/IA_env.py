from onpolicy.envs.IA.environment import IAMultiAgentEnv
from onpolicy.envs.IA.scenarios import load
from onpolicy.envs.IA.scenarios.ia_simple import Scenario
import imageio
from matplotlib import pyplot as plt
import numpy as np

def IAEnv(args):
    '''
    Creates a MultiAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.

    Input:
        scenario_name   :   name of the scenario from ./scenarios/ to be Returns
                            (without the .py extension)
        benchmark       :   whether you want to produce benchmarking data
                            (usually only done during evaluation)

    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    '''

    # load scenario from script
    scenario:Scenario = load(args.scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world(args)
    # create multiagent environment
    env = IAMultiAgentEnv(world, scenario.reset_world,
                        scenario.reward, scenario.observation, scenario.info, scenario.done)
    return env

if __name__ == "__main__":
    import argparse
    import time
    import random
    from onpolicy.config import get_config
    from onpolicy.envs.IA.utils import test_graph

    np.random.seed(3)
    random.seed(1)

    parser = get_config()
    parser.add_argument('--scenario_name', type=str,
                        default='ia_simple', help="Which scenario to run on")
    parser.add_argument("--num_harvester", type=int, default=3, help="number of harvesters")
    parser.add_argument('--num_transporter', type=int,
                        default=2, help="number of transporters")
    parser.add_argument('--test_graph',  action='store_true',
                        default=False)

    all_args = parser.parse_known_args()[0]

    env = IAEnv(all_args)

    # img = env.render("rgb_array")[0]
    # imageio.imsave(f"figs/single_{all_args.num_harvester}_{all_args.num_transporter}.jpg", img)

    reward_ls = []
    dist_ls = []
    wait_ls = []
    trans_num_ls = []
    for episode in range(1):
        image_list = []
        print("Episode: ", episode)
        env.reset()

        for i in range(len(env.world.harvesters)):
            print(f"Harv {i} speed and cap: ", env.world.harvesters[i].speed, env.world.harvesters[i].capacity)
            # print(f"harv {i} state: ", env.world.harvesters[i].get_state())
        for i in range(len(env.world.transporters)):
            print(f"Trans {i} speed and cap: ", env.world.transporters[i].speed, env.world.transporters[i].capacity)
            
        rewards_total = []
        for i in range(all_args.episode_length):
            # print(i)
            if all_args.test_graph and i % 100 == 0:
                test_graph(env.world.field.graph, time=i)
         
            img = env.render("rgb_array")[0]
            # image_list.append(img)
            # if i % 100 == 0:
                # imageio.imsave(f"/home/wenbo/Documents/MAIA/figs/graph_test/field_{i}.jpg", img)

            # auto trans mode
            actions = np.zeros([env.world.num_transporter, 1])
            # obs, rews,dones,infos = env.step(actions, auto_trans_mode=True, decPt=0.8, show_graph=True, time = i)
            obs, rews,dones,infos = env.step(actions, auto_trans_mode=True, decPt=0.5, transDP=1.0)
            # obs, rews,dones,infos = env.step(actions, no_trans_mode=True, decPt=0.8)
            rewards_total.append(rews)

            # # random policy
            # actions = np.random.randint(0, 5, size=env.world.num_transporter)
            # print(actions)
            # obs, rews, dones, infos = env.step(actions)
            # print("Rewards: ", rews)
            # print("step:", env.current_step)
            # print("done: ", dones)
            # print(obs)

            if np.all(dones):
                print(i)
                if all_args.test_graph:
                    test_graph(env.world.field.graph, time=i)
                # img = env.render("rgb_array")[0]
                # image_list.append(img)
                break

        # imageio.mimsave('/home/wenbo/Documents/MAIA/figs/tmp.mp4', image_list)
            
        rewards_total = np.sum(np.array(rewards_total), axis=0)
        reward_ls.append(np.mean(rewards_total))
        wait_tmp = []
        dist_tmp = []
        trans_num_tmp = []
        for h in range(all_args.num_harvester):
            print(f"Harvester{h} total wait time: ", env.world.harvesters[h].total_wait_time)
            wait_tmp.append(env.world.harvesters[h].total_wait_time)
        for t in range(all_args.num_transporter):
            print(f"Transporter{t} total trip: ", env.world.transporters[t].total_trip)
            print(f"Transporter{t} total transport times: ", env.world.transporters[t].trans_times)
            dist_tmp.append(env.world.transporters[t].total_trip)
            trans_num_tmp.append(env.world.transporters[t].trans_times)
        wait_ls.append(np.mean(wait_tmp))
        dist_ls.append(np.mean(dist_tmp))
        trans_num_ls.append(np.mean(trans_num_tmp))
        print("*"*10)
        
    # print(reward_ls, np.mean(reward_ls), np.std(reward_ls))
    print("mean wait time: ", np.mean(wait_ls))
    print("mean trans distance: ", np.mean(dist_ls))
    print("mean trans times: ", np.mean(trans_num_ls))
    print("reuslt: ", '%.2f' % np.mean(wait_ls), " & ", '%.2f' % np.mean(dist_ls)," & ", '%.2f' % np.mean(trans_num_ls))
    # imageio.mimsave('render/env_auto_trans_mode_100_per_3_2.mp4', image_list)
