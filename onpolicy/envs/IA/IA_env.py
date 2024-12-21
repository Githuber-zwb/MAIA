from onpolicy.envs.IA.environment import IAMultiAgentEnv
from onpolicy.envs.IA.scenarios import load
from onpolicy.envs.IA.scenarios.ia_simple import Scenario
import imageio

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
    import numpy as np
    import time
    from onpolicy.config import get_config

    # np.random.seed(7)
    parser = get_config()
    parser.add_argument('--scenario_name', type=str,
                        default='ia_simple', help="Which scenario to run on")
    parser.add_argument("--num_harvester", type=int, default=3, help="number of harvesters")
    parser.add_argument('--num_transporter', type=int,
                        default=2, help="number of transporters")

    all_args = parser.parse_known_args()[0]
    env = IAEnv(all_args)

    # env.world.harvesters[0].speed = 1.8
    # env.world.harvesters[0].capacity = 1600
    # env.world.harvesters[1].speed = 1.6
    # env.world.harvesters[1].capacity = 2000

    image_list = []
    reward_ls = []
    for episode in range(1):
        env.reset()
        for i in range(len(env.world.harvesters)):
            print(f"Harv {i} speed and cap: ", env.world.harvesters[i].speed, env.world.harvesters[i].capacity)
            print(f"harv {i} state: ", env.world.harvesters[i].get_state())
        for i in range(len(env.world.transporters)):
            print(f"Trans {i} speed and cap: ", env.world.transporters[i].speed, env.world.transporters[i].capacity)
            
        rewards_total = []
        for i in range(all_args.episode_length):
            # print("step: ", i, ", environment step: ", env.world.world_step)
            # print(env.world.transporters[0].trans_times)
            # print(env.world.harvesters[0].new_wait_time)
            # print(env.world.harvesters[1].new_wait_time)
            # print("trans length: ", env.world.transporters[0].new_trip_len)

            # auto trans mode
            actions = np.zeros([env.world.num_transporter, 1])
            obs, rews,dones,infos = env.step(actions, auto_trans_mode=True, decPt=0.6)
            # obs, rews,dones,infos = env.step(actions, no_trans_mode=True)
            rewards_total.append(rews)
            # print("Rewards: ", rews)
            # print("done: ", dones)
            # print("step:", env.current_step, "\n")

            # # random policy
            # actions = np.random.randint(0, 5, size=env.world.num_transporter)
            # print(actions)
            # obs, rews, dones, infos = env.step(actions)
            # print("Rewards: ", rews)
            # print("step:", env.current_step)
            # print("done: ", dones)
            # print(obs)

            img = env.render("rgb_array")[0]
            # image_list.append(img)
            if np.all(dones):
                break
            # if i % 100 == 0:
            #     imageio.imsave(f"env_auto_trans_mode{i}.jpg", img)
        rewards_total = np.sum(np.array(rewards_total), axis=0)
        print(rewards_total)
        reward_ls.append(np.mean(rewards_total))
        for h in range(all_args.num_harvester):
            print(f"Harvester{h} total wait time: ", env.world.harvesters[h].total_wait_time)
        for t in range(all_args.num_transporter):
            print(f"Transporter{t} total trip: ", env.world.transporters[t].total_trip)
            print(f"Transporter{t} total transport times: ", env.world.transporters[t].trans_times)
    print(reward_ls, np.mean(reward_ls), np.std(reward_ls))
    # imageio.mimsave('render/env_auto_trans_mode_100_per_3_2.mp4', image_list)
