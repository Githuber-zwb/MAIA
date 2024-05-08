from onpolicy.envs.IA.environment import IAMultiAgentEnv
from onpolicy.envs.IA.scenarios import load
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
    scenario = load(args.scenario_name + ".py").Scenario()
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

    np.random.seed(10)
    parser = argparse.ArgumentParser(
        description='ia', formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--scenario_name', type=str,
                        default='ia_simple', help="Which scenario to run on")
    parser.add_argument("--num_harvester", type=int, default=3, help="number of harvesters")
    parser.add_argument('--num_transporter', type=int,
                        default=2, help="number of transporters")
    
    # environment settings
    parser.add_argument('--dt', type=float, default=0.1, help="simulation interval")
    parser.add_argument("--episode_length", type=int, default=100)  
    # Other settings
    parser.add_argument("--shared_reward", action='store_false', default=True, help='Whether agent share the same rewadr')

    all_args = parser.parse_known_args()[0]
    env = IAEnv(all_args)

    env.world.harvesters[0].speed = 1.8
    env.world.harvesters[0].capacity = 1600
    env.world.harvesters[1].speed = 1.6
    env.world.harvesters[1].capacity = 2000

    world = env.world

    image_list = []
    for i in range(5000):
        # print("step: ", i, "\n")
        # print(i, world.harvesters[0].pos, "moving:", world.harvesters[0].moving, "\tfull:", world.harvesters[0].full, "\ttrasporting:", world.harvesters[0].transporting, "\tload:", world.harvesters[0].load, "\tin field: ",world.harvesters[0].in_harvest_field())
        actions = np.zeros([env.world.num_transporter, 1])
        # actions = np.zeros([2,1])
        # print("actions:", actions)
        # actions = np.random.ramd(env.n, 2)
        # print("actions: ", actions)
        # print("step: ", i, "actions: \n", actions, "\n")
        # obs, rews,dones,infos = env.step(actions, image_list)
        obs, rews,dones,infos = env.step(actions, auto_trans_mode=True)

        # for i in range(all_args.num_harvester):
        #     print("pos:", env.world.harvesters[i].pos, "moving:", env.world.harvesters[i].moving, "\tfull:",env. world.harvesters[i].full, "\ttrasporting:", 
        #         env.world.harvesters[i].transporting, "\tload:", env.world.harvesters[i].load, "\tin field: ", env.world.harvesters[i].in_harvest_field(),
        #             "\tload percentage: ", env.world.harvesters[i].load_percent)
        # for i in range(all_args.num_transporter):
        #     print("pos:", env.world.transporters[i].pos, "vel: ", env.world.transporters[i].vel, 
        #           "\ttrasporting:", env.world.transporters[i].transporting, "\tunloading: ", env.world.transporters[i].unloading,
        #           "\tempty: ", env.world.transporters[i].empty, "\tfull: ", env.world.transporters[i].full, 
        #           "\tload:", env.world.transporters[i].load, "\tload percentage: ", env.world.transporters[i].load_percent)
        
        # print("obs: ", obs, "\nrews: ", rews, "\ndones: ", dones, "\ninfos: ", infos, "\n")
        # print("harv1 load per:", obs[0][20])
        # print("harv1 load per:", obs[0][30])
        # print("harv1 load per:", obs[0][40])
        # print(env.world.transporters[0].has_dispatch_task)
        # print(env.world.harvesters[0].chosen)
        img = env.render("rgb_array")[0]
        image_list.append(img)
        if np.all(dones):
            break
        # if i % 100 == 0:
        #     imageio.imsave(f"env_auto_trans_mode{i}.jpg", img)

    for h in range(all_args.num_harvester):
        print(f"Harvester{h} total wait time: ", env.world.harvesters[h].total_wait_time)
    for t in range(all_args.num_transporter):
        print(f"Transporter{t} total trip: ", env.world.transporters[t].total_trip)
    print(i)
    # imageio.mimsave('render/env_auto_trans_mode_80_per.mp4', image_list)
