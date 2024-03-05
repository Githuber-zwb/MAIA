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
    
    # Field settings
    parser.add_argument("--field_width", type=float, default=100.0, help="width of the farmland")
    parser.add_argument("--field_length", type=float, default=500.0, help="length of the farmland")
    parser.add_argument("--working_width", type=float, default=10.0, help="working width of the harvester")
    parser.add_argument("--headland_width", type=float, default=3.0, help="width of the headland")
    parser.add_argument("--ridge_width", type=float, default=0.3, help="width of the ridge")

    # Harvester parameters
    parser.add_argument("--yeild_per_meter", type=float, default=1.5, help="yeild of the harvester per meter")
    parser.add_argument("--transporting_speed", type=float, default=10.0, help="transporting speed of the harvester per second")
    parser.add_argument("--harvester_max_v", nargs='+', help='<Required> The maximum velocity of each harvester', required=True)
    parser.add_argument("--cap_harvester", nargs='+', help='<Required> The load capacity of each harvester', required=True) 

    # Transpoter parameters
    parser.add_argument("--unloading_speed", type=float, default=30.0, help="unloading speed of the transporter to the depot per second")
    parser.add_argument("--transporter_max_v", nargs='+', help='<Required> The maximum velocity of each transporter', required=True)  
    parser.add_argument("--cap_transporter", nargs='+', help='<Required> The load capacity of each harvester', required=True)
    
    # Other settings
    parser.add_argument("--shared_reward", action='store_false', default=True, help='Whether agent share the same rewadr')
    parser.add_argument("--d_range", type=float, default=5.0, help="the distance between the harv and trans to transporting")

    all_args = parser.parse_known_args()[0]
    env = IAEnv(all_args)
    world = env.world
    # for harv in world.harvesters:
    #     print(harv.i, harv.name)
    # for trans in world.transporters:
    #     print(trans.i, trans.name)
    # print("navigation points:", world.harvesters[0].nav_points)
    # for harv in env.world.harvesters:
        # print(harv.i, harv.name, "\nnav points: \n", harv.nav_points, "pos:", harv.pos)
    env.world.transporters[0].init_pos(env.world.harvesters[0].pos - [0, 4.9])
    # env.world.transporters[0].init_pos(env.world.field.depot - [0, 4.9])
    # env.world.transporters[0].load = 20.0
    # env.world.transporters[0].empty = False
    # env.render_world()
    # time.sleep(10)

    image_list = []
    for i in range(500):
        print("step: ", i, "\n")
        # print(i, world.harvesters[0].pos, "moving:", world.harvesters[0].moving, "\tfull:", world.harvesters[0].full, "\ttrasporting:", world.harvesters[0].transporting, "\tload:", world.harvesters[0].load, "\tin field: ",world.harvesters[0].in_harvest_field())
        action0 = np.array([0.0, 3.0]) if env.world.harvesters[0].start_side == 0 else np.array([0.0, -3.0])
        # action0 = np.zeros(2)
        action1 = np.random.rand(2, )
        actions = np.stack([action0, action1])
        # print("actions:", actions)
        # actions = np.random.ramd(env.n, 2)
        # print("actions: ", actions)
        # print("step: ", i, "actions: \n", actions, "\n")
        obs, rews,dones,infos = env.step(actions)

        # for i in range(all_args.num_harvester):
        #     print("pos:", env.world.harvesters[i].pos, "moving:", env.world.harvesters[i].moving, "\tfull:",env. world.harvesters[i].full, "\ttrasporting:", 
        #         env.world.harvesters[i].transporting, "\tload:", env.world.harvesters[i].load, "\tin field: ", env.world.harvesters[i].in_harvest_field(),
        #             "\tload percentage: ", env.world.harvesters[i].load_percent)
        # for i in range(all_args.num_transporter):
        #     print("pos:", env.world.transporters[i].pos, "vel: ", env.world.transporters[i].vel, 
        #           "\ttrasporting:", env.world.transporters[i].transporting, "\tunloading: ", env.world.transporters[i].unloading,
        #           "\tempty: ", env.world.transporters[i].empty, "\tfull: ", env.world.transporters[i].full, 
        #           "\tload:", env.world.transporters[i].load, "\tload percentage: ", env.world.transporters[i].load_percent)
        print("obs: ", obs, "\nrews: ", rews, "\ndones: ", dones, "\ninfos: ", infos, "\n")
        img = env.render("rgb_array")[0]
        # image_list.append(img)
        # print(img.shape)
        # if i == 0:
            # imageio.imsave("env2.jpg", img)
        # time.sleep(10)
    # imageio.mimsave('env2.gif', image_list, duration=0.03)
    # world.harvesters[0].transporting = True
    # for i in range(200):
        # world.harvesters[0].update_state()
        # print(i, world.harvesters[0].pos, "moving:", world.harvesters[0].moving, "\tfull:", world.harvesters[0].full, "\ttrasporting:", world.harvesters[0].transporting, "\tload:", world.harvesters[0].load, "\tin field: ",world.harvesters[0].in_harvest_field())
    

    # world.transporters[0].vel = np.array([0.3,0.4])
    # for _ in range(100):
        # world.transporters[0].update_state()
        # world.harvesters[0].update_state()
        # ob = scenario.observation(world.transporters[0],world)
        # rew = scenario.reward(world.transporters[0])
        # print(i, ob, rew, scenario.done(world))
        # scenario.observation(world.transporters[0],world)
        # print(world.transporters[0].pos)

    # for harv in world.harvesters:
        # harv.complete_task = True
    # print(scenario.done(world))