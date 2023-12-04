from onpolicy.envs.mpe.environment import IAMultiAgentEnv
from onpolicy.envs.mpe.scenarios import load


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
    parser = argparse.ArgumentParser(
        description='ia', formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--scenario_name', type=str,
                        default='ia_simple', help="Which scenario to run on")
    parser.add_argument("--num_harvester", type=int, default=3, help="number of harvesters")
    parser.add_argument('--num_transporter', type=int,
                        default=2, help="number of transporters")
    parser.add_argument('--dt', type=float, default=0.1, help="simulation interval")
    
    parser.add_argument("--episode_length", type=int, default=100)  
    
    parser.add_argument("--field_width", type=float, default=10.0, help="width of the farmland")
    parser.add_argument("--field_length", type=float, default=50.0, help="length of the farmland")
    parser.add_argument("--working_width", type=float, default=1.0, help="working width of the harvester")
    parser.add_argument("--headland_width", type=float, default=3.0, help="width of the headland")
    parser.add_argument("--ridge_width", type=float, default=0.3, help="width of the ridge")
    parser.add_argument("--harvester_max_v", type=float, default=3.0)
    parser.add_argument("--transporter_max_v", type=float, default=5.0)  
    parser.add_argument("--yeild_per_meter", type=float, default=1.5, help="yeild of the harvester per meter") 
    parser.add_argument("--cap_harvester", type=float, default=30.0, help="capacity of the harvester") 
    parser.add_argument("--cap_transporter", type=float, default=100.0, help="capacity of the transporter")
    parser.add_argument("--transporting_speed", type=float, default=10.0, help="transporting speed of the harvester per second")
    parser.add_argument("--unloading_speed", type=float, default=30.0, help="unloading speed of the transporter per second")
    parser.add_argument("--shared_reward", action='store_false', default=True, help='Whether agent share the same rewadr')
    parser.add_argument("--d_range", type=float, default=1.0, help="the distance between the harv and trans to transporting")

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
    # world.transporters[0].vel = np.array([0.3,0.4])

    for i in range(100):
        # print(i, world.harvesters[0].pos, "moving:", world.harvesters[0].moving, "\tfull:", world.harvesters[0].full, "\ttrasporting:", world.harvesters[0].transporting, "\tload:", world.harvesters[0].load, "\tin field: ",world.harvesters[0].in_harvest_field())
        actions = np.random.rand(env.n, 2)
        print("step: ", i, "actions: \n", actions, "\n")
        obs, rews,dones,infos = env.step(actions)
        for harv in env.world.transporters:
            print(harv.vel, "\n")
        print("obs: ", obs, "\nrews: ", rews, "\ndones: ", dones, "\ninfos: ", infos, "\n")
        env.render()
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