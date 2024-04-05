import numpy as np
from numpy import random
from onpolicy.envs.IA.ia_core import World, Harvester, Transporter, Field
from onpolicy.envs.IA.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self, args):
        world = World(args)
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        world.assign_agent_colors()
        arr = np.array(range(world.field.num_working_lines))
        random.shuffle(arr)
        newarr = np.array_split(arr, world.num_harvester)
        for i, harv in enumerate(world.harvesters):
            start_side = random.randint(0,2)
            harv.assign_nav_points(newarr[i],start_side)
            # print(i, harv.working_lines, harv.start_side, harv.nav_points, harv.full, harv.moving, harv.transporting, harv.complete_task)
        for i, trans in enumerate(world.transporters):
            start_side = random.randint(0,2)
            x = random.uniform(0.0, world.field.field_width)
            if not start_side: # start side = 0
                y = random.uniform(0.0, world.field.headland_width)
            else:
                y = random.uniform(world.field.field_length - world.field.headland_width, world.field.field_length)
            trans.init_pos(np.array([x, y]))
            # print(trans.pos)

    def reward(self, trans, world):
        # Harvesters should try to move as short as they can
        # rew = - 0.0001 * np.linalg.norm(trans.vel)
        rew = 0
        if trans.transporting:
            rew += 10
        if trans.unloading:
            rew += 10
        if trans.full:
            rew -= 5
        dis_mat = []
        left_mat = []
        for har in world.harvesters:
            dis = np.linalg.norm(trans.pos - har.pos)
            # weighted_dis = dis * (1.0 - har.load_percent)
            dis_mat.append(dis)
            left_mat.append(har.capacity - har.load)
        # dis_mat.append(np.linalg.norm(trans.pos - world.field.depot) * (1.0 - trans.load_percent))
        dis_mat.append(np.linalg.norm(trans.pos - world.field.depot))
        left_mat.append(trans.capacity - trans.load)
        # print(left_mat, np.argmin(left_mat))
        rew = -0.01 * dis_mat[np.argmin(left_mat)]

        # min_dis = min(dis_mat)
        # sum_dis = np.sum(dis_mat)

        # print("min dis:", min_dis)
        # rew -= 0.05 * dis_mat
        return rew

    def observation(self, trans, world):
        # self_state = np.concatenate([[trans.i], trans.pos, trans.vel, [trans.load], [trans.load_percent],
        #                              [float(trans.transporting)], [float(trans.unloading)]])
        self_state = np.concatenate([trans.pos, trans.vel, [trans.capacity - trans.load], [trans.load_percent], 
                                     [float(trans.transporting)], [float(trans.unloading)]])
        # print(self_state)
        harv_states = []
        for harv in world.harvesters:
            harv_state = np.concatenate([harv.pos,  harv.dir, [harv.capacity - harv.load], [harv.load_percent], 
                                         [float(harv.full)], [float(harv.moving)], 
                                         [float(harv.transporting)], [float(harv.complete_task)]])
            harv_states.append(harv_state)
        harv_states = np.array(harv_states).reshape(-1)
        # print(harv_states)
        # print(np.concatenate([self_state, harv_states]))
        return np.concatenate([world.field.depot, self_state, harv_states])
    
    def done(self, world):
        flag = []
        for harv in world.harvesters:
            flag.append(harv.complete_task)
        if np.all(flag):
            return True
        return False

if __name__ == "__main__":
    import argparse
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
    parser.add_argument("--d_range", type=float, default=1.0, help="the distance between the harv and trans to transporting")


    all_args = parser.parse_known_args()[0]
    scenario = Scenario()
    world = scenario.make_world(all_args)
    # world.field.depot = np.array([0.0, 50.0])
    # print("depot: ", world.field.depot)
    # for harv in world.harvesters:
    #     print(harv.i, harv.name)
    # for trans in world.transporters:
    #     print(trans.i, trans.name)
    for harv in world.harvesters:
        print("navigation points:", harv.nav_points)
    world.transporters[0].vel = np.array([0.0,3.0])
    world.transporters[0].load = 20.0
    world.transporters[0].init_pos(np.array([0.5, 1.5]))
    for i in range(200):
        # print(i, world.harvesters[0].pos, "moving:", world.harvesters[0].moving, "\tfull:", world.harvesters[0].full, "\ttrasporting:", world.harvesters[0].transporting, "\tload:", world.harvesters[0].load, "\tin field: ",world.harvesters[0].in_harvest_field())
        world.step()
        print("step:", i, world.harvesters[0].pos, "moving:", world.harvesters[0].moving, "\tfull:", world.harvesters[0].full, 
              "\ttrasporting:", world.harvesters[0].transporting, "\tload:", world.harvesters[0].load, "\tin field: ",world.harvesters[0].in_harvest_field())
        print("step:", i, world.transporters[0].pos, "transporting speed:", world.transporters[0].transporting_speed, 
              "load:", world.transporters[0].load, "\tfull:", world.transporters[0].full, "\ttrasporting:", world.transporters[0].transporting, "\tunloading:", world.transporters[0].unloading, "\tempty ",world.transporters[0].empty)
        # ob = scenario.observation(world.transporters[0],world)
        # rew = scenario.reward(world.transporters[0])
        # print(i, ob, rew, scenario.done(world))
    # world.harvesters[0].transporting = True
    # for i in range(200):
    #     world.harvesters[0].update_state()
        # print(i, world.harvesters[0].pos, "moving:", world.harvesters[0].moving, "\tfull:", world.harvesters[0].full, "\ttrasporting:", world.harvesters[0].transporting, "\tload:", world.harvesters[0].load, "\tin field: ",world.harvesters[0].in_harvest_field())
    

    # world.transporters[0].vel = np.array([0.3,0.4])
    # for _ in range(100):
    #     world.transporters[0].update_state()
    #     world.harvesters[0].update_state()
    #     ob = scenario.observation(world.transporters[0], world)
    #     rew = scenario.reward(world.transporters[0], world)
    #     # print(i, ob, rew, scenario.done(world))
    #     # scenario.observation(world.transporters[0],world)
    #     # print(world.transporters[0].pos)

    # for harv in world.harvesters:
    #     harv.complete_task = True
    # # print(scenario.done(world))