import numpy as np
from numpy import random
from onpolicy.envs.IA.ia_core import World, Harvester, Transporter, Field
from onpolicy.envs.IA.scenario import BaseScenario

class Scenario(BaseScenario):
    def make_world(self, args):
        world = World(args)
        self.reset_world(world)
        return world

    def reset_world(self, world: World):
        world.reset()
        arr = np.array(range(world.field.num_working_lines))
        random.shuffle(arr)
        newarr = np.array_split(arr, world.num_harvester)
        for i, harv in enumerate(world.harvesters):
            harv.dispatch_tasks(newarr[i])

    def reward(self, trans: Transporter, world: World):
        # transporters reward
        # 计算智能体自身的新增加奖励，即新增加的路程
        # 计算收割机的等待时间成本（主要惩罚）
        # total_wait_time = 0.0
        # for harv in world.harvesters:
        #     total_wait_time += harv.new_wait_time
        # if trans.new_trans_times == 1:
        #     print("TRANS!")
        return - trans.new_trip_len, - trans.new_trans_times

    def observation(self, trans:Transporter, world: World):
        # self_state = np.concatenate([[trans.i], trans.pos, trans.vel, [trans.load], [trans.load_percent],
        #                              [float(trans.transporting)], [float(trans.unloading)]])
        self_state = trans.get_state()
        # print(self_state)
        harv_states = []
        for harv in world.harvesters:
            harv_state = harv.get_state()
            harv_states.append(harv_state)
        harv_states = np.array(harv_states).reshape(-1)
        # print(harv_states)
        # print(np.concatenate([self_state, harv_states]))
        return np.concatenate([np.array(world.field.depot) / 100, self_state, harv_states])
    
    def done(self, world):
        flag = []
        for harv in world.harvesters:
            flag.append(harv.complete_traj)
        if np.all(flag):
            return True
        return False

if __name__ == "__main__":
    import argparse
    import numpy as np
    import time

    # np.random.seed(10)
    parser = argparse.ArgumentParser(
        description='ia', formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--scenario_name', type=str,
                        default='ia_simple', help="Which scenario to run on")
    parser.add_argument("--num_harvester", type=int, default=3, help="number of harvesters")
    parser.add_argument('--num_transporter', type=int,
                        default=2, help="number of transporters")
    parser.add_argument("--harv_vmin", type=float, default=1.0)
    parser.add_argument("--harv_vmax", type=float, default=2.0)
    parser.add_argument("--harv_capmin", type=int, default=14)
    parser.add_argument("--harv_capmax", type=int, default=20)
    parser.add_argument("--trans_vmin", type=float, default=5.0)
    parser.add_argument("--trans_vmax", type=float, default=8.0)
    parser.add_argument("--trans_capmin", type=int, default=70)
    parser.add_argument("--trans_capmax", type=int, default=100)
    
    # environment settings
    parser.add_argument('--dt', type=float, default=0.1, help="simulation interval")
    parser.add_argument('--decision_dt', type=float, default=5.0, help="decision interval")
    parser.add_argument("--episode_length", type=int, default=100)  
    # Other settings
    parser.add_argument("--shared_reward", action='store_false', default=True, help='Whether agent share the same rewadr')
    parser.add_argument('--wait_time_factor', type=float, default=10.0, help="wait time factor")
    parser.add_argument('--distance_factor', type=float, default=0.01, help="distanc factor")
    parser.add_argument('--trans_times_factor', type=float, default=10.0, help="trans times factor")

    all_args = parser.parse_known_args()[0]
    scenario = Scenario()
    world = scenario.make_world(all_args)
    # for harv in world.harvesters:
    #     print(harv.i, harv.name)
    # for trans in world.transporters:
    #     print(trans.i, trans.name)
    for harv in world.harvesters:
        print("navigation points:", harv.nav_points)

    for i in range(200):
        # print(i, world.harvesters[0].pos, "moving:", world.harvesters[0].moving, "\tfull:", world.harvesters[0].full, "\ttrasporting:", world.harvesters[0].transporting, "\tload:", world.harvesters[0].load, "\tin field: ",world.harvesters[0].in_harvest_field())
        world.step()
        print("step:", i, world.harvesters[0].get_state())
        print("step:", i, world.transporters[0].get_state())
