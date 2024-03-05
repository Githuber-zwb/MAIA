import numpy as np
import math
import seaborn as sns
from numpy import random

class Field(object):
    def __init__(self, args):
        # farm properties
        self.field_length = args.field_length
        self.field_width = args.field_width
        self.working_width = args.working_width
        self.headland_width = args.headland_width
        self.ridge_width = args.ridge_width

        depot = np.array([[0,0], [0, self.field_length], [self.field_width, 0], [self.field_width, self.field_length]])
        self.depot = depot[random.randint(0,4), :]
        self.num_working_lines = math.ceil(self.field_width/self.working_width)
        self.ridge_length = self.field_length - 2 * self.headland_width
        self.working_line_length = self.field_length - self.headland_width  # length of a working line

        self.compute_ridges()
        self.compute_nav_points()

    def compute_ridges(self):
        # return the center of all the ridges
        # return: a (self.num_working_lines - 1, 2) numpt array, each row is the center of a ridge.
        self.ridges = np.array([[(i+1) * self.working_width, self.field_length/2] for i in range(self.num_working_lines - 1)])

    def compute_nav_points(self):
        #computes the navigation points
        #return: a [2, num_working_lines, 2] numpy array. 
        x_coor_1 = [self.working_width/2 + i * self.working_width for i in range(self.num_working_lines)]
        y_coor_1 = [self.headland_width/2 for _ in range(self.num_working_lines)]
        x_coor_2 = [self.working_width/2 + i * self.working_width for i in range(self.num_working_lines)]
        y_coor_2 = [self.field_length - self.headland_width/2 for _ in range(self.num_working_lines)]
        self.nav_points = np.array([[x_coor_1, y_coor_1],[x_coor_2, y_coor_2]]).transpose(0,2,1)

class Harvester(object):
    def __init__(self, args, field, speed = 3.0, capacity = 30.0, start_side = 0):
        self.i = 0
        self.name = ''
        self.color = None
        self.field = field
        self.dt = args.dt

        self.yeild_per_meter = args.yeild_per_meter
        self.transporting_speed = args.transporting_speed
        self.start_side = start_side    # which side to start(0 or 1)
        self.speed = float(speed)   # drive at maximum speed
        self.capacity = float(capacity)
        
        self.field_x_range = [0, self.field.field_width]
        self.field_y_range = [self.field.headland_width, self.field.field_length - self.field.headland_width]

        self.load = 0.0
        self.load_percent = self.load / self.capacity
        self.moving = True
        self.full = False
        self.transporting = False
        self.complete_task = False

    def assign_nav_points(self, working_lines, start_side):
        self.working_lines = working_lines  # the working lines
        self.start_side = start_side
        nav_points = []
        if not self.start_side: # start side = 0
            for i in range(len(self.working_lines)):
                nav_points.append(self.field.nav_points[i % 2, self.working_lines[i], :])
                nav_points.append(self.field.nav_points[(i + 1) % 2, self.working_lines[i], :])
        else:   # start side = 1
            for i in range(len(self.working_lines)):
                nav_points.append(self.field.nav_points[(i + 1) % 2, self.working_lines[i], :])
                nav_points.append(self.field.nav_points[i % 2, self.working_lines[i], :])
        self.nav_points = np.array(nav_points)

        self.pos = self.nav_points[0]
        self.nav = 1    # curr nav point
        self.old_nav_point = self.nav_points[self.nav - 1]
        self.curr_nav_point = self.nav_points[self.nav]
        self.dir = (self.curr_nav_point - self.old_nav_point) / np.linalg.norm(self.curr_nav_point - self.old_nav_point)

    def update_pos(self):
        # if the harvester capacity is full, do not update position(it is stop)
        # if the car is transporting or has complete task, do not move
        if self.full or self.transporting or self.complete_task:  
            return
        else:
            pred_new_pos = self.pos + self.dir * self.dt * self.speed
            while (pred_new_pos - self.old_nav_point) @ (pred_new_pos - self.curr_nav_point) > 0:
                left_dis = np.linalg.norm(pred_new_pos - self.curr_nav_point)
                self.nav += 1
                if self.nav == len(self.nav_points):    #complete task
                    self.complete_task = True
                    return
                self.curr_nav_point = self.nav_points[self.nav]
                self.old_nav_point = self.nav_points[self.nav - 1]
                self.dir = (self.curr_nav_point - self.old_nav_point) / np.linalg.norm(self.curr_nav_point - self.old_nav_point)
                pred_new_pos = self.old_nav_point + left_dis * self.dir
            self.pos = pred_new_pos

    def update_load(self):
        if self.transporting:
            self.stop = True
            self.load = max(0, self.load - self.transporting_speed * self.dt)
            if self.load == 0:
                self.stop = False
                self.transporting = False
        if self.stop or self.complete_task:
            return
        elif self.in_harvest_field():
            self.load = self.load + self.yeild_per_meter * self.speed * self.dt
            if self.load > self.capacity:
                self.load = self.capacity
                self.stop = True

    def in_harvest_field(self):
        return self.field_x_range[0] <= self.pos[0] <= self.field_x_range[1] and self.field_y_range[0] <= self.pos[1] <= self.field_y_range[1]

    def update_state(self):
        # self.update_pos()
        # self.update_load()
        if self.full and self.in_harvest_field():
            self.moving = False

        if self.moving:
            pred_new_pos = self.pos + self.dir * self.dt * self.speed
            while (pred_new_pos - self.old_nav_point) @ (pred_new_pos - self.curr_nav_point) > 0:
                left_dis = np.linalg.norm(pred_new_pos - self.curr_nav_point)
                self.nav += 1
                if self.nav == len(self.nav_points):    #complete task
                    self.complete_task = True
                    self.moving = False
                    return
                self.curr_nav_point = self.nav_points[self.nav]
                self.old_nav_point = self.nav_points[self.nav - 1]
                self.dir = (self.curr_nav_point - self.old_nav_point) / np.linalg.norm(self.curr_nav_point - self.old_nav_point)
                pred_new_pos = self.old_nav_point + left_dis * self.dir
            self.pos = pred_new_pos
            if self.in_harvest_field():
                self.load = self.load + self.yeild_per_meter * self.speed * self.dt
                if self.load > self.capacity:
                    self.load = self.capacity
                    self.full = True

        if self.transporting:
            # self.moving = False
            if not self.complete_task:
                self.moving = True
            self.full = False
            self.load = max(0, self.load - self.transporting_speed * self.dt)
            # if self.load == 0:  #complete transporting
                # if not self.complete_task:
                    # self.moving = True
                # self.transporting = False
            
        self.load_percent = self.load / self.capacity

class Transporter(object):
    def __init__(self, args, field, speed = 10.0, capacity = 100.0):
        self.i = 0
        self.name = ''
        self.color = None
        self.field = field
        self.dt = args.dt
        self.transporting_speed = args.transporting_speed
        self.unloading_speed = args.unloading_speed
        self.capacity = float(capacity)
        self.max_velocity = float(speed)

        self.pos = np.array([0.0, 0.0])
        self.vel = np.array([0.0, 0.0])
        self.load = 0
        self.load_percent = self.load / self.capacity
        self.transporting = False
        self.unloading = False
        self.empty = True
        self.full = False

    def init_pos(self, pos):
        self.pos = pos

    def update_state(self):
        if self.transporting:
            self.empty = False
            self.load = min(self.capacity, self.load + self.transporting_speed * self.dt)
            if self.load == self.capacity:
                self.full = True
        elif self.unloading:
            self.full = False
            self.load = max(0, self.load - self.unloading_speed * self.dt)
            if self.load == 0:
                self.empty = True
        
        if np.linalg.norm(self.vel) > self.max_velocity:
            self.vel = self.max_velocity * self.vel / np.linalg.norm(self.vel)
        self.pos += self.vel * self.dt
        self.load_percent = self.load / self.capacity
            
# multi-agent world
class World(object):
    def __init__(self, args, field):
        # farm properties
        self.world_step = 0
        self.field = field

        self.num_harvester = args.num_harvester
        self.num_transporter = args.num_transporter

        self.dt = args.dt
        self.episode_length = args.episode_length

        self.d_range = args.d_range
        self.shared_reward = args.shared_reward

        self.transporting_speed = args.transporting_speed

        self.dim_color = 3
        self.dim_c = 2

        assert len(args.harvester_max_v) == args.num_harvester and len(args.cap_harvester) == args.num_harvester, "Harvesters speed or cap not right"
        assert len(args.transporter_max_v) == args.num_transporter and len(args.cap_transporter) == args.num_transporter, "Transporters speed or cap not right"
        self.harvesters = [Harvester(args, field, args.harvester_max_v[i], args.cap_harvester[i]) for i in range(self.num_harvester)]
        self.transporters = [Transporter(args, field, args.transporter_max_v[i], args.cap_transporter[i]) for i in range(self.num_transporter)]
        for i, harv in enumerate(self.harvesters):
            harv.i = i
            harv.name = 'harvester %d' % i
        for j, harv in enumerate(self.transporters):
            harv.i = j + self.num_harvester
            harv.name = 'transporter %d' % j
        
    def assign_agent_colors(self):
        # harv_colors = [(0.25, 0.75, 0.25)] * self.num_harvester
        # for color, agent in zip(harv_colors, self.harvesters):
        #     agent.color = color

        # trans_colors = [(0.75, 0.25, 0.25)] * self.num_transporter
        # for color, agent in zip(trans_colors, self.transporters):
        #     agent.color = color
        for harv in self.harvesters:
            harv.color = np.random.rand(3)

        for trans in self.transporters:
            trans.color = np.random.rand(3)

    # update state of the world
    def step(self):
        # zoe 20200420
        self.world_step += 1

        for harv in self.harvesters:
            harv.transporting = False
        for trans in self.transporters:
            trans.transporting = False
            trans.transporting_speed = self.transporting_speed

        d_mat = np.zeros((self.num_transporter, self.num_harvester))
        for i, trans in enumerate(self.transporters):
            for j, harv in enumerate(self.harvesters):
                d_mat[i,j] = np.linalg.norm(harv.pos - trans.pos)
                # print("pos: ", harv.pos, trans.pos)
        # print(d_mat)

        for i, trans in enumerate(self.transporters):
            for j, harv in enumerate(self.harvesters):
                if d_mat[i, j] < self.d_range and not trans.full:
                    trans.transporting = True
                    harv.transporting = True
                    if harv.load == 0:
                        if harv.in_harvest_field():
                            trans.transporting_speed = harv.yeild_per_meter * harv.speed
                        else:
                            trans.transporting = False
                            harv.transporting = False
                            trans.transporting_speed = 0
            
            if np.linalg.norm(trans.pos - self.field.depot) < self.d_range and not trans.empty:
                trans.unloading = True
            else:
                trans.unloading = False

        for harv in self.harvesters:
            harv.update_state()
        for trans in self.transporters:
            trans.update_state()
                

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
    field = Field(all_args)
    # harv1 = Harvester(all_args, field, all_args.harvester_max_v[0], all_args.cap_harvester[0])
    # harv1.assign_nav_points(np.array([0,3,2,1]), 0)
    # for i in range(100):
    #     harv1.update_state()
    #     print("step:", i, harv1.pos, "moving:", harv1.moving, "\tfull:", harv1.full, "\ttrasporting:", 
    #           harv1.transporting, "\tload:", harv1.load, "\tin field: ",harv1.in_harvest_field(), "\tload percentage: ", harv1.load_percent)
    # harv1.transporting = True
    # for i in range(100):
    #     harv1.update_state()
    #     print("step:", i, harv1.pos, "moving:", harv1.moving, "\tfull:", harv1.full, "\ttrasporting:", 
    #           harv1.transporting, "\tload:", harv1.load, "\tin field: ",harv1.in_harvest_field(), "\tload percentage: ", harv1.load_percent)

    # trans1 = Transporter(all_args, field, all_args.transporter_max_v[0], all_args.cap_transporter[0])
    # trans1.vel = np.array([2.0, 3.0])
    # trans1.transporting = True
    # for i in range(150):
    #     trans1.update_state()
    #     print("step:", i, trans1.pos, "empty:", trans1.empty, "\tfull:", trans1.full, "\ttrasporting:", 
    #           trans1.transporting, "\tunloading: ",trans1.unloading, "\tload:", trans1.load, "\tload percentage: ", trans1.load_percent)

    world = World(all_args, field)
    for i in range(all_args.num_harvester):
        print(world.harvesters[i].speed, world.harvesters[i].capacity)
    for i in range(all_args.num_transporter):
        print(world.transporters[i].vel, world.transporters[i].capacity)
    world.harvesters[0].assign_nav_points(np.array([0,3,2,1]), 0)
    for i in range(100):
        world.harvesters[0].update_state()
        print("step:", i, world.harvesters[0].pos, "moving:", world.harvesters[0].moving, "\tfull:", world.harvesters[0].full, "\ttrasporting:", 
              world.harvesters[0].transporting, "\tload:", world.harvesters[0].load, "\tin field: ",world.harvesters[0].in_harvest_field(), "\tload percentage: ", world.harvesters[0].load_percent)
    # world.harvesters[0].assign_nav_points(np.array([0,3,2,1]), 0)
    # world.harvesters[1].assign_nav_points(np.array([4,5,6]), 1)
    # world.harvesters[2].assign_nav_points(np.array([7,8,9]), 0)
    # world.transporters[0].vel = np.array([0.0,3.0])
    # world.transporters[0].init_pos(np.array([0.5, 1.5]))
    # print("nav potins: ", field.nav_points)
    # print("nav potins 0: ", field.nav_points[0,0,:])
    # print("nav points shape: ", field.nav_points.shape)
    # print("ridges: ", field.ridges)
    # print("ridges shape: ", field.ridges.shape)
    # print(field.working_line_length)
    # hav1 = Harvester(all_args, field, 0, [0,3,2])
    # print("navigation points:", world.harvesters[0].nav_points)
    # print(hav1.dir)
    # for i in range(200):
        # world.harvesters[0].update_state()
        # world.step()
        # print("step:", i, world.harvesters[0].pos, "moving:", world.harvesters[0].moving, "\tfull:", world.harvesters[0].full, "\ttrasporting:", world.harvesters[0].transporting, "\tload:", world.harvesters[0].load, "\tin field: ",world.harvesters[0].in_harvest_field())
        # print("step:", i, world.transporters[0].pos, "transporting speed:", world.transporters[0].transporting_speed, "load:", world.transporters[0].load, "\tfull:", world.transporters[0].full, "\ttrasporting:", world.transporters[0].transporting, "\tunloading:", world.transporters[0].unloading, "\tempty ",world.transporters[0].empty)
    # world.harvesters[0].transporting = True
    # for i in range(100):
        # world.harvesters[0].update_state()
        # print(i, world.harvesters[0].pos, "moving:", world.harvesters[0].moving, "\tfull:", world.harvesters[0].full, "\ttrasporting:", world.harvesters[0].transporting, "\tload:", world.harvesters[0].load, "\tin field: ",world.harvesters[0].in_harvest_field())
        # print(i, hav1.pos, "moving:", hav1.moving, "\tfull:", hav1.full, "\ttrasporting:", hav1.transporting, "\tload:", hav1.load, "\tin field: ",hav1.in_harvest_field())
    # world.harvesters[0].transporting = False
    # for i in range(100):
    #     world.harvesters[0].update_state()
        # print(i, world.harvesters[0].pos, "moving:", world.harvesters[0].moving, "\tfull:", world.harvesters[0].full, "\ttrasporting:", world.harvesters[0].transporting, "\tload:", world.harvesters[0].load, "\tin field: ",world.harvesters[0].in_harvest_field())
       
    # trans1 = Transporter(all_args, field, [1.0,2.0])
    # trans1.vel = np.array([0.5,0.6])
    # for _ in range(100):
    #     trans1.update_state()
    #     print(trans1.pos)
    # world.transporters[0].vel = np.array([0.5,0.6])
    # world.transporters[0].init_pos(np.array([1.0, 2.0]))
    # for _ in range(100):
    #     world.transporters[0].update_state()
    #     print(world.transporters[0].pos)