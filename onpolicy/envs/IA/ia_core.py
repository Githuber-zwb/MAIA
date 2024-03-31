import numpy as np
import math
import seaborn as sns
from numpy import random

class Field(object):
    def __init__(self, field_length = 520, field_width = 60, working_width = 3, yeild_per_m2 = 0.8, headland_width = 10, depot_pos = "fixed"):
        # farm properties
        self.field_length = field_length
        self.field_width = field_width
        self.working_width = working_width  # 作业行宽度 (m)
        self.yeild_per_m2 = yeild_per_m2    # 单位面积产量 (kg/m2)
        self.headland_width = headland_width    # 地头宽度

        depot = np.array([[0,0], [self.field_width, 0], [0, self.field_length], [self.field_width, self.field_length]])
        depot_nav_point = np.array([[0,self.headland_width/2], [self.field_width, self.headland_width/2], \
                                    [0, self.field_length - self.headland_width/2], [self.field_width, self.field_length - self.headland_width/2]])
        if depot_pos == "random":
            randInt = random.randint(0,4)
            randIntops = (randInt + 2) % 4
            self.depot = depot[randInt, :]
            self.depot_nav_point = depot_nav_point[randInt, :]
            self.depot_nav_point_ops = depot_nav_point[randIntops, :]
        elif depot_pos == "fixed":
            self.depot = depot[0, :]
            self.depot_nav_point = depot_nav_point[0, :]
            self.depot_nav_point_ops = depot_nav_point[2, :]
        else: 
            raise NotImplementedError
        self.num_working_lines = math.ceil(self.field_width/self.working_width) # 向上取整
        self.working_line_length = self.field_length - self.headland_width  # length of a working line
        self.crop_x_range = [0, self.field_width]
        self.crop_y_range = [self.headland_width, self.field_length - self.headland_width]

        self.compute_nav_points()

    def compute_nav_points(self):
        # computes the navigation points
        # return: a [2, num_working_lines, 2] numpy array. 
        x_coor_1 = [self.working_width/2 + i * self.working_width for i in range(self.num_working_lines - 1)]
        x_coor_1.append(self.field_width - self.working_width / 2)  #最后一个作业行用来封边
        y_coor_1 = [self.headland_width/2 for _ in range(self.num_working_lines)]
        x_coor_2 = x_coor_1
        y_coor_2 = [self.field_length - self.headland_width/2 for _ in range(self.num_working_lines)]
        self.nav_points = np.array([[x_coor_1, y_coor_1],[x_coor_2, y_coor_2]]).transpose(0,2,1)

class Harvester(object):
    def __init__(self, field: Field, speed = 1.5, capacity = 1800, transporting_speed = 200, dt = 0.1):
        self.id = 0
        self.name = ''
        self.color = None
        self.field = field
        self.speed = float(speed)   # 收割机的运行速度
        self.capacity = float(capacity)    # 容量，以kg为单位，1800kg约折合2.4立方米小麦。
        self.transporting_speed = transporting_speed
        self.dt = dt
        self.yeild_per_second = self.field.yeild_per_m2 * self.field.working_width * self.speed    # 计算得到收割机以最大速度行驶时每秒的产量(kg/s)，default得到的是3.6
        
        self.pos = self.field.depot  # 所有收割机都初始化在粮仓位置
        self.time = 0.0 # 记录当前时刻
        self.last_trans_time = 0.0  # 记录上一次转运完成时间
        self.total_wait_time = 0.0  # 记录总的等待时间
        self.load = 0.0
        self.cur_working_line = -1  # 开始时的作业行标识为-1
        self.load_percent = self.load / self.capacity
        self.complete_traj = False
        self.has_a_trans = False
        self.able_to_trans = False

    def assign_nav_points(self, working_lines):
        self.working_lines = working_lines  # the working lines
        if self.field.depot[1] == 0:    # start_side标识起始作业位置，0表示南侧（y==0)，1表示北侧
            start_side = 0
        else:
            start_side = 1
        nav_points = []
        nav_points.append(self.field.depot)
        nav_points.append(self.field.depot_nav_point)   # 从机库出发
        if not start_side: # start side = 0
            for i in range(len(self.working_lines)):
                nav_points.append(self.field.nav_points[i % 2, self.working_lines[i], :])
                nav_points.append(self.field.nav_points[(i + 1) % 2, self.working_lines[i], :])
        else:   # start side = 1
            for i in range(len(self.working_lines)):
                nav_points.append(self.field.nav_points[(i + 1) % 2, self.working_lines[i], :])
                nav_points.append(self.field.nav_points[i % 2, self.working_lines[i], :])
        if len(self.working_lines) % 2 == 1:
            nav_points.append(self.field.depot_nav_point_ops)   # 如果奇数个作业行，农机会停在对面位置，需要先导航到机库对面的导航点再回机库
            nav_points.append(self.field.depot) # 回到机库
        else:
            nav_points.append(self.field.depot_nav_point)   # 如果偶数作业行，返回临近机库的导航点，再回到机库
            nav_points.append(self.field.depot) # 回到机库
        self.nav_points = np.array(nav_points)
        # self.nav_points = nav_points

        self.nav = 1    # curr nav point
        self.old_nav_point = self.nav_points[self.nav - 1]
        self.curr_nav_point = self.nav_points[self.nav]
        self.dir = (self.curr_nav_point - self.old_nav_point) / np.linalg.norm(self.curr_nav_point - self.old_nav_point)

    def in_harvest_field(self):
        return self.field.crop_x_range[0] <= self.pos[0] <= self.field.crop_x_range[1] and self.field.crop_y_range[0] <= self.pos[1] <= self.field.crop_y_range[1]

    def move(self):
        if self.complete_traj: 
            return
        pred_new_pos = self.pos + self.dir * self.dt * self.speed
        if (pred_new_pos - self.old_nav_point) @ (pred_new_pos - self.curr_nav_point) > 0:  # 不支持连续一个time step拐弯多次，要设置dt足够小。
            left_dis = np.linalg.norm(pred_new_pos - self.curr_nav_point)
            self.nav += 1
            if self.nav == len(self.nav_points):    #complete task
                self.pos =self.nav_points[-1]
                self.complete_traj = True
                return
            self.curr_nav_point = self.nav_points[self.nav]
            self.old_nav_point = self.nav_points[self.nav - 1]
            self.dir = (self.curr_nav_point - self.old_nav_point) / np.linalg.norm(self.curr_nav_point - self.old_nav_point)
            pred_new_pos = self.old_nav_point + left_dis * self.dir
        self.pos = pred_new_pos
        if 0 <= self.nav - 2 < len(self.working_lines) * 2:
            self.cur_working_line = self.working_lines[math.floor((self.nav - 2) / 2)]
        else:
            self.cur_working_line = -1
        if self.in_harvest_field():
            self.load += self.yeild_per_second * self.dt
            if self.load > self.capacity:
                self.load = self.capacity
        self.load_percent = self.load / self.capacity

    def transport(self):
        self.load = max(0.0, self.load - self.transporting_speed * self.dt)
        self.load_percent = self.load / self.capacity
        if self.load_percent == 0.0:
            self.able_to_trans = False
            self.last_trans_time = self.time

    def update_state(self):
        self.time += self.dt
        if self.time - self.last_trans_time < 60:   # 两次转运的间隔大于60s
            self.able_to_trans = False
        else:
            self.able_to_trans = True

        if self.has_a_trans and self.able_to_trans:
            self.transport()
        elif self.load_percent != 1.0:
            self.move()
        else:
            self.total_wait_time += self.dt # 产生等待时间
            # return

class Transporter(object):
    def __init__(self, field: Field, speed = 10, capacity = 10000, transporting_speed = 200, dt = 0.1, pos_error = 0.8):
        self.i = 0
        self.name = ''
        self.color = None
        self.field = field
        self.dt = dt
        self.transporting_speed = transporting_speed
        self.capacity = float(capacity)
        self.speed = float(speed)
        self.pos_error = pos_error

        self.pos = self.field.depot # 运粮车初始位置在粮仓
        self.load = 0
        self.load_percent = self.load / self.capacity
        self.unloading = False
        self.returning_to_depot = False
        self.has_dispatch_task = False  # 当前是否有调运任务
        self.last_nav_point = np.zeros(2)   # 保存上一个导航点
        # 转运任务分成三个阶段：寻找收割机，转运，回到地头
        self.searching_for_harv = False
        self.transporting = False
        self.returning_to_headland = False

    def vehicle_in_north_side(self):
        return self.pos[1] > self.field.field_length / 2
    
    def point_in_north_side(self, point):
        return point[1] > self.field.field_length / 2

    def in_same_side(self, point):  # 判断当前车辆位置和传入的点是不是在一侧
        if self.vehicle_in_north_side() and self.point_in_north_side(point):
            return True
        elif not self.vehicle_in_north_side() and not self.point_in_north_side(point):
            return True
        else:
            return False

    def assign_search_nav_points(self, harv: Harvester):
        if self.has_dispatch_task:
            return  # 如果有调运任务，直接返回
        self.nav_points = []
        self.nav_points.append(self.pos)    # 每次调运任务结束都保证运粮车在地头导航点
        if np.all(self.pos == self.field.depot):
            self.nav_points.append(self.field.depot_nav_point)  # 第一次出发是在机库，将机库旁边的点加入导航点
        if harv.cur_working_line == -1: # 保证收割机的 nav >= 2
            print("Current harvester has just started or has completed task.")
            return
        harv_old_nav_point = harv.old_nav_point
        harv_curr_nav_point = harv.curr_nav_point
        # 当前位置和收割机上一个导航点在同侧，则依此假如上一个和当前导航点
        if self.in_same_side(harv_old_nav_point):  
            if not np.all(self.nav_points[-1] == harv_old_nav_point):
                self.nav_points.append(harv_old_nav_point)
            self.nav_points.append(harv_curr_nav_point)
            if harv.nav + 1 < len(harv.nav_points):
                self.nav_points.append(harv.nav_points[harv.nav + 1])
            if harv.nav + 2 < len(harv.nav_points):
                self.nav_points.append(harv.nav_points[harv.nav + 2])
        # old_nav_point的上一个导航点和农机在同侧，则加入上一个导航点
        elif harv.nav - 2 >= 0 and self.in_same_side(harv.nav_points[harv.nav - 2]):
            if not np.all(self.nav_points[-1] == harv.nav_points[harv.nav - 2]):
                self.nav_points.append(harv.nav_points[harv.nav - 2])
            self.nav_points.append(harv_old_nav_point)
            self.nav_points.append(harv_curr_nav_point)
            if harv.nav + 1 < len(harv.nav_points):
                self.nav_points.append(harv.nav_points[harv.nav + 1])
            if harv.nav + 2 < len(harv.nav_points):
                self.nav_points.append(harv.nav_points[harv.nav + 2])
        # old_nav_point的上两个个导航点和农机在同侧，则加入导航点
        elif harv.nav - 3 >= 0 and self.in_same_side(harv.nav_points[harv.nav - 3]):
            if not np.all(self.nav_points[-1] == harv.nav_points[harv.nav - 3]):
                self.nav_points.append(harv.nav_points[harv.nav - 3])
            self.nav_points.append(harv.nav_points[harv.nav - 2])
            self.nav_points.append(harv_old_nav_point)
            self.nav_points.append(harv_curr_nav_point)
            if harv.nav + 1 < len(harv.nav_points):
                self.nav_points.append(harv.nav_points[harv.nav + 1])
            if harv.nav + 2 < len(harv.nav_points):
                self.nav_points.append(harv.nav_points[harv.nav + 2])
        else:
            raise NotImplementedError
        self.nav_points = np.array(self.nav_points)
        self.nav = 1    # curr nav point
        self.old_nav_point = self.nav_points[self.nav - 1]
        self.curr_nav_point = self.nav_points[self.nav]
        self.dir = (self.curr_nav_point - self.old_nav_point) / np.linalg.norm(self.curr_nav_point - self.old_nav_point)
        self.searching_for_harv = True
    
    def in_head_lines(self):
        return self.pos[1] == self.field.headland_width/2 or self.pos[1] == self.field.field_length - self.field.headland_width/2
    
    def assign_return_head_nav_points(self):
        if self.in_head_lines():
            return
        self.nav_points = []
        self.nav_points.append(self.pos)
        self.nav_points.append(self.last_nav_point)
        self.nav_points = np.array(self.nav_points)

        self.nav = 1    # curr nav point
        self.old_nav_point = self.nav_points[self.nav - 1]
        self.curr_nav_point = self.nav_points[self.nav]
        self.dir = (self.curr_nav_point - self.old_nav_point) / np.linalg.norm(self.curr_nav_point - self.old_nav_point)
        self.returning_to_headland = True

    def assign_return_depot_nav_points(self):
        self.nav_points = []
        self.nav_points.append(self.pos)
        if self.in_same_side(self.field.depot_nav_point):
            self.nav_points.append(self.field.depot_nav_point)
        else:
            self.nav_points.append(self.field.depot_nav_point_ops)
        self.nav_points.append(self.field.depot)
        self.nav_points = np.array(self.nav_points)
        
        self.nav = 1    # curr nav point
        self.old_nav_point = self.nav_points[self.nav - 1]
        self.curr_nav_point = self.nav_points[self.nav]
        self.dir = (self.curr_nav_point - self.old_nav_point) / np.linalg.norm(self.curr_nav_point - self.old_nav_point)
        self.returning_to_headland = True

    def move(self, harv: Harvester = None):
        if self.searching_for_harv:
            assert harv != None, "The harvester must be provided."
            pred_new_pos = self.pos + self.dir * self.dt * self.speed
            if (pred_new_pos - self.old_nav_point) @ (pred_new_pos - self.curr_nav_point) > 0:  # 不支持连续一个time step拐弯多次，要设置dt足够小。
                left_dis = np.linalg.norm(pred_new_pos - self.curr_nav_point)
                self.nav += 1
                if self.nav == len(self.nav_points):    # 轨迹结束都没有找到收割机，返回
                    self.pos =self.nav_points[-1]
                    print("Cannot find the harvester!!")
                    self.searching_for_harv = False
                    return
                self.curr_nav_point = self.nav_points[self.nav]
                self.old_nav_point = self.nav_points[self.nav - 1]
                self.dir = (self.curr_nav_point - self.old_nav_point) / np.linalg.norm(self.curr_nav_point - self.old_nav_point)
                pred_new_pos = self.old_nav_point + left_dis * self.dir
            self.pos = pred_new_pos
            if np.linalg.norm(self.pos - harv.pos) < self.pos_error:
                self.searching_for_harv = False
                self.transporting = True
                self.last_nav_point = self.old_nav_point
        elif self.transporting:
            assert harv != None, "The harvester must be provided."
            if harv.able_to_trans == True and self.load_percent != 1.0:
                harv.has_a_trans = True
                self.load = min(self.capacity, self.load + self.transporting_speed * self.dt)
                self.load_percent = self.load / self.load_percent
            elif harv.able_to_trans == False:
                harv.has_a_trans = False
                self.transporting = False
                self.returning_to_headland = True
        elif self.returning_to_headland:
            if not self.in_head_lines():
                pred_new_pos = self.pos + self.dir * self.dt * self.speed
                if (pred_new_pos - self.old_nav_point) @ (pred_new_pos - self.curr_nav_point) > 0:  # 不支持连续一个time step拐弯多次，要设置dt足够小。
                    left_dis = np.linalg.norm(pred_new_pos - self.curr_nav_point)
                    self.nav += 1
                    if self.nav == len(self.nav_points):    # 结束导航
                        self.pos =self.nav_points[-1]
                        self.returning_to_headland = False
                        return
                    self.curr_nav_point = self.nav_points[self.nav]
                    self.old_nav_point = self.nav_points[self.nav - 1]
                    self.dir = (self.curr_nav_point - self.old_nav_point) / np.linalg.norm(self.curr_nav_point - self.old_nav_point)
                    pred_new_pos = self.old_nav_point + left_dis * self.dir
            else:
                self.returning_to_depot = False
        elif self.returning_to_depot:
            pred_new_pos = self.pos + self.dir * self.dt * self.speed
            if (pred_new_pos - self.old_nav_point) @ (pred_new_pos - self.curr_nav_point) > 0:  # 不支持连续一个time step拐弯多次，要设置dt足够小。
                left_dis = np.linalg.norm(pred_new_pos - self.curr_nav_point)
                self.nav += 1
                if self.nav == len(self.nav_points):    # 结束导航
                    self.pos =self.nav_points[-1]
                    self.returning_to_depot = False
                    return
                self.curr_nav_point = self.nav_points[self.nav]
                self.old_nav_point = self.nav_points[self.nav - 1]
                self.dir = (self.curr_nav_point - self.old_nav_point) / np.linalg.norm(self.curr_nav_point - self.old_nav_point)
                pred_new_pos = self.old_nav_point + left_dis * self.dir
            self.pos = pred_new_pos



    # def update_state(self):
    #     if self.transporting:
    #         self.empty = False
    #         self.load = min(self.capacity, self.load + self.transporting_speed * self.dt)
    #         if self.load == self.capacity:
    #             self.full = True
    #     elif self.unloading:
    #         self.full = False
    #         self.load = max(0, self.load - self.unloading_speed * self.dt)
    #         if self.load == 0:
    #             self.empty = True
        
    #     if np.linalg.norm(self.vel) > self.max_velocity:
    #         self.vel = self.max_velocity * self.vel / np.linalg.norm(self.vel)
    #     self.pos += self.vel * self.dt
    #     self.load_percent = self.load / self.capacity
            
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
                

def test_harv():
    # field = Field(field_length=500, field_width=50, working_width=3.2, depot_pos="random")
    field = Field(depot_pos="random")
    print(field.depot, field.depot_nav_point, field.depot_nav_point_ops)
    print(field.nav_points)
    har1 = Harvester(field)
    har1.assign_nav_points(working_lines=[2,3,4])
    print(har1.nav_points)
    for i in range(5600):
        if i % 10 == 0: # 每隔一秒打印一次位置
            print(i, har1.pos, har1.load, har1.time, har1.last_trans_time, har1.able_to_trans, har1.cur_working_line)
        har1.update_state()
    har1.has_a_trans = True
    for i in range(400):
        if i % 10 == 0: 
            print(i, har1.pos, har1.load, har1.time, har1.last_trans_time, har1.able_to_trans, har1.cur_working_line)
        har1.update_state()
    har1.has_a_trans = False
    for i in range(5000):
        if i % 10 == 0:
            print(i, har1.pos, har1.load, har1.time, har1.last_trans_time, har1.able_to_trans, har1.cur_working_line)
        har1.update_state()

def test_trans():
    # field = Field(field_length=500, field_width=50, working_width=3.2, depot_pos="random")
    field = Field(depot_pos="random")
    # print(field.depot, field.depot_nav_point, field.depot_nav_point_ops)
    # print(field.nav_points)
    har1 = Harvester(field)
    har1.assign_nav_points(working_lines=[2,3,4])
    print(har1.nav_points)
    for i in range(5600):
        # if i % 10 == 0: # 每隔一秒打印一次位置
            # print(i, har1.pos, har1.load, har1.time, har1.last_trans_time, har1.able_to_trans, har1.cur_working_line)
        har1.update_state()
    har1.has_a_trans = True
    for i in range(400):
        # if i % 10 == 0: 
            # print(i, har1.pos, har1.load, har1.time, har1.last_trans_time, har1.able_to_trans, har1.cur_working_line)
        har1.update_state()
    har1.has_a_trans = False

    print(har1.nav, har1.old_nav_point, har1.curr_nav_point)
    trans1 = Transporter(field)
    trans1.assign_search_nav_points(har1)
    print(trans1.nav_points)


if __name__ == "__main__":
    # test_harv()
    test_trans()

