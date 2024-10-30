import numpy as np
import math
import seaborn as sns
from numpy import random

# 定义农田类
class Field(object):
    def __init__(self, field_length = 310, field_width = 100, working_width = 3, yeild_per_m2 = 0.8, headland_width = 5, depot_pos = "random"):
        # farm properties
        self.field_length = field_length    # 农田长度（m）
        self.field_width = field_width      # 农田宽度（m）
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
        self.num_working_lines = math.ceil(self.field_width / self.working_width) # 向上取整
        self.working_line_length = self.field_length - self.headland_width 
        self.crop_x_range = [0, self.field_width]
        self.crop_y_range = [self.headland_width, self.field_length - self.headland_width]

        self.compute_nav_points()

    def compute_nav_points(self):
        # 计算农田的导航点
        # 返回: [2, num_working_lines, 2] numpy array. 
        x_coor_1 = [self.working_width/2 + i * self.working_width for i in range(self.num_working_lines - 1)]
        x_coor_1.append(self.field_width - self.working_width / 2)  #最后一个作业行用来封边
        y_coor_1 = [self.headland_width/2 for _ in range(self.num_working_lines)]
        x_coor_2 = x_coor_1[:]
        y_coor_2 = [self.field_length - self.headland_width/2 for _ in range(self.num_working_lines)]
        self.nav_points = np.array([[x_coor_1, y_coor_1],[x_coor_2, y_coor_2]]).transpose(0,2,1)

# 定义收割机类
class Harvester(object):
    def __init__(self, field: Field, speed = 1.5, capacity = 1800, transporting_speed = 200, dt = 0.1):
        self.id = 0
        self.name = ''
        self.color = None
        self.field = field
        self.speed = float(speed)   # 收割机的运行速度(m/s)
        self.capacity = float(capacity)    # 容量，以kg为单位，1800kg约折合2.4立方米小麦。
        self.transporting_speed = transporting_speed    # 转运速度，kg/s
        self.dt = dt
        self.yeild_per_second = self.field.yeild_per_m2 * self.field.working_width * self.speed    # 计算得到收割机以最大速度行驶时每秒的产量(kg/s)，default得到的是3.6
        
        self.pos = self.field.depot  # 所有收割机都初始化在粮仓位置
        self.time = 0.0 # 记录当前时刻
        self.last_trans_time = 0.0  # 记录上一次转运完成时间
        self.total_wait_time = 0.0  # 记录总的等待时间
        self.new_wait_time = 0.0    # 记录新增加的等待时间
        self.load = 0.0 # 当前收割机的总负载
        self.cur_working_line = -1  # 开始时的作业行标识为-1
        self.last_working_line = -1 # 上一个作业行标记为-1
        self.load_percent = self.load / self.capacity
        self.complete_traj = False
        self.has_a_trans = False
        self.able_to_trans = False
        self.chosen = False

    def dispatch_tasks(self, working_lines):
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
        return self.field.crop_x_range[0] < self.pos[0] < self.field.crop_x_range[1] and self.field.crop_y_range[0] <= self.pos[1] <= self.field.crop_y_range[1]

    def move(self):
        if self.complete_traj: 
            return
        
        # 更新自身位置
        pred_new_pos = self.pos + self.dir * self.dt * self.speed
        while (pred_new_pos - self.old_nav_point) @ (pred_new_pos - self.curr_nav_point) > 0: 
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

        # 更新作业行信息
        if 0 <= self.nav - 2 < len(self.working_lines) * 2:
            if self.cur_working_line != self.working_lines[math.floor((self.nav - 2) / 2)]:
                self.last_working_line = self.cur_working_line
                self.cur_working_line = self.working_lines[math.floor((self.nav - 2) / 2)]
        elif self.nav - 2 >= len(self.working_lines) * 2:
            self.last_working_line = self.working_lines[-1]
            self.cur_working_line = -1

        # 更新粮仓储量
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
        old_wait_time = self.total_wait_time

        if self.complete_traj:
            self.able_to_trans = False

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

        self.new_wait_time = self.total_wait_time - old_wait_time

    def get_state(self):    # 13 dim
        # state = np.concatenate([[self.id], self.pos, self.dir, [self.capacity - self.load], [self.load_percent], \
        #                         [int(self.able_to_trans)], [self.cur_working_line], [self.last_trans_time], [self.time - self.last_trans_time]])
        state = np.concatenate([[self.id], [self.speed], np.array(self.pos) / 100, self.dir, [(self.capacity - self.load) / 100],\
                                 np.array(self.curr_nav_point) / 100, np.array(self.old_nav_point) / 100, [float(self.chosen)], \
                                [float(self.complete_traj)], [(self.time - self.last_trans_time) / 100]])
        return state

class Transporter(object):
    def __init__(self, field: Field, speed = 6, capacity = 8000, transporting_speed = 200, dt = 0.1, pos_error = 2):
        self.id = 0
        self.name = ''
        self.color = None
        self.field = field
        self.dt = dt
        self.transporting_speed = transporting_speed    # 转运速度，kg/s，需要和harvester的转运速度一致
        self.capacity = float(capacity) # 容量，单位kg，运粮车的容量要明显大于收割机
        self.speed = float(speed)   # 行驶速度，单位m/s
        self.pos_error = pos_error  # 认为收割机和运粮车相距多远即可开始转运。需要根据二者速度和dt计算：(v_1 + v_2) * dt / 2
        
        self.total_trip = 0.0   #总行驶路程，单位m
        self.trans_times: int = 0   #总转运次数
        self.nav_points =[]
        self.pos = self.field.depot # 运粮车初始位置在粮仓
        self.dir = np.zeros(2)
        self.load = 0.0
        self.load_percent = self.load / self.capacity
        self.has_dispatch_task = False  # 当前是否有调运任务。调运任务包括返回机库卸粮和前往指定收割机转运
        # 返回机库卸载粮食
        self.returning_to_depot = False # 当前是否在返回机库
        self.unloading = False  # 当前是否在机库卸粮
        # 前往指定收割机分成三个阶段：寻找收割机，转运，回到地头
        self.searching_for_harv = False
        self.transporting = False
        self.returning_to_headland = False
        self.last_nav_point = np.zeros(2)   # 保存上一个导航点
        self.serving_harv = None

        self.new_trip_len = 0.0
        self.new_trans_times = 0
    
    def get_state(self):    # 5 dim
        # state = np.concatenate([[self.id], self.pos, self.dir, [self.capacity - self.load], [self.load_percent], \
        #                         [float(self.has_dispatch_task)], [float(self.returning_to_depot)], [float(self.unloading)], \
        #                         [float(self.searching_for_harv)], [float(self.transporting)], [float(self.returning_to_headland)]])
        state = np.concatenate([[self.speed], np.array(self.pos) / 100, [(self.capacity - self.load) / 100], [float(self.has_dispatch_task)]])
        return state

    def add_nav_point(self, point):
        assert point.size == 2, "The point must has 2 dimensions."
        if len(self.nav_points) > 0 and np.all(self.nav_points[-1] == point): return    # 最后一个导航点和新导航点不重合
        self.nav_points.append(point)
        assert point[0] == self.nav_points[-1][0] or point[1] == self.nav_points[-1][1], "The harvester can only move Horizontally and Vertically."

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

    def reset_nav_and_dir(self):
        assert len(self.nav_points) >= 2, "The nav_points list must have at least two points."
        self.nav = 1    # curr nav point
        self.old_nav_point = self.nav_points[self.nav - 1]
        self.curr_nav_point = self.nav_points[self.nav]
        self.dir = (self.curr_nav_point - self.old_nav_point) / np.linalg.norm(self.curr_nav_point - self.old_nav_point)

    # 为returning_to_depot过程分配导航点
    def assign_return_depot_nav_points(self):
        assert self.nav_points == [], "Cannot assign nav points when former task has not finished."
        if np.all(self.pos == self.field.depot):     # 如果当前就在depot，直接return
            self.returning_to_depot = False
            return
        self.returning_to_depot = True
        self.add_nav_point(self.pos)
        if self.in_same_side(self.field.depot_nav_point):
            self.add_nav_point(self.field.depot_nav_point)
        else:
            self.add_nav_point(self.field.depot_nav_point_ops)
        self.add_nav_point(self.field.depot)
        self.reset_nav_and_dir()

    # 为searching_for_harv分配导航点
    def assign_search_nav_points(self, harv: Harvester):
        assert self.serving_harv == None, "Cannot do tasks while other tasks is doing"
        assert self.nav_points == [], "Cannot assign nav points when former task has not finished."
        if harv.chosen or harv.cur_working_line == -1:
            # 收割机已经被选择，或者刚开始工作，或者已经完成所有作业返回仓库，则不进行转运。保证收割机的 nav >= 2
            # print("Current harvester has just started or has completed task.")
            self.searching_for_harv = False
            self.serving_harv = None
            harv.chosen = False
            return
        assert harv.nav >= 2, "Strange case ???"
        self.searching_for_harv = True
        self.serving_harv = harv
        harv.chosen = True
        self.add_nav_point(self.pos)    # 每次调运任务结束都保证运粮车在地头导航点
        if np.all(self.pos == self.field.depot):
            self.add_nav_point(self.field.depot_nav_point)  # 如果出发是在机库，加入机库导航点
        harv_old_nav_point = harv.old_nav_point
        harv_curr_nav_point = harv.curr_nav_point
        # 当前位置和收割机上一个导航点在同侧，则依此加入上一个和当前导航点
        if self.in_same_side(harv_old_nav_point):  
            self.add_nav_point(harv_old_nav_point)
            self.add_nav_point(harv_curr_nav_point)
            # 再多加上两个导航点
            if harv.nav + 1 < len(harv.nav_points):
                self.add_nav_point(harv.nav_points[harv.nav + 1])
            if harv.nav + 2 < len(harv.nav_points):
                self.add_nav_point(harv.nav_points[harv.nav + 2])
        # old_nav_point的上一个导航点和农机在同侧，则加入上一个导航点
        elif harv.nav - 2 >= 0 and self.in_same_side(harv.nav_points[harv.nav - 2]):
            self.add_nav_point(harv.nav_points[harv.nav - 2])
            self.add_nav_point(harv_old_nav_point)
            self.add_nav_point(harv_curr_nav_point)
            if harv.nav + 1 < len(harv.nav_points):
                self.add_nav_point(harv.nav_points[harv.nav + 1])
            if harv.nav + 2 < len(harv.nav_points):
                self.add_nav_point(harv.nav_points[harv.nav + 2])
            if harv.nav + 3 < len(harv.nav_points):
                self.add_nav_point(harv.nav_points[harv.nav + 3])
        # old_nav_point的上两个个导航点和农机在同侧，则加入导航点
        elif harv.nav >= 3 and self.in_same_side(harv.nav_points[harv.nav - 3]):
            self.add_nav_point(harv.nav_points[harv.nav - 3])
            self.add_nav_point(harv.nav_points[harv.nav - 2])
            self.add_nav_point(harv_old_nav_point)
            self.add_nav_point(harv_curr_nav_point)
            if harv.nav + 1 < len(harv.nav_points):
                self.add_nav_point(harv.nav_points[harv.nav + 1])
            if harv.nav + 2 < len(harv.nav_points):
                self.add_nav_point(harv.nav_points[harv.nav + 2])
            if harv.nav + 3 < len(harv.nav_points):
                self.add_nav_point(harv.nav_points[harv.nav + 3])
            if harv.nav + 4 < len(harv.nav_points):
                self.add_nav_point(harv.nav_points[harv.nav + 4])
        else:
            self.searching_for_harv = False
            self.serving_harv = None
            self.nav_points = []
            harv.chosen = False
            # print("STRANGE CASE")
            return
        
        self.reset_nav_and_dir()
        return
    
    def in_head_lines(self):
        return self.pos[1] == self.field.headland_width/2 or self.pos[1] == self.field.field_length - self.field.headland_width / 2
    
    def assign_return_head_nav_points(self):
        assert self.nav_points == [], "Cannot assign nav points when former task has not finished."
        if self.in_head_lines():
            self.returning_to_headland = False
            return
        self.returning_to_headland = True
        self.add_nav_point(self.pos)
        self.add_nav_point(self.last_nav_point)
        self.reset_nav_and_dir()

    def update_state(self):
        self.check_dispatching()

        old_total_trip = self.total_trip
        old_trans_time = self.trans_times
        if self.searching_for_harv:
            assert self.serving_harv != None, "The harvester must be provided."
            assert len(self.nav_points) >= 2, "The nav_points list must have at least two points."
            pred_new_pos = self.pos + self.dir * self.dt * self.speed
            self.total_trip += self.dt * self.speed # 增加总路程
            while (pred_new_pos - self.old_nav_point) @ (pred_new_pos - self.curr_nav_point) > 0: 
                left_dis = np.linalg.norm(pred_new_pos - self.curr_nav_point)
                self.nav += 1
                if self.nav == len(self.nav_points):    # 轨迹结束都没有找到收割机，返回
                    self.pos =self.nav_points[-1]
                    print("Cannot find the harvester!!")
                    self.nav_points = []
                    self.searching_for_harv = False
                    # self.returning_to_headland = True
                    self.serving_harv.chosen = False
                    self.serving_harv = None
                    return
                self.curr_nav_point = self.nav_points[self.nav]
                self.old_nav_point = self.nav_points[self.nav - 1]
                self.dir = (self.curr_nav_point - self.old_nav_point) / np.linalg.norm(self.curr_nav_point - self.old_nav_point)
                pred_new_pos = self.old_nav_point + left_dis * self.dir
            self.pos = pred_new_pos
            # 判断是否找到收割机
            if np.linalg.norm(self.pos - self.serving_harv.pos) < self.pos_error:
                self.nav_points = []
                self.searching_for_harv = False
                self.transporting = True
                self.last_nav_point = self.old_nav_point
                # self.assign_return_head_nav_points()
                if self.serving_harv.able_to_trans and self.load_percent != 1.0:
                    self.trans_times += 1

        elif self.transporting:
            assert self.serving_harv != None, "The harvester must be provided."
            if self.serving_harv.able_to_trans and self.load_percent != 1.0:
                self.serving_harv.has_a_trans = True
                self.load = min(self.capacity, self.load + self.transporting_speed * self.dt)
                self.load_percent = self.load / self.capacity
            elif self.serving_harv.able_to_trans == False:   # 收割机空了，或者本身就因为间隔太短无法卸粮食
                self.serving_harv.has_a_trans = False
                self.serving_harv.chosen = False
                self.serving_harv = None
                self.transporting = False
                self.assign_return_head_nav_points()
            else:   # self.load_percentage == 1.0，在这里可以加负的奖励
                self.serving_harv.has_a_trans = False
                self.serving_harv.chosen = False
                self.serving_harv = None
                self.transporting = False
                self.assign_return_head_nav_points()

        elif self.returning_to_headland:
            assert self.serving_harv == None, "No harvester should be provided."
            assert len(self.nav_points) >= 2, "The nav_points list must have at least two points."
            pred_new_pos = self.pos + self.dir * self.dt * self.speed
            self.total_trip += self.dt * self.speed
            while (pred_new_pos - self.old_nav_point) @ (pred_new_pos - self.curr_nav_point) > 0:  # 不支持连续一个time step拐弯多次，要设置dt足够小。
                left_dis = np.linalg.norm(pred_new_pos - self.curr_nav_point)
                self.nav += 1
                if self.nav == len(self.nav_points):    # 结束导航
                    self.pos =self.nav_points[-1]
                    self.nav_points = []
                    self.returning_to_headland = False
                    return
                self.curr_nav_point = self.nav_points[self.nav]
                self.old_nav_point = self.nav_points[self.nav - 1]
                self.dir = (self.curr_nav_point - self.old_nav_point) / np.linalg.norm(self.curr_nav_point - self.old_nav_point)
                pred_new_pos = self.old_nav_point + left_dis * self.dir
            self.pos = pred_new_pos

        elif self.returning_to_depot:
            assert len(self.nav_points) >= 2, "The nav_points list must have at least two points."
            pred_new_pos = self.pos + self.dir * self.dt * self.speed
            self.total_trip += self.dt * self.speed # 增加总行程
            while (pred_new_pos - self.old_nav_point) @ (pred_new_pos - self.curr_nav_point) > 0: 
                left_dis = np.linalg.norm(pred_new_pos - self.curr_nav_point)
                self.nav += 1
                if self.nav == len(self.nav_points):    # 结束导航
                    self.pos =self.nav_points[-1]
                    self.nav_points = []
                    self.returning_to_depot = False
                    self.unloading = True
                    return
                self.curr_nav_point = self.nav_points[self.nav]
                self.old_nav_point = self.nav_points[self.nav - 1]
                self.dir = (self.curr_nav_point - self.old_nav_point) / np.linalg.norm(self.curr_nav_point - self.old_nav_point)
                pred_new_pos = self.old_nav_point + left_dis * self.dir
            self.pos = pred_new_pos

        elif self.unloading:
            assert np.all(self.pos == self.field.depot), "The harvester is not in the depot!"
            self.load = max(0, self.load - self.transporting_speed * self.dt)
            self.load_percent = self.load / self.capacity
            if self.load_percent == 0:
                self.unloading = False

        self.new_trip_len = self.total_trip - old_total_trip
        self.new_trans_times = self.trans_times - old_trans_time
        # else:
        #     return
        
    def check_dispatching(self):
        if self.searching_for_harv or self.transporting or self.returning_to_headland or self.returning_to_depot or self.unloading:
            self.has_dispatch_task = True
        else:
            self.has_dispatch_task = False
    
    def set_action(self, action, harv = None):
        self.check_dispatching()
        # 如果有调运任务，直接返回
        if self.has_dispatch_task:
            return
        # action == 0, 不采取任何动作
        if action == 0: 
            return
        # action == 1， 返回depot卸载粮食
        elif action == 1:   #1，前往depot卸载粮食
            self.assign_return_depot_nav_points()
        # action >= 2，前往其它收割机卸载粮食
        elif action >= 2: 
            self.assign_search_nav_points(harv)
            # 设置转运成本
            
# multi-agent world
class World(object):
    def __init__(self, args):
        # farm properties
        self.world_step = 0
        # self.field = Field(field_width=99.01)
        self.field = Field()

        self.color_list = np.array([[1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0.6, 0.34],
        [1, 0, 1],
        [0, 1, 1],
        [0, 0, 0], 
        [1, 0.5, 0],
        [1, 0.9, 0.8]  ])

        self.dt = args.dt
        self.decision_dt = args.decision_dt
        self.wait_time_factor = args.wait_time_factor
        self.distance_factor = args.distance_factor
        self.trans_times_factor = args.trans_times_factor
        self.num_harvester = args.num_harvester
        self.num_transporter = args.num_transporter

        self.episode_length = args.episode_length
        self.shared_reward = args.shared_reward

        self.harv_vmin = args.harv_vmin
        self.harv_vmax = args.harv_vmax
        self.harv_capmin = args.harv_capmin
        self.harv_capmmax = args.harv_capmax
        self.trans_vmin = args.trans_vmin
        self.trans_vmax = args.trans_vmax
        self.trans_capmin = args.trans_capmin
        self.trans_capmax = args.trans_capmax

        self.harvesters = [Harvester(field=self.field, speed=np.random.uniform(self.harv_vmin, self.harv_vmax), \
                                     capacity=int(np.random.uniform(self.harv_capmin, self.harv_capmmax)) * 100, \
                                     dt = self.dt) for i in range(self.num_harvester)]
        self.transporters = [Transporter(field=self.field, speed=np.random.uniform(self.trans_vmin, self.trans_vmax), \
                                         capacity=int(np.random.uniform(self.trans_capmin, self.trans_capmax)) * 100, \
                                         dt = self.dt) for i in range(self.num_transporter)]
        for i, harv in enumerate(self.harvesters):
            harv.id = i
            harv.name = 'harvester %d' % i
        for j, harv in enumerate(self.transporters):
            harv.id = j + self.num_harvester
            harv.name = 'transporter %d' % j
        self.assign_agent_colors()

    def reset(self):
        self.field = None
        self.harvesters.clear()
        self.transporters.clear()
        self.world_step = 0

        self.field = Field()
        self.harvesters = [Harvester(field=self.field, speed=np.random.uniform(self.harv_vmin, self.harv_vmax), \
                                     capacity=int(np.random.uniform(self.harv_capmin, self.harv_capmmax)) * 100, \
                                     dt = self.dt) for _ in range(self.num_harvester)]
        self.transporters = [Transporter(field=self.field, speed=np.random.uniform(self.trans_vmin, self.trans_vmax), \
                                         capacity=int(np.random.uniform(self.trans_capmin, self.trans_capmax)) * 100, \
                                         dt = self.dt) for _ in range(self.num_transporter)]
        for i, harv in enumerate(self.harvesters):
            harv.id = i
            harv.name = 'harvester %d' % i
        for j, harv in enumerate(self.transporters):
            harv.id = j + self.num_harvester
            harv.name = 'transporter %d' % j
        self.assign_agent_colors()
        
    def assign_agent_colors(self, color_mode="random"):
        if color_mode == "fixed":
            harv_colors = [(0.25, 0.75, 0.25)] * self.num_harvester
            for color, agent in zip(harv_colors, self.harvesters):
                agent.color = color
            trans_colors = [(0.75, 0.25, 0.25)] * self.num_transporter
            for color, agent in zip(trans_colors, self.transporters):
                agent.color = color
        elif color_mode == "random":
            i = 0
            for harv in self.harvesters:
                harv.color = self.color_list[i % len(self.color_list)]
                i += 1
            for trans in self.transporters:
                trans.color = self.color_list[i % len(self.color_list)]
                i += 1
        else:
            raise NotImplementedError

    # update state of the world
    def step(self):
        self.world_step += 1
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
    har1.reset_harv(working_lines=[2,3,4])
    print(har1.nav_points)
    for i in range(5600):
        if i % 10 == 0: # 每隔一秒打印一次位置
            state = har1.get_state()
            print(i, state)
            print(i, har1.pos, har1.load, har1.time, har1.last_trans_time, har1.able_to_trans, har1.cur_working_line)
        har1.update_state()
    har1.has_a_trans = True
    for i in range(400):
        if i % 10 == 0: 
            print(i, har1.pos, har1.load, har1.time, har1.last_trans_time, har1.able_to_trans, har1.cur_working_line)
        har1.update_state()
    # har1.has_a_trans = False
    # for i in range(5000):
    #     if i % 10 == 0:
    #         print(i, har1.pos, har1.load, har1.time, har1.last_trans_time, har1.able_to_trans, har1.cur_working_line)
    #     har1.update_state()

def test_trans():
    # field = Field(field_length=500, field_width=50, working_width=3.2, depot_pos="random")
    field = Field(depot_pos="random")
    # print(field.depot, field.depot_nav_point, field.depot_nav_point_ops)
    # print(field.nav_points)
    har1 = Harvester(field)
    har1.reset_harv(working_lines=[2,3,4])
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

    print(har1.nav, har1.old_nav_point, har1.curr_nav_point, har1.pos)
    trans1 = Transporter(field)
    trans1.assign_search_nav_points(har1)
    print(trans1.nav_points)
    print(trans1.get_state())


if __name__ == "__main__":
    test_harv()
    # test_trans()
    # field = Field()
    # harv = Harvester(field)
    # field.field_width = 200
    # print(harv.field.field_width)

